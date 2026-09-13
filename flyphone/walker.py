"""Controlador de bajo nivel: la política de marcha preentrenada de flybody.

`WalkPolicy` es la red DMPO de imitación de marcha (Vaxenburg et al. 2025) portada a
numpy desde el SavedModel de TensorFlow (`scripts/export_walk_policy.py`), así corre en
macOS/arm64 sin TF 2.8 ni dm-reverb. Espera las 10 observaciones propioceptivas y
vestibulares del walker más una trayectoria de referencia de 65 pasos
(`ref_displacement` (65, 3) y `ref_root_quat` (65, 4), egocéntricas) y devuelve las 59
señales de actuadores.

`SteerSelfieGym` la usa como controlador de bajo nivel: la política de alto nivel (PPO)
emite un comando de rumbo (velocidad, giro) cada `steer_every` pasos de control, el
wrapper sintetiza la trayectoria de referencia que ese rumbo implica y la política de
marcha produce los actuadores. Es el esquema de "reutilización de controlador" del
paper de flybody, con la trayectoria generada al vuelo en lugar de leída de datos.
"""
import os
import numpy as np
import gymnasium as gym

from flybody.quaternions import get_dquat_local
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

from flyphone.arena import SCREEN_TOP
from flyphone.gym_env import FlyPhoneGym

ASSET = os.path.join(os.path.dirname(__file__), 'assets', 'walk_policy.npz')
FUTURE_STEPS = 64        # la política ve el punto actual + 64 futuros de la referencia
CONTROL_DT = 2e-3
REF_HEIGHT = 0.1278      # altura del tórax en las trayectorias de referencia (cm sobre el piso)


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.)))


class WalkPolicy:
    """Forward pass en numpy de la política de marcha: batch_concat(obs ordenadas por
    clave) → Linear → LayerNorm → tanh → 3×(Linear → ELU) → Linear (media)."""

    KEYS = ('walker/accelerometer', 'walker/actuator_activation', 'walker/appendages_pos',
            'walker/force', 'walker/gyro', 'walker/joints_pos', 'walker/joints_vel',
            'walker/ref_displacement', 'walker/ref_root_quat', 'walker/touch',
            'walker/velocimeter', 'walker/world_zaxis')

    def __init__(self, path: str = ASSET):
        with np.load(path) as f:
            self._w = {k: f[k] for k in f.files}
        self._n_hidden = int(self._w['n_hidden'])

    def __call__(self, obs: dict) -> np.ndarray:
        w = self._w
        x = np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self.KEYS])
        h = x @ w['in_w'] + w['in_b']
        h = (h - h.mean()) / np.sqrt(h.var() + 1e-5) * w['ln_scale'] + w['ln_offset']
        h = np.tanh(h)
        for i in range(self._n_hidden):
            h = _elu(h @ w[f'h{i}_w'] + w[f'h{i}_b'])
        return h @ w['mean_w'] + w['mean_b']


def reference_from_steering(fly_pos, fly_quat, speed, yaw_speed, ground_z=SCREEN_TOP):
    """Trayectoria de referencia egocéntrica (65 pasos) para "avanza a `speed` cm/s
    girando a `yaw_speed` rad/s" desde la pose actual de la mosca."""
    w, x, y, z = fly_quat
    heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    qpos, _ = constant_speed_trajectory(
        n_steps=FUTURE_STEPS + 1, speed=speed, yaw_speed=yaw_speed,
        init_pos=(fly_pos[0], fly_pos[1], ground_z + REF_HEIGHT),
        init_heading=heading, control_timestep=CONTROL_DT)
    return qpos[:, :3], qpos[:, 3:]


class SteerSelfieGym(FlyPhoneGym):
    """TakeSelfie con la política de marcha como controlador de bajo nivel.

    Acción de alto nivel en [-1, 1]²: velocidad hacia delante en [0, max_speed] cm/s y
    giro en [-max_yaw, max_yaw] rad/s, mantenida `steer_every` pasos de control.
    La observación es la misma que en `FlyPhoneGym` (290); la recompensa se suma
    sobre los pasos de control del comando."""

    def __init__(self, max_speed: float = 3., max_yaw: float = 6., steer_every: int = 5,
                 policy_path: str = ASSET, **kwargs):
        # La política de marcha se entrenó con la fricción de garras de flybody; con la
        # claw_friction=1.0 de TakeSelfie tropieza y vuelca. Con física a 0.4 ms anda bien
        # (igual que a los 0.2 ms de flybody, que costarían el doble).
        kwargs.setdefault('claw_friction', None)
        super().__init__(**kwargs)
        self._walk = WalkPolicy(policy_path)
        self._max_speed, self._max_yaw, self._steer_every = max_speed, max_yaw, steer_every
        spec = self._env.action_spec()
        self._act_lo, self._act_hi = spec.minimum, spec.maximum
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self._obs = None

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        ts = self._env.reset()
        self._obs = ts.observation
        return self._flatten(self._obs), {}

    def low_level_action(self, speed, yaw_speed):
        """Actuadores que la política de marcha manda para un rumbo dado (obs actual)."""
        walker = self._env.task.walker
        pos, quat = walker.get_pose(self._env.physics)
        ref_pos, ref_quat = reference_from_steering(pos, quat, speed, yaw_speed)
        obs = dict(self._obs)
        obs['walker/ref_displacement'] = walker.transform_vec_to_egocentric_frame(
            self._env.physics, ref_pos - pos)
        dquat = get_dquat_local(quat, ref_quat)
        # q y -q son la misma rotación, pero para la red no: con yaw de spawn > π el
        # cuaternión raíz tiene w < 0 y el delta salía [-1, 0, 0, 0] → la mosca volcaba.
        obs['walker/ref_root_quat'] = dquat * np.sign(dquat[:, :1] + 1e-12)
        return np.clip(self._walk(obs), self._act_lo, self._act_hi)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1., 1.)
        speed = (a[0] + 1.) / 2. * self._max_speed
        yaw_speed = a[1] * self._max_yaw
        total, task = 0., self._env.task
        for _ in range(self._steer_every):
            ts = self._env.step(self.low_level_action(speed, yaw_speed))
            self._obs = ts.observation
            total += float(ts.reward)
            if ts.last():
                break
        info = {'pressed': task.pressed, 'wrong_button': task.wrong_button,
                'dist': task._fly_button_dist(self._env.physics)}
        terminated = bool(ts.last() and (ts.discount == 0. or task.pressed))
        truncated = bool(ts.last() and not terminated)
        return self._flatten(self._obs), total, terminated, truncated, info
