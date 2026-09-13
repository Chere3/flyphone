"""Wrapper Gymnasium del entorno TakeSelfie (observación aplanada)."""
import numpy as np
import gymnasium as gym

from flyphone.task import make_env

EYE_KEYS = ("walker/left_eye", "walker/right_eye")


class FlyPhoneGym(gym.Env):
    """Observación: vector de 290 (claves ordenadas alfabéticamente). Con `vision=True`
    es un Dict: `vec` (propiocepción, sin `button_displacement`) y `eyes`, los dos ojos
    compuestos en gris, uint8 (2, 32, 32)."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0, time_limit: float = 3., capture_photos: bool = False,
                 spawn_radius=(0.8, 1.4), render_camera: str = "phone_side",
                 camera_app: bool = False, vision: bool = False, **task_kwargs):
        self._env = make_env(random_state=np.random.RandomState(seed),
                             time_limit=time_limit, capture_photos=capture_photos,
                             spawn_radius=spawn_radius, camera_app=camera_app,
                             vision=vision, **task_kwargs)
        self._render_camera = render_camera
        self._vision = vision
        spec = self._env.action_spec()
        # La política actúa en [-1, 1]; se mapea linealmente al rango real de cada actuador.
        self._act_mid = (spec.maximum + spec.minimum) / 2.
        self._act_half = (spec.maximum - spec.minimum) / 2.
        self.action_space = gym.spaces.Box(-1., 1., shape=spec.shape, dtype=np.float32)
        obs_spec = self._env.observation_spec()
        self._obs_keys = sorted(k for k in obs_spec if k not in EYE_KEYS)
        n = sum(int(np.prod(obs_spec[k].shape)) for k in self._obs_keys)
        vec = gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        if vision:
            h, w = obs_spec[EYE_KEYS[0]].shape[:2]
            self.observation_space = gym.spaces.Dict(
                {"vec": vec, "eyes": gym.spaces.Box(0, 255, shape=(2, h, w), dtype=np.uint8)})
        else:
            self.observation_space = vec

    @property
    def task(self):
        return self._env.task

    @property
    def physics(self):
        return self._env.physics

    def _flatten(self, obs):
        vec = np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel()
                              for k in self._obs_keys])
        if not self._vision:
            return vec
        eyes = np.stack([obs[k].mean(axis=-1) for k in EYE_KEYS]).astype(np.uint8)
        return {"vec": vec, "eyes": eyes}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ts = self._env.reset()
        return self._flatten(ts.observation), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1., 1.)
        ts = self._env.step(self._act_mid + self._act_half * a)
        task = self._env.task
        obs = self._flatten(ts.observation)
        info = {"pressed": task.pressed, "wrong_button": task.wrong_button,
                "dist": task._fly_button_dist(self._env.physics)}
        terminated = bool(ts.last() and (ts.discount == 0. or task.pressed))
        truncated = bool(ts.last() and not terminated)
        return obs, float(ts.reward), terminated, truncated, info

    def render(self):
        return self._env.physics.render(camera_id=self._render_camera, width=480, height=360)
