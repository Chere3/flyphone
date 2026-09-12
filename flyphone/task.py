"""Tarea: la mosca camina sobre la pantalla y pisa el botón para tomar una selfie."""
# ruff: noqa: F821

import numpy as np
from dm_control import composer, mujoco
from dm_control.composer.observation import observable

from flybody.fruitfly import fruitfly
from flybody.tasks import base as flybody_base
from flybody.tasks.base import Walking

# flybody usa 0.2 ms de paso de física para caminar. 0.4 ms es estable en esta tarea
# y ~45% más rápido. Walking lo lee de esta constante del módulo al construir la tarea.
PHYSICS_TIMESTEP = 4e-4
flybody_base._WALK_PHYSICS_TIMESTEP = PHYSICS_TIMESTEP
from flybody.tasks.constants import _TERMINAL_ANGVEL, _TERMINAL_LINVEL

from flyphone.arena import (PhoneArena, BUTTON_TRAVEL, PHONE_HALF, SCREEN_TOP)


class TakeSelfie(Walking):
    """La mosca aparece en la pantalla cerca del botón y debe presionarlo.

    Recompensa por paso: acercamiento al botón (0..1) + bono grande al presionar.
    Al presionar se captura una foto desde la cámara selfie y el episodio termina
    con descuento 1 (terminación "buena").
    """

    def __init__(self,
                 spawn_radius: tuple[float, float] = (0.8, 1.4),  # mín > radio botón + alcance de patas
                 press_fraction: float = 0.5,
                 press_bonus: float = 100.,
                 approach_scale: float = 10.,
                 photo_size: tuple[int, int] = (320, 240),
                 capture_photos: bool = True,
                 claw_friction: float = 1.0,
                 **kwargs):
        super().__init__(add_ghost=False, **kwargs)
        self._spawn_radius = spawn_radius
        self._press_depth = press_fraction * BUTTON_TRAVEL
        self._press_bonus = press_bonus
        self._approach_scale = approach_scale
        self._photo_size = photo_size
        self._capture_photos = capture_photos
        self._prev_dist = None
        self._pressed = False
        self._last_photo = None
        self._photos_taken = 0

        if claw_friction is not None:
            self._walker.mjcf_model.find(
                'default', 'adhesion-collision').geom.friction = (claw_friction,)

        # Observables de la tarea: vector egocéntrico al botón y estado del botón.
        self._walker.observables.add_observable('button_displacement',
                                                self.button_displacement)
        self._walker.observables.add_observable('button_state',
                                                self.button_state)

    # --- observables -------------------------------------------------------
    @composer.observable
    def button_displacement(self):
        def get(physics):
            fly_pos, _ = self._walker.get_pose(physics)
            btn = physics.bind(self._arena.button_body).xpos
            return self._walker.transform_vec_to_egocentric_frame(
                physics, btn - fly_pos)
        return observable.Generic(get)

    @composer.observable
    def button_state(self):
        def get(physics):
            q = physics.bind(self._arena.button_joint).qpos
            return np.array([-q[0] / BUTTON_TRAVEL])   # 0 = suelto, 1 = a fondo
        return observable.Generic(get)

    # --- ciclo de episodio -------------------------------------------------
    def initialize_episode_mjcf(self, random_state):
        mujoco.set_mjcb_control(None)
        super().initialize_episode_mjcf(random_state)

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._pressed = False
        self._last_photo = None
        # Posición inicial: anillo alrededor del botón, orientación aleatoria.
        btn = physics.bind(self._arena.button_body).xpos
        r = random_state.uniform(*self._spawn_radius)
        ang = random_state.uniform(0, 2 * np.pi)
        yaw = random_state.uniform(0, 2 * np.pi)
        xyz = [btn[0] + r * np.cos(ang), btn[1] + r * np.sin(ang),
               SCREEN_TOP + self._walker.upright_pose.xpos[2]]
        quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.place_fly(physics, xyz, quat)
        physics.bind(self._arena.button_joint).qpos = 0.
        physics.bind(self._wing_joints).qpos = self._wing_springrefs
        self._prev_dist = self._fly_button_dist(physics)

    def place_fly(self, physics, xyz, quat):
        physics.bind(self._root_joint).qpos = np.hstack([xyz, quat])
        physics.bind(self._root_joint).qvel = 0.

    def before_step(self, physics, action, random_state):
        action[np.isnan(action)] = 0.
        super().before_step(physics, action, random_state)

    # --- recompensa y terminación -------------------------------------------
    def _button_depth(self, physics):
        return -physics.bind(self._arena.button_joint).qpos[0]

    def _fly_button_dist(self, physics):
        fly_pos, _ = self._walker.get_pose(physics)
        btn = physics.bind(self._arena.button_body).xpos
        return float(np.linalg.norm((btn - fly_pos)[:2]))

    def get_reward_factors(self, physics):
        return (1.,)   # no se usa; get_reward está sobrescrito.

    def get_reward(self, physics):
        """Shaping potencial (progreso hacia el botón, en cm) + hundimiento parcial
        del botón + bono al presionar. Sin término denso por "estar cerca", para que
        no convenga quedarse junto al botón sin presionarlo."""
        self._should_terminate = self.check_termination(physics)
        dist = self._fly_button_dist(physics)
        reward = self._approach_scale * (self._prev_dist - dist)
        self._prev_dist = dist
        reward += 5. * min(self._button_depth(physics) / BUTTON_TRAVEL, 1.)
        if self._pressed:
            reward += self._press_bonus
        return float(reward)

    def check_termination(self, physics):
        # ¿Se presionó el botón?
        if not self._pressed and self._button_depth(physics) >= self._press_depth:
            self._pressed = True
            self._last_photo = self._capture_photo(physics)
            self._photos_taken += 1
            return True
        fly_pos, _ = self._walker.get_pose(physics)
        fell_off = (abs(fly_pos[0]) > PHONE_HALF[0] or abs(fly_pos[1]) > PHONE_HALF[1]
                    or fly_pos[2] < SCREEN_TOP + 0.03)
        linvel = np.linalg.norm(self._walker.observables.velocimeter(physics))
        angvel = np.linalg.norm(self._walker.observables.gyro(physics))
        return (fell_off or linvel > _TERMINAL_LINVEL or angvel > _TERMINAL_ANGVEL
                or super().check_termination(physics))

    def get_discount(self, physics):
        if self._should_terminate and not self._pressed:
            return 0.
        return 1.

    def _capture_photo(self, physics):
        if not self._capture_photos:
            return None
        w, h = self._photo_size
        return physics.render(camera_id='selfie_cam', width=w, height=h)

    def take_photo(self, physics):
        """Renderiza la selfie a demanda (p. ej. en evaluación)."""
        w, h = self._photo_size
        return physics.render(camera_id='selfie_cam', width=w, height=h)

    # --- acceso externo ------------------------------------------------------
    @property
    def pressed(self):
        return self._pressed

    @property
    def last_photo(self):
        return self._last_photo

    @property
    def photos_taken(self):
        return self._photos_taken


def make_env(random_state=None, time_limit: float = 2., **task_kwargs):
    """Entorno dm_control listo para usar."""
    task = TakeSelfie(walker=fruitfly.FruitFly, arena=PhoneArena(),
                      time_limit=time_limit, joint_filter=0.01,
                      adhesion_filter=0.007, **task_kwargs)
    mujoco.set_mjcb_control(None)
    return composer.Environment(time_limit=time_limit, task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True,
                                recompile_mjcf_every_episode=False)
