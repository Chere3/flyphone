"""Tarea: la mosca camina sobre la pantalla y pisa el botón para tomar una selfie."""
# ruff: noqa: F821

import numpy as np
from dm_control import composer, mujoco
from dm_control.composer.observation import observable
from dm_control.mujoco import wrapper as mj_wrapper

from flybody.fruitfly import fruitfly
from flybody.tasks import base as flybody_base
from flybody.tasks.base import Walking

# flybody usa 0.2 ms de paso de física para caminar. 0.4 ms es estable en esta tarea
# y ~45% más rápido. Walking lo lee de esta constante del módulo al construir la tarea.
PHYSICS_TIMESTEP = 4e-4
flybody_base._WALK_PHYSICS_TIMESTEP = PHYSICS_TIMESTEP
EYE_SIZE = 32                 # píxeles por ojo (alto = ancho)
EYE_UPDATE_CONTROL_STEPS = 4  # los ojos se refrescan cada 4 pasos de control (8 ms, 125 Hz)
from flybody.tasks.constants import _TERMINAL_ANGVEL, _TERMINAL_LINVEL

from flyphone.arena import (PhoneArena, BUTTON_TRAVEL, DECOY_BUTTONS, PHONE_HALF,
                            SCREEN_HALF, SCREEN_TOP)


class TakeSelfie(Walking):
    """La mosca aparece en la pantalla cerca del botón y debe presionarlo.

    Recompensa por paso: acercamiento al botón (0..1) + bono grande al presionar.
    Al presionar se captura una foto desde la cámara selfie y el episodio termina
    con descuento 1 (terminación "buena").

    Si la arena tiene app de cámara (`PhoneArena(camera_app=True)`), pisar un botón
    señuelo termina el episodio con descuento 0 y `wrong_button_penalty`.
    Con `vision=True` se activan las cámaras de los ojos compuestos (32×32 RGB cada
    una) y se apaga `button_displacement`: la mosca tiene que ver el botón.
    """

    def __init__(self,
                 spawn_radius: tuple[float, float] = (0.8, 1.4),  # mín > radio botón + alcance de patas
                 press_fraction: float = 0.5,
                 press_bonus: float = 100.,
                 approach_scale: float = 10.,
                 upright_weight: float = 0.05,
                 angvel_cost: float = 5e-4,
                 flip_threshold: float = 0.3,
                 flip_penalty: float = 10.,
                 time_penalty: float = 0.01,
                 photo_size: tuple[int, int] = (320, 240),
                 capture_photos: bool = True,
                 claw_friction: float = 1.0,
                 wrong_button_penalty: float = 10.,
                 vision: bool = False,
                 **kwargs):
        super().__init__(add_ghost=False, **kwargs)
        self._spawn_radius = spawn_radius
        self._press_depth = press_fraction * BUTTON_TRAVEL
        self._press_bonus = press_bonus
        self._approach_scale = approach_scale
        self._upright_weight = upright_weight
        self._angvel_cost = angvel_cost
        self._flip_threshold = flip_threshold
        self._flip_penalty = flip_penalty
        self._time_penalty = time_penalty
        self._photo_size = photo_size
        self._capture_photos = capture_photos
        self._wrong_button_penalty = wrong_button_penalty
        self._vision = vision
        self._prev_dist = None
        self._pressed = False
        self._wrong_button = False
        self._last_photo = None
        self._photos_taken = 0

        if claw_friction is not None:
            self._walker.mjcf_model.find(
                'default', 'adhesion-collision').geom.friction = (claw_friction,)

        # Observables de la tarea: vector egocéntrico al botón y estado de los botones.
        # Con visión el vector al botón se apaga y entran los ojos (v0.3).
        self._walker.observables.add_observable('button_displacement',
                                                self.button_displacement,
                                                enabled=not vision)
        self._walker.observables.add_observable('button_state',
                                                self.button_state)
        if vision:
            self._add_fast_eyes()

    def _add_fast_eyes(self):
        """Sustituye los observables de ojos de flybody por versiones baratas: sin
        sombras, reflejos ni cielo, sin las mallas cosméticas de la propia mosca
        (grupo 1, 800k vértices) y refrescadas cada EYE_UPDATE_CONTROL_STEPS pasos de
        control en vez de cada subpaso de física. Con las originales un paso costaba
        ~200 ms en macOS/glfw (10 renders con sombras); así cuesta ~4 ms extra."""
        opt = mj_wrapper.MjvOption()
        opt.geomgroup[1] = 0
        every = EYE_UPDATE_CONTROL_STEPS * self.physics_steps_per_control_step
        for name, cam in (('left_eye', self._walker.left_eye), ('right_eye', self._walker.right_eye)):
            self._walker.observables.add_observable(name, observable.MJCFCamera(
                cam, height=EYE_SIZE, width=EYE_SIZE, update_interval=every, scene_option=opt,
                render_flag_overrides=dict(shadow=False, reflection=False, skybox=False)))

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
        """Hundimiento de cada botón, 0 = suelto, 1 = a fondo. El disparador va
        primero; con app de cámara siguen los señuelos (forma (3,))."""
        joints = [self._arena.button_joint] + self._arena.decoy_joints

        def get(physics):
            return np.array([-physics.bind(j).qpos[0] / BUTTON_TRAVEL for j in joints])
        return observable.Generic(get)

    # --- ciclo de episodio -------------------------------------------------
    def initialize_episode_mjcf(self, random_state):
        mujoco.set_mjcb_control(None)
        super().initialize_episode_mjcf(random_state)

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._pressed = False
        self._wrong_button = False
        self._last_photo = None
        # Posición inicial: anillo alrededor del botón, orientación aleatoria.
        # Con app de cámara se rechazan los puntos fuera de la pantalla o a menos
        # de 0.4 cm (alcance de patas) de un señuelo, para no empezar pisándolo.
        btn = physics.bind(self._arena.button_body).xpos
        while True:
            r = random_state.uniform(*self._spawn_radius)
            ang = random_state.uniform(0, 2 * np.pi)
            x, y = btn[0] + r * np.cos(ang), btn[1] + r * np.sin(ang)
            if not self._arena.camera_app:
                break
            on_screen = abs(x) < SCREEN_HALF[0] - 0.3 and abs(y) < SCREEN_HALF[1] - 0.3
            clear = all(np.hypot(x - dx, y - dy) > dr + 0.4
                        for dx, dy, dr in DECOY_BUTTONS.values())
            if on_screen and clear:
                break
        yaw = random_state.uniform(0, 2 * np.pi)
        xyz = [x, y, SCREEN_TOP + self._walker.upright_pose.xpos[2]]
        quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.place_fly(physics, xyz, quat)
        physics.bind(self._arena.button_joint).qpos = 0.
        for j in self._arena.decoy_joints:
            physics.bind(j).qpos = 0.
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

    def _upright(self, physics):
        """Componente z del eje z del mundo en el marco de la mosca: 1 erguida, -1 volcada."""
        return float(self._walker.observables.world_zaxis(physics)[2])

    def _fly_button_dist(self, physics):
        fly_pos, _ = self._walker.get_pose(physics)
        btn = physics.bind(self._arena.button_body).xpos
        return float(np.linalg.norm((btn - fly_pos)[:2]))

    def get_reward_factors(self, physics):
        return (1.,)   # no se usa; get_reward está sobrescrito.

    def get_reward(self, physics):
        """Shaping potencial (progreso hacia el botón, en cm) + hundimiento parcial
        del botón + bono al presionar, más términos de postura: penalización por no
        estar erguida y por velocidad angular. Sin término denso por "estar cerca",
        para que no convenga quedarse junto al botón sin presionarlo.

        run1 (sin términos de postura) aprendió a lanzarse y caer volcada sobre el
        botón; por eso el hundimiento y el bono solo cuentan estando erguida."""
        self._should_terminate = self.check_termination(physics)
        dist = self._fly_button_dist(physics)
        upright = self._upright(physics)
        # run2: abalanzarse cobraba el progreso antes de volcarse. Ahora el progreso
        # también se pondera por la postura y el vuelco tiene castigo explícito.
        reward = self._approach_scale * (self._prev_dist - dist) * max(upright, 0.)
        self._prev_dist = dist
        if upright < self._flip_threshold:
            reward -= self._flip_penalty
        if upright > 0.5:
            reward += 5. * min(self._button_depth(physics) / BUTTON_TRAVEL, 1.)
        if self._pressed:
            reward += self._press_bonus
        if self._wrong_button:   # v0.3: pisó un señuelo de la app de cámara
            reward -= self._wrong_button_penalty
        reward += self._upright_weight * (upright - 1.)
        reward -= self._time_penalty   # run4: la política determinista llegaba tarde; premia llegar antes
        reward -= self._angvel_cost * float(np.linalg.norm(
            self._walker.observables.gyro(physics)))
        return float(reward)

    def check_termination(self, physics):
        upright = self._upright(physics)
        # ¿Se presionó el botón (estando erguida)?
        if (not self._pressed and upright > 0.5
                and self._button_depth(physics) >= self._press_depth):
            self._pressed = True
            self._last_photo = self._capture_photo(physics)
            self._photos_taken += 1
            return True
        # ¿Pisó un señuelo? Cierra la app: fin sin foto (descuento 0).
        for j in self._arena.decoy_joints:
            if -physics.bind(j).qpos[0] >= self._press_depth:
                self._wrong_button = True
                return True
        fly_pos, _ = self._walker.get_pose(physics)
        fell_off = (abs(fly_pos[0]) > PHONE_HALF[0] or abs(fly_pos[1]) > PHONE_HALF[1]
                    or fly_pos[2] < SCREEN_TOP + 0.03
                    or upright < self._flip_threshold)   # volcada: terminación fatal
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
    def wrong_button(self):
        return self._wrong_button

    @property
    def vision(self):
        return self._vision

    @property
    def last_photo(self):
        return self._last_photo

    @property
    def photos_taken(self):
        return self._photos_taken


def make_env(random_state=None, time_limit: float = 2., camera_app: bool = False,
             **task_kwargs):
    """Entorno dm_control listo para usar."""
    task = TakeSelfie(walker=fruitfly.FruitFly, arena=PhoneArena(camera_app=camera_app),
                      time_limit=time_limit, joint_filter=0.01,
                      adhesion_filter=0.007, **task_kwargs)
    mujoco.set_mjcb_control(None)
    return composer.Environment(time_limit=time_limit, task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True,
                                recompile_mjcf_every_episode=False)
