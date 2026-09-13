"""Wrapper Gymnasium del entorno TakeSelfie (observación aplanada)."""
import numpy as np
import gymnasium as gym

from flyphone.task import make_env


class FlyPhoneGym(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0, time_limit: float = 3., capture_photos: bool = False,
                 spawn_radius=(0.8, 1.4), render_camera: str = "phone_side", **task_kwargs):
        self._env = make_env(random_state=np.random.RandomState(seed),
                             time_limit=time_limit, capture_photos=capture_photos,
                             spawn_radius=spawn_radius, **task_kwargs)
        self._render_camera = render_camera
        spec = self._env.action_spec()
        # La política actúa en [-1, 1]; se mapea linealmente al rango real de cada actuador.
        self._act_mid = (spec.maximum + spec.minimum) / 2.
        self._act_half = (spec.maximum - spec.minimum) / 2.
        self.action_space = gym.spaces.Box(-1., 1., shape=spec.shape, dtype=np.float32)
        self._obs_keys = sorted(self._env.observation_spec().keys())
        n = sum(int(np.prod(self._env.observation_spec()[k].shape)) for k in self._obs_keys)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    @property
    def task(self):
        return self._env.task

    @property
    def physics(self):
        return self._env.physics

    def _flatten(self, obs):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel()
                               for k in self._obs_keys])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ts = self._env.reset()
        return self._flatten(ts.observation), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1., 1.)
        ts = self._env.step(self._act_mid + self._act_half * a)
        task = self._env.task
        obs = self._flatten(ts.observation)
        info = {"pressed": task.pressed, "dist": task._fly_button_dist(self._env.physics)}
        terminated = bool(ts.last() and (ts.discount == 0. or task.pressed))
        truncated = bool(ts.last() and not terminated)
        return obs, float(ts.reward), terminated, truncated, info

    def render(self):
        return self._env.physics.render(camera_id=self._render_camera, width=480, height=360)
