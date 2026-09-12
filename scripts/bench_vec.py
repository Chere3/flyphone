import os, sys, time
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym

class Ping(gym.Env):
    """Entorno vacío: mide solo la latencia de IPC."""
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, (300,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (59,), np.float32)
    def reset(self, seed=None, options=None): return np.zeros(300, np.float32), {}
    def step(self, a): return np.zeros(300, np.float32), 0., False, False, {}

def bench(venv, label, n=150):
    venv.reset(); a = np.random.uniform(-1,1,(venv.num_envs,59)).astype(np.float32)*0.3
    venv.step(a); t0=time.time()
    for _ in range(n): venv.step(a)
    dt=(time.time()-t0)/n; print(f"{label}: {dt*1000:.1f} ms por vec-step -> {venv.num_envs/dt:.0f} fps", flush=True); venv.close()

if __name__ == "__main__":
    from flyphone.gym_env import FlyPhoneGym
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for sm in ("spawn", "forkserver"):
        bench(SubprocVecEnv([Ping for _ in range(9)], start_method=sm), f"IPC vacío x9 ({sm})", n=500)
    bench(DummyVecEnv([lambda: FlyPhoneGym(seed=0)]), "Dummy x1")
    bench(DummyVecEnv([lambda i=i: FlyPhoneGym(seed=i) for i in range(4)]), "Dummy x4 (secuencial)")
    bench(SubprocVecEnv([lambda i=i: FlyPhoneGym(seed=i) for i in range(4)], start_method="spawn"), "Subproc x4")
    bench(SubprocVecEnv([lambda i=i: FlyPhoneGym(seed=i) for i in range(9)], start_method="spawn"), "Subproc x9")
    bench(SubprocVecEnv([lambda i=i: FlyPhoneGym(seed=i) for i in range(9)], start_method="forkserver"), "Subproc x9 (forkserver)")
