import os, sys, time
for v in ("OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
if __name__ == "__main__":
    from flyphone.gym_env import FlyPhoneGym
    for n in (2, 4, 6, 8):
        venv = SubprocVecEnv([lambda i=i: FlyPhoneGym(seed=i) for i in range(n)], start_method="spawn")
        venv.reset(); a = np.random.uniform(-1,1,(n,59)).astype(np.float32)*0.3
        venv.step(a); t0=time.time()
        for _ in range(150): venv.step(a)
        dt=(time.time()-t0)/150; print(f"hilos=1 Subproc x{n}: {dt*1000:.1f} ms por vec-step -> {n/dt:.0f} fps", flush=True); venv.close()
