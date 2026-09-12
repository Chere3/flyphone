"""¿Es estable la mosca con un paso de física mayor? Mide velocidad y terminaciones por física."""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from flybody.tasks import constants
for pts in (2e-4, 4e-4, 5e-4):
    constants._WALK_PHYSICS_TIMESTEP = pts
    import flybody.tasks.base as base; base._WALK_PHYSICS_TIMESTEP = pts
    from flyphone.task import make_env
    env = make_env(random_state=np.random.RandomState(0), time_limit=3., capture_photos=False)
    spec = env.action_spec(); mid=(spec.maximum+spec.minimum)/2; half=(spec.maximum-spec.minimum)/2
    bad = 0; steps = 0; t0 = time.time()
    for ep in range(6):
        ts = env.reset()
        while not ts.last():
            ts = env.step(mid + half*np.random.uniform(-1,1,59)); steps += 1
        if ts.discount == 0.: bad += 1
    print(f"physics_timestep={pts}: {steps/(time.time()-t0):.0f} pasos/s, episodios con terminación fatal {bad}/6", flush=True)
