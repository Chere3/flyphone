"""Evaluación robusta de un checkpoint: N episodios deterministas y N estocásticos."""
import os, sys, glob, re, collections
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from flyphone.gym_env import FlyPhoneGym

ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
vn = VecNormalize.load(ck.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl"), DummyVecEnv([lambda: FlyPhoneGym(seed=0)])); vn.training = False
model = PPO.load(ck, device="cpu")
for det in (True, False):
    env = FlyPhoneGym(seed=2024, time_limit=6.); ends = collections.Counter(); lens, dists, tpress = [], [], []
    for ep in range(n):
        obs, _ = env.reset(); done = False; t = 0
        while not done:
            act, _ = model.predict(vn.normalize_obs(obs), deterministic=det)
            obs, r, term, trunc, info = env.step(act); done = term or trunc; t += 1
        up = env.task._upright(env.physics)
        kind = "foto" if info["pressed"] else ("tiempo" if trunc else ("volcada" if up < 0.3 else "caída"))
        ends[kind] += 1; lens.append(t); dists.append(info["dist"])
        if info["pressed"]: tpress.append(t * 0.002)
    print(f"{os.path.basename(ck)} {'determinista' if det else 'estocástico'} n={n}: {dict(ends)} | "
          f"éxito {ends['foto']/n:.0%} | dist final media {np.mean(dists):.2f} cm | "
          f"tiempo hasta foto {np.mean(tpress) if tpress else float('nan'):.2f} s", flush=True)
