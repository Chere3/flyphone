"""Evaluación robusta de un checkpoint: N episodios deterministas y N estocásticos.

Uso: python scripts/eval_final.py runs/run4/ppo_14999940_steps.zip 20 [--camera-app] [--vision] [--llc] [--spawn 0.8 2.6]
Los flags deben ser los mismos con los que se entrenó el checkpoint."""
import os, sys, argparse, collections
os.environ.setdefault("MUJOCO_GL", "glfw")
# Un hilo BLAS: con los de OpenBLAS por defecto, cada gemv de la política de marcha se reparte
# entre hilos que quedan girando (300 % de CPU y 10× más lento). Debe ir antes de importar numpy.
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_ppo import make_gym

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("n", type=int, nargs="?", default=20)
ap.add_argument("--spawn", type=float, nargs=2, default=(0.8, 1.4)); ap.add_argument("--time-limit", type=float, default=6.)
ap.add_argument("--camera-app", action="store_true"); ap.add_argument("--vision", action="store_true"); ap.add_argument("--llc", action="store_true")
a = ap.parse_args(); ck, n = a.ckpt, a.n
variant = dict(camera_app=a.camera_app, vision=a.vision, llc=a.llc)
vn = VecNormalize.load(ck.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl"), DummyVecEnv([lambda: make_gym(0, tuple(a.spawn), a.time_limit, **variant)])); vn.training = False
model = PPO.load(ck, device="cpu")
for det in (True, False):
    env = make_gym(2024, tuple(a.spawn), a.time_limit, **variant); ends = collections.Counter(); lens, dists, tpress = [], [], []
    for ep in range(n):
        obs, _ = env.reset(); done = False; t = 0
        while not done:
            act, _ = model.predict(vn.normalize_obs(obs), deterministic=det)
            obs, r, term, trunc, info = env.step(act); done = term or trunc; t += 1
        up = env.task._upright(env.physics)
        kind = "foto" if info["pressed"] else ("señuelo" if info["wrong_button"] else ("tiempo" if trunc else ("volcada" if up < 0.3 else "caída")))
        ends[kind] += 1; lens.append(t); dists.append(info["dist"])
        if info["pressed"]: tpress.append(t * 0.002 * (env._steer_every if a.llc else 1))
    print(f"{os.path.basename(ck)} {'determinista' if det else 'estocástico'} n={n}: {dict(ends)} | "
          f"éxito {ends['foto']/n:.0%} | dist final media {np.mean(dists):.2f} cm | "
          f"tiempo hasta foto {np.mean(tpress) if tpress else float('nan'):.2f} s", flush=True)
