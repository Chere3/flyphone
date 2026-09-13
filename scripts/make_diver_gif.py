"""GIF documental: la política de run1 (sin recompensa de postura) se lanza sobre el botón."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mediapy
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from flyphone.gym_env import FlyPhoneGym

ck = sys.argv[1] if len(sys.argv) > 1 else "runs/run1/ppo_3999984_steps.zip"
vn_path = ck.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
vn = VecNormalize.load(vn_path, DummyVecEnv([lambda: FlyPhoneGym(seed=0)])); vn.training = False
model = PPO.load(ck, device="cpu")
best, best_dist = None, 9.
for seed in range(12345, 12345 + 12):
    env = FlyPhoneGym(seed=seed, capture_photos=False, render_camera="closeup", flip_threshold=-2.)
    obs, _ = env.reset(); frames, done, t, min_up = [], False, 0, 1.
    while not done and t < 900:
        act, _ = model.predict(vn.normalize_obs(obs), deterministic=False)
        obs, r, term, trunc, info = env.step(act); done = term or trunc
        up = env.task._upright(env.physics); min_up = min(min_up, up)
        if t % 3 == 0:
            im = Image.fromarray(env.render()); d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 480, 22], fill=(0, 0, 0))
            d.text((6, 5), f"run1 @4.0M steps  t={t*0.002:.2f}s  dist={info['dist']:.2f}cm  upright={up:+.2f}", fill=(255, 255, 255))
            frames.append(np.asarray(im))
        t += 1
    depth = env.task._button_depth(env.physics)
    print(f"seed {seed}: dist {info['dist']:.2f} min_upright {min_up:+.2f} depth {depth:.4f}", flush=True)
    # episodio volcado que más se acerca al botón (honesto: no siempre llega encima)
    if min_up < 0 and info["dist"] < best_dist:
        best, best_dist = frames, info["dist"]
sel = best[::2][:200] + [best[-1]] * 12
mediapy.write_video("docs/run1_diver.gif", [np.asarray(Image.fromarray(f).resize((360, 270))) for f in sel], fps=15, codec="gif")
print("docs/run1_diver.gif", len(sel), "cuadros")
