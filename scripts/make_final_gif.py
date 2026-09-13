"""GIF de portada con la política entrenada: camina hasta el botón, lo pisa y sale la selfie."""
import os, sys, glob
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mediapy
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from flyphone.gym_env import FlyPhoneGym
from flyphone.viz import ActivationProbe, compose

ck = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "docs/hero_trained.gif"
EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 6   # pasos de simulación por cuadro (1 = cámara lenta)
vn = VecNormalize.load(ck.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl"), DummyVecEnv([lambda: FlyPhoneGym(seed=0)])); vn.training = False
model = PPO.load(ck, device="cpu"); probe = ActivationProbe(model)
best = None
for seed in range(4242, 4242 + 30):
    env = FlyPhoneGym(seed=seed, capture_photos=False, render_camera="closeup", time_limit=6.)
    obs, _ = env.reset(); frames, done, t = [], False, 0
    while not done:
        act = probe.predict(vn.normalize_obs(obs), deterministic=True)
        obs, r, term, trunc, info = env.step(act); done = term or trunc
        if t % EVERY == 0:
            scene = np.asarray(Image.fromarray(env.render()).resize((360, 270)))
            im = Image.fromarray(compose(scene, probe.acts)); d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 360, 18], fill=(0, 0, 0)); d.text((5, 4), f"PPO policy  t={t*0.002:.2f}s" + ("   slow motion" if EVERY < 6 else ""), fill=(255, 255, 255))
            frames.append(im)
        t += 1
    print(f"seed {seed}: pressed={info['pressed']} t={t*0.002:.2f}s dist={info['dist']:.2f}", flush=True)
    # preferimos un episodio en el que se la vea caminar (1–4 s hasta la foto)
    secs = t * 0.002
    lo, hi, target = (1.0, 4.0, 2.0) if EVERY >= 6 else (0.1, 0.6, 0.2)
    score = abs(secs - target) if lo <= secs <= hi else 10 + secs
    # y la mosca debe acabar del lado visible para la cámara closeup (y menor que el botón)
    fly_y = env.task._walker.get_pose(env.physics)[0][1]
    btn_y = env.physics.bind(env.task._arena.button_body).xpos[1]
    if fly_y > btn_y - 0.1:
        score += 5
    if info["pressed"] and (best is None or score < best[3]):
        best = (frames, t, env.task.take_photo(env.physics), score)
        if score < 0.5: break
frames, t, photo, _ = best
last = frames[-1].copy(); d = ImageDraw.Draw(last)
d.rectangle([0, 0, 360, 18], fill=(0, 0, 0)); d.text((5, 4), f"PPO policy  t={t*0.002:.2f}s   PHOTO TAKEN", fill=(235, 104, 52))
last.paste(Image.fromarray(photo).resize((160, 120)), (360 - 166, 24))
seq = [np.asarray(f) for f in frames] + [np.asarray(last)] * 20
mediapy.write_video(out, seq, fps=15, codec="gif")
print(out, len(seq), "cuadros")
