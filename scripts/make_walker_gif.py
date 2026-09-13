"""GIF del walker preentrenado de flybody sobre el celular con rumbo fijo (recto), sin
política de alto nivel: muestra la marcha natural del controlador de bajo nivel.
Uso: python scripts/make_walker_gif.py [docs/walker_gait.gif] [pasos_por_cuadro]"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
# Un hilo BLAS: con los de OpenBLAS por defecto, cada gemv de la política de marcha se reparte
# entre hilos que quedan girando (300 % de CPU y 10× más lento). Debe ir antes de importar numpy.
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mediapy
from PIL import Image, ImageDraw
from flyphone.walker import SteerSelfieGym

out = sys.argv[1] if len(sys.argv) > 1 else "docs/walker_gait.gif"
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 3   # comandos (10 ms) por cuadro
best = None
for seed in range(100, 130):
    env = SteerSelfieGym(seed=seed, time_limit=4., render_camera="closeup", steer_every=5)
    obs, _ = env.reset(); frames, done, t = [], False, 0
    while not done:
        obs, r, term, trunc, info = env.step(np.array([1 / 3, 0.]))   # 2 cm/s, sin giro
        done = term or trunc
        if t % EVERY == 0:
            im = Image.fromarray(env.render()).resize((360, 270)); d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 360, 18], fill=(0, 0, 0))
            d.text((5, 4), f"flybody walker (pretrained), fixed heading  t={t*0.01:.2f}s", fill=(255, 255, 255))
            frames.append(im)
        t += 1
    secs = t * 0.01
    print(f"seed {seed}: pressed={info['pressed']} t={secs:.2f}s dist={info['dist']:.2f}", flush=True)
    if info["pressed"] and (best is None or secs > best[1]) and secs < 2.5:
        best = (frames, secs, env.task.take_photo(env.physics))
frames, secs, photo = best
last = frames[-1].copy(); d = ImageDraw.Draw(last)
d.rectangle([0, 0, 360, 18], fill=(0, 0, 0)); d.text((5, 4), f"flybody walker (pretrained)  t={secs:.2f}s   PHOTO TAKEN", fill=(235, 104, 52))
last.paste(Image.fromarray(photo).resize((160, 120)), (360 - 166, 24))
seq = [np.asarray(f) for f in frames] + [np.asarray(last)] * 15
mediapy.write_video(out, seq, fps=12, codec="gif")
print(out, len(seq), "cuadros")
