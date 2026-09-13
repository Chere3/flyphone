"""Gráficas y video de la evolución del entrenamiento.

Uso: MUJOCO_GL=glfw python scripts/make_evolution.py --runs run2 run3 run4 --out final
Genera en runs/<out>/: evolucion.png (curvas), selfies.png (mosaico) y evolucion.mp4
(un episodio determinista por checkpoint, mismo punto de partida, con etiqueta del paso).
Varias corridas encadenadas (reanudadas una de otra) se tratan como una sola línea temporal.
"""
import argparse, glob, os, re, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mediapy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.join(os.path.dirname(__file__), "..")
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#c9c8c2", "#52514e"


def load_scalars(tb_dir, runs):
    """Concatena los escalares de las corridas indicadas (ordenados por paso)."""
    out = {}
    for run in runs:
        for d in sorted(glob.glob(os.path.join(tb_dir, f"{run}_*"))):
            acc = EventAccumulator(d, size_guidance={"scalars": 0}); acc.Reload()
            for tag in acc.Tags()["scalars"]:
                ev = acc.Scalars(tag)
                x, y = out.get(tag, (np.array([]), np.array([])))
                out[tag] = (np.concatenate([x, [e.step for e in ev]]), np.concatenate([y, [e.value for e in ev]]))
    for tag, (x, y) in out.items():
        o = np.argsort(x, kind="stable"); out[tag] = (x[o], y[o])
    return out


def plot_curves(scalars, path):
    panels = [("rollout/ep_rew_mean", "Recompensa media por episodio", BLUE),
              ("rollout/ep_len_mean", "Duración media del episodio (pasos)", BLUE),
              ("eval/success_rate", "Tasa de éxito en evaluación (5 episodios)", ORANGE),
              ("eval/final_dist", "Distancia final al botón en evaluación (cm)", ORANGE)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (tag, title, color) in zip(axes.ravel(), panels):
        ax.set_facecolor("#fcfcfb"); ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GREY)
        ax.grid(axis="y", color=GREY, lw=0.5); ax.tick_params(colors=INK, labelsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:g}M"))
        if tag in scalars:
            x, y = scalars[tag]
            ax.plot(x, y, color=color, lw=1.6, marker="o" if len(x) < 40 else None, ms=3)
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
        else:
            ax.text(0.5, 0.5, "sin datos todavía", ha="center", va="center", transform=ax.transAxes, color=INK)
        if tag == "eval/success_rate": ax.set_ylim(-0.02, 1.02)
    fig.supxlabel("pasos de entrenamiento", color=INK, fontsize=9)
    fig.tight_layout(); fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)


def selfie_grid(run_dirs, path):
    files = sorted([f for d in run_dirs for f in glob.glob(os.path.join(d, "selfie_*.png"))],
                   key=lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[0]))
    if not files:
        return False
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB").resize((320, 240))
        d = ImageDraw.Draw(im); step = int(re.findall(r"(\d+)", os.path.basename(f))[0])
        d.rectangle([0, 0, 110, 18], fill=(0, 0, 0)); d.text((4, 3), f"{step/1e6:.2f}M pasos", fill=(255, 255, 255))
        thumbs.append(im)
    cols = min(4, len(thumbs)); rows = int(np.ceil(len(thumbs) / cols))
    grid = Image.new("RGB", (cols * 320, rows * 240), (252, 252, 251))
    for i, t in enumerate(thumbs):
        grid.paste(t, ((i % cols) * 320, (i // cols) * 240))
    grid.save(path); return True


def annotate(frame, text, sub=None):
    im = Image.fromarray(frame); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0)); d.text((8, 6), text, fill=(255, 255, 255))
    if sub: d.text((im.width - 8 - len(sub) * 6, 6), sub, fill=(235, 104, 52))
    return np.asarray(im)


def rollout(model, vecnorm, steps_label, seed=777, every=3, time_limit=6.):
    from flyphone.gym_env import FlyPhoneGym
    env = FlyPhoneGym(seed=seed, capture_photos=False, render_camera="closeup", time_limit=time_limit)
    obs, _ = env.reset(); frames, done, t = [], False, 0
    while not done:
        if model is None:
            act = env.action_space.sample() * 0.3
        else:
            o = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            act, _ = model.predict(o, deterministic=True)
        obs, r, term, trunc, info = env.step(act); done = term or trunc
        if t % every == 0:
            frames.append(annotate(env.render(), f"{steps_label}   t={t*0.002:.2f}s   dist={info['dist']:.2f}cm"))
        t += 1
    status = "FOTO TOMADA" if info["pressed"] else "sin foto"
    last = annotate(env.render(), f"{steps_label}   t={t*0.002:.2f}s   dist={info['dist']:.2f}cm", status)
    if info["pressed"]:
        photo = Image.fromarray(env.task.take_photo(env.physics)).resize((160, 120))
        base = Image.fromarray(last); base.paste(photo, (base.width - 168, 34)); last = np.asarray(base)
    frames += [last] * 30
    return frames, info["pressed"]


def make_video(run_dirs, path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from flyphone.gym_env import FlyPhoneGym
    ckpts = sorted([c for d in run_dirs for c in glob.glob(os.path.join(d, "ppo_*_steps.zip"))],
                   key=lambda p: int(re.findall(r"ppo_(\d+)_steps", p)[0]))
    frames, results = [], []
    f, ok = rollout(None, None, "0 pasos (sin entrenar)"); frames += f; results.append((0, ok))
    dummy = DummyVecEnv([lambda: FlyPhoneGym(seed=0)])
    for ck in ckpts:
        steps = int(re.findall(r"ppo_(\d+)_steps", ck)[0])
        vn_path = ck.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
        vn = VecNormalize.load(vn_path, dummy) if os.path.exists(vn_path) else None
        if vn is not None: vn.training = False
        model = PPO.load(ck, device="cpu")
        f, ok = rollout(model, vn, f"{steps/1e6:.1f}M pasos"); frames += f; results.append((steps, ok))
        print(f"  checkpoint {steps:,}: {'foto' if ok else 'sin foto'}", flush=True)
    mediapy.write_video(path, frames, fps=30)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["run1"], help="corridas encadenadas, en orden")
    ap.add_argument("--out", default=None, help="carpeta de salida en runs/ (por defecto la última corrida)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    run_dirs = [os.path.join(ROOT, "runs", r) for r in args.runs]
    out_dir = os.path.join(ROOT, "runs", args.out or args.runs[-1]); os.makedirs(out_dir, exist_ok=True)
    scalars = load_scalars(os.path.join(ROOT, "runs", "tb"), args.runs)
    plot_curves(scalars, os.path.join(out_dir, "evolucion.png")); print("gráficas: evolucion.png")
    if selfie_grid(run_dirs, os.path.join(out_dir, "selfies.png")): print("mosaico: selfies.png")
    if not args.no_video:
        res = make_video(run_dirs, os.path.join(out_dir, "evolucion.mp4")); print("video: evolucion.mp4", res)
