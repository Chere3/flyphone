"""Entrena a la mosca con PPO (Stable-Baselines3) en N procesos.

Uso: MUJOCO_GL=glfw python scripts/train_ppo.py --steps 15_000_000 --envs 6 --run run1
Variantes v0.3: --camera-app (UI con señuelos; usar --spawn 0.8 2.6), --vision (ojos
compuestos, MultiInputPolicy) y --llc (la política de marcha de flybody como controlador
de bajo nivel; PPO aprende solo el rumbo). Se pueden combinar.
"""
import argparse, os, sys, time
os.environ.setdefault("MUJOCO_GL", "glfw")
# Un hilo BLAS por proceso: con más, los 6-9 workers se pisan entre sí en el M4.
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import mediapy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from flyphone.gym_env import FlyPhoneGym
from flyphone.vision import policy_kwargs as vision_policy_kwargs

ROOT = os.path.join(os.path.dirname(__file__), "..")


def make_gym(seed, spawn, time_limit, camera_app=False, vision=False, llc=False, **kw):
    """Construye el entorno de la variante pedida (mismos flags que la CLI)."""
    kw.update(seed=seed, spawn_radius=spawn, time_limit=time_limit, camera_app=camera_app, vision=vision)
    if llc:
        from flyphone.walker import SteerSelfieGym   # carga la política de marcha (numpy)
        return SteerSelfieGym(**kw)
    return FlyPhoneGym(**kw)


def make(seed, spawn, time_limit, **variant):
    def _f():
        # Monitor registra recompensa/duración por episodio (rollout/ep_rew_mean en TensorBoard).
        return Monitor(make_gym(seed, spawn, time_limit, **variant))
    return _f


class EvalSelfie(BaseCallback):
    """Cada `every` pasos: N episodios deterministas en un entorno con render;
    registra tasa de éxito y guarda la selfie + video del primer éxito."""

    def __init__(self, run_dir, spawn, every=250_000, n_episodes=5, time_limit=3., **variant):
        super().__init__()
        self.run_dir, self.spawn, self.every, self.n = run_dir, spawn, every, n_episodes
        self.time_limit, self.variant = time_limit, variant
        self._last = 0
        self._env = None

    def _on_step(self):
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        if self._env is None:
            self._env = make_gym(12345, self.spawn, self.time_limit, render_camera="closeup", **self.variant)
        vec = self.model.get_vec_normalize_env()
        successes, dists, frames, saved = 0, [], [], False
        for ep in range(self.n):
            obs, _ = self._env.reset()
            done, t = False, 0
            while not done:
                o = vec.normalize_obs(obs) if vec is not None else obs
                act, _ = self.model.predict(o, deterministic=True)
                obs, r, term, trunc, info = self._env.step(act)
                done = term or trunc
                if ep == 0 and t % 4 == 0:
                    frames.append(self._env.render())
                t += 1
            successes += int(info["pressed"]); dists.append(info["dist"])
            if info["pressed"] and not saved:
                photo = self._env.task.take_photo(self._env.physics)
                mediapy.write_image(os.path.join(self.run_dir, f"selfie_{self.num_timesteps:09d}.png"), photo)
                saved = True
        mediapy.write_video(os.path.join(self.run_dir, f"eval_{self.num_timesteps:09d}.mp4"), frames, fps=30)
        self.logger.record("eval/success_rate", successes / self.n)
        self.logger.record("eval/final_dist", float(np.mean(dists)))
        print(f"[eval @ {self.num_timesteps:,}] éxito {successes}/{self.n}, dist final media {np.mean(dists):.2f} cm", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15_000_000)
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--run", default="run1")
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.8, 1.4))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--time-limit", type=float, default=6.)
    ap.add_argument("--camera-app", action="store_true", help="UI de app de cámara con botones señuelo")
    ap.add_argument("--vision", action="store_true", help="ojos compuestos en vez de button_displacement")
    ap.add_argument("--llc", action="store_true", help="walker preentrenado de flybody como controlador de bajo nivel")
    args = ap.parse_args()
    torch.set_num_threads(2)
    variant = dict(camera_app=args.camera_app, vision=args.vision, llc=args.llc)

    run_dir = os.path.join(ROOT, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    venv = SubprocVecEnv([make(i, tuple(args.spawn), args.time_limit, **variant) for i in range(args.envs)], start_method="spawn")
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99,
                        norm_obs_keys=["vec"] if args.vision else None)   # los ojos (uint8) no se normalizan

    if args.resume:
        # Carga también las estadísticas de normalización del checkpoint.
        vn_path = args.resume.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
        if os.path.exists(vn_path):
            venv = VecNormalize.load(vn_path, venv.venv); venv.training = True
        model = PPO.load(args.resume, env=venv, device="cpu")
    else:
        if args.vision:
            policy, policy_kwargs = "MultiInputPolicy", vision_policy_kwargs()
        else:
            policy, policy_kwargs = "MlpPolicy", dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0)
        model = PPO(policy, venv, device="cpu", verbose=1,
                    n_steps=1024, batch_size=4096, n_epochs=8, learning_rate=2e-4, target_kl=0.05,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=os.path.join(ROOT, "runs", "tb"))
    callbacks = [CheckpointCallback(500_000 // args.envs, run_dir, name_prefix="ppo", save_vecnormalize=True),
                 EvalSelfie(run_dir, tuple(args.spawn), every=args.eval_every, time_limit=args.time_limit, **variant)]
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name=args.run,
                reset_num_timesteps=args.resume is None)
    model.save(os.path.join(run_dir, "final"))
    venv.save(os.path.join(run_dir, "final_vecnorm.pkl"))
    print(f"listo en {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
