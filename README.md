<h1 align="center">🪰📱 flyphone</h1>
<p align="center"><b>Teaching DeepMind's virtual fruit fly to take selfies with a phone.</b></p>
<p align="center">
  <a href="https://github.com/TuragaLab/flybody"><img alt="flybody" src="https://img.shields.io/badge/body%20model-flybody-8A2BE2"></a>
  <a href="https://mujoco.org"><img alt="MuJoCo" src="https://img.shields.io/badge/physics-MuJoCo%203-blue"></a>
  <a href="https://stable-baselines3.readthedocs.io"><img alt="SB3" src="https://img.shields.io/badge/RL-Stable--Baselines3%20PPO-orange"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"></a>
</p>

<p align="center"><img src="docs/hero.gif" width="480" alt="The fly lands on the shutter button, presses it and the phone takes a selfie"></p>

`flybody` is the anatomically detailed *Drosophila* model built by Google DeepMind and HHMI Janelia
([Nature 2025](https://www.nature.com/articles/s41586-025-09029-4)). It can walk and fly.
This repo asks a sillier question: **can it learn to use a phone?**

The fly is dropped on the screen of a life‑size phone lying on a table. Somewhere on the screen there is a
physical shutter button on a spring. If the fly walks over and pushes it down, the phone's front camera fires
and you get a selfie. Everything is simulated in MuJoCo; the policy is trained from scratch with PPO on a laptop.

<p align="center">
  <img src="docs/scene.png" width="420" alt="Fly standing on the shutter button">
  <img src="docs/selfie.png" width="280" alt="Selfie taken by the phone camera">
  <br><sub>Left: third‑person view. Right: the actual photo rendered from the phone's selfie camera.</sub>
</p>

## Quickstart (2 minutes)

```bash
git clone https://github.com/Chere3/flyphone && cd flyphone
uv venv --python 3.11 .venv && source .venv/bin/activate     # or python -m venv
uv pip install -e ".[train]"                                  # or pip install -e ".[train]"

export MUJOCO_GL=glfw   # macOS; use egl on a headless Linux box
python scripts/demo_press.py     # validates the button → writes outputs/selfie_000.png
python scripts/train_ppo.py --steps 15_000_000 --envs 6 --run run1
python scripts/make_evolution.py --run run1   # curves + a video of the fly at every checkpoint
```

Runs on an Apple M4 laptop at ~750 environment steps/s with 6 worker processes (no GPU needed).

## How it works

| Piece | Where | What |
|---|---|---|
| Scene | `flyphone/arena.py` | Table, 7×15 cm phone, screen, shutter button on a slide joint + spring, selfie camera that always targets the button. |
| Task | `flyphone/task.py` | `TakeSelfie`: spawns the fly 0.8–1.4 cm from the button with a random heading; adds `button_displacement` (egocentric vector to the button) and `button_state` to flybody's proprioceptive/vestibular observations. |
| Reward | `flyphone/task.py` | Potential‑based progress toward the button + partial button depression + **+100** when the button is pressed **while upright**; small posture and angular‑velocity costs. Falling off the phone or flipping over ends the episode with discount 0. |
| Gym wrapper | `flyphone/gym_env.py` | Flattens observations to a 290‑vector, maps a [-1, 1] action box onto the 59 real actuator ranges (6 adhesion + 53 joints). |
| Training | `scripts/train_ppo.py` | Stable‑Baselines3 PPO, `SubprocVecEnv` + `VecNormalize`, periodic evaluation that saves a video and the first selfie. |

Units are flybody's CGS units: the fly weighs ~1 mg (0.97 dyn) and the spring is tuned so that the fly's own
weight bottoms the button out. The button body uses gravity compensation so it rests at exactly zero without a fly on it.

## Results so far

**run1** (4.5M steps, no posture terms) learned to take selfies… by throwing itself at the button and landing on its back.
Every one of 12 rollouts of the 4.0M checkpoint ends with the fly flipped, most of them short of the button. Textbook reward hacking.

<p align="center"><img src="docs/run1_diver.gif" width="360" alt="run1 policy diving onto the button"><br>
<sub>run1 @ 4.0M steps: lunges toward the button and ends up on its back. The selfies it did manage to take show it upside down on the button.</sub></p>

**run2** adds an upright factor: button presses only count on its feet, flipping ends the episode, and there is
a small per-step posture and angular-velocity cost. It is training now; curves and video will land here.

Full details, curves, selfie mosaics and the bug list: **[docs/TRAINING_LOG.md](docs/TRAINING_LOG.md)**.

## Lessons learned (so you don't repeat them)

- **Reward what you mean.** Without an upright term the fly found that flipping onto the button is cheaper than walking. See the training log.

- **Don't recompile the MJCF every episode.** `composer.Environment(recompile_mjcf_every_episode=False)` turned a 1.1 s, memory‑leaking reset into 0.1 s.
- **On Apple Silicon, 6 worker processes beat 9.** More workers than performance cores just contend. Set `OMP_NUM_THREADS=1` per worker.
- **flybody's pretrained walking policies need TensorFlow 2.8 + dm‑reverb**, which have no macOS/arm64 wheels. That's why this trains from scratch.
- **Physics timestep 0.4 ms** (instead of flybody's 0.2 ms) is stable for this task and ~45 % faster.
- Three reward bugs caught before they ate a training run: the button sagging under its own cap, spawns that started with a leg already on the button, and a forgotten `Monitor` wrapper hiding episode stats.

## Roadmap

- [x] Publish run1 curves and the reward‑hacking post‑mortem
- [ ] Publish run2 curves and the checkpoint‑by‑checkpoint evolution video
- [ ] Use flybody's pretrained walker as a low‑level controller (Linux/Colab) for a natural gait
- [ ] Multiple buttons / a camera app UI on the screen
- [ ] Vision: let the fly find the button with its own compound‑eye cameras

Ideas and PRs welcome. If you get the fly to take a selfie faster, open an issue with the video 🙂

## Citation & credits

The body model, physics and task base classes are from **flybody** (Apache 2.0):

> Vaxenburg et al., *Whole-body physics simulation of fruit fly locomotion*, Nature 643, 1312–1320 (2025).

This repo is licensed under Apache 2.0.
