# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es

La mosca virtual de `flybody` (DeepMind/Janelia, MuJoCo) aprende con PPO a caminar sobre la pantalla de un celular y pisar el botón de disparo para tomarse una selfie. Todo se simula en MuJoCo y se entrena en CPU (MacBook Air M4, ~750 pasos/s con 6 workers). Estado actual: v0.2 = 100 % de éxito determinista a 15M pasos; v0.3 añade tres variantes opcionales de la tarea (walker preentrenado como controlador de bajo nivel, app de cámara con señuelos, visión con ojos compuestos), implementadas y probadas de punta a punta pero **sin corridas largas todavía** (ver `docs/TRAINING_LOG.md`, sección v0.3).

## Entorno y comandos

Python 3.11 en `.venv/` (creado con `uv venv`); no hay `uv.lock`, se instala con pip:

```bash
source .venv/bin/activate
uv pip install -e ".[train]"      # flybody viene de git; [train] añade torch, SB3, gymnasium, tensorboard, matplotlib
export MUJOCO_GL=glfw             # macOS; egl en Linux headless. Los scripts hacen setdefault a glfw.
```

Los scripts insertan `..` en `sys.path`, así que se corren **desde la raíz del repo** con `python scripts/x.py`.

```bash
python scripts/demo_press.py                                   # sanity check del botón; escribe outputs/selfie_000.png y un video
python scripts/train_ppo.py --steps 15_000_000 --envs 6 --run run5          # entrena; checkpoints en runs/run5/, TB en runs/tb/run5_N
python scripts/train_ppo.py --resume runs/run4/ppo_14999940_steps.zip --run run5   # reanuda (carga también el VecNormalize)
python scripts/eval_final.py runs/run4/ppo_14999940_steps.zip 20            # 20 episodios deterministas + 20 estocásticos
python scripts/make_evolution.py --runs run2 run3 run4 --out final          # curvas + mosaico de selfies + video por checkpoint
python scripts/make_final_gif.py <ckpt.zip> docs/hero_trained.gif [pasos_por_cuadro]
tensorboard --logdir runs/tb
# v0.3: las variantes se eligen con flags, combinables; eval_final necesita los mismos flags (y --spawn) que el entrenamiento
python scripts/train_ppo.py --llc --steps 2_000_000 --run run5_llc                  # PPO solo aprende el rumbo (acción 2-D)
python scripts/train_ppo.py --camera-app --spawn 0.8 2.6 --run run6_app             # señuelos; obs 292
python scripts/train_ppo.py --vision --run run7_eyes                                 # obs Dict, MultiInputPolicy
python scripts/eval_final.py runs/run5_llc/ppo_<N>_steps.zip 20 --llc
python scripts/make_walker_gif.py docs/walker_gait.gif   # walker preentrenado con rumbo fijo, sin política de alto nivel
# Reexportar la política de marcha (solo si cambia flybody): necesita un TF moderno en otro venv, no el del proyecto
uv venv /tmp/tfvenv && uv pip install --python /tmp/tfvenv/bin/python tensorflow
/tmp/tfvenv/bin/python scripts/export_walk_policy.py <trained-fly-policies/policies/walking> flyphone/assets/walk_policy.npz
```

Prueba de humo de una variante (unos 5 min, deja `runs/_smoke_x/`): `train_ppo.py --steps 4096 --envs 2 --eval-every 2000 <flags> --run _smoke_x`.

`scripts/bench_*.py` son microbenchmarks (nº de workers, timestep de física, método de spawn de `SubprocVecEnv`) que justifican las decisiones de rendimiento del README.

No hay tests ni linter configurados. La verificación es empírica: `demo_press.py` para la mecánica del botón, `eval_final.py` para un checkpoint.

## Arquitectura

Pipeline de cuatro capas, cada una en un archivo:

1. **`flyphone/arena.py` — `PhoneArena`** (`composer.Arena`). Mesa, celular 7×15 cm, botón sobre una junta `slide` con resorte (`BUTTON_STIFFNESS=15 dyn/cm`, calibrado para que el peso de la mosca, ~0.97 dyn, lo hunda), sensor `touch` y cámara `selfie_cam` en modo `targetbody` al botón. Unidades CGS de flybody (cm, g, s). El cuerpo del botón lleva `gravcomp=1`; sin eso la tapa hunde el resorte sola y regala recompensa. Los botones se crean con `_add_button`; con `camera_app=True` se dibuja la UI (solo visual, `contype=0`) y se añaden los señuelos de `DECOY_BUTTONS` (galería y voltear, a ±2.3 cm del disparador), expuestos como `decoy_joints`.
2. **`flyphone/task.py` — `TakeSelfie(flybody.tasks.base.Walking)`**. Añade dos observables al walker (`button_displacement` egocéntrico y `button_state`, un valor en [0,1] por botón: (1,) normal, (3,) con app de cámara), spawnea la mosca en un anillo de 0.8–1.4 cm alrededor del botón con yaw aleatorio (con app de cámara rechaza spawns fuera de pantalla o a < 0.4 cm de un señuelo), y sobrescribe `get_reward` / `check_termination` / `get_discount`. Pisar un señuelo → `wrong_button`, −10, fin con descuento 0. Con `vision=True` apaga `button_displacement` y registra los ojos con `_add_fast_eyes` (sin sombras/reflejos/cielo, sin las mallas cosméticas de la mosca, refresco cada `EYE_UPDATE_CONTROL_STEPS=4` pasos de control): los observables de ojos de flybody se renderizan en cada subpaso de física con sombras y cuestan 200 ms/paso. `make_env(camera_app=...)` construye el `composer.Environment` con `recompile_mjcf_every_episode=False` (obligatorio: si no, cada reset tarda 1.1 s y fuga memoria). Al importar, el módulo parchea `flybody_base._WALK_PHYSICS_TIMESTEP = 4e-4` antes de que `Walking` lo lea; el control timestep queda en 2 ms (1 paso de política = 0.002 s, de ahí los `t * 0.002` en los scripts).
3. **`flyphone/gym_env.py` — `FlyPhoneGym`**. Wrapper Gymnasium: aplana la observación a un vector de 290 concatenando las claves **ordenadas alfabéticamente** (292 con app de cámara), y mapea acciones en [-1, 1] linealmente al rango real de los 59 actuadores (6 adhesión + 53 articulaciones). Con `vision=True` la observación es un `Dict`: `vec` (287, sin `button_displacement`) y `eyes` (2, 32, 32) uint8 en gris. Semántica de fin de episodio: `terminated` = caída/vuelco/señuelo (discount 0) **o** foto tomada; `truncated` = se acabó el `time_limit`. `info` trae `pressed`, `wrong_button` y `dist`.
4. **`flyphone/walker.py` — `WalkPolicy` y `SteerSelfieGym`**. `WalkPolicy` es la política de marcha por imitación de flybody (DMPO) en numpy: `batch_concat` de 12 observables ordenados por clave (741) → Linear 512 → LayerNorm → tanh → 3×(Linear 512 → ELU) → Linear 59 (media). Los pesos vienen de `flyphone/assets/walk_policy.npz`, exportados del SavedModel de TF con `scripts/export_walk_policy.py`. `SteerSelfieGym(FlyPhoneGym)` convierte un comando (velocidad 0–3 cm/s, giro ±6 rad/s), mantenido `steer_every=5` pasos de control, en la trayectoria de referencia de 65 pasos (`ref_displacement`, `ref_root_quat`) que la política espera, sintetizada desde la pose actual con `constant_speed_trajectory`. Dos trampas ya resueltas y que no hay que deshacer: usa `claw_friction=None` (con la fricción 1.0 de la tarea el walker vuelca) y canonicaliza el signo del cuaternión delta (w ≥ 0; con yaw de spawn > π salía −identidad y la mosca caía).
5. **`flyphone/vision.py` — `EyesExtractor`**. Extractor SB3 para `MultiInputPolicy`: CNN 3×3 de tres capas sobre los ojos (normalizados (x−128)/64) → 32 rasgos, concatenados con `vec`. `policy_kwargs()` devuelve la config completa.
6. **`scripts/train_ppo.py`**. SB3 PPO (`MlpPolicy` 290→256→256→59, `log_std_init=-1`, `target_kl=0.05`), `SubprocVecEnv(start_method="spawn")` envuelto en `Monitor` (sin él TensorBoard no muestra `rollout/*`) y `VecNormalize` (obs + reward; con `--vision`, `norm_obs_keys=["vec"]`). `make_gym(seed, spawn, time_limit, camera_app, vision, llc)` es la fábrica común que también usa `eval_final.py`. Callback `EvalSelfie` cada 250k pasos: 5 episodios deterministas, guarda `eval_N.mp4` y la primera `selfie_N.png`, registra `eval/success_rate` y `eval/final_dist`.

### Recompensa (la parte que más se ha iterado)

Está en `TakeSelfie.get_reward` y cada término lleva un comentario con la corrida que lo motivó. Resumen: progreso potencial hacia el botón × `max(upright, 0)`, hundimiento parcial del botón solo si `upright > 0.5`, +100 al presionar erguida, −10 al volcar (y termina el episodio), pequeños costos de postura, velocidad angular y por paso. `upright` es `world_zaxis[2]` en el marco de la mosca. run1 sin términos de postura aprendió a lanzarse y caer de espaldas sobre el botón; toda modificación de la recompensa debe anotarse en `docs/TRAINING_LOG.md` y correrse como una nueva `--run`.

### Checkpoints y corridas

- `CheckpointCallback` guarda cada 500k pasos **dos archivos**: `ppo_<N>_steps.zip` y `ppo_vecnormalize_<N>_steps.pkl`. Cualquier script que cargue un checkpoint debe cargar ambos (`VecNormalize.load(..., DummyVecEnv([...]))`, `vn.training = False`) y normalizar la observación con `vn.normalize_obs` antes de `model.predict`; el patrón de reemplazo `ppo_` → `ppo_vecnormalize_` está repetido en todos los scripts.
- Las corridas se encadenan con `--resume` (run2 → run3 → run4 comparten una sola línea temporal de pasos); `make_evolution.py --runs a b c` las concatena leyendo `runs/tb/<run>_*`.
- `runs/` y `outputs/` están en `.gitignore`; solo se versionan `docs/*.png|gif` y `docs/TRAINING_LOG.md`. `docs/_*.png` también se ignora (scratch).
- Un checkpoint solo carga en la variante con la que se entrenó (la forma de la observación cambia): `eval_final.py` y cualquier script nuevo deben recibir los mismos `--llc/--camera-app/--vision` y `--spawn`.

### Visualización

- `flyphone/viz.py` — `ActivationProbe` engancha forward hooks a la política SB3 y expone `probe.predict()` (igual que `model.predict` pero deja `probe.acts`). `compose(frame, acts, mode="brain"|"grid")` pega el panel a la derecha del cuadro.
- `flyphone/brainviz.py` — `BrainMap` proyecta obs/ocultas/acciones sobre 139k posiciones de neuronas del conectoma FlyWire (`flyphone/assets/flywire_neurons.npz`). Cada unidad de la red se asigna a un subconjunto fijo de neuronas por región (semilla 0). Es visualización, no simulación neuronal; el README lo deja explícito y hay que mantenerlo así. Solo está cableado para la política `MlpPolicy` de 290 entradas; las políticas `--vision`/`--llc` no pasan por el mapa todavía.
- Assets versionados: `flywire_neurons.npz` (1.1 MB) y `walk_policy.npz` (4.5 MB, Apache 2.0 de flybody).

### Valores por defecto que difieren entre capas

`time_limit` es 2 s en `make_env`, 3 s en `FlyPhoneGym` y 6 s en `train_ppo.py` / `eval_final.py` (v0.1+ entrena y evalúa a 6 s). `capture_photos` es `True` en la tarea pero `False` en el gym (los scripts renderizan la selfie a demanda con `task.take_photo`). Al escribir un script nuevo, fijar estos valores explícitamente.

## Convenciones

- **Idioma**: código, comentarios, docstrings y mensajes de CLI en español; `README.md`, `docs/TRAINING_LOG.md` y los mensajes de commit en inglés (conventional commits: `feat(reward):`, `docs:`, `feat(viz):`). Mantener esa separación.
- Los scripts son procedurales y compactos (varias sentencias por línea, sin `main()` salvo en `train_ppo.py` y `make_evolution.py`); seguir el estilo del script vecino.
- Rendimiento en Apple Silicon: 6 workers y un hilo BLAS por proceso (`train_ppo.py`, `eval_final.py` y `make_walker_gif.py` fijan `OMP_NUM_THREADS=1` y afines **antes de importar numpy**); más workers que núcleos de rendimiento empeora el throughput. Cualquier script nuevo que use `WalkPolicy` debe hacer lo mismo: con los hilos de OpenBLAS por defecto cada gemv se reparte en hilos que quedan girando (300 % de CPU, 10× más lento).
- flybody exige TF 2.8 + dm-reverb para *cargar* sus políticas preentrenadas, sin wheels arm64; por eso la política de marcha se exportó a numpy (`walker.py`) y el resto se entrena desde cero.
- Renderizado en macOS es glfw y funciona dentro de los workers `spawn`; pero es caro (≥6 ms por render aunque sea 32×32). Nada nuevo debe renderizar en cada subpaso de física.
