"""Valida el mecanismo: (1) mosca colocada sobre el botón → se dispara la foto;
(2) episodio con acciones aleatorias desde una posición inicial normal, con video."""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mediapy
from flyphone.task import make_env
from flyphone.arena import SCREEN_TOP, BUTTON_TRAVEL, BUTTON_HALF_H

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

env = make_env(random_state=np.random.RandomState(0))
task = env.task
print("acciones:", env.action_spec().shape, "| obs:",
      {k: v.shape for k, v in env.observation_spec().items()
       if k.startswith("walker/button")})

# --- 1) Mosca directamente sobre el botón: su peso debe hundirlo.
ts = env.reset()
btn = env.physics.bind(task._arena.button_body).xpos.copy()
task.place_fly(env.physics, [btn[0], btn[1], SCREEN_TOP + 2*BUTTON_HALF_H + 0.13], [1, 0, 0, 0])
frames = []
for i in range(150):
    ts = env.step(np.zeros(env.action_spec().shape))
    if i % 5 == 0:
        frames.append(env.physics.render(camera_id="phone_side", width=480, height=360))
    if i % 25 == 0 or ts.last():
        print(f"paso {i:3d} profundidad botón = {-env.physics.bind(task._arena.button_joint).qpos[0]/BUTTON_TRAVEL:5.2f} "
              f"touch = {env.physics.bind(task._arena.button_touch_sensor).sensordata[0]:.3f} dyn "
              f"reward = {ts.reward} pressed = {task.pressed}")
    if ts.last():
        break
if not task.pressed:
    d = env.physics.data; m = env.physics.model
    for i in range(d.ncon):
        c = d.contact[i]
        print("contacto", m.id2name(c.geom1,'geom'), m.id2name(c.geom2,'geom'), "dist", c.dist)
    raise SystemExit("El botón no se presionó con la mosca encima")
mediapy.write_image(os.path.join(OUT, "selfie_000.png"), task.last_photo)
mediapy.write_image(os.path.join(OUT, "scene_side.png"), frames[-1])
mediapy.write_image(os.path.join(OUT, "scene_top.png"),
                    env.physics.render(camera_id="phone_top", width=640, height=480))
print("foto guardada en outputs/selfie_000.png")

# --- 2) Episodio normal con acciones aleatorias (sanity check + video).
ts = env.reset()
frames, t0 = [], time.time()
n = 0
while not ts.last():
    ts = env.step(np.random.uniform(-1, 1, env.action_spec().shape) * 0.3)
    n += 1
    if n % 4 == 0:
        frames.append(env.physics.render(camera_id="phone_side", width=480, height=360))
print(f"episodio aleatorio: {n} pasos en {time.time()-t0:.1f}s, pressed={task.pressed}, "
      f"dist final={task._fly_button_dist(env.physics):.2f} cm")
mediapy.write_video(os.path.join(OUT, "random_episode.mp4"), frames, fps=30)
print("video en outputs/random_episode.mp4")
