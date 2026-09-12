"""GIF de portada: la mosca sobre el botón, lo hunde, se toma la selfie."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mediapy
from PIL import Image
from flyphone.task import make_env
from flyphone.arena import SCREEN_TOP, BUTTON_HALF_H

env = make_env(random_state=np.random.RandomState(0)); task = env.task
env.reset(); btn = env.physics.bind(task._arena.button_body).xpos.copy()
# arranca un poco por encima para que se vea el "aterrizaje" y la pulsación
task.place_fly(env.physics, [btn[0], btn[1], SCREEN_TOP + 2*BUTTON_HALF_H + 0.16], [1, 0, 0, 0])
frames = []
for i in range(60):
    ts = env.step(np.zeros(59))
    frames.append(env.physics.render(camera_id="closeup", width=480, height=360))
    if ts.last(): break
photo = Image.fromarray(task.last_photo).resize((200, 150))
last = Image.fromarray(frames[-1]); last.paste(photo, (480-208, 8))
frames += [np.asarray(last)] * 24
mediapy.write_video(os.path.join("docs", "hero.gif"), frames, fps=12, codec="gif")
mediapy.write_image(os.path.join("docs", "selfie.png"), task.last_photo)
mediapy.write_image(os.path.join("docs", "scene.png"), env.physics.render(camera_id="closeup", width=960, height=720))
print("docs/hero.gif", len(frames), "cuadros")
