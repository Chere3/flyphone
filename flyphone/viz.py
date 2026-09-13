"""Panel de activaciones de la política PPO (SB3 MlpPolicy) para GIFs y videos.

Capas mostradas: observación normalizada (290), capas ocultas de la política (256, 256),
media de la acción (59) y valor estimado V(s) del crítico.
Colores: divergente azul (−) · gris claro (0) · naranja (+), todo en [-1, 1].
"""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

BLUE = np.array([42, 120, 214]); MID = np.array([236, 236, 232]); ORANGE = np.array([235, 104, 52])
INK = (11, 11, 11); MUTED = (82, 81, 78); BG = (252, 252, 251)


class ActivationProbe:
    """Captura activaciones de una PPO de Stable-Baselines3 con forward hooks."""

    def __init__(self, model):
        self.model = model
        self.acts = {}
        pol = model.policy
        pol.mlp_extractor.policy_net[1].register_forward_hook(self._hook("pi_h1"))
        pol.mlp_extractor.policy_net[3].register_forward_hook(self._hook("pi_h2"))
        pol.mlp_extractor.value_net[1].register_forward_hook(self._hook("vf_h1"))
        pol.mlp_extractor.value_net[3].register_forward_hook(self._hook("vf_h2"))
        pol.action_net.register_forward_hook(self._hook("action"))
        pol.value_net.register_forward_hook(self._hook("value"))

    def _hook(self, name):
        def f(module, inp, out):
            self.acts[name] = out.detach().cpu().numpy().ravel()
        return f

    def predict(self, obs_norm, deterministic=True):
        """Como model.predict, pero además deja las activaciones en self.acts."""
        self.acts = {"obs": np.asarray(obs_norm, dtype=np.float32).ravel()}
        with torch.no_grad():
            t = torch.as_tensor(obs_norm, dtype=torch.float32).reshape(1, -1)
            self.model.policy.forward(t, deterministic=deterministic)  # dispara los hooks
        act, _ = self.model.predict(obs_norm, deterministic=deterministic)
        return act


def _colorize(v):
    """v en [-1, 1] -> RGB uint8 (H, W, 3) con mapa divergente."""
    v = np.clip(v, -1, 1)
    pos = np.clip(v, 0, 1)[..., None]; neg = np.clip(-v, 0, 1)[..., None]
    rgb = MID * (1 - pos - neg) + ORANGE * pos + BLUE * neg
    return rgb.astype(np.uint8)


def _grid(values, rows, cols, cell, gap=1):
    """Rejilla de celdas coloreadas; rellena con NaN (gris más claro) si sobran huecos."""
    v = np.full(rows * cols, np.nan); v[:len(values)] = values[:rows * cols]
    img = np.full((rows * (cell + gap) - gap, cols * (cell + gap) - gap, 3), BG, np.uint8)
    col = _colorize(np.nan_to_num(v.reshape(rows, cols)))
    empty = np.isnan(v.reshape(rows, cols))
    for r in range(rows):
        for c in range(cols):
            y, x = r * (cell + gap), c * (cell + gap)
            img[y:y + cell, x:x + cell] = (BG if empty[r, c] else col[r, c])
    return img


def render_panel(acts, height=270, width=300):
    """Devuelve un array RGB (height, width, 3) con las capas de la política."""
    im = Image.new("RGB", (width, height), BG); d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 10)
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 11)
    except OSError:
        font = bold = ImageFont.load_default()
    d.text((8, 4), "PPO policy activations", fill=INK, font=bold)
    y = 22
    cell = 6 if height >= 330 else 5          # 270 px de alto -> celda 5; 360 -> celda 6
    obs = np.clip(acts.get("obs", np.zeros(290)) / 3., -1, 1)     # obs normalizada, ~3σ -> ±1
    layers = [("obs · 290 (normalized, ±3σ)", obs, 10, 29),
              ("π hidden 1 · 256 (tanh)", acts.get("pi_h1", np.zeros(256)), 8, 32),
              ("π hidden 2 · 256 (tanh)", acts.get("pi_h2", np.zeros(256)), 8, 32),
              ("action mean · 59", np.clip(acts.get("action", np.zeros(59)), -1, 1), 2, 30)]
    for label, vals, rows, cols in layers:
        d.text((8, y), label, fill=MUTED, font=font); y += 12
        g = _grid(np.asarray(vals, dtype=float), rows, cols, cell)
        im.paste(Image.fromarray(g), (8, y)); y += g.shape[0] + 5
    v = float(acts.get("value", [0.])[0])
    d.text((8, y), f"V(s) = {v:+.1f}", fill=MUTED, font=font)
    vb = np.clip(v / 100., -1, 1); x0 = 80; w = width - x0 - 10
    d.rectangle([x0, y + 2, x0 + w, y + 9], fill=(236, 236, 232))
    if vb >= 0: d.rectangle([x0 + w // 2, y + 2, x0 + w // 2 + int(vb * w / 2), y + 9], fill=tuple(ORANGE))
    else: d.rectangle([x0 + w // 2 + int(vb * w / 2), y + 2, x0 + w // 2, y + 9], fill=tuple(BLUE))
    return np.asarray(im)


_brain_cache = {}


def compose(frame, acts, panel_width=None, mode="brain"):
    """Escena a la izquierda, panel a la derecha, misma altura.

    mode="brain": nube de puntos del conectoma FlyWire iluminada con la actividad (flyphone.brainviz).
    mode="grid":  rejillas de activaciones por capa (render_panel).
    """
    h = frame.shape[0]
    if mode == "brain":
        from flyphone.brainviz import BrainMap
        w = panel_width or int(h * 1.4)
        key = (h, w)
        if key not in _brain_cache:
            _brain_cache[key] = BrainMap(height=h, width=w)
        panel = _brain_cache[key].render(acts)
    else:
        panel = render_panel(acts, height=h, width=panel_width or 300)
    return np.concatenate([frame, panel], axis=1)
