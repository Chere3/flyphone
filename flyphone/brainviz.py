"""Mapa cerebral: nube de puntos del conectoma FlyWire iluminada con la actividad de la política.

Posiciones y clases de 139k neuronas: Schlegel et al. 2024 (flyconnectome/flywire_annotations, CC-BY 4.0).
La mosca simulada no tiene neuronas; aquí se PROYECTA la actividad de la red PPO sobre el mapa:
  · observación normalizada (290)  -> neuronas sensoriales y ascendentes
  · capas ocultas de la política (512) -> cerebro central
  · media de la acción (59)          -> neuronas descendentes y motoras
  · lóbulos ópticos                  -> apagados (esta tarea no usa visión)
Cada unidad de la red se asigna a un subconjunto fijo de neuronas de su región (semilla fija).
"""
import os
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

_ASSET = os.path.join(os.path.dirname(__file__), "assets", "flywire_neurons.npz")
# legend colours (as in the FlyWire viewer style)
COLORS = {0: (96, 150, 255), 1: (255, 200, 80), 2: (120, 230, 150), 3: (255, 110, 100)}
LABELS = {0: "optic lobe (no vision in this task)", 1: "central brain  <- policy hidden layers",
          2: "sensory in  <- observations", 3: "descending + motor out  <- actions"}


class BrainMap:
    def __init__(self, height=270, width=380, seed=0):
        d = np.load(_ASSET); pos, self.cls = d["pos"], d["cls"]
        # Vista frontal: x lateral, y dorso-ventral (y crece hacia ventral en FAFB -> invertimos).
        x, y = pos[:, 0], -pos[:, 1]
        self.H, self.W = height, width
        margin = 0.06; area_h = height - 62          # deja sitio a la leyenda
        sx = (x - x.min()) / (x.max() - x.min()); sy = (y - y.min()) / (y.max() - y.min())
        scale = min(width * (1 - 2 * margin) / 1., area_h * (1 - 2 * margin) / ((y.max() - y.min()) / (x.max() - x.min())))
        aspect = (y.max() - y.min()) / (x.max() - x.min())
        self.px = (margin * width + sx * (width * (1 - 2 * margin))).astype(int).clip(0, width - 1)
        self.py = (8 + (1 - sy) * (area_h - 16) * 1.0).astype(int).clip(0, area_h - 1)
        rng = np.random.RandomState(seed)
        self.unit = np.zeros(len(self.cls), np.int32)
        self.n_units = {1: 512, 2: 290, 3: 59}
        for c, n in self.n_units.items():
            idx = np.where(self.cls == c)[0]; self.unit[idx] = rng.randint(0, n, len(idx))
        self.base = self._density(np.ones(len(self.cls)), sigma=1.2)
        self.base = np.sqrt(self.base / self.base.max())
        self._font = None

    def _density(self, w, sigma=1.5):
        img = np.zeros((self.H, self.W), np.float32)
        np.add.at(img, (self.py, self.px), w.astype(np.float32))
        return ndimage.gaussian_filter(img, sigma)

    def render(self, acts):
        """acts: dict con 'obs' (290), 'pi_h1'/'pi_h2' (256 c/u), 'action' (59)."""
        unit_act = {
            2: np.clip(np.abs(np.asarray(acts.get("obs", np.zeros(290))) / 3.), 0, 1),
            1: np.abs(np.concatenate([acts.get("pi_h1", np.zeros(256)), acts.get("pi_h2", np.zeros(256))])),
            3: np.clip(np.abs(np.asarray(acts.get("action", np.zeros(59)))), 0, 1),
        }
        rgb = np.zeros((self.H, self.W, 3), np.float32)
        rgb += self.base[..., None] * 0.28 * np.array([210, 215, 225]) / 255.   # silueta gris
        levels = {}
        for c, col in COLORS.items():
            mask = self.cls == c
            if c == 0:
                levels[c] = 0.; continue
            w = np.zeros(len(self.cls), np.float32); w[mask] = unit_act[c][self.unit[mask]]
            dens = self._density(w, sigma=1.6)
            ref = self._density(mask.astype(np.float32), sigma=1.6).max() + 1e-6
            dens = np.clip(dens / ref * 1.6, 0, 1)
            levels[c] = float(unit_act[c].mean())
            rgb += dens[..., None] * np.array(col) / 255.
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        if self._font is None:
            try: self._font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 10)
            except OSError: self._font = ImageFont.load_default()
        y = self.H - 56
        for c in (0, 1, 2, 3):
            d.ellipse([10, y + 2, 17, y + 9], fill=COLORS[c])
            bar = int(levels[c] * 60)
            d.text((24, y), LABELS[c], fill=(225, 225, 225), font=self._font)
            if c != 0:
                d.rectangle([self.W - 72, y + 3, self.W - 12, y + 8], fill=(45, 45, 50))
                d.rectangle([self.W - 72, y + 3, self.W - 72 + bar, y + 8], fill=COLORS[c])
            y += 13
        d.text((10, 6), "FlyWire connectome · 139k neurons · policy activity projected", fill=(150, 150, 160), font=self._font)
        return np.asarray(im)
