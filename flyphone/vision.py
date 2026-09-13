"""Extractor de características para la observación Dict con ojos (SB3 MultiInputPolicy).

Los dos ojos compuestos (2 × 32 × 32, gris) pasan por una CNN pequeña —la NatureCNN de
SB3 no cabe en 32 px— y su salida se concatena con el vector propioceptivo, que entra
tal cual (ya normalizado por VecNormalize). Misma idea que la VisNet de flybody.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EyesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, eye_features: int = 32):
        n_vec = observation_space["vec"].shape[0]
        super().__init__(observation_space, features_dim=n_vec + eye_features)
        c, h, w = observation_space["eyes"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 8, 3, stride=2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.head = nn.Sequential(nn.Linear(n_flat, eye_features), nn.ReLU())

    def forward(self, obs):
        eyes = (obs["eyes"].float() - 128.) / 64.
        return torch.cat([obs["vec"], self.head(self.cnn(eyes))], dim=1)


def policy_kwargs(eye_features: int = 32):
    """policy_kwargs para PPO("MultiInputPolicy", ...) con este extractor."""
    return dict(features_extractor_class=EyesExtractor,
                features_extractor_kwargs=dict(eye_features=eye_features),
                net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0)
