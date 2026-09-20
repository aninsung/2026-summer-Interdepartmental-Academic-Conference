"""3D convolutional observation encoder for local PPO refinement."""
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BoundaryFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(observation_space.shape[0], 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(64, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.encoder(observations)
