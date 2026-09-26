"""Small patch energy model for GT-free test-time component selection."""
import torch
from torch import nn

class EnergyModel(nn.Module):
    def __init__(self, in_channels=4, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.InstanceNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, 1, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x))
