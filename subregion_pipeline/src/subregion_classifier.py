"""
subregion_pipeline/src/subregion_classifier.py
------------------------------------------------
Subregion Classifier (ET / TC / WT 판별기)
MRI 슬라이스 입력을 받아 3가지 서브리전 유형을 분류합니다:
  - Class 0: ET (Enhancing Tumor 우세)
  - Class 1: TC (Tumor Core 우세)
  - Class 2: WT (Whole Tumor / Edema 우세)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubregionClassifier(nn.Module):
    """
    Subregion CNN Classifier (Input: 2-channel 128x128 -> Output: 3 classes)
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # 64x64
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)            # 32x32
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)           # 16x16
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)          # 8x8
        self.bn4   = nn.BatchNorm2d(256)

        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits
