"""BraTS 2021 최종 2위(NVAUTO, NVIDIA)의 SegResNet 기반 방법을 2D로 각색한 베이스라인.

원 논문
    Myronenko, A. et al. "Redundancy Reduction in Semantic Segmentation of
    3D Brain Tumor MRIs." BrainLes @ MICCAI 2021.
    https://arxiv.org/abs/2111.00742
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class ProjectionHead(nn.Module):
    """Barlow Twins 용 3층 MLP. 논문과 같이 추론 시에는 사용하지 않는다."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NVAutoSegResNet2D(nn.Module):
    """SegResNet 백본 + Barlow Twins 프로젝션 브랜치 (2D 버전)."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 3,
        init_filters: int = 32,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
        pool_factor: int = 8,
        projection_dim: int = 256,
    ):
        super().__init__()
        self.net = SegResNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            norm=("INSTANCE", {"affine": True}),
            dropout_prob=None,
        )
        self.pool = nn.AvgPool2d(pool_factor)
        self.projector = ProjectionHead(init_filters, out_dim=projection_dim)

        self._pre_final: torch.Tensor | None = None
        self.net.conv_final.register_forward_pre_hook(self._capture_features)

    def _capture_features(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        self._pre_final = inputs[0]

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        if not return_embedding:
            self._pre_final = None
            return logits

        feat = self._pre_final
        if feat is None:
            raise RuntimeError("최종 정규화 직전 특징을 가져오지 못했습니다.")
        self._pre_final = None

        pooled = self.pool(feat)
        b, c, h, w = pooled.shape
        regions = pooled.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return logits, self.projector(regions)


# Alias for backwards compatibility
NVAutoSegResNet3D = NVAutoSegResNet2D


def build_nvauto_segresnet(
    in_channels: int = 2,
    out_channels: int = 3,
    init_filters: int = 32,
    pool_factor: int = 8,
) -> NVAutoSegResNet2D:
    """2D NVAUTO SegResNet 생성 함수."""
    return NVAutoSegResNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        pool_factor=pool_factor,
    )
