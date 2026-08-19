"""BraTS 2021 최종 2위(NVAUTO, NVIDIA)의 SegResNet 기반 방법을 2D로 각색한 베이스라인.

원 논문
    Myronenko, A. et al. "Redundancy Reduction in Semantic Segmentation of
    3D Brain Tumor MRIs." BrainLes @ MICCAI 2021.
    https://arxiv.org/abs/2111.00742

논문의 구성 요소를 그대로 옮긴다.
    1. MONAI SegResNet 백본. 인코더는 ResNet 블록(Conv-Norm-ReLU 2개 + 항등
       스킵), 디코더는 업샘플 후 같은 해상도 인코더 출력을 더한다(concat 아님).
    2. 정규화는 InstanceNorm 을 쓴다. 논문이 GroupNorm 대비 메모리가 낮고
       성능은 동등하다고 밝혀 InstanceNorm 으로 바꾼 부분을 따른다.
    3. 최종 정규화 직전 특징에 프로젝션 브랜치를 붙여 Barlow Twins 손실을
       계산한다. 특징을 16배 average pooling 한 뒤 공간 영역들을 배치로 펼치고
       3층 MLP 를 통과시킨다. 추론 시에는 이 브랜치를 쓰지 않는다.

논문의 confidence 기반 앙상블은 여러 모델을 학습해야 하므로 여기서는
단일 모델 학습까지만 구현하고, 앙상블은 baselines/eval_baseline.py 의
--ensemble 옵션에서 같은 heuristic(분할 영역 평균 확률)으로 처리한다.
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
    """SegResNet 백본 + Barlow Twins 프로젝션 브랜치.

    Args:
        in_channels: 입력 모달리티 채널 수.
        out_channels: 출력 채널 수(이진 분할이면 1).
        init_filters: 첫 블록 채널 수. 논문은 32를 썼다.
        blocks_down / blocks_up: SegResNet 단계별 블록 수.
        pool_factor: 프로젝션 전 average pooling 배수. 논문 값은 16.
        projection_dim: 프로젝션 출력 차원.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        init_filters: int = 32,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
        pool_factor: int = 16,
        projection_dim: int = 256,
    ):
        super().__init__()
        # 논문은 드롭아웃을 쓰지 않는다.
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

        # 최종 정규화 직전 특징을 가로챈다. conv_final = [Norm, ReLU, Conv] 이므로
        # 이 모듈의 입력이 곧 논문이 말한 "one level before the final normalization" 이다.
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

        # 공간 축을 눌러 영역 단위 특징 배치를 만든다.
        pooled = self.pool(feat)
        b, c, h, w = pooled.shape
        regions = pooled.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return logits, self.projector(regions)


def build_nvauto_segresnet(
    in_channels: int = 2,
    out_channels: int = 1,
    init_filters: int = 32,
    pool_factor: int = 16,
) -> NVAutoSegResNet2D:
    """논문 설정(init_filters=32, InstanceNorm, 드롭아웃 없음)을 기본값으로 한다."""
    return NVAutoSegResNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        pool_factor=pool_factor,
    )
