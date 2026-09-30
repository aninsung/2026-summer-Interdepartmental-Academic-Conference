"""지도학습 윤곽 진화 네트워크 (MARL-MambaContour 타당성 검증용).

MARL-MambaContour(ACM MM 2025)는 윤곽점마다 에이전트를 두고 SAC로 위치를 옮긴다.
그 논문의 ablation에서 "거리 손실로 윤곽점을 직접 지도학습"하는 베이스라인은
MARL 대비 mDice 4.02%p 낮은 정도였다. 따라서 윤곽 표현 자체가 이 데이터에서
통하는지는 RL 없이 지도학습만으로 먼저 판별할 수 있다. 이 모듈이 그 베이스라인이다.

논문과의 대응:
  - 윤곽점 = 에이전트          → 여기서는 점마다 오프셋을 회귀
  - Mamba + BCHFM (점 간 교류) → 순환 1D 합성곱 (닫힌 윤곽이므로 circular padding)
  - 반복 진화 5회              → n_iters 단계, 단계마다 별도 가중치
  - 협력 정규화(윤곽 매끄러움) → 학습 손실의 2차 차분 항 (train 스크립트 쪽)

입력은 영상 모달리티와 초기 마스크·확률맵이며, Stage 2 Expert가 만든 마스크에서
윤곽을 초기화한다. 논문의 검출 헤드는 Stage 2가 대신하므로 여기엔 없다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ContourEvolveNet", "build_contour_evolve", "sample_features"]


def sample_features(feat_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """윤곽점 위치에서 특징맵을 이중선형 보간으로 뽑는다.

    Parameters
    ----------
    feat_map : (B, C, H, W)
    points   : (B, N, 2) — (row, col) 픽셀 좌표

    Returns
    -------
    (B, C, N)
    """
    b, _, h, w = feat_map.shape
    rows, cols = points[..., 0], points[..., 1]
    # grid_sample은 [-1, 1] 정규화 좌표에 (x, y) 순서를 쓴다.
    x = 2.0 * cols / max(w - 1, 1) - 1.0
    y = 2.0 * rows / max(h - 1, 1) - 1.0
    grid = torch.stack([x, y], dim=-1).view(b, -1, 1, 2)
    sampled = F.grid_sample(feat_map, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
    return sampled.squeeze(-1)


class CircularConv1d(nn.Module):
    """닫힌 윤곽용 1D 합성곱. 시퀀스가 고리이므로 순환 패딩을 쓴다."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode="circular")
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ImageEncoder(nn.Module):
    """윤곽점이 참조할 특징맵을 만드는 소형 U-Net."""

    def __init__(self, in_channels: int, feat_dim: int, width: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = ConvBlock(width * 2, width)
        self.head = nn.Conv2d(width, feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class EvolveStage(nn.Module):
    """한 번의 윤곽 진화 단계. 점 특징을 이웃과 섞은 뒤 오프셋을 낸다."""

    def __init__(self, feat_dim: int, hidden: int = 128, dilations=(1, 2, 4, 8)):
        super().__init__()
        # 점 입력: 샘플된 영상 특징 + 중심 기준 상대좌표(2) + 이웃 방향 접선(2)
        self.inp = CircularConv1d(feat_dim + 4, hidden, 3)
        self.blocks = nn.ModuleList([CircularConv1d(hidden, hidden, 3, d) for d in dilations])
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in dilations])
        self.out = nn.Conv1d(hidden, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        b, n, _ = points.shape
        sampled = sample_features(feat_map, points)               # (B, C, N)

        center = points.mean(dim=1, keepdim=True)
        scale = points.std(dim=1, keepdim=True).clamp(min=1.0)
        rel = ((points - center) / scale).transpose(1, 2)          # (B, 2, N)
        tangent = (torch.roll(points, -1, dims=1) - torch.roll(points, 1, dims=1))
        tangent = (tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)).transpose(1, 2)

        h = F.relu(self.inp(torch.cat([sampled, rel, tangent], dim=1)))
        for block, norm in zip(self.blocks, self.norms):
            h = h + F.relu(norm(block(h)))
        return self.out(h).transpose(1, 2)                         # (B, N, 2) 오프셋


class ContourEvolveNet(nn.Module):
    """초기 윤곽을 GT 경계로 반복 이동시키는 네트워크.

    forward는 단계별 윤곽 목록을 돌려준다. 각 단계에 손실을 걸어 학습한다.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_dim: int = 64,
        width: int = 32,
        hidden: int = 128,
        n_iters: int = 3,
        max_shift: float = 8.0,
    ):
        super().__init__()
        self.encoder = ImageEncoder(in_channels, feat_dim, width)
        self.stages = nn.ModuleList([EvolveStage(feat_dim, hidden) for _ in range(n_iters)])
        self.n_iters = n_iters
        self.max_shift = max_shift

    def forward(self, image: torch.Tensor, init_points: torch.Tensor) -> list[torch.Tensor]:
        feat_map = self.encoder(image)
        h, w = image.shape[-2:]
        points = init_points
        outputs = []
        for stage in self.stages:
            delta = stage(feat_map, points).tanh() * self.max_shift
            points = points + delta
            points = torch.stack([
                points[..., 0].clamp(0.0, h - 1.0),
                points[..., 1].clamp(0.0, w - 1.0),
            ], dim=-1)
            outputs.append(points)
        return outputs


def build_contour_evolve(in_channels: int = 4, n_iters: int = 3, **kwargs) -> ContourEvolveNet:
    return ContourEvolveNet(in_channels=in_channels, n_iters=n_iters, **kwargs)
