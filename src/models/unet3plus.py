"""
Light-weight 2D UNet 3+ (Huang et al., 2020)
Step 1: 초기 뇌종양 마스크 생성 모델

MONAI에 UNet 3+ 구현이 없어 직접 구현한 모델입니다.
기존 U-Net / Attention U-Net 과 동일한 4단계 인코더(16, 32, 64, 128)를 사용하고,
각 디코더 스테이지가 모든 인코더 스케일 + 하위 디코더 스케일을 리사이즈해
concat 하는 Full-scale Skip Connection 으로 연결됩니다.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv-IN-PReLU x2 블록."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _CatConv(nn.Module):
    """스케일을 맞춘 뒤 채널을 CatChannels로 투영하는 단위 블록 (Conv-IN-PReLU)."""

    def __init__(self, in_ch: int, cat_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, cat_channels, 3, padding=1),
            nn.InstanceNorm2d(cat_channels),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3Plus(nn.Module):
    """
    Light-weight 2D UNet 3+ (Huang et al., 2020 의 Full-scale Skip Connection 구조).

    기존 프로젝트의 UNet/AttentionUNet과 동일하게 4단계 인코더
    (channels 기본값 16,32,64,128)를 사용하고, 각 디코더 스테이지는
    모든 인코더 스케일 + 하위 디코더 스케일을 리사이즈하여 concat하는
    Full-scale skip connection으로 연결됩니다.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple = (16, 32, 64, 128),
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        cat_channels = c1                      # 각 스케일을 투영할 통일 채널 수
        up_channels = cat_channels * 4         # 디코더 스테이지 출력 채널 (4개 스케일 concat)

        # ── 인코더 ──────────────────────────────────────────
        self.enc1 = _ConvBlock(in_channels, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.enc4 = _ConvBlock(c3, c4)          # bottleneck
        self.pool = nn.MaxPool2d(2)

        # ── 디코더 d3 (H/4 스케일) ──────────────────────────
        self.d3_from_e1 = _CatConv(c1, cat_channels)   # maxpool 4x
        self.d3_from_e2 = _CatConv(c2, cat_channels)   # maxpool 2x
        self.d3_from_e3 = _CatConv(c3, cat_channels)   # scale 유지
        self.d3_from_e4 = _CatConv(c4, cat_channels)   # upsample 2x
        self.d3_fuse = _CatConv(up_channels, up_channels)

        # ── 디코더 d2 (H/2 스케일) ──────────────────────────
        self.d2_from_e1 = _CatConv(c1, cat_channels)   # maxpool 2x
        self.d2_from_e2 = _CatConv(c2, cat_channels)   # scale 유지
        self.d2_from_d3 = _CatConv(up_channels, cat_channels)  # upsample 2x
        self.d2_from_e4 = _CatConv(c4, cat_channels)   # upsample 4x
        self.d2_fuse = _CatConv(up_channels, up_channels)

        # ── 디코더 d1 (H 스케일, 최종) ───────────────────────
        self.d1_from_e1 = _CatConv(c1, cat_channels)   # scale 유지
        self.d1_from_d2 = _CatConv(up_channels, cat_channels)  # upsample 2x
        self.d1_from_d3 = _CatConv(up_channels, cat_channels)  # upsample 4x
        self.d1_from_e4 = _CatConv(c4, cat_channels)   # upsample 8x
        self.d1_fuse = _CatConv(up_channels, up_channels)

        self.out_conv = nn.Conv2d(up_channels, out_channels, kernel_size=1)

    @staticmethod
    def _resize(x: torch.Tensor, size) -> torch.Tensor:
        return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        h3, w3 = e3.shape[-2:]
        h2, w2 = e2.shape[-2:]
        h1, w1 = e1.shape[-2:]

        # d3
        d3 = torch.cat([
            self.d3_from_e1(self._resize(e1, (h3, w3))),
            self.d3_from_e2(self._resize(e2, (h3, w3))),
            self.d3_from_e3(e3),
            self.d3_from_e4(self._resize(e4, (h3, w3))),
        ], dim=1)
        d3 = self.d3_fuse(d3)

        # d2
        d2 = torch.cat([
            self.d2_from_e1(self._resize(e1, (h2, w2))),
            self.d2_from_e2(e2),
            self.d2_from_d3(self._resize(d3, (h2, w2))),
            self.d2_from_e4(self._resize(e4, (h2, w2))),
        ], dim=1)
        d2 = self.d2_fuse(d2)

        # d1
        d1 = torch.cat([
            self.d1_from_e1(e1),
            self.d1_from_d2(self._resize(d2, (h1, w1))),
            self.d1_from_d3(self._resize(d3, (h1, w1))),
            self.d1_from_e4(self._resize(e4, (h1, w1))),
        ], dim=1)
        d1 = self.d1_fuse(d1)

        return self.out_conv(d1)


def build_unet3plus(
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple = (16, 32, 64, 128),
    strides: tuple = (2, 2, 2),  # 다른 build_* 함수와 시그니처 통일용 (미사용)
) -> nn.Module:
    """
    Light-weight 2D UNet 3+ 를 반환합니다.
    """
    return UNet3Plus(in_channels=in_channels, out_channels=out_channels, channels=channels)
