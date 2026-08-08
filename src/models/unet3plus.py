"""
UNet 3+ (UNet3+) 외부 경량 코드 기반 프로젝트 적용 버전 (BatchNorm + 32ch 상향 튜닝)
"""

import torch
import torch.nn as nn

class _ConvBlock(nn.Module):
    """Conv-BN-PReLU x2 블록 (일반화 성능 강화를 위해 BatchNorm2d 적용)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _CatConv(nn.Module):
    """스케일을 맞춘 뒤 채널을 CatChannels로 투영하는 단위 블록 (Conv-BN-PReLU)."""

    def __init__(self, in_ch: int, cat_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, cat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cat_channels),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3Plus(nn.Module):
    """
    Light-weight 2D UNet 3+ (Huang et al., 2020 의 Full-scale Skip Connection 구조).
    가져오신 외부 코드를 기반으로 하되, 채널 크기를 한 단계 상향(32~256)하고 정규화를 BatchNorm으로
    개선하여 표현력과 수렴 성능을 비약적으로 업그레이드하였습니다.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple = (32, 64, 128, 256),  # 표현력을 위해 기본값을 32채널 시작으로 상향
        DSV: bool = False,
    ):
        super().__init__()
        self.DSV = DSV
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

        # 출력 헤드 정의
        self.out_conv = nn.Conv2d(up_channels, out_channels, kernel_size=1)
        
        # Deep Supervision 용 보조 출력 헤드 (DSV=True 시 프로젝트 호환용)
        if self.DSV:
            self.out_d2 = nn.Conv2d(up_channels, out_channels, kernel_size=1)
            self.out_d3 = nn.Conv2d(up_channels, out_channels, kernel_size=1)
            self.out_e4 = nn.Conv2d(c4, out_channels, kernel_size=1)

    @staticmethod
    def _resize(x: torch.Tensor, size) -> torch.Tensor:
        return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
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

        # 최종 출력
        out0 = self.out_conv(d1)

        if self.DSV:
            # Deep Supervision 연동 활성화 시 (출력 크기를 H, W 로 보간하여 리스트 반환)
            out1 = nn.functional.interpolate(self.out_d2(d2), size=(h1, w1), mode='bilinear', align_corners=False)
            out2 = nn.functional.interpolate(self.out_d3(d3), size=(h1, w1), mode='bilinear', align_corners=False)
            out3 = nn.functional.interpolate(self.out_e4(e4), size=(h1, w1), mode='bilinear', align_corners=False)
            # 프로젝트 loss 함수 시그니처 호환을 위해 5개 출력 스케일 리스트 반환
            return [out0, out1, out2, out3, out3]
        else:
            return out0


def build_unet3plus(
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple = (32, 64, 128, 256),  # 32채널 기본값
    DSV: bool = False,
    strides: tuple = (2, 2, 2),  # 다른 build_* 함수와 시그니처 통일용 (미사용)
) -> nn.Module:
    """
    Light-weight 2D UNet 3+ 를 반환합니다.
    """
    return UNet3Plus(in_channels=in_channels, out_channels=out_channels, channels=channels, DSV=DSV)
