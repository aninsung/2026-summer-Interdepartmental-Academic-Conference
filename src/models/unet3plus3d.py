import torch
import torch.nn as nn

class _ConvBlock3D(nn.Module):
    """Conv-BN-PReLU x2 블록 3D 버전."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _CatConv3D(nn.Module):
    """스케일을 맞춘 뒤 채널을 투영하는 단위 블록 3D 버전."""
    def __init__(self, in_ch: int, cat_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, cat_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(cat_channels),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3Plus3D(nn.Module):
    """
    Light-weight 3D UNet 3+ (Huang et al., 2020 의 Full-scale Skip Connection 구조를 3D로 확장).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: tuple = (32, 64, 128, 256),
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        cat_channels = c1                      # 각 스케일을 투영할 통일 채널 수
        up_channels = cat_channels * 4         # 디코더 스테이지 출력 채널 (4개 스케일 concat)

        # ── 인코더 ──────────────────────────────────────────
        self.enc1 = _ConvBlock3D(in_channels, c1)
        self.enc2 = _ConvBlock3D(c1, c2)
        self.enc3 = _ConvBlock3D(c2, c3)
        self.enc4 = _ConvBlock3D(c3, c4)          # bottleneck
        self.pool = nn.MaxPool3d(2)

        # ── 디코더 d3 (H/4 스케일) ──────────────────────────
        self.d3_from_e1 = _CatConv3D(c1, cat_channels)   # maxpool 4x
        self.d3_from_e2 = _CatConv3D(c2, cat_channels)   # maxpool 2x
        self.d3_from_e3 = _CatConv3D(c3, cat_channels)   # scale 유지
        self.d3_from_e4 = _CatConv3D(c4, cat_channels)   # upsample 2x
        self.d3_fuse = _CatConv3D(up_channels, up_channels)

        # ── 디코더 d2 (H/2 스케일) ──────────────────────────
        self.d2_from_e1 = _CatConv3D(c1, cat_channels)   # maxpool 2x
        self.d2_from_e2 = _CatConv3D(c2, cat_channels)   # scale 유지
        self.d2_from_d3 = _CatConv3D(up_channels, cat_channels)  # upsample 2x
        self.d2_from_e4 = _CatConv3D(c4, cat_channels)   # upsample 4x
        self.d2_fuse = _CatConv3D(up_channels, up_channels)

        # ── 디코더 d1 (H 스케일, 최종) ───────────────────────
        self.d1_from_e1 = _CatConv3D(c1, cat_channels)   # scale 유지
        self.d1_from_d2 = _CatConv3D(up_channels, cat_channels)  # upsample 2x
        self.d1_from_d3 = _CatConv3D(up_channels, cat_channels)  # upsample 4x
        self.d1_from_e4 = _CatConv3D(c4, cat_channels)   # upsample 8x
        self.d1_fuse = _CatConv3D(up_channels, up_channels)

        # 출력 헤드 정의
        self.out_conv = nn.Conv3d(up_channels, out_channels, kernel_size=1)

    @staticmethod
    def _resize(x: torch.Tensor, size) -> torch.Tensor:
        return nn.functional.interpolate(x, size=size, mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d, h, w = e3.shape[-3:]
        d2_, h2, w2 = e2.shape[-3:]
        d1, h1, w1 = e1.shape[-3:]

        # d3
        d3_cat = torch.cat([
            self.d3_from_e1(self._resize(e1, (d, h, w))),
            self.d3_from_e2(self._resize(e2, (d, h, w))),
            self.d3_from_e3(e3),
            self.d3_from_e4(self._resize(e4, (d, h, w))),
        ], dim=1)
        d3 = self.d3_fuse(d3_cat)

        # d2
        d2_cat = torch.cat([
            self.d2_from_e1(self._resize(e1, (d2_, h2, w2))),
            self.d2_from_e2(e2),
            self.d2_from_d3(self._resize(d3, (d2_, h2, w2))),
            self.d2_from_e4(self._resize(e4, (d2_, h2, w2))),
        ], dim=1)
        d2 = self.d2_fuse(d2_cat)

        # d1
        d1_cat = torch.cat([
            self.d1_from_e1(e1),
            self.d1_from_d2(self._resize(d2, (d1, h1, w1))),
            self.d1_from_d3(self._resize(d3, (d1, h1, w1))),
            self.d1_from_e4(self._resize(e4, (d1, h1, w1))),
        ], dim=1)
        d1 = self.d1_fuse(d1_cat)

        # 최종 출력
        out0 = self.out_conv(d1)
        return out0
