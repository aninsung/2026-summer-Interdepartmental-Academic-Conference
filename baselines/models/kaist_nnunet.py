"""BraTS 2021 최종 1위(KAIST MRI Lab)의 nnU-Net 확장을 2D로 각색한 베이스라인.

원 논문
    Luu, H.M., Park, S.-H. "Extending nn-UNet for Brain Tumor Segmentation."
    BrainLes @ MICCAI 2021. https://arxiv.org/abs/2112.04653
    코드: https://github.com/rixez/Brats21_KAIST_MRI_Lab

논문이 nnU-Net 대비 제시한 세 가지 변경을 그대로 옮긴다.
    1. 인코더만 비대칭으로 2배 확장한다. 디코더 필터 수는 nnU-Net 원본을
       유지하고, 인코더 최대 필터 수는 512로 제한한다.
    2. 모든 BatchNorm 을 GroupNorm(32 그룹)으로 교체한다.
    3. 디코더에 axial attention 을 넣어 장거리 의존성을 선형 비용으로 얻는다.

본 프로젝트는 2D 단일 슬라이스 이진 분할이므로 3D 패치, 4모달리티,
region-based 3채널 출력, 5-fold 앙상블은 제외했다. 원 논문과의 차이는
baselines/README.md 에 정리해 두었다.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """32 그룹을 기본으로 하되 채널 수로 나누어떨어지게 보정한다."""
    return nn.GroupNorm(math.gcd(max_groups, channels), channels)


class ConvGnLReLU(nn.Module):
    """nnU-Net 기본 단위: 3x3 Conv - Norm - LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = _group_norm(out_ch)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Stage(nn.Module):
    """Conv 두 개로 이루어진 해상도 단계. 첫 Conv 의 stride 로 다운샘플한다."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, n_convs: int = 2):
        super().__init__()
        layers = [ConvGnLReLU(in_ch, out_ch, stride=stride)]
        layers += [ConvGnLReLU(out_ch, out_ch) for _ in range(n_convs - 1)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AxialAttention2d(nn.Module):
    """행 방향과 열 방향에 각각 self-attention 을 적용한다.

    2D 전체에 대한 attention 은 비용이 (HW)^2 로 늘지만, 축별로 나누면
    H*W^2 + W*H^2 로 줄어든다. 원 논문이 3D 에서 쓴 축 분해를 2D 로 옮긴 것이다.
    """

    def __init__(self, channels: int, size: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = math.gcd(channels, num_heads) or 1

        self.norm_row = _group_norm(channels)
        self.norm_col = _group_norm(channels)
        self.attn_row = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_col = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        # 축별 학습형 위치 임베딩
        self.pos_row = nn.Parameter(torch.zeros(1, size, channels))
        self.pos_col = nn.Parameter(torch.zeros(1, size, channels))
        nn.init.trunc_normal_(self.pos_row, std=0.02)
        nn.init.trunc_normal_(self.pos_col, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # 행 방향: 각 행을 길이 W 인 시퀀스로 본다.
        r = self.norm_row(x).permute(0, 2, 3, 1).reshape(b * h, w, c)
        r = r + self.pos_row[:, :w, :]
        r, _ = self.attn_row(r, r, r, need_weights=False)
        x = x + r.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # 열 방향: 각 열을 길이 H 인 시퀀스로 본다.
        col = self.norm_col(x).permute(0, 3, 2, 1).reshape(b * w, h, c)
        col = col + self.pos_col[:, :h, :]
        col, _ = self.attn_col(col, col, col, need_weights=False)
        x = x + col.reshape(b, w, h, c).permute(0, 3, 2, 1)
        return x


class KaistNNUNet2D(nn.Module):
    """비대칭 인코더 + GroupNorm + axial attention 디코더를 갖춘 2D U-Net.

    Args:
        in_channels: 입력 모달리티 채널 수.
        out_channels: 출력 채널 수(이진 분할이면 1).
        encoder_channels: 인코더 필터 수. nnU-Net 기본값을 2배로 키우고 512 로 제한한 값.
        decoder_channels: 디코더 필터 수. nnU-Net 원본 값을 그대로 둔다.
        input_size: 입력 한 변의 길이. axial attention 위치 임베딩 크기 계산에 쓴다.
        attention_max_size: 이 해상도 이하의 디코더 단계에만 attention 을 넣는다.
        deep_supervision: 학습 시 보조 출력을 함께 반환한다.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        encoder_channels: Sequence[int] = (64, 128, 256, 512, 512),
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        input_size: int = 128,
        attention_max_size: int = 32,
        deep_supervision: bool = True,
    ):
        super().__init__()
        enc = list(encoder_channels)
        dec = list(decoder_channels)
        if len(dec) != len(enc) - 1:
            raise ValueError(
                f"디코더 단계 수({len(dec)})는 인코더 단계 수 - 1({len(enc) - 1})이어야 합니다."
            )

        self.deep_supervision = deep_supervision
        self.n_levels = len(enc)

        # ── 인코더 ──
        self.encoder = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(enc):
            self.encoder.append(Stage(prev, ch, stride=1 if i == 0 else 2))
            prev = ch

        # ── 디코더 ──
        # 인코더 i 단계의 해상도는 input_size / 2**i 이다.
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.ds_heads = nn.ModuleList()

        prev = enc[-1]
        for j, ch in enumerate(dec):
            skip_level = self.n_levels - 2 - j
            size = input_size // (2 ** skip_level)
            self.upsamples.append(nn.ConvTranspose2d(prev, ch, kernel_size=2, stride=2))
            self.dec_blocks.append(Stage(ch + enc[skip_level], ch))
            self.attentions.append(
                AxialAttention2d(ch, size=size) if size <= attention_max_size else nn.Identity()
            )
            self.ds_heads.append(nn.Conv2d(ch, out_channels, kernel_size=1))
            prev = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        skips: List[torch.Tensor] = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)

        outputs: List[torch.Tensor] = []
        x = skips[-1]
        for j, (up, block, attn, head) in enumerate(
            zip(self.upsamples, self.dec_blocks, self.attentions, self.ds_heads)
        ):
            skip = skips[self.n_levels - 2 - j]
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
            x = attn(x)
            outputs.append(head(x))

        # 최고 해상도 출력을 앞에 두고 반환한다.
        outputs = outputs[::-1]
        if self.deep_supervision and self.training:
            return outputs
        return outputs[0]


def build_kaist_nnunet(
    in_channels: int = 2,
    out_channels: int = 1,
    input_size: int = 128,
    deep_supervision: bool = True,
    attention_max_size: int = 32,
) -> KaistNNUNet2D:
    """논문 설정(인코더 2배 확장, 최대 512 필터)을 기본값으로 하는 생성 함수."""
    return KaistNNUNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        encoder_channels=(64, 128, 256, 512, 512),
        decoder_channels=(256, 128, 64, 32),
        input_size=input_size,
        attention_max_size=attention_max_size,
        deep_supervision=deep_supervision,
    )
