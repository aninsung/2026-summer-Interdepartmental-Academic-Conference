# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

__all__ = ["SegResNet", "SegResNetVAE", "build_segresnet"]


class SegResNet(nn.Module):
    """
    Autoencoder Regularization을 활용한 3D MRI 뇌종양 세그멘테이션 논문
    <https://arxiv.org/pdf/1810.11654.pdf>에 기반한 SegResNet 아키텍처입니다.
    이 모듈은 Variational Autoencoder (VAE) 분기를 포함하지 않습니다.
    2D 및 3D 입력을 모두 지원합니다.

    매개변수(Args):
        spatial_dims: 입력 데이터의 공간 차원. 기본값은 3.
        init_filters: 초기 컨볼루션 레이어의 출력 채널 수. 기본값은 8.
        in_channels: 네트워크의 입력 채널 수. 기본값은 1.
        out_channels: 네트워크의 출력 채널 수. 기본값은 2.
        dropout_prob: 드롭아웃 확률. 기본값은 None (비활성화).
        act: 활성화 함수 종류 및 인자. 기본값은 RELU.
        norm: 특징 정규화 종류 및 인자. 기본값은 GROUP (Group Normalization).
        norm_name: 지원 중단 예정인 정규화 이름 옵션.
        num_groups: Group Norm용 그룹 수. 기본값은 8.
        use_conv_final: 최종 출력을 위한 컨볼루션 블록 추가 여부. 기본값은 True.
        blocks_down: 인코더 각 레이어의 다운샘플링 블록 수. 기본값은 [1,2,2,4].
        blocks_up: 디코더 각 레이어의 업샘플링 블록 수. 기본값은 [1,1,1].
        upsample_mode: 업샘플링 연산 모드 ["deconv", "nontrainable", "pixelshuffle"]. 기본값은 nontrainable.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims`는 2 또는 3만 가능합니다.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"지원 중단 예정 옵션 'norm_name={norm_name}'입니다. 대신 'norm'을 사용하세요.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x


class SegResNetVAE(SegResNet):
    """
    Autoencoder Regularization을 활용한 3D MRI 뇌종양 세그멘테이션 논문에 기반한 SegResNetVAE 아키텍처입니다.
    이 모듈은 Variational Autoencoder (VAE) 분기를 포함합니다.
    2D 및 3D 입력을 모두 지원합니다.

    매개변수(Args):
        input_image_size: 네트워크에 입력되는 이미지 크기. VAE FC 레이어의 입력 채널 크기를 결정하는 데 사용됩니다.
        vae_estimate_std: VAE에서 표준 편차(std)를 추정할지 여부. 기본값은 False.
        vae_default_std: 표준 편차를 추정하지 않는 경우 사용하는 기본값. 기본값은 0.3.
        vae_nz: VAE의 잠재 변수(latent variables) 크기. 기본값은 256 (128은 평균, 128은 표준편차 표현).
        spatial_dims: 입력 데이터의 공간 차원. 기본값은 3.
        init_filters: 초기 컨볼루션 레이어의 출력 채널 수. 기본값은 8.
        in_channels: 네트워크의 입력 채널 수. 기본값은 1.
        out_channels: 네트워크의 출력 채널 수. 기본값은 2.
        dropout_prob: 드롭아웃 확률. 기본값은 None (비활성화).
        act: 활성화 함수 종류 및 인자. 기본값은 RELU.
        norm: 특징 정규화 종류 및 인자. 기본값은 GROUP.
        use_conv_final: 최종 출력을 위한 컨볼루션 블록 추가 여부. 기본값은 True.
        blocks_down: 인코더 각 레이어의 다운샘플링 블록 수.
        blocks_up: 디코더 각 레이어의 업샘플링 블록 수.
        upsample_mode: 업샘플링 연산 모드.
    """

    def __init__(
        self,
        input_image_size: Sequence[int],
        vae_estimate_std: bool = False,
        vae_default_std: float = 0.3,
        vae_nz: int = 256,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: str | tuple = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            use_conv_final=use_conv_final,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode=upsample_mode,
        )

        self.input_image_size = input_image_size
        self.smallest_filters = 16

        zoom = 2 ** (len(self.blocks_down) - 1)
        self.fc_insize = [s // (2 * zoom) for s in self.input_image_size]

        self.vae_estimate_std = vae_estimate_std
        self.vae_default_std = vae_default_std
        self.vae_nz = vae_nz
        self._prepare_vae_modules()
        self.vae_conv_final = self._make_final_conv(in_channels)

    def _prepare_vae_modules(self):
        zoom = 2 ** (len(self.blocks_down) - 1)
        v_filters = self.init_filters * zoom
        total_elements = int(self.smallest_filters * np.prod(self.fc_insize))

        self.vae_down = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, v_filters, self.smallest_filters, stride=2, bias=True),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.smallest_filters),
            self.act_mod,
        )
        self.vae_fc1 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc2 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc3 = nn.Linear(self.vae_nz, total_elements)

        self.vae_fc_up_sample = nn.Sequential(
            get_conv_layer(self.spatial_dims, self.smallest_filters, v_filters, kernel_size=1),
            get_upsample_layer(self.spatial_dims, v_filters, upsample_mode=self.upsample_mode),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
        )

    def _get_vae_loss(self, net_input: torch.Tensor, vae_input: torch.Tensor):
        x_vae = self.vae_down(vae_input)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)
        z_mean = self.vae_fc1(x_vae)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)

        if self.vae_estimate_std:
            z_sigma = self.vae_fc2(x_vae)
            z_sigma = F.softplus(z_sigma)
            vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)

            x_vae = z_mean + z_sigma * z_mean_rand
        else:
            z_sigma = self.vae_default_std
            vae_reg_loss = torch.mean(z_mean**2)

            x_vae = z_mean + z_sigma * z_mean_rand

        x_vae = self.vae_fc3(x_vae)
        x_vae = self.act_mod(x_vae)
        x_vae = x_vae.view([-1, self.smallest_filters] + self.fc_insize)
        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = up(x_vae)
            x_vae = upl(x_vae)

        x_vae = self.vae_conv_final(x_vae)
        vae_mse_loss = F.mse_loss(net_input, x_vae)
        vae_loss = vae_reg_loss + vae_mse_loss
        return vae_loss

    def forward(self, x):
        net_input = x
        x, down_x = self.encode(x)
        down_x.reverse()

        vae_input = x
        x = self.decode(x, down_x)

        if self.training:
            vae_loss = self._get_vae_loss(net_input, vae_input)
            return x, vae_loss

        return x, None


# ── 기존 프로젝트 인터페이스 및 외부 모듈 공유 ───────────────────
from src.models.unet import DiceLoss, compute_dice

def build_segresnet(
    in_channels: int = 1,
    out_channels: int = 1,
    init_filters: int = 16,
    dropout_prob: float = 0.2,
) -> nn.Module:
    """
    프로젝트의 기존 SegResNet 연동 인터페이스입니다.
    """
    model = SegResNet(
        spatial_dims=2,
        init_filters=init_filters,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout_prob,
    )
    return model
