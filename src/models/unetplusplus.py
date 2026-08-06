"""
MONAI BasicUNetPlusPlus 기반 2D UNet++ 모델 구현
Step 1: 초기 뇌종양 마스크 생성 모델
"""

import torch
import torch.nn as nn
from monai.networks.nets import BasicUNetPlusPlus

class UNetPlusPlus(nn.Module):
    """
    MONAI BasicUNetPlusPlus의 래퍼 클래스입니다.
    출력으로 리스트가 아닌 단일 텐서를 반환하여 기존 pipeline과의 호환성을 유지합니다.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple = (16, 32, 64, 128, 256, 16),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            deep_supervision=False,
            act="PRELU",
            norm="INSTANCE",
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        # deep_supervision=False 이므로 outputs는 [output_tensor] 형태의 리스트입니다.
        return outputs[0]

def build_unetplusplus(
    in_channels: int = 1,
    out_channels: int = 1,
    features: tuple = (16, 32, 64, 128, 256, 16),
    dropout: float = 0.0,
) -> nn.Module:
    """
    UNetPlusPlus 모델 객체를 생성하여 반환하는 헬퍼 함수입니다.
    """
    return UNetPlusPlus(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        dropout=dropout,
    )
