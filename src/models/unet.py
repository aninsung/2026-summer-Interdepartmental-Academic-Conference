"""
Light-weight 2D U-Net (MONAI 기반)
Step 1: 초기 뇌종양 마스크 생성 모델
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet


def build_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple = (16, 32, 64, 128),
    strides: tuple = (2, 2, 2),
) -> nn.Module:
    """
    MONAI 기반 Light-weight 2D U-Net을 반환합니다.

    Parameters
    ----------
    in_channels  : 입력 채널 수 (그레이스케일 MRI → 1)
    out_channels : 출력 채널 수 (이진 마스크 → 1)
    channels     : 각 인코더 레이어의 특성 맵 채널 수
    strides      : 각 인코더 다운샘플링 스트라이드
    """
    model = MonaiUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=1,
        act="PRELU",
        norm="INSTANCE",
    )
    return model


class DiceLoss(nn.Module):
    """
    이진 Dice Loss.
    DSC = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - DSC
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dsc = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dsc.mean()


def compute_dice(pred_binary: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """이진 예측과 GT 간의 DSC 계산."""
    pred_flat = pred_binary.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dsc = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dsc.item()
