"""
MONAI AttentionUnet 기반 2D Attention U-Net 모델
Step 1: 초기 뇌종양 마스크 생성 모델

기존 U-Net(src/models/unet.py)과 동일한 인코더 폭(16, 32, 64, 128)을 사용하되,
디코더의 skip connection 에 Attention Gate 를 적용하여
종양 영역에 해당하는 특성 맵만 선택적으로 통과시킵니다.
"""

import torch.nn as nn
from monai.networks.nets import AttentionUnet


def build_attention_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple = (16, 32, 64, 128),
    strides: tuple = (2, 2, 2),
    dropout: float = 0.0,
) -> nn.Module:
    """
    MONAI 기반 2D Attention U-Net 을 반환합니다.

    Parameters
    ----------
    in_channels  : 입력 채널 수 (그레이스케일 MRI → 1)
    out_channels : 출력 채널 수 (이진 마스크 → 1)
    channels     : 각 인코더 레이어의 특성 맵 채널 수 (경량 16채널 모드 기본 설정)
    strides      : 각 인코더 다운샘플링 스트라이드
    dropout      : 드롭아웃 확률 (0.0 = 비활성화)
    """
    model = AttentionUnet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        dropout=dropout,
    )
    return model

if __name__ == "__main__":
    import torch
    model = build_attention_unet(1, 1)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print("Attention U-Net Output Shape:", list(out.shape))
