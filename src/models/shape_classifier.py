import logging

import torch
import torch.nn as nn
import torchvision.models as models

log = logging.getLogger(__name__)

SUPPORTED_BACKBONES = ("resnet18", "efficientnet_b0")


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int, pretrained_weight: torch.Tensor | None):
    if in_channels == conv.in_channels:
        return conv
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False,
    )
    if pretrained_weight is not None:
        # RGB → 합산 후 채널 균등 분배
        merged = pretrained_weight.sum(dim=1, keepdim=True) / in_channels
        new.weight.data.copy_(merged.repeat(1, in_channels, 1, 1))
    return new


def infer_backbone_from_state_dict(state_dict: dict) -> str:
    keys = state_dict.keys()
    if any(k.startswith("features.") for k in keys):
        return "efficientnet_b0"
    if "conv1.weight" in state_dict or any(k.startswith("layer1.") for k in keys):
        return "resnet18"
    return "resnet18"


def infer_in_channels_from_state_dict(state_dict: dict, default: int = 1) -> int:
    if "features.0.0.weight" in state_dict:
        return int(state_dict["features.0.0.weight"].shape[1])
    if "conv1.weight" in state_dict:
        return int(state_dict["conv1.weight"].shape[1])
    return default


def build_shape_classifier(
    in_channels=1,
    num_classes=3,
    pretrained=False,
    backbone: str = "resnet18",
):
    """
    뇌종양 크기/형태 분류 모델 (3-class: Small / Medium / Large).
    backbone: 'resnet18' | 'efficientnet_b0'
    """
    backbone = (backbone or "resnet18").lower().replace("-", "_")
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone={backbone!r}; choose from {SUPPORTED_BACKBONES}")

    if backbone == "resnet18":
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception as e:
                log.warning("ImageNet 가중치를 쓸 수 없어 무작위 초기화합니다 (%s).", e)
        model = models.resnet18(weights=weights)
        old_w = model.conv1.weight.data.clone() if weights is not None else None
        model.conv1 = _adapt_first_conv(model.conv1, in_channels, old_w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # efficientnet_b0
    weights = None
    if pretrained:
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        except Exception as e:
            log.warning("ImageNet 가중치를 쓸 수 없어 무작위 초기화합니다 (%s).", e)
    model = models.efficientnet_b0(weights=weights)
    old_conv = model.features[0][0]
    old_w = old_conv.weight.data.clone() if weights is not None else None
    model.features[0][0] = _adapt_first_conv(old_conv, in_channels, old_w)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    return model
