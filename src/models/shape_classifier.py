import logging

import torch
import torch.nn as nn
import torchvision.models as models

log = logging.getLogger(__name__)

def build_shape_classifier(in_channels=1, num_classes=3, pretrained=False):
    """
    ResNet18을 기반으로 하는 경량화 뇌종양 크기/형태 분류 모델
    - 입력: in_channels 채널 MRI (t1ce, t1ce+flair 등)
    - 출력: 3개 클래스 (0: Small, 1: Medium, 2: Large)
    - pretrained: ImageNet 가중치로 초기화 (학습 시 권장, 추론 시 불필요)
    """
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except Exception as e:  # 오프라인 등
            log.warning("ImageNet 가중치를 쓸 수 없어 무작위 초기화합니다 (%s).", e)

    model = models.resnet18(weights=weights)

    # 첫 번째 Conv 레이어를 in_channels 입력에 맞게 수정
    if in_channels != 3:
        old_weight = model.conv1.weight.data
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if weights is not None:
            # RGB 커널을 채널 축으로 합친 뒤 균등 분배한다.
            # 입력 채널이 동일한 값일 때 원래 응답 크기가 유지된다.
            merged = old_weight.sum(dim=1, keepdim=True) / in_channels
            model.conv1.weight.data.copy_(merged.repeat(1, in_channels, 1, 1))

    # 마지막 Fully Connected 레이어를 num_classes(3)에 맞게 수정
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
