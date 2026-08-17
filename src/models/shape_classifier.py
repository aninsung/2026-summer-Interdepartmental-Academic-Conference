import torch
import torch.nn as nn
import torchvision.models as models

def build_shape_classifier(in_channels=1, num_classes=3):
    """
    ResNet18을 기반으로 하는 경량화 뇌종양 크기/형태 분류 모델
    - 입력: in_channels 채널 MRI (t1ce, t1ce+flair 등)
    - 출력: 3개 클래스 (0: Small, 1: Medium, 2: Large)
    """
    model = models.resnet18(weights=None)
    
    # 첫 번째 Conv 레이어를 in_channels 입력에 맞게 수정
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 마지막 Fully Connected 레이어를 num_classes(3)에 맞게 수정
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
