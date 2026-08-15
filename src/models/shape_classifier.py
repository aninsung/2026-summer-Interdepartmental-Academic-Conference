import torch
import torch.nn as nn
import torchvision.models as models

def build_shape_classifier(num_classes=3):
    """
    ResNet18을 기반으로 하는 경량화 뇌종양 크기/형태 분류 모델
    - 입력: 1채널 흑백 MRI (t1ce)
    - 출력: 3개 클래스 (0: Small, 1: Medium, 2: Large)
    """
    # 1. ImageNet 사전학습 가중치가 없는 빈 ResNet18 생성 (의료영상은 채널이 다르므로 빈 모델 사용)
    model = models.resnet18(weights=None)
    
    # 2. 첫 번째 Conv 레이어를 1채널 입력으로 수정
    # 원본: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 3. 마지막 Fully Connected 레이어를 num_classes(3)에 맞게 수정
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
