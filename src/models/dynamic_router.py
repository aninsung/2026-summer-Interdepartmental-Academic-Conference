import torch
import torch.nn as nn
from src.models.shape_classifier import build_shape_classifier
from src.models import build_unet, build_unetplusplus, build_segresnet, build_attention_unet
import numpy as np

class AdaptivePipeline(nn.Module):
    """
    3-Stage Adaptive Pipeline - Stage 1 & 2 
    1. MRI(T1ce) 입력 -> Shape Classifier가 종양 크기 클래스(0, 1, 2) 예측
    2. 클래스에 맞춰 알맞은 Expert Backbone(UNet, UNet++, SegResNet)을 선택하여 분할(Segmentation) 수행
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # 1. Shape Classifier 로드
        self.classifier = build_shape_classifier(num_classes=3).to(device)
        self.classifier.load_state_dict(torch.load('checkpoints/shape_classifier_best.pt', map_location=device, weights_only=True))
        self.classifier.eval()
        
        # 2. Expert 백본 모델 로드
        print("[Router] Loading Expert 0 (Small): Attention U-Net...")
        self.expert_small = build_attention_unet(in_channels=1, out_channels=1).to(device)
        self.expert_small.load_state_dict(torch.load('checkpoints/attention_unet_best.pt', map_location=device, weights_only=True))
        self.expert_small.eval()
        
        print("[Router] Loading Expert 1 (Medium): UNet++...")
        self.expert_medium = build_unetplusplus(in_channels=1, out_channels=1).to(device)
        self.expert_medium.load_state_dict(torch.load('checkpoints/unetplusplus_best.pt', map_location=device, weights_only=True))
        self.expert_medium.eval()
        
        print("[Router] Loading Expert 2 (Large): SegResNet...")
        self.expert_large = build_segresnet(in_channels=1, out_channels=1).to(device)
        self.expert_large.load_state_dict(torch.load('checkpoints/segresnet_best.pt', map_location=device, weights_only=True))
        self.expert_large.eval()
        
    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, 1, H, W) 텐서
        반환: rough_mask (B, 1, H, W) 텐서, class_preds (B,) 텐서
        """
        # 1. 형태/크기 분류
        class_logits = self.classifier(x)
        _, class_preds = torch.max(class_logits, 1)
        
        # 2. 결과 저장용 텐서
        rough_masks = torch.zeros_like(x)
        
        # 3. 클래스별로 라우팅 (배치 내 개별 처리)
        for i in range(x.size(0)):
            c = class_preds[i].item()
            img_slice = x[i:i+1] # (1, 1, H, W)
            
            if c == 0:
                out = self.expert_small(img_slice)
            elif c == 1:
                out = self.expert_medium(img_slice)
            else:
                out = self.expert_large(img_slice)
                
            rough_masks[i:i+1] = torch.sigmoid(out)
            
        return rough_masks, class_preds
