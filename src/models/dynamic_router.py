import os
import torch
import torch.nn as nn
from src.models.shape_classifier import build_shape_classifier
from src.models import build_unet, build_unetplusplus, build_segresnet, build_attention_unet, build_caranet
from src.models.unet3plus import build_unet3plus
import numpy as np

class AdaptivePipeline(nn.Module):
    """
    3-Stage Adaptive Pipeline - Stage 1 & 2 
    1. MRI(T1ce) 입력 -> Shape Classifier가 종양 크기 클래스(0, 1, 2) 예측
    2. 클래스에 맞춰 알맞은 Expert Backbone(Attention U-Net, UNet++, SegResNet)을 선택하여 분할(Segmentation) 수행
    """
    def __init__(self, device, in_channels=1):
        super().__init__()
        self.device = device
        
        # 1. Shape Classifier 로드 (가중치 파일의 채널 수 자동 감지)
        cls_ckpt = 'checkpoints/shape_classifier_best.pt'
        cls_in_channels = in_channels
        if os.path.exists(cls_ckpt):
            state_dict = torch.load(cls_ckpt, map_location=device, weights_only=True)
            if 'conv1.weight' in state_dict:
                cls_in_channels = state_dict['conv1.weight'].shape[1]
            self.classifier = build_shape_classifier(in_channels=cls_in_channels, num_classes=3).to(device)
            self.classifier.load_state_dict(state_dict)
        else:
            self.classifier = build_shape_classifier(in_channels=in_channels, num_classes=3).to(device)
        self.classifier.eval()
        
        # 2. Expert 백본 모델 로드 (가중치 파일의 채널 수 자동 감지)
        # Expert 0 (Small): CaraNet (Sniper for Small Objects)
        self.expert_small = self._load_expert(
            'checkpoints/caranet_best.pt',
            build_caranet,
            'checkpoints/attention_unet_best.pt',
            build_attention_unet,
            fallback_fn=build_caranet,
            device=device,
            in_channels=in_channels,
            desc_primary="Expert 0 (Small): CaraNet",
            desc_secondary="Expert 0 (Small): Attention U-Net (Fallback)",
            desc_fallback="Expert 0 (Small): Default CaraNet"
        )
        self.expert_small.eval()
        
        # Expert 1 (Medium): UNet++ / UNet 3+
        self.expert_medium = self._load_expert(
            'checkpoints/unetplusplus_best.pt',
            build_unetplusplus,
            'checkpoints/unet3plus_best.pt',
            build_unet3plus,
            fallback_fn=build_unetplusplus,
            device=device,
            in_channels=in_channels,
            desc_primary="Expert 1 (Medium): UNet++",
            desc_secondary="Expert 1 (Medium): UNet 3+",
            desc_fallback="Expert 1 (Medium): Default UNet++"
        )
        self.expert_medium.eval()
        
        # Expert 2 (Large): SegResNet
        self.expert_large = self._load_expert(
            'checkpoints/segresnet_best.pt',
            build_segresnet,
            None,
            None,
            fallback_fn=build_segresnet,
            device=device,
            in_channels=in_channels,
            desc_primary="Expert 2 (Large): SegResNet",
            desc_secondary="",
            desc_fallback="Expert 2 (Large): Default SegResNet"
        )
        self.expert_large.eval()

    def _load_expert(self, primary_ckpt, primary_fn, secondary_ckpt, secondary_fn, fallback_fn, device, in_channels, desc_primary, desc_secondary, desc_fallback):
        for ckpt, fn, desc in [(primary_ckpt, primary_fn, desc_primary), (secondary_ckpt, secondary_fn, desc_secondary)]:
            if ckpt and os.path.exists(ckpt):
                print(f"[Router] Loading {desc} ({ckpt})...")
                st = torch.load(ckpt, map_location=device, weights_only=True)
                ex_in_ch = in_channels
                for v in st.values():
                    if hasattr(v, 'ndim') and v.ndim == 4:
                        ex_in_ch = v.shape[1]
                        break
                model = fn(in_channels=ex_in_ch, out_channels=1).to(device)
                model.load_state_dict(st)
                return model
        print(f"[Router] Loading {desc_fallback}...")
        return fallback_fn(in_channels=in_channels, out_channels=1).to(device)
        
    @torch.no_grad()
    def _forward_expert(self, expert, img):
        exp_in_ch = 1
        for m in expert.modules():
            if isinstance(m, nn.Conv2d):
                exp_in_ch = m.weight.shape[1]
                break
        if img.shape[1] != exp_in_ch:
            if img.shape[1] == 1:
                img_inp = img.repeat(1, exp_in_ch, 1, 1)
            else:
                img_inp = img[:, :exp_in_ch, :, :]
        else:
            img_inp = img
        return expert(img_inp)

    def forward(self, x, true_class_preds=None):
        """
        x: (B, C, H, W) 텐서
        true_class_preds: (B,) 정답 기반 수학적 클래스 (옵션)
        반환: rough_mask (B, 1, H, W) 텐서, class_preds (B,) 텐서
        """
        # 1. 형태/크기 분류 (채널 수 차이 발생 시 자동 적응)
        if true_class_preds is not None:
            class_preds = true_class_preds
        else:
            cls_in_ch = self.classifier.conv1.weight.shape[1]
            if x.shape[1] != cls_in_ch:
                if x.shape[1] == 1:
                    x_cls = x.repeat(1, cls_in_ch, 1, 1)
                else:
                    x_cls = x[:, :cls_in_ch, :, :]
            else:
                x_cls = x
            class_logits = self.classifier(x_cls)
            _, class_preds = torch.max(class_logits, 1)
        
        # 2. 결과 저장용 텐서 (마스크는 항상 1채널)
        B, C, H, W = x.shape
        rough_masks = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
        
        # 3. 클래스별 배치 라우팅
        experts = [self.expert_small, self.expert_medium, self.expert_large]
        for c, expert in enumerate(experts):
            idx = (class_preds == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            out = torch.sigmoid(self._forward_expert(expert, x[idx]))
            rough_masks[idx] = out
                
        return rough_masks, class_preds
