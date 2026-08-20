import os
import torch
import torch.nn as nn
from src.models.shape_classifier import build_shape_classifier
from src.models import build_unet, build_unetplusplus, build_segresnet, build_attention_unet, build_caranet
from src.models.unet3plus import build_unet3plus
from src.models.segresnet import region_logits_to_wt
import numpy as np

class AdaptivePipeline(nn.Module):
    """
    3-Stage Adaptive Pipeline - Stage 1 & 2
    Small Expert는 2.5D(prev/center/next) 입력을 쓰고,
    Large Expert는 ED/TC 2채널 출력을 WT로 합친다.
    """
    def __init__(self, device, in_channels=1):
        super().__init__()
        self.device = device
        self.base_in_channels = in_channels
        
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
        
        self.expert_small = self._load_expert(
            'checkpoints/caranet_best.pt',
            build_caranet,
            'checkpoints/attention_unet_best.pt',
            build_attention_unet,
            fallback_fn=build_caranet,
            device=device,
            in_channels=in_channels * 3,
            out_channels=1,
            desc_primary="Expert 0 (Small): CaraNet",
            desc_secondary="Expert 0 (Small): Attention U-Net (Fallback)",
            desc_fallback="Expert 0 (Small): Default CaraNet",
        )
        self.expert_small.eval()
        self.small_zoom = True
        self.small_zoom_patch = 64
        
        self.expert_medium = self._load_expert(
            'checkpoints/unetplusplus_best.pt',
            build_unetplusplus,
            'checkpoints/unet3plus_best.pt',
            build_unet3plus,
            fallback_fn=build_unetplusplus,
            device=device,
            in_channels=in_channels,
            out_channels=1,
            desc_primary="Expert 1 (Medium): UNet++",
            desc_secondary="Expert 1 (Medium): UNet 3+",
            desc_fallback="Expert 1 (Medium): Default UNet++",
        )
        self.expert_medium.eval()
        
        self.expert_large = self._load_expert(
            'checkpoints/segresnet_best.pt',
            build_segresnet,
            None,
            None,
            fallback_fn=build_segresnet,
            device=device,
            in_channels=in_channels,
            out_channels=2,
            desc_primary="Expert 2 (Large): SegResNet",
            desc_secondary="",
            desc_fallback="Expert 2 (Large): Default SegResNet",
        )
        self.expert_large.eval()

    def _load_expert(
        self,
        primary_ckpt,
        primary_fn,
        secondary_ckpt,
        secondary_fn,
        fallback_fn,
        device,
        in_channels,
        out_channels,
        desc_primary,
        desc_secondary,
        desc_fallback,
    ):
        for ckpt, fn, desc in [(primary_ckpt, primary_fn, desc_primary), (secondary_ckpt, secondary_fn, desc_secondary)]:
            if ckpt and os.path.exists(ckpt):
                print(f"[Router] Loading {desc} ({ckpt})...")
                st = torch.load(ckpt, map_location=device, weights_only=True)
                convs = [v for v in st.values() if hasattr(v, "ndim") and v.ndim == 4]
                ex_in_ch = convs[0].shape[1] if convs else in_channels
                ex_out_ch = convs[-1].shape[0] if convs else out_channels
                try:
                    model = fn(in_channels=ex_in_ch, out_channels=ex_out_ch).to(device)
                    model.load_state_dict(st)
                    return model
                except Exception as e:
                    print(f"[Router] {desc} load failed ({e}), trying default out_channels={out_channels}")
                    model = fn(in_channels=ex_in_ch, out_channels=out_channels).to(device)
                    model.load_state_dict(st, strict=False)
                    return model
        print(f"[Router] Loading {desc_fallback}...")
        return fallback_fn(in_channels=in_channels, out_channels=out_channels).to(device)

    def _match_in_channels(self, img, exp_in_ch):
        c = img.shape[1]
        if c == exp_in_ch:
            return img
        if c % 3 == 0:
            base = c // 3
            if exp_in_ch == base:
                return img[:, base:2 * base]
            if exp_in_ch == c:
                return img
        if c < exp_in_ch and exp_in_ch % c == 0:
            return img.repeat(1, exp_in_ch // c, 1, 1)
        if c > exp_in_ch:
            if c % 3 == 0:
                base = c // 3
                start = base if exp_in_ch <= base else 0
                return img[:, start:start + exp_in_ch]
            return img[:, :exp_in_ch]
        return img.repeat(1, exp_in_ch, 1, 1)[:, :exp_in_ch]

    @torch.no_grad()
    def _forward_expert(self, expert, img):
        exp_in_ch = 1
        for m in expert.modules():
            if isinstance(m, nn.Conv2d):
                exp_in_ch = m.weight.shape[1]
                break
        return expert(self._match_in_channels(img, exp_in_ch))

    def _to_wt_prob(self, logits):
        prob = torch.sigmoid(logits)
        if prob.shape[1] == 1:
            return prob
        return region_logits_to_wt(prob, from_logits=False)

    def forward(self, x, true_class_preds=None):
        """
        x: (B, C, H, W) — C는 중심 모달리티이거나 2.5D(3C).
        """
        if true_class_preds is not None:
            class_preds = true_class_preds
        else:
            cls_in_ch = self.classifier.conv1.weight.shape[1]
            x_cls = self._match_in_channels(x, cls_in_ch)
            class_logits = self.classifier(x_cls)
            _, class_preds = torch.max(class_logits, 1)
        
        B, C, H, W = x.shape
        rough_masks = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
        
        experts = [self.expert_small, self.expert_medium, self.expert_large]
        for c, expert in enumerate(experts):
            idx = (class_preds == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            logits = self._forward_expert(expert, x[idx])
            out = self._to_wt_prob(logits)
            if c == 0 and self.small_zoom:
                from src.utils.zoom_crop import refine_with_zoom
                out = refine_with_zoom(
                    lambda z: self._forward_expert(expert, z),
                    x[idx],
                    out,
                    patch_size=self.small_zoom_patch,
                )
                if out.shape[1] != 1:
                    out = self._to_wt_prob(out)
            rough_masks[idx] = out
                
        return rough_masks, class_preds
