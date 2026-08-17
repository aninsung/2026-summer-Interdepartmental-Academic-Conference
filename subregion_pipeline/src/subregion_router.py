"""
subregion_pipeline/src/subregion_router.py
--------------------------------------------
Subregion Dynamic Router
Subregion Classifier 결과를 바탕으로 입력 슬라이스를 해당하는 전문가(Expert) 백본 모델로 라우팅합니다.

  - Class 0 (ET): Expert 0 (Attention U-Net + Zoom-Refiner)
  - Class 1 (TC): Expert 1 (UNet++)
  - Class 2 (WT): Expert 2 (SegResNet)
"""

import os
from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from subregion_pipeline.src.subregion_classifier import SubregionClassifier
from src.models.attention_unet import build_attention_unet
from src.models.unetplusplus import build_unetplusplus
from src.models.segresnet import build_segresnet


class SubregionAdaptivePipeline(nn.Module):
    def __init__(
        self,
        classifier_path: str = "subregion_pipeline/checkpoints/subregion_classifier.pt",
        expert_et_path: str = "subregion_pipeline/checkpoints/expert_et_best.pt",
        expert_tc_path: str = "subregion_pipeline/checkpoints/expert_tc_best.pt",
        expert_wt_path: str = "subregion_pipeline/checkpoints/expert_wt_best.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        # 1. Subregion Classifier
        self.classifier = SubregionClassifier(in_channels=2, num_classes=3).to(self.device)
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.classifier.eval()

        # 2. Experts
        self.expert_et = build_attention_unet(in_channels=2, out_channels=1).to(self.device)
        if os.path.exists(expert_et_path):
            self.expert_et.load_state_dict(torch.load(expert_et_path, map_location=self.device))
        self.expert_et.eval()

        self.expert_tc = build_unetplusplus(in_channels=2, out_channels=1).to(self.device)
        if os.path.exists(expert_tc_path):
            self.expert_tc.load_state_dict(torch.load(expert_tc_path, map_location=self.device))
        self.expert_tc.eval()

        self.expert_wt = build_segresnet(in_channels=2, out_channels=1).to(self.device)
        if os.path.exists(expert_wt_path):
            self.expert_wt.load_state_dict(torch.load(expert_wt_path, map_location=self.device))
        self.expert_wt.eval()

    def route_slice(self, img_tensor: torch.Tensor) -> int:
        """
        img_tensor: (1, 2, H, W) -> predicted class (0=ET, 1=TC, 2=WT)
        """
        with torch.no_grad():
            logits = self.classifier(img_tensor)
            pred_c = torch.argmax(logits, dim=1).item()
        return pred_c

    def predict_mask(self, img_tensor: torch.Tensor, class_override: int = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Returns: (rough_mask_binary, soft_prob_map, chosen_class)
        """
        if class_override is not None:
            c = class_override
        else:
            c = self.route_slice(img_tensor)

        with torch.no_grad():
            if c == 0:
                # ET Expert (Attention U-Net + Zoom-Refiner)
                out = torch.sigmoid(self.expert_et(img_tensor))
                mask_50 = (out > 0.35).squeeze()
                area_50 = torch.sum(mask_50).item()
                if 0 < area_50 < 300:
                    y_pts, x_pts = torch.where(mask_50)
                    cy, cx = int(torch.mean(y_pts.float()).item()), int(torch.mean(x_pts.float()).item())
                    half = 28
                    H, W = img_tensor.shape[-2:]
                    y1, y2 = max(0, cy - half), min(H, cy + half)
                    x1, x2 = max(0, cx - half), min(W, cx + half)
                    crop_img = img_tensor[:, :, y1:y2, x1:x2]
                    crop_zoom = F.interpolate(crop_img, size=(H, W), mode='bilinear', align_corners=False)
                    out_zoom = torch.sigmoid(self.expert_et(crop_zoom))
                    out_crop_back = F.interpolate(out_zoom, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
                    out_full = out.clone()
                    out_full[:, :, y1:y2, x1:x2] = (out_full[:, :, y1:y2, x1:x2] + out_crop_back) / 2.0
                    out = out_full
                thresh = 0.38
            elif c == 1:
                # TC Expert (UNet++)
                out = torch.sigmoid(self.expert_tc(img_tensor))
                thresh = 0.5
            else:
                # WT Expert (SegResNet)
                out = torch.sigmoid(self.expert_wt(img_tensor))
                thresh = 0.5

            prob_np = out.squeeze().cpu().numpy()
            binary_np = (prob_np > thresh).astype(np.float32)

        return binary_np, prob_np, c
