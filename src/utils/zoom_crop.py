"""Small-tumor zoom-crop: train on a magnified patch, infer with a two-pass crop."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, label


def crop_box(h: int, w: int, cy: int, cx: int, patch: int) -> Tuple[int, int, int, int]:
    patch = min(patch, h, w)
    half = patch // 2
    y1 = max(0, min(h - patch, cy - half))
    x1 = max(0, min(w - patch, cx - half))
    return y1, x1, y1 + patch, x1 + patch


def _com_from_mask(mask: np.ndarray) -> Tuple[int, int]:
    h, w = mask.shape
    if float(mask.sum()) <= 0:
        return h // 2, w // 2
    cy, cx = center_of_mass(mask > 0.5)
    return int(cy), int(cx)


def largest_component_com(mask: np.ndarray) -> Tuple[int, int]:
    binary = mask > 0.5
    if not np.any(binary):
        h, w = mask.shape
        return h // 2, w // 2
    labeled, n = label(binary)
    if n == 0:
        return _com_from_mask(mask)
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    k = 1 + int(np.argmax(sizes))
    return _com_from_mask((labeled == k).astype(np.float32))


def zoom_tensors(
    image: torch.Tensor,
    gt: torch.Tensor,
    patch_size: int = 64,
    out_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GT 중심으로 patch를 잘라 out_size로 확대. image (C,H,W), gt (1,H,W)."""
    _, h, w = image.shape
    gt2d = gt.squeeze(0).cpu().numpy() if gt.ndim == 3 else gt.cpu().numpy()
    cy, cx = _com_from_mask(gt2d)
    y1, x1, y2, x2 = crop_box(h, w, cy, cx, patch_size)
    img_c = image[:, y1:y2, x1:x2].unsqueeze(0)
    gt_c = gt[:, y1:y2, x1:x2].unsqueeze(0)
    img_z = F.interpolate(img_c, size=(out_size, out_size), mode="bilinear", align_corners=False)
    gt_z = F.interpolate(gt_c, size=(out_size, out_size), mode="nearest")
    return img_z.squeeze(0), gt_z.squeeze(0)


@torch.no_grad()
def refine_with_zoom(
    forward_fn,
    img: torch.Tensor,
    coarse_prob: torch.Tensor,
    patch_size: int = 64,
    loc_thr: float = 0.30,
) -> torch.Tensor:
    """
    1차 확률맵에서 가장 큰 연결요소 중심으로 crop → 확대 재추론 → 원위치 paste.
    img (B,C,H,W), coarse_prob (B,1,H,W) in [0,1].
    """
    bsz, _, h, w = img.shape
    refined = coarse_prob.clone()
    for b in range(bsz):
        loc = (coarse_prob[b, 0] > loc_thr).detach().cpu().numpy()
        if loc.sum() < 5:
            continue
        cy, cx = largest_component_com(loc.astype(np.float32))
        y1, x1, y2, x2 = crop_box(h, w, cy, cx, patch_size)
        crop = img[b : b + 1, :, y1:y2, x1:x2]
        crop_up = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
        pred_up = torch.sigmoid(forward_fn(crop_up))
        pred_patch = F.interpolate(pred_up, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
        refined[b : b + 1, :, y1:y2, x1:x2] = pred_patch
    return refined
