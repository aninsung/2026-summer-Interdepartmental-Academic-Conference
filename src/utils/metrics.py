"""Segmentation metrics shared across training, RL env, and evaluation."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice(a: np.ndarray, b: np.ndarray, smooth: float = 1e-5) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    return float((2.0 * (a * b).sum() + smooth) / (a.sum() + b.sum() + smooth))


def hd95(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    dist_b: Optional[np.ndarray] = None,
) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not np.any(a) or not np.any(b):
        return min(30.0, float(mask_a.shape[0]))

    dist_a = distance_transform_edt(~a)
    if dist_b is None:
        dist_b = distance_transform_edt(~b)
    d_ab = dist_b[a]
    d_ba = dist_a[b]
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def precision(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    """TP / (TP + FP). 낮으면 과분할(경계가 GT 밖으로 번짐)."""
    p, g = pred.ravel().astype(float), gt.ravel().astype(float)
    tp = (p * g).sum()
    return float((tp + smooth) / (p.sum() + smooth))


def recall(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    """TP / (TP + FN). 낮으면 과소분할(종양을 놓침)."""
    p, g = pred.ravel().astype(float), gt.ravel().astype(float)
    tp = (p * g).sum()
    return float((tp + smooth) / (g.sum() + smooth))


def apply_monotonic_dsc_gate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    if dice(candidate, gt) + eps < dice(baseline, gt):
        return baseline.copy()
    return candidate.copy()


def gt_size_class(gt: np.ndarray) -> int:
    area = float(np.sum(gt))
    if area < 300:
        return 0
    if area < 700:
        return 1
    return 2


def filter_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """min_size 미만 연결요소를 제거한다. 전부 사라지면 원본 이진화 마스크를 유지한다."""
    from scipy.ndimage import label as cc_label

    if min_size <= 0:
        return mask.astype(np.float32)
    binary = (mask > 0.5).astype(np.uint8)
    labeled, n = cc_label(binary)
    if n == 0:
        return binary.astype(np.float32)
    out = np.zeros_like(binary, dtype=np.float32)
    kept = 0
    for i in range(1, n + 1):
        comp = labeled == i
        if int(comp.sum()) >= min_size:
            out[comp] = 1.0
            kept += 1
    if kept == 0:
        return binary.astype(np.float32)
    return out
