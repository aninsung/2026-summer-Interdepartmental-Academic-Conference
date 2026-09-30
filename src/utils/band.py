"""Stage 2 마스크의 경계 띠. 띠 안에서만 픽셀을 켜거나 끈다.

고칠 곳은 Stage 2 확률이 애매하거나, 마스크 경계에서 몇 픽셀 안쪽·바깥이다.
띠 밖은 Stage 2 라벨을 유지한다.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

OFF, KEEP, ON = 0, 1, 2


def edit_band(
    probability: np.ndarray,
    mask: np.ndarray,
    prob_lo: float = 0.35,
    prob_hi: float = 0.65,
    radius: int = 2,
) -> np.ndarray:
    """수정 가능한 픽셀. 확률 구간 또는 마스크 경계 ±radius."""
    prob = np.asarray(probability, dtype=np.float32)
    binary = np.asarray(mask) > 0.5
    uncertain = (prob >= prob_lo) & (prob <= prob_hi)
    if radius <= 0:
        return uncertain
    struct = np.ones((3, 3), dtype=bool)
    outer = binary_dilation(binary, structure=struct, iterations=radius)
    inner = binary_erosion(binary, structure=struct, iterations=radius)
    boundary = outer & ~inner
    return uncertain | boundary


def action_target(stage2: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """픽셀 정답 동작. GT와 Stage 2가 같으면 유지, 다르면 켜기 또는 끄기."""
    pred = np.asarray(stage2) > 0.5
    truth = np.asarray(gt) > 0.5
    target = np.full(pred.shape, KEEP, dtype=np.int64)
    target[truth & ~pred] = ON
    target[~truth & pred] = OFF
    return target


def apply_actions(stage2: np.ndarray, actions: np.ndarray, band: np.ndarray) -> np.ndarray:
    """띠 안의 켜기/끄기만 반영한 이진 마스크. 띠 밖과 유지는 Stage 2 그대로."""
    out = (np.asarray(stage2) > 0.5).astype(np.uint8)
    editable = np.asarray(band).astype(bool)
    act = np.asarray(actions)
    out[editable & (act == ON)] = 1
    out[editable & (act == OFF)] = 0
    return out


def signed_boundary_distance(gt: np.ndarray) -> np.ndarray:
    """GT 경계까지 거리. 내부는 음수, 외부는 양수."""
    truth = np.asarray(gt) > 0.5
    if not truth.any() or truth.all():
        return np.zeros(truth.shape, dtype=np.float32)
    outside = distance_transform_edt(~truth)
    inside = distance_transform_edt(truth)
    return (outside - inside).astype(np.float32)


def apply_flair_guard(
    flair: np.ndarray,
    stage2: np.ndarray,
    actions: np.ndarray,
    band: np.ndarray,
) -> np.ndarray:
    """띠 안에서 FLAIR가 상대적으로 밝을 때만 켜고, 어두울 때만 끈다.

    기준은 그 슬라이스 띠의 평균과 표준편차다. 절대 밝기 임계값은 쓰지 않는다.
    """
    image = np.asarray(flair, dtype=np.float32)
    editable = np.asarray(band).astype(bool)
    if int(editable.sum()) < 8:
        return (np.asarray(stage2) > 0.5).astype(np.uint8)
    values = image[editable]
    center = float(values.mean())
    spread = float(values.std())
    if spread < 1e-6:
        return (np.asarray(stage2) > 0.5).astype(np.uint8)
    z = (image - center) / spread
    guarded = np.full(actions.shape, KEEP, dtype=np.int64)
    turn_on = editable & (np.asarray(actions) == ON) & (z > 0)
    turn_off = editable & (np.asarray(actions) == OFF) & (z < 0)
    guarded[turn_on] = ON
    guarded[turn_off] = OFF
    return apply_actions(stage2, guarded, editable)


def flair_z_map(flair: np.ndarray, band: np.ndarray) -> np.ndarray:
    """띠 안 FLAIR의 상대 밝기. 픽셀 수가 적거나 분산이 없으면 0."""
    image = np.asarray(flair, dtype=np.float32)
    editable = np.asarray(band).astype(bool)
    z = np.zeros(image.shape, dtype=np.float32)
    if int(editable.sum()) < 8:
        return z
    values = image[editable]
    spread = float(values.std())
    if spread < 1e-6:
        return z
    return ((image - float(values.mean())) / spread).astype(np.float32)


def pixel_flip_reward(
    prev: np.ndarray,
    new: np.ndarray,
    gt: np.ndarray,
    hd_prev: float | None,
    hd_new: float | None,
    hd_coef: float,
) -> np.ndarray:
    """뒤집은 픽셀만 보상. GT와 맞으면 +1, 틀리면 -1. HD95가 줄면 그 양을 더한다."""
    prev_b = np.asarray(prev) > 0.5
    new_b = np.asarray(new) > 0.5
    truth = np.asarray(gt) > 0.5
    flipped = prev_b != new_b
    reward = np.zeros(prev_b.shape, dtype=np.float32)
    reward[flipped & (new_b == truth)] = 1.0
    reward[flipped & (new_b != truth)] = -1.0
    if hd_prev is not None and hd_new is not None and flipped.any():
        reward[flipped] += np.float32(hd_coef * (float(hd_prev) - float(hd_new)))
    return reward
