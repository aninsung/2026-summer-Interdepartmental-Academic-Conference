"""Supervised dual-head refiner: candidate + local fix."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DualHeadRefiner(nn.Module):
    def __init__(self, in_ch: int = 4):
        # default 4 = 2 MRI (t1ce+flair) + rough + prob
        super().__init__()
        self.in_ch = in_ch
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(True),
        )
        self.cand = nn.Conv2d(32, 1, 1)
        self.fix = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        f = self.enc(x)
        return self.cand(f), self.fix(f)


def _stack_sl_input(image, rough, prob) -> np.ndarray:
    """Build (C_img+2, H, W) float32 array."""
    img = np.asarray(image, dtype=np.float32)
    r = np.asarray(rough, dtype=np.float32)
    p = np.asarray(prob, dtype=np.float32)
    if img.ndim == 2:
        img = img[None]
    if r.ndim == 3:
        r = r[0]
    if p.ndim == 3:
        p = p[0]
    return np.concatenate([img, r[None], p[None]], axis=0).astype(np.float32)


def _zoom2(arr: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


def _crop_pad(arr: np.ndarray, cy: int, cx: int, size: int, h: int, w: int):
    half = size // 2
    y1, x1 = cy - half, cx - half
    out = np.zeros((size, size), dtype=np.float32)
    sy1, sx1 = max(0, y1), max(0, x1)
    sy2, sx2 = min(h, y1 + size), min(w, x1 + size)
    dy1, dx1 = sy1 - y1, sx1 - x1
    out[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = arr[sy1:sy2, sx1:sx2]
    return out, (y1, x1)


def _sample_band_centers(
    band: np.ndarray,
    weights: np.ndarray | None,
    k: int,
    rng: np.random.Generator,
):
    ys, xs = np.where(band)
    if len(ys) == 0:
        return []
    if weights is None:
        p = None
    else:
        w = np.asarray(weights[ys, xs], dtype=np.float64)
        s = float(w.sum())
        p = (w / s) if s > 0 else None
    idx = rng.choice(len(ys), size=min(k, len(ys)), replace=False, p=p)
    return [(int(ys[i]), int(xs[i])) for i in idx]


@torch.no_grad()
def _zoom_band_delete(
    net: DualHeadRefiner,
    image_2d: np.ndarray,
    rough: np.ndarray,
    prob: np.ndarray,
    base_pred: np.ndarray,
    device: torch.device,
    band_n: int,
    n_patches: int,
    patch: int,
    cand_thr: float,
    seed: int = 0,
) -> np.ndarray:
    """외곽 rem_band 패치를 2× 확대해 SL 후, rem_band에서만 OFF(삭제) 반영.

    base_pred에서 시작해 추가 삭제만 허용(면적 증가 금지). Large/Medium 과분할 완화용.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    x_np = _stack_sl_input(image_2d, rough, prob)
    img_ch = x_np[:-2]  # (C,H,W)
    r = x_np[-2]
    p = x_np[-1]
    h, w = r.shape
    rough_b = r > 0.5
    if not rough_b.any() or n_patches <= 0 or band_n <= 0:
        return base_pred.astype(np.float32)

    ero = binary_erosion(rough_b, iterations=band_n)
    dil = binary_dilation(rough_b, iterations=band_n)
    rem_band = np.logical_and(rough_b, np.logical_not(ero))
    outer_band = np.logical_xor(dil, ero)
    if not rem_band.any():
        return base_pred.astype(np.float32)

    out = (base_pred > 0.5).copy()
    # 낮은 확률 외곽을 더 자주 샘플 → FP 후보 우선
    wmap = outer_band.astype(np.float32) * (1.0 + (1.0 - p))
    rng = np.random.default_rng(seed)
    net.eval()
    expected = getattr(net, "in_ch", None)

    for cy, cx in _sample_band_centers(outer_band, wmap, n_patches, rng):
        crops = []
        for ch in range(img_ch.shape[0]):
            c_img, (y1, x1) = _crop_pad(img_ch[ch], cy, cx, patch, h, w)
            crops.append(_zoom2(c_img))
        r_c, (y1, x1) = _crop_pad(r, cy, cx, patch, h, w)
        p_c, _ = _crop_pad(p, cy, cx, patch, h, w)
        x_list = crops + [_zoom2(r_c), _zoom2(p_c)]
        x = torch.from_numpy(np.stack(x_list, 0)[None]).float().to(device)
        if expected is not None and x.shape[1] != expected:
            raise ValueError(
                f"SL zoom in_ch={expected} but patch input has {x.shape[1]} channels"
            )
        lc, lf = net(x)
        cand = torch.sigmoid(lc) > cand_thr
        blended = torch.where(cand, torch.sigmoid(lf), x[:, -2:-1])
        pred_c = (blended > 0.5).float()[0, 0].cpu().numpy()[::2, ::2]

        sy1, sx1 = max(0, y1), max(0, x1)
        sy2, sx2 = min(h, y1 + patch), min(w, x1 + patch)
        py1, px1 = sy1 - y1, sx1 - x1
        patch_off = pred_c[py1 : py1 + (sy2 - sy1), px1 : px1 + (sx2 - sx1)] < 0.5
        region = rem_band[sy1:sy2, sx1:sx2] & patch_off
        out[sy1:sy2, sx1:sx2] = np.where(region, False, out[sy1:sy2, sx1:sx2])

    # 안전: zoom은 삭제만 — base/rough 대비 ON 추가 금지
    out = np.logical_and(out, rough_b)
    return out.astype(np.float32)


@torch.no_grad()
def apply_sl_refiner(
    net: DualHeadRefiner,
    image_2d: np.ndarray,
    rough: np.ndarray,
    prob: np.ndarray,
    device: torch.device,
    cand_thr: float = 0.4,
    morph_small: bool = False,
    boundary_band_px: int = 0,
    boundary_mode: str = "replace",
    zoom_n_patches: int = 0,
    zoom_patch: int = 48,
    zoom_seed: int = 0,
) -> np.ndarray:
    """image: (H,W) or (C,H,W). rough/prob: (H,W). Returns refined binary float mask.

    If boundary_band_px > 0, SL edits are kept only inside the morphological
    boundary band of the rough mask (dilate ⊕ erode by N px); elsewhere rough
    is preserved.

    boundary_mode:
      - "replace": band 안에서 SL 결과로 교체 (수축·확장 모두 가능)
      - "expand":  외곽에서만 픽셀 추가 (rough | SL), 삭제 금지
      - "shrink":  외곽에서만 픽셀 삭제 (rough & SL), 추가 금지
        → Medium/Large 경계 FP 완화용

    zoom_n_patches > 0 (shrink 모드 권장):
      외곽 패치 2× 확대 다중 크롭으로 rem_band 추가 삭제.
    """
    from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_erosion

    if boundary_mode not in ("replace", "expand", "shrink"):
        raise ValueError(
            f"boundary_mode must be 'replace', 'expand', or 'shrink', got {boundary_mode!r}"
        )

    x_np = _stack_sl_input(image_2d, rough, prob)
    r = x_np[-2]
    x = torch.from_numpy(x_np[None]).float().to(device)
    expected = getattr(net, "in_ch", None)
    if expected is not None and x.shape[1] != expected:
        raise ValueError(
            f"SL refiner in_ch={expected} but input has {x.shape[1]} channels "
            f"(image channels + rough + prob)."
        )
    net.eval()
    lc, lf = net(x)
    cand = (torch.sigmoid(lc) > cand_thr).float()
    rb = x[:, -2:-1]
    blended = torch.where(cand > 0.5, torch.sigmoid(lf), rb)
    pred = (blended > 0.5).float()[0, 0].cpu().numpy()
    if float(pred.sum()) < 5:
        pred = (r > 0.5).astype(np.float32)
    if morph_small:
        m = pred > 0.5
        m = binary_closing(binary_opening(m, iterations=1), iterations=1)
        if m.sum() >= 3:
            pred = m.astype(np.float32)

    band_n = int(boundary_band_px) if boundary_band_px else 0
    if band_n > 0:
        rough_b = r > 0.5
        pred_b = pred > 0.5
        if rough_b.any():
            dil = binary_dilation(rough_b, iterations=band_n)
            ero = binary_erosion(rough_b, iterations=band_n)
            band = np.logical_xor(dil, ero)
            if boundary_mode == "expand":
                add_band = np.logical_and(dil, np.logical_not(rough_b))
                out = rough_b.copy()
                out[add_band] = np.logical_or(out[add_band], pred_b[add_band])
                pred = out.astype(np.float32)
            elif boundary_mode == "shrink":
                rem_band = np.logical_and(rough_b, np.logical_not(ero))
                out = rough_b.copy()
                out[rem_band] = np.logical_and(out[rem_band], pred_b[rem_band])
                pred = out.astype(np.float32)
            else:
                out = rough_b.copy()
                out[band] = pred_b[band]
                pred = out.astype(np.float32)
        else:
            pred = rough_b.astype(np.float32)

    # Medium/Large: 외곽 zoom 다중 크롭으로 추가 삭제 (shrink에서만 의미 있음)
    if int(zoom_n_patches) > 0 and boundary_mode == "shrink":
        zn = band_n if band_n > 0 else 3
        pred = _zoom_band_delete(
            net,
            image_2d,
            rough,
            prob,
            pred,
            device,
            band_n=zn,
            n_patches=int(zoom_n_patches),
            patch=int(zoom_patch),
            cand_thr=float(cand_thr),
            seed=int(zoom_seed),
        )

    return pred.astype(np.float32)


def build_sl_refiner(in_ch: int = 4) -> DualHeadRefiner:
    return DualHeadRefiner(in_ch=in_ch)
