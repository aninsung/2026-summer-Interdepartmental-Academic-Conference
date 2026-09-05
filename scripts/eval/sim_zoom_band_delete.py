"""
Medium/Large: 외곽 band 확대 → 잘못된 경계 삭제 시뮬레이션 (강화 학습).

비교 (합성 blob, Stage2 체크포인트 미사용):
  - Rough
  - Morph shrink     : 전역 erosion 1px
  - Full SL shrink   : 전체 128 DualHead + shrink band (풀슬라이스 학습)
  - Zoom SL delete   : 외곽 패치 2× 확대, FP-가중 삭제 학습 → rem_band OFF만

결과: results/sim_zoom_band_delete.md
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.models.sl_refiner import DualHeadRefiner, apply_sl_refiner
from src.utils.metrics import dice, hd95

OUT = ROOT / "results" / "sim_zoom_band_delete.md"
H = W = 128
PATCH = 48
BAND = 3
N_TRAIN = 400
N_EVAL = 80
EPOCHS_FULL = 25
EPOCHS_ZOOM = 40
PATCHES_PER = 12
INFER_PATCHES = 24


def make_blob(h, w, cy, cx, ry, rx):
    yy, xx = np.ogrid[:h, :w]
    return (((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0).astype(np.float32)


def synth_batch(n: int, mode: str, seed: int = 0):
    """Medium/Large: 외곽 FP를 의도적으로 많이 넣음 (large_correction_1 유형)."""
    rng = np.random.default_rng(seed)
    images, gts, roughs, probs = [], [], [], []
    for _ in range(n):
        img = np.clip(rng.normal(0.35, 0.12, (H, W)), 0, 1).astype(np.float32)
        if mode == "medium":
            cy, cx = rng.integers(40, 90), rng.integers(40, 90)
            gt = make_blob(H, W, cy, cx, rng.integers(12, 18), rng.integers(12, 18))
        else:
            cy, cx = rng.integers(35, 95), rng.integers(35, 95)
            gt = make_blob(H, W, cy, cx, rng.integers(22, 32), rng.integers(22, 32))

        r = binary_dilation(gt > 0.5, iterations=int(rng.integers(1, 3))).astype(np.float32)
        shell = np.logical_and(binary_dilation(r > 0.5, iterations=2), r <= 0.5)
        bumps = (rng.random((H, W)) < 0.08).astype(np.float32) * shell.astype(np.float32)
        r = (np.clip(r + bumps, 0, 1) > 0.5).astype(np.float32)
        if r.sum() < 10:
            r = gt.copy()

        img = np.clip(img + 0.28 * gaussian_filter(gt, 1.2), 0, 1).astype(np.float32)
        prob = np.clip(
            0.65 * gaussian_filter(gt, 1.0) + 0.20 * r + rng.normal(0, 0.04, (H, W)),
            0,
            1,
        ).astype(np.float32)
        fp_shell = np.logical_and(r > 0.5, gt <= 0.5)
        prob = np.where(fp_shell, prob * 0.35, prob).astype(np.float32)

        images.append(img)
        gts.append(gt.astype(np.float32))
        roughs.append(r)
        probs.append(prob)
    return np.stack(images), np.stack(gts), np.stack(roughs), np.stack(probs)


def score(preds, gts):
    dscs, hds = [], []
    for p, g in zip(preds, gts):
        pb = (p > 0.5).astype(np.float32)
        gb = (g > 0.5).astype(np.float32)
        dscs.append(dice(pb, gb))
        hds.append(hd95(pb, gb))
    return float(np.mean(dscs)), float(np.mean(hds))


def outer_band(mask, n=BAND):
    b = mask > 0.5
    if not b.any():
        return np.zeros_like(b)
    dil = binary_dilation(b, iterations=n)
    ero = binary_erosion(b, iterations=n)
    return np.logical_xor(dil, ero)


def rem_band_of(mask, n=BAND):
    b = mask > 0.5
    return np.logical_and(b, ~binary_erosion(b, iterations=n))


def sample_patch_centers(band: np.ndarray, rng, k=8, weights: np.ndarray | None = None):
    ys, xs = np.where(band)
    if len(ys) == 0:
        return []
    if weights is None:
        w = None
    else:
        w = weights[ys, xs].astype(np.float64)
        w = w / max(w.sum(), 1e-8)
    idx = rng.choice(len(ys), size=min(k, len(ys)), replace=False, p=w)
    return [(int(ys[i]), int(xs[i])) for i in idx]


def crop_pad(arr, cy, cx, size=PATCH):
    half = size // 2
    y1, x1 = cy - half, cx - half
    out = np.zeros((size, size), dtype=np.float32)
    sy1, sx1 = max(0, y1), max(0, x1)
    sy2, sx2 = min(H, y1 + size), min(W, x1 + size)
    dy1, dx1 = sy1 - y1, sx1 - x1
    out[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = arr[sy1:sy2, sx1:sx2]
    return out, (y1, x1)


def soft_dice_bce(logits, targets, weight=None):
    if weight is None:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    else:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weight, reduction="mean"
        )
    p = torch.sigmoid(logits)
    num = 2 * (p * targets).sum(dim=(2, 3)) + 1e-5
    den = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-5
    return 0.5 * bce + 0.5 * (1.0 - (num / den).mean())


def zoom2(arr):
    return np.repeat(np.repeat(arr, 2, 0), 2, 1)


def build_full_trainset(images, gts, roughs, probs):
    xs = np.stack(
        [np.stack([img, r, p], 0) for img, r, p in zip(images, roughs, probs)],
        axis=0,
    ).astype(np.float32)
    ys = gts[:, None].astype(np.float32)
    # candidate = rem_band pixels that differ from GT (주로 FP 삭제 위치)
    cands = []
    for r, g in zip(roughs, gts):
        rem = rem_band_of(r, BAND)
        diff = (r > 0.5) != (g > 0.5)
        cands.append(np.logical_and(rem, diff).astype(np.float32))
    cs = np.stack(cands, 0)[:, None].astype(np.float32)
    return xs, ys, cs


def build_zoom_trainset(images, gts, roughs, probs, seed=0):
    """외곽 FP-가중 패치: fix→GT, cand→삭제할 rem_band FP."""
    rng = np.random.default_rng(seed)
    xs, ys, cs, ws = [], [], [], []
    for img, gt, r, p in zip(images, gts, roughs, probs):
        band = outer_band(r, BAND)
        rem = rem_band_of(r, BAND)
        fp = np.logical_and(r > 0.5, gt <= 0.5)
        # FP가 많은 외곽을 더 자주 샘플
        wmap = band.astype(np.float32) + 4.0 * np.logical_and(band, fp).astype(np.float32)
        for cy, cx in sample_patch_centers(band, rng, k=PATCHES_PER, weights=wmap):
            img_c, _ = crop_pad(img, cy, cx)
            r_c, _ = crop_pad(r, cy, cx)
            p_c, _ = crop_pad(p, cy, cx)
            g_c, _ = crop_pad(gt, cy, cx)
            rem_c, _ = crop_pad(rem.astype(np.float32), cy, cx)
            fp_c, _ = crop_pad(fp.astype(np.float32), cy, cx)

            img_z, r_z, p_z = zoom2(img_c), zoom2(r_c), zoom2(p_c)
            g_z = zoom2(g_c)
            rem_z = zoom2(rem_c) > 0.5
            fp_z = zoom2(fp_c) > 0.5
            cand_z = np.logical_and(rem_z, fp_z).astype(np.float32)
            # pixel weight: FP 삭제에 5×, rem_band 2×
            wt = np.ones_like(g_z, dtype=np.float32)
            wt = np.where(rem_z, wt * 2.0, wt)
            wt = np.where(fp_z, wt * 5.0, wt)

            xs.append(np.stack([img_z, r_z, p_z], 0).astype(np.float32))
            ys.append(g_z[None].astype(np.float32))
            cs.append(cand_z[None].astype(np.float32))
            ws.append(wt[None].astype(np.float32))
    return np.stack(xs), np.stack(ys), np.stack(cs), np.stack(ws)


def train_sl(x, y, c, device, epochs, seed=0, w=None, lr=1e-3):
    torch.manual_seed(seed)
    net = DualHeadRefiner(in_ch=3).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tensors = [torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(c)]
    if w is not None:
        tensors.append(torch.from_numpy(w))
    loader = DataLoader(TensorDataset(*tensors), batch_size=16, shuffle=True)
    t0 = time.time()
    net.train()
    for _ in range(epochs):
        for batch in loader:
            if w is None:
                xb, yb, cb = batch
                wb = None
            else:
                xb, yb, cb, wb = batch
                wb = wb.to(device)
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            opt.zero_grad(set_to_none=True)
            lc, lf = net(xb)
            loss = soft_dice_bce(lf, yb, weight=wb) + 0.5 * soft_dice_bce(lc, cb)
            loss.backward()
            opt.step()
    return net, time.time() - t0


@torch.no_grad()
def refine_full_shrink(net, images, roughs, probs, device):
    out = []
    for img, r, p in zip(images, roughs, probs):
        pred = apply_sl_refiner(
            net,
            img,
            r,
            p,
            device,
            cand_thr=0.40,
            boundary_band_px=BAND,
            boundary_mode="shrink",
        )
        out.append(pred)
    return np.stack(out)


@torch.no_grad()
def refine_zoom_delete(net, images, roughs, probs, device, seed=1):
    """외곽 패치 확대 → SL → rem_band에서만 OFF (삭제 우선)."""
    rng = np.random.default_rng(seed)
    net.eval()
    outs = []
    for img, r, p in zip(images, roughs, probs):
        out = (r > 0.5).copy()
        band = outer_band(r, BAND)
        rem = rem_band_of(r, BAND)
        # 낮은 prob 외곽을 더 자주 봄
        wmap = band.astype(np.float32) * (1.0 + (1.0 - p))
        for cy, cx in sample_patch_centers(band, rng, k=INFER_PATCHES, weights=wmap):
            img_c, (y1, x1) = crop_pad(img, cy, cx)
            r_c, _ = crop_pad(r, cy, cx)
            p_c, _ = crop_pad(p, cy, cx)
            x = torch.from_numpy(
                np.stack([zoom2(img_c), zoom2(r_c), zoom2(p_c)], 0)[None]
            ).float().to(device)
            lc, lf = net(x)
            cand = torch.sigmoid(lc) > 0.35
            # cand가 켜진 곳에서만 fix로 덮기; 결과는 shrink-only로 적용
            blended = torch.where(cand, torch.sigmoid(lf), x[:, 1:2])
            pred_c = (blended > 0.5).float()[0, 0].cpu().numpy()[::2, ::2]

            sy1, sx1 = max(0, y1), max(0, x1)
            sy2, sx2 = min(H, y1 + PATCH), min(W, x1 + PATCH)
            py1, px1 = sy1 - y1, sx1 - x1
            patch_off = pred_c[py1 : py1 + (sy2 - sy1), px1 : px1 + (sx2 - sx1)] < 0.5
            region = rem[sy1:sy2, sx1:sx2] & patch_off
            out[sy1:sy2, sx1:sx2] = np.where(region, False, out[sy1:sy2, sx1:sx2])
        outs.append(out.astype(np.float32))
    return np.stack(outs)


def morph_shrink(roughs):
    return np.stack(
        [binary_erosion(r > 0.5, iterations=1).astype(np.float32) for r in roughs]
    )


def fp_removed(preds, roughs, gts):
    vals = []
    for pred, r, g in zip(preds, roughs, gts):
        fp0 = np.logical_and(r > 0.5, g <= 0.5)
        fp1 = np.logical_and(pred > 0.5, g <= 0.5)
        vals.append(float(fp0.sum() - fp1.sum()))
    return float(np.mean(vals))


def fn_added(preds, roughs, gts):
    """GT를 잘못 지운 픽셀(평균)."""
    vals = []
    for pred, r, g in zip(preds, roughs, gts):
        # rough에서 TP였는데 pred에서 사라진 것
        lost = np.logical_and(np.logical_and(r > 0.5, g > 0.5), pred <= 0.5)
        vals.append(float(lost.sum()))
    return float(np.mean(vals))


def run_mode(mode: str, device: torch.device, seed: int = 42):
    print(f"\n=== {mode.upper()} (강화) ===", flush=True)
    tr_img, tr_gt, tr_r, tr_p = synth_batch(N_TRAIN, mode, seed=seed)
    ev_img, ev_gt, ev_r, ev_p = synth_batch(N_EVAL, mode, seed=seed + 7)

    rough_d, rough_h = score(ev_r, ev_gt)
    morph_p = morph_shrink(ev_r)
    morph_d, morph_h = score(morph_p, ev_gt)

    fx, fy, fc = build_full_trainset(tr_img, tr_gt, tr_r, tr_p)
    print(f"[{mode}] train Full SL ({fx.shape[0]} slices × {EPOCHS_FULL} ep)...", flush=True)
    net_full, sec_f = train_sl(fx, fy, fc, device, epochs=EPOCHS_FULL, seed=seed, lr=1e-3)

    zx, zy, zc, zw = build_zoom_trainset(tr_img, tr_gt, tr_r, tr_p, seed=seed)
    print(
        f"[{mode}] train Zoom SL ({zx.shape[0]} patches × {EPOCHS_ZOOM} ep)...",
        flush=True,
    )
    net_zoom, sec_z = train_sl(
        zx, zy, zc, device, epochs=EPOCHS_ZOOM, seed=seed + 1, w=zw, lr=8e-4
    )

    full = refine_full_shrink(net_full, ev_img, ev_r, ev_p, device)
    zoom = refine_zoom_delete(net_zoom, ev_img, ev_r, ev_p, device, seed=seed + 1)
    full_d, full_h = score(full, ev_gt)
    zoom_d, zoom_h = score(zoom, ev_gt)

    row = {
        "mode": mode,
        "rough": (rough_d, rough_h, 0.0, 0.0),
        "morph": (morph_d, morph_h, fp_removed(morph_p, ev_r, ev_gt), fn_added(morph_p, ev_r, ev_gt)),
        "full_sl": (full_d, full_h, fp_removed(full, ev_r, ev_gt), fn_added(full, ev_r, ev_gt)),
        "zoom_sl": (zoom_d, zoom_h, fp_removed(zoom, ev_r, ev_gt), fn_added(zoom, ev_r, ev_gt)),
        "sec": sec_f + sec_z,
    }
    print(
        f"[{mode}] Rough={rough_d:.4f} | Morph={morph_d:.4f} | "
        f"FullSL={full_d:.4f} | ZoomSL={zoom_d:.4f} "
        f"(FPΔ zoom={row['zoom_sl'][2]:+.1f}, FNΔ={row['zoom_sl'][3]:+.1f}) "
        f"train={row['sec']:.1f}s",
        flush=True,
    )
    return row


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    t0 = time.time()
    rows = [run_mode(m, device) for m in ("medium", "large")]

    lines = [
        "# Zoom-band delete 시뮬레이션 (강화: FP-가중 + 더 긴 학습)",
        "",
        "합성 과분할 rough + 외곽 FP. Stage2 체크포인트 미사용.",
        "",
        f"- Train {N_TRAIN} / Eval {N_EVAL}",
        f"- Full SL: {EPOCHS_FULL} ep (풀슬라이스), Zoom SL: {EPOCHS_ZOOM} ep (패치 {PATCHES_PER}/slice, FP-가중)",
        "- Morph shrink: 전역 erosion 1px",
        "- Zoom SL delete: 외곽 2× 확대 → rem_band에서만 OFF",
        "",
        "| Size | Method | DSC ↑ | HD95 ↓ | ΔDSC vs Rough | FP removed | TP lost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        base = r["rough"][0]
        for name, key in (
            ("Rough", "rough"),
            ("Morph shrink", "morph"),
            ("Full SL shrink", "full_sl"),
            ("**Zoom SL delete**", "zoom_sl"),
        ):
            d, h, fp, fn = r[key]
            lines.append(
                f"| {r['mode']} | {name} | {d:.4f} | {h:.2f} | {d - base:+.4f} | {fp:+.1f} | {fn:+.1f} |"
            )

    lines += [
        "",
        f"총 소요: {time.time() - t0:.1f}s",
        "",
        "> 합성 실험. BraTS 절대수치와 비교하지 말고 상대 효과만 볼 것.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
