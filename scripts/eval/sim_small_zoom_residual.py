"""
Small 전용 Refiner 설계 시뮬레이션 (메인 파이프라인 미변경).

비교 (Small only):
  - Rough
  - Morphology
  - Global Refiner      : 128x128 전역 Tiny-UNet
  - Zoom Residual Refiner: 64x64 crop + residual Δ on soft-rough
  - PPO (짧게)

결과: results/sim_small_zoom_residual.md
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening, center_of_mass, gaussian_filter
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.mask_refinement_env import MaskRefinementEnv
from src.utils.metrics import dice, hd95

OUT_MD = ROOT / "results" / "sim_small_zoom_residual.md"
PATCH = 64
H = W = 128


def make_blob(h, w, cy, cx, ry, rx):
    yy, xx = np.ogrid[:h, :w]
    return (((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0).astype(np.float32)


def synth_small(n: int, seed: int = 0, oversample_micro: bool = True):
    """Small blobs; optionally oversample <50px like CaraNet fragment policy."""
    rng = np.random.default_rng(seed)
    images, gts, roughs, probs = [], [], [], []
    while len(images) < n:
        img = np.clip(rng.normal(0.35, 0.12, (H, W)), 0, 1).astype(np.float32)
        cy, cx = int(rng.integers(40, 90)), int(rng.integers(40, 90))
        # mix micro (<50) and normal small (<300)
        if oversample_micro and rng.random() < 0.45:
            ry = rx = int(rng.integers(2, 5))
        else:
            ry, rx = int(rng.integers(4, 9)), int(rng.integers(4, 9))
        gt = make_blob(H, W, cy, cx, ry, rx)
        if gt.sum() < 3 or gt.sum() >= 300:
            continue
        r = gt.copy()
        if rng.random() < 0.55:
            r = binary_dilation(r, iterations=int(rng.integers(1, 3))).astype(np.float32)
        else:
            r = binary_erosion(r, iterations=1).astype(np.float32)
        if r.sum() < 3:
            r = gt.copy()
        r = np.clip(r + ((rng.random((H, W)) < 0.008).astype(np.float32) * (1 - gt)), 0, 1)
        r = (r > 0.5).astype(np.float32)
        img = np.clip(img + 0.3 * gaussian_filter(gt, 1.2), 0, 1).astype(np.float32)
        prob = np.clip(
            0.55 * gaussian_filter(gt, 0.8) + 0.35 * r + rng.normal(0, 0.04, (H, W)), 0, 1
        ).astype(np.float32)
        images.append(img)
        gts.append(gt)
        roughs.append(r)
        probs.append(prob)
        # fragment oversample
        if oversample_micro and gt.sum() < 50 and len(images) < n:
            for _ in range(2):
                images.append(img.copy())
                gts.append(gt.copy())
                roughs.append(r.copy())
                probs.append(prob.copy())
    return (
        np.stack(images[:n]),
        np.stack(gts[:n]),
        np.stack(roughs[:n]),
        np.stack(probs[:n]),
    )


def score_masks(preds, gts):
    dscs, hds = [], []
    for p, g in zip(preds, gts):
        pb = (p > 0.5).astype(np.float32)
        gb = (g > 0.5).astype(np.float32)
        dscs.append(dice(pb, gb))
        hds.append(hd95(pb, gb))
    return {"dsc": float(np.mean(dscs)), "hd95": float(np.mean(hds)), "n": len(dscs)}


def morph_refine(roughs):
    out = []
    for r in roughs:
        m = r > 0.5
        m = binary_opening(m, iterations=1)
        m = binary_closing(m, iterations=1)
        if m.sum() < 3:
            m = r > 0.5
        out.append(m.astype(np.float32))
    return np.stack(out)


def crop_center_from_mask(mask, patch=PATCH):
    m = mask > 0.5
    if m.sum() < 1:
        cy, cx = H // 2, W // 2
    else:
        cy, cx = center_of_mass(m)
        cy, cx = int(round(cy)), int(round(cx))
    half = patch // 2
    y0 = int(np.clip(cy - half, 0, H - patch))
    x0 = int(np.clip(cx - half, 0, W - patch))
    return y0, x0


def extract_crop(arr, y0, x0, patch=PATCH):
    return arr[y0 : y0 + patch, x0 : x0 + patch].copy()


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(True))
        self.bot = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True))
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True))
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True))
        self.head = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bot(e2)
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def soft_dice_loss(logits, targets, eps=1e-5):
    p = torch.sigmoid(logits)
    num = 2 * (p * targets).sum((2, 3)) + eps
    den = p.sum((2, 3)) + targets.sum((2, 3)) + eps
    return 1 - (num / den).mean()


def boundary_dice_loss(logits, targets, band=2, eps=1e-5):
    # soft approx: dilate target with max-pool
    t = targets
    for _ in range(band):
        t = torch.nn.functional.max_pool2d(t, 3, stride=1, padding=1)
    er = targets
    for _ in range(band):
        er = -torch.nn.functional.max_pool2d(-er, 3, stride=1, padding=1)
    band_m = (t - er).clamp(0, 1)
    return soft_dice_loss(logits * band_m, targets * band_m, eps)


class GlobalDS(Dataset):
    def __init__(self, images, gts, roughs, probs):
        self.x = np.stack([images, roughs, probs], 1).astype(np.float32)
        self.y = gts[:, None].astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


class ZoomResidualDS(Dataset):
    """Crop around rough COM; target = GT crop; also keep rough_soft for residual."""

    def __init__(self, images, gts, roughs, probs):
        self.samples = []
        for img, gt, r, p in zip(images, gts, roughs, probs):
            y0, x0 = crop_center_from_mask(r)
            self.samples.append(
                (
                    extract_crop(img, y0, x0),
                    extract_crop(r, y0, x0),
                    extract_crop(p, y0, x0),
                    extract_crop(gt, y0, x0),
                    y0,
                    x0,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, r, p, gt, y0, x0 = self.samples[i]
        x = np.stack([img, r, p], 0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(gt[None].astype(np.float32)), torch.from_numpy(r[None].astype(np.float32))


def train_global(images, gts, roughs, probs, epochs=12, device="cpu", seed=0):
    torch.manual_seed(seed)
    loader = DataLoader(GlobalDS(images, gts, roughs, probs), batch_size=16, shuffle=True)
    model = TinyUNet(3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = 0.5 * bce(logits, yb) + 0.5 * soft_dice_loss(logits, yb)
            loss.backward()
            opt.step()
    return model, time.time() - t0


@torch.no_grad()
def eval_global(model, images, roughs, probs, device="cpu"):
    model.eval()
    x = torch.from_numpy(np.stack([images, roughs, probs], 1)).float().to(device)
    outs = []
    for i in range(0, len(x), 16):
        pred = (torch.sigmoid(model(x[i : i + 16])) > 0.5).float().cpu().numpy()[:, 0]
        outs.append(pred)
    return np.concatenate(outs, 0)


def train_zoom_residual(images, gts, roughs, probs, epochs=20, device="cpu", seed=0):
    torch.manual_seed(seed)
    loader = DataLoader(ZoomResidualDS(images, gts, roughs, probs), batch_size=16, shuffle=True)
    model = TinyUNet(3).to(device)
    # prior skip strength (learnable)
    alpha = nn.Parameter(torch.tensor(0.5, device=device))
    opt = torch.optim.Adam(list(model.parameters()) + [alpha], lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for xb, yb, rb in loader:
            xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
            opt.zero_grad(set_to_none=True)
            delta = model(xb)
            prior = torch.logit(rb.clamp(1e-4, 1 - 1e-4))
            # crop 직접 보정: 네트워크가 주도, rough prior는 약한 스킵
            logits = delta + alpha.clamp(0, 2) * prior
            loss = (
                0.4 * bce(logits, yb)
                + 0.4 * soft_dice_loss(logits, yb)
                + 0.2 * boundary_dice_loss(logits, yb)
                + 0.01 * (delta - prior).abs().mean()  # 너무 멀어지지 않게
            )
            loss.backward()
            opt.step()
    model._prior_alpha = float(alpha.detach().cpu())
    return model, time.time() - t0


@torch.no_grad()
def eval_zoom_residual(model, images, roughs, probs, device="cpu"):
    model.eval()
    alpha = float(getattr(model, "_prior_alpha", 0.5))
    preds = np.zeros_like(roughs)
    for i in range(len(images)):
        y0, x0 = crop_center_from_mask(roughs[i])
        img_c = extract_crop(images[i], y0, x0)
        r_c = extract_crop(roughs[i], y0, x0)
        p_c = extract_crop(probs[i], y0, x0)
        xb = torch.from_numpy(np.stack([img_c, r_c, p_c], 0)[None]).float().to(device)
        rb = torch.from_numpy(r_c[None, None]).float().to(device)
        delta = model(xb)
        prior = torch.logit(rb.clamp(1e-4, 1 - 1e-4))
        logits = delta + alpha * prior
        crop_pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.float32)
        # tiny: 완전 소멸만 방지
        if crop_pred.sum() < 2 and r_c.sum() >= 2:
            crop_pred = r_c.copy()
        out = roughs[i].copy()
        out[y0 : y0 + PATCH, x0 : x0 + PATCH] = crop_pred
        if out.sum() < 2:
            out = roughs[i].copy()
        preds[i] = out
    return preds


def train_eval_ppo(train_img, train_gt, train_r, train_p, eval_img, eval_gt, eval_r, eval_p, timesteps=4000, seed=0):
    def make_train():
        return MaskRefinementEnv(
            images=train_img,
            gt_masks=train_gt,
            rough_masks=train_r,
            uncertainty_maps=train_p,
            max_steps=15,
            target_dsc=1.0,
            step_penalty=0.001,
            refinement_mode="small",
            enable_stop=False,
        )

    venv = DummyVecEnv([make_train])
    t0 = time.time()
    model = PPO(
        "CnnPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=3,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=dict(normalize_images=False, net_arch=[64, 64]),
        verbose=0,
        seed=seed,
        device="cpu",
    )
    model.learn(total_timesteps=timesteps)
    sec = time.time() - t0
    env = MaskRefinementEnv(
        images=eval_img,
        gt_masks=eval_gt,
        rough_masks=eval_r,
        uncertainty_maps=eval_p,
        max_steps=15,
        target_dsc=1.0,
        step_penalty=0.001,
        refinement_mode="small",
        enable_stop=False,
    )
    preds = []
    for i in range(len(eval_img)):
        obs, _ = env.reset(seed=i)
        for _ in range(15):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        preds.append(env._current_mask.copy())
    return np.stack(preds), sec


def main():
    t_all = time.time()
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    train_img, train_gt, train_r, train_p = synth_small(96, seed=seed)
    eval_img, eval_gt, eval_r, eval_p = synth_small(48, seed=seed + 99, oversample_micro=False)
    print(
        f"train n={len(train_img)} mean_gt_area={train_gt.sum((1,2)).mean():.1f} | "
        f"eval n={len(eval_img)} mean_gt_area={eval_gt.sum((1,2)).mean():.1f}",
        flush=True,
    )

    rough_s = score_masks(eval_r, eval_gt)
    morph_s = score_masks(morph_refine(eval_r), eval_gt)

    g_model, g_sec = train_global(train_img, train_gt, train_r, train_p, epochs=12, device=device, seed=seed)
    g_pred = eval_global(g_model, eval_img, eval_r, eval_p, device=device)
    g_s = score_masks(g_pred, eval_gt)
    print(f"Global Refiner DSC={g_s['dsc']:.4f} HD95={g_s['hd95']:.2f} ({g_sec:.1f}s)", flush=True)

    z_model, z_sec = train_zoom_residual(
        train_img, train_gt, train_r, train_p, epochs=12, device=device, seed=seed
    )
    z_pred = eval_zoom_residual(z_model, eval_img, eval_r, eval_p, device=device)
    z_s = score_masks(z_pred, eval_gt)
    print(f"Zoom+Residual DSC={z_s['dsc']:.4f} HD95={z_s['hd95']:.2f} ({z_sec:.1f}s)", flush=True)

    ppo_pred, ppo_sec = train_eval_ppo(
        train_img, train_gt, train_r, train_p, eval_img, eval_gt, eval_r, eval_p, timesteps=4000, seed=seed
    )
    ppo_s = score_masks(ppo_pred, eval_gt)
    print(f"PPO DSC={ppo_s['dsc']:.4f} HD95={ppo_s['hd95']:.2f} ({ppo_sec:.1f}s)", flush=True)

    # micro subset
    micro = eval_gt.sum((1, 2)) < 50
    def sub(preds):
        if micro.sum() == 0:
            return {"dsc": float("nan"), "hd95": float("nan"), "n": 0}
        return score_masks(preds[micro], eval_gt[micro])

    rows = [
        ("Rough", rough_s, score_masks(eval_r[micro], eval_gt[micro]) if micro.any() else None, 0.0),
        ("Morphology", morph_s, sub(morph_refine(eval_r)), 0.0),
        ("Global Refiner", g_s, sub(g_pred), g_sec),
        ("Zoom+Residual", z_s, sub(z_pred), z_sec),
        ("PPO", ppo_s, sub(ppo_pred), ppo_sec),
    ]

    lines = [
        "# Small Zoom+Residual Refiner 시뮬레이션",
        "",
        "메인 파이프라인 미변경. 합성 Small blob (`area<300`), micro(`<50`) 오버샘플 학습.",
        "",
        "- Global: 128×128 Tiny-UNet `[MRI,rough,prob]→mask`",
        "- Zoom+Residual: 64×64 COM crop, `logit = f(x) + α·logit(rough)`, boundary Dice, empty-crop fallback",
        "- PPO: MaskRefinementEnv small, 4k steps",
        "",
        f"eval n={len(eval_img)}, micro n={int(micro.sum())}",
        "",
        "| Method | DSC ↑ | HD95 ↓ | ΔDSC | Micro DSC | Train sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    base = rough_s["dsc"]
    for name, s, ms, sec in rows:
        md = f"{ms['dsc']:.4f}" if ms and ms["n"] else "—"
        bold = "**" if name == "Zoom+Residual" else ""
        lines.append(
            f"| {bold}{name}{bold} | {s['dsc']:.4f} | {s['hd95']:.2f} | {s['dsc']-base:+.4f} | {md} | {sec:.1f} |"
        )
    lines += [
        "",
        f"총 소요: {time.time()-t_all:.1f}s",
        "",
        "> 합성·짧은 학습. BraTS 절대수치와 비교하지 말고 Small 설계 상대 효과만 볼 것.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
