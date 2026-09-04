"""
Stage 3 대체 후보 짧은 시뮬레이션 (메인 파이프라인/체크포인트 미변경).

비교:
  - Rough          : 초기 rough mask
  - Morphology      : 고정 open/close
  - PPO (짧게)      : MaskRefinementEnv 경계 보정
  - Refiner U-Net   : MRI + rough(+prob) → 보정 마스크 지도 학습

결과: results/sim_refiner_vs_ppo.md
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening, gaussian_filter
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.mask_refinement_env import MaskRefinementEnv
from src.utils.metrics import dice, hd95


OUT_MD = ROOT / "results" / "sim_refiner_vs_ppo.md"


def make_blob(h, w, cy, cx, ry, rx):
    yy, xx = np.ogrid[:h, :w]
    return (((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0).astype(np.float32)


def synth_batch(n: int, mode: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    h = w = 128
    images, gts, roughs, probs = [], [], [], []
    for _ in range(n):
        img = rng.normal(0.35, 0.12, (h, w)).astype(np.float32)
        img = np.clip(img, 0, 1)
        if mode == "small":
            cy, cx = rng.integers(40, 90), rng.integers(40, 90)
            gt = make_blob(h, w, cy, cx, rng.integers(4, 8), rng.integers(4, 8))
        elif mode == "medium":
            cy, cx = rng.integers(35, 95), rng.integers(35, 95)
            gt = make_blob(h, w, cy, cx, rng.integers(12, 18), rng.integers(12, 18))
        else:
            cy, cx = rng.integers(30, 100), rng.integers(30, 100)
            gt = make_blob(h, w, cy, cx, rng.integers(22, 32), rng.integers(22, 32))

        r = gt.copy()
        if rng.random() < 0.5:
            r = binary_dilation(r, iterations=int(rng.integers(1, 3))).astype(np.float32)
        else:
            r = binary_erosion(r, iterations=1).astype(np.float32)
        if r.sum() < 5:
            r = gt.copy()
        # 경계 노이즈
        noise = (rng.random((h, w)) < 0.01).astype(np.float32)
        r = np.clip(r + noise * (1 - gt), 0, 1)
        r = (r > 0.5).astype(np.float32)

        img = np.clip(img + 0.25 * gaussian_filter(gt, 1.5), 0, 1).astype(np.float32)
        # Expert 확률맵 근사: GT soft + rough 쪽 편향
        prob = np.clip(0.55 * gaussian_filter(gt, 1.0) + 0.35 * r + rng.normal(0, 0.05, (h, w)), 0, 1).astype(
            np.float32
        )

        images.append(img)
        gts.append(gt)
        roughs.append(r)
        probs.append(prob)
    return np.stack(images), np.stack(gts), np.stack(roughs), np.stack(probs)


def score_masks(preds: np.ndarray, gts: np.ndarray) -> dict:
    dscs, hds = [], []
    for p, g in zip(preds, gts):
        pb = (p > 0.5).astype(np.float32)
        gb = (g > 0.5).astype(np.float32)
        dscs.append(dice(pb, gb))
        hds.append(hd95(pb, gb))
    return {
        "dsc": float(np.mean(dscs)),
        "hd95": float(np.mean(hds)),
        "n": len(dscs),
    }


def morph_refine(roughs: np.ndarray) -> np.ndarray:
    out = []
    for r in roughs:
        m = (r > 0.5)
        m = binary_opening(m, iterations=1)
        m = binary_closing(m, iterations=1)
        if m.sum() < 5:
            m = r > 0.5
        out.append(m.astype(np.float32))
    return np.stack(out)


class TinyRefiner(nn.Module):
    """얕은 U-Net 스타일 refiner: [MRI, rough, prob] → logit."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.bot = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bot(e2)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def soft_dice_loss(logits, targets, eps=1e-5):
    p = torch.sigmoid(logits)
    num = 2 * (p * targets).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


def train_refiner(images, gts, roughs, probs, epochs=8, batch_size=16, device="cpu", seed=0):
    torch.manual_seed(seed)
    x = torch.from_numpy(np.stack([images, roughs, probs], axis=1)).float()
    y = torch.from_numpy(gts[:, None]).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    model = TinyRefiner(3).to(device)
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
def eval_refiner(model, images, roughs, probs, device="cpu") -> np.ndarray:
    model.eval()
    x = torch.from_numpy(np.stack([images, roughs, probs], axis=1)).float().to(device)
    out = []
    for i in range(0, len(x), 16):
        logits = model(x[i : i + 16])
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[:, 0]
        out.append(pred)
    return np.concatenate(out, axis=0)


def train_eval_ppo(mode, train_img, train_gt, train_r, train_p, eval_img, eval_gt, eval_r, eval_p, timesteps=4000, seed=0):
    def make_train():
        return MaskRefinementEnv(
            images=train_img,
            gt_masks=train_gt,
            rough_masks=train_r,
            uncertainty_maps=train_p,
            max_steps=15,
            target_dsc=1.0,
            step_penalty=0.001,
            refinement_mode=mode,
            enable_stop=False,
        )

    venv = DummyVecEnv([make_train])
    policy_kwargs = dict(normalize_images=False, net_arch=[64, 64])
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
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device="cpu",
    )
    model.learn(total_timesteps=timesteps)
    train_sec = time.time() - t0

    env = MaskRefinementEnv(
        images=eval_img,
        gt_masks=eval_gt,
        rough_masks=eval_r,
        uncertainty_maps=eval_p,
        max_steps=15,
        target_dsc=1.0,
        step_penalty=0.001,
        refinement_mode=mode,
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
    return np.stack(preds), train_sec


def run_mode(mode: str, seed: int = 0):
    print(f"\n=== MODE {mode.upper()} ===", flush=True)
    train_img, train_gt, train_r, train_p = synth_batch(64, mode, seed=seed)
    eval_img, eval_gt, eval_r, eval_p = synth_batch(32, mode, seed=seed + 77)

    rough_s = score_masks(eval_r, eval_gt)
    morph_pred = morph_refine(eval_r)
    morph_s = score_masks(morph_pred, eval_gt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    refiner, ref_sec = train_refiner(train_img, train_gt, train_r, train_p, epochs=10, device=device, seed=seed)
    ref_pred = eval_refiner(refiner, eval_img, eval_r, eval_p, device=device)
    ref_s = score_masks(ref_pred, eval_gt)

    ppo_pred, ppo_sec = train_eval_ppo(
        mode,
        train_img,
        train_gt,
        train_r,
        train_p,
        eval_img,
        eval_gt,
        eval_r,
        eval_p,
        timesteps=4000,
        seed=seed,
    )
    ppo_s = score_masks(ppo_pred, eval_gt)

    row = {
        "mode": mode,
        "rough": rough_s,
        "morph": morph_s,
        "ppo": ppo_s,
        "refiner": ref_s,
        "ppo_sec": ppo_sec,
        "refiner_sec": ref_sec,
    }
    print(
        f"[{mode}] Rough DSC={rough_s['dsc']:.4f} | Morph={morph_s['dsc']:.4f} | "
        f"PPO={ppo_s['dsc']:.4f} ({ppo_sec:.1f}s) | Refiner={ref_s['dsc']:.4f} ({ref_sec:.1f}s)",
        flush=True,
    )
    return row


def main():
    t_all = time.time()
    rows = [run_mode(m, seed=42) for m in ("small", "medium", "large")]

    lines = [
        "# Refiner vs PPO 짧은 시뮬레이션",
        "",
        "메인 파이프라인·체크포인트는 변경하지 않음. 합성 blob + Expert 노이즈 rough.",
        "",
        "- Rough: 초기 마스크",
        "- Morphology: opening→closing",
        "- PPO: `MaskRefinementEnv` 4k steps (CPU, 짧은 학습)",
        "- Refiner: Tiny U-Net, 입력 `[MRI, rough, prob]`, BCE+Dice 10 epoch",
        "",
        "| Size | Method | DSC ↑ | HD95 ↓ | ΔDSC vs Rough | Train sec |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        base = r["rough"]["dsc"]
        for name, key, sec in (
            ("Rough", "rough", 0.0),
            ("Morphology", "morph", 0.0),
            ("PPO", "ppo", r["ppo_sec"]),
            ("**Refiner**", "refiner", r["refiner_sec"]),
        ):
            s = r[key]
            lines.append(
                f"| {r['mode']} | {name} | {s['dsc']:.4f} | {s['hd95']:.2f} | "
                f"{s['dsc'] - base:+.4f} | {sec:.1f} |"
            )

    # overall mean
    def mean_of(key, field):
        return float(np.mean([r[key][field] for r in rows]))

    lines += [
        "",
        "## Overall (3 size mean)",
        "",
        f"| Method | DSC | HD95 |",
        f"|---|---:|---:|",
        f"| Rough | {mean_of('rough','dsc'):.4f} | {mean_of('rough','hd95'):.2f} |",
        f"| Morphology | {mean_of('morph','dsc'):.4f} | {mean_of('morph','hd95'):.2f} |",
        f"| PPO | {mean_of('ppo','dsc'):.4f} | {mean_of('ppo','hd95'):.2f} |",
        f"| **Refiner** | **{mean_of('refiner','dsc'):.4f}** | **{mean_of('refiner','hd95'):.2f}** |",
        "",
        f"총 소요: {time.time() - t_all:.1f}s",
        "",
        "> 합성 데이터·짧은 학습이므로 절대 수치는 BraTS 논문 수치와 직접 비교하지 말 것. "
        "같은 조건에서 **상대 순위**만 본다.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_MD}", flush=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
