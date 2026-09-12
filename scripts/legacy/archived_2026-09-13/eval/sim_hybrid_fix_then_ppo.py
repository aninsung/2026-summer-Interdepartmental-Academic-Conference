"""
하이브리드 Stage 3 시뮬레이션 (메인 파이프라인 미변경)

흐름:
  1) Supervised "어디를 고칠지" 헤드 → candidate mask (경계∩저신뢰 근사 + 학습)
  2) Supervised local fix → candidate 위에서만 rough 수정
  3) PPO → 그 마스크를 초기값으로 RL 추가 보정

비교:
  Rough | Morph | FixOnly | PPOOnly | Hybrid(Fix→PPO)

결과: results/sim_hybrid_fix_then_ppo.md
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    gaussian_filter,
)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.mask_refinement_env import MaskRefinementEnv
from src.utils.metrics import dice, hd95

OUT = ROOT / "results" / "sim_hybrid_fix_then_ppo.md"
H = W = 128


def make_blob(h, w, cy, cx, ry, rx):
    yy, xx = np.ogrid[:h, :w]
    return (((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0).astype(np.float32)


def synth_batch(n: int, mode: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    images, gts, roughs, probs = [], [], [], []
    for _ in range(n):
        img = np.clip(rng.normal(0.35, 0.12, (H, W)), 0, 1).astype(np.float32)
        if mode == "small":
            cy, cx = rng.integers(40, 90), rng.integers(40, 90)
            gt = make_blob(H, W, cy, cx, rng.integers(4, 8), rng.integers(4, 8))
        elif mode == "medium":
            cy, cx = rng.integers(35, 95), rng.integers(35, 95)
            gt = make_blob(H, W, cy, cx, rng.integers(12, 18), rng.integers(12, 18))
        else:
            cy, cx = rng.integers(30, 100), rng.integers(30, 100)
            gt = make_blob(H, W, cy, cx, rng.integers(22, 32), rng.integers(22, 32))
        r = gt.copy()
        if rng.random() < 0.5:
            r = binary_dilation(r, iterations=int(rng.integers(1, 3))).astype(np.float32)
        else:
            r = binary_erosion(r, iterations=1).astype(np.float32)
        if r.sum() < 5:
            r = gt.copy()
        r = np.clip(r + ((rng.random((H, W)) < 0.01).astype(np.float32) * (1 - gt)), 0, 1)
        r = (r > 0.5).astype(np.float32)
        img = np.clip(img + 0.25 * gaussian_filter(gt, 1.5), 0, 1).astype(np.float32)
        prob = np.clip(
            0.55 * gaussian_filter(gt, 1.0) + 0.35 * r + rng.normal(0, 0.05, (H, W)), 0, 1
        ).astype(np.float32)
        images.append(img)
        gts.append(gt)
        roughs.append(r)
        probs.append(prob)
    return np.stack(images), np.stack(gts), np.stack(roughs), np.stack(probs)


def score(preds, gts):
    d, h = [], []
    for p, g in zip(preds, gts):
        pb = (p > 0.5).astype(np.float32)
        gb = (g > 0.5).astype(np.float32)
        d.append(dice(pb, gb))
        h.append(hd95(pb, gb))
    return {"dsc": float(np.mean(d)), "hd95": float(np.mean(h))}


def morph_refine(roughs):
    out = []
    for r in roughs:
        m = binary_closing(binary_opening(r > 0.5, iterations=1), iterations=1)
        if m.sum() < 5:
            m = r > 0.5
        out.append(m.astype(np.float32))
    return np.stack(out)


def heuristic_candidate(rough, prob, band=2):
    """배포 가능 proxy: 경계 밴드 ∩ 저신뢰."""
    m = rough > 0.5
    dil = m.copy()
    er = m.copy()
    for _ in range(band):
        dil = binary_dilation(dil)
        er = binary_erosion(er)
    boundary = dil ^ er
    uncertain = np.abs(prob - 0.5) < 0.25
    return (boundary & uncertain).astype(np.float32)


def gt_error_candidate(rough, gt, band=2):
    """학습 타깃용(GT): 틀린 화소 ∪ 경계."""
    err = (rough > 0.5) != (gt > 0.5)
    m = rough > 0.5
    dil = m.copy()
    for _ in range(band):
        dil = binary_dilation(dil)
    er = m.copy()
    for _ in range(band):
        er = binary_erosion(er)
    boundary = dil ^ er
    return (err | boundary).astype(np.float32)


class DualHeadNet(nn.Module):
    """공유 인코더 + (candidate logits, fix logits)."""

    def __init__(self, in_ch=3):
        super().__init__()
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


def soft_dice(logits, targets, eps=1e-5):
    p = torch.sigmoid(logits)
    num = 2 * (p * targets).sum((2, 3)) + eps
    den = p.sum((2, 3)) + targets.sum((2, 3)) + eps
    return 1 - (num / den).mean()


def train_supervised(images, gts, roughs, probs, epochs=12, device="cpu", seed=0):
    torch.manual_seed(seed)
    x = np.stack([images, roughs, probs], 1).astype(np.float32)
    cand = np.stack([gt_error_candidate(r, g) for r, g in zip(roughs, gts)])[:, None].astype(np.float32)
    y = gts[:, None].astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(cand), torch.from_numpy(y), torch.from_numpy(roughs[:, None].astype(np.float32)))
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    net = DualHeadNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    t0 = time.time()
    net.train()
    for _ in range(epochs):
        for xb, cb, yb, rb in loader:
            xb, cb, yb, rb = xb.to(device), cb.to(device), yb.to(device), rb.to(device)
            opt.zero_grad(set_to_none=True)
            lc, lf = net(xb)
            # candidate: where to fix
            loss_c = bce(lc, cb) + soft_dice(lc, cb)
            # fix: predict GT, but weight loss on candidate (+ weak global)
            w = cb * 0.8 + 0.2
            loss_f = F.binary_cross_entropy_with_logits(lf, yb, weight=w) + soft_dice(lf, yb)
            # residual consistency: outside candidate stay near rough
            with torch.no_grad():
                outside = 1.0 - cb
            pref = torch.sigmoid(lf)
            loss_keep = ((pref - rb).abs() * outside).mean()
            loss = loss_c + loss_f + 0.3 * loss_keep
            loss.backward()
            opt.step()
    return net, time.time() - t0


@torch.no_grad()
def supervised_fix(net, images, roughs, probs, device="cpu", cand_thr=0.4):
    net.eval()
    x = torch.from_numpy(np.stack([images, roughs, probs], 1)).float().to(device)
    outs, cands = [], []
    for i in range(0, len(x), 16):
        lc, lf = net(x[i : i + 16])
        cand = (torch.sigmoid(lc) > cand_thr).float()
        # deploy-ish gate: also intersect heuristic uncertain boundary
        fix_p = torch.sigmoid(lf)
        # only overwrite inside predicted candidate; else keep rough
        rb = x[i : i + 16, 1:2]
        blended = torch.where(cand > 0.5, fix_p, rb)
        pred = (blended > 0.5).float()
        # safety: empty → rough
        for j in range(pred.size(0)):
            if pred[j].sum() < 5:
                pred[j] = rb[j]
        outs.append(pred[:, 0].cpu().numpy())
        cands.append(cand[:, 0].cpu().numpy())
    return np.concatenate(outs, 0), np.concatenate(cands, 0)


def run_ppo(mode, train_img, train_gt, train_r, train_p, eval_img, eval_gt, init_masks, eval_p, timesteps=3500, seed=0):
    """PPO trained on train rough; evaluated starting from init_masks (rough or fixed)."""

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
        rough_masks=init_masks,  # start from supervised fix for hybrid
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
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        preds.append(env._current_mask.copy())
    return np.stack(preds), sec


def run_mode(mode: str, seed: int = 42):
    print(f"\n=== {mode.upper()} ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_img, tr_gt, tr_r, tr_p = synth_batch(64, mode, seed)
    ev_img, ev_gt, ev_r, ev_p = synth_batch(32, mode, seed + 7)

    rough_s = score(ev_r, ev_gt)
    morph_s = score(morph_refine(ev_r), ev_gt)

    net, fix_sec = train_supervised(tr_img, tr_gt, tr_r, tr_p, epochs=12, device=device, seed=seed)
    fix_pred, cand = supervised_fix(net, ev_img, ev_r, ev_p, device=device)
    fix_s = score(fix_pred, ev_gt)
    cand_cov = float(cand.mean())
    print(f"FixOnly DSC={fix_s['dsc']:.4f} cand_frac={cand_cov:.3f} ({fix_sec:.1f}s)", flush=True)

    ppo_only, ppo_sec = run_ppo(mode, tr_img, tr_gt, tr_r, tr_p, ev_img, ev_gt, ev_r, ev_p, seed=seed)
    ppo_s = score(ppo_only, ev_gt)
    print(f"PPOOnly DSC={ppo_s['dsc']:.4f} ({ppo_sec:.1f}s)", flush=True)

    # Hybrid: train PPO on rough, but eval starts from Fix (same policy transfer)
    # Better hybrid: also train PPO starting from supervised-fixed train set
    tr_fix, _ = supervised_fix(net, tr_img, tr_r, tr_p, device=device)
    hybrid_pred, hy_sec = run_ppo(
        mode, tr_img, tr_gt, tr_fix, tr_p, ev_img, ev_gt, fix_pred, ev_p, timesteps=3500, seed=seed + 1
    )
    hy_s = score(hybrid_pred, ev_gt)
    print(f"Hybrid DSC={hy_s['dsc']:.4f} ({hy_sec:.1f}s)", flush=True)

    return {
        "mode": mode,
        "rough": rough_s,
        "morph": morph_s,
        "fix": fix_s,
        "ppo": ppo_s,
        "hybrid": hy_s,
        "fix_sec": fix_sec,
        "ppo_sec": ppo_sec,
        "hy_sec": hy_sec,
        "cand_frac": cand_cov,
    }


def main():
    t0 = time.time()
    rows = [run_mode(m) for m in ("small", "medium", "large")]
    lines = [
        "# Hybrid: Supervised Fix → PPO 시뮬레이션",
        "",
        "메인 파이프라인 미변경. 합성 데이터.",
        "",
        "1. Dual-head 지도 학습: candidate(고칠 곳, GT-error∪경계로 학습) + fix(후보 가중 GT)",
        "2. 추론: candidate 안에서만 rough 덮어쓰기 (밖은 유지)",
        "3. Hybrid: Fix된 마스크를 rough로 두고 PPO 학습·평가",
        "",
        "| Size | Method | DSC ↑ | HD95 ↓ | ΔDSC | sec |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        base = r["rough"]["dsc"]
        for name, key, sec in (
            ("Rough", "rough", 0.0),
            ("Morph", "morph", 0.0),
            ("FixOnly", "fix", r["fix_sec"]),
            ("PPOOnly", "ppo", r["ppo_sec"]),
            ("**Hybrid Fix→PPO**", "hybrid", r["hy_sec"]),
        ):
            s = r[key]
            lines.append(
                f"| {r['mode']} | {name} | {s['dsc']:.4f} | {s['hd95']:.2f} | {s['dsc']-base:+.4f} | {sec:.1f} |"
            )
                lines.append(f"| {r['mode']} | (cand frac) | {r['cand_frac']:.3f} | - | - | - |")

    def mean(key, field):
        return float(np.mean([r[key][field] for r in rows]))

    lines += [
        "",
        "## Overall mean",
        "",
        "| Method | DSC | HD95 |",
        "|---|---:|---:|",
        f"| Rough | {mean('rough','dsc'):.4f} | {mean('rough','hd95'):.2f} |",
        f"| Morph | {mean('morph','dsc'):.4f} | {mean('morph','hd95'):.2f} |",
        f"| FixOnly | {mean('fix','dsc'):.4f} | {mean('fix','hd95'):.2f} |",
        f"| PPOOnly | {mean('ppo','dsc'):.4f} | {mean('ppo','hd95'):.2f} |",
        f"| **Hybrid** | **{mean('hybrid','dsc'):.4f}** | **{mean('hybrid','hd95'):.2f}** |",
        "",
        f"총 소요: {time.time()-t0:.1f}s",
        "",
        "> candidate 학습은 GT-error를 쓰지만, 추론 덮어쓰기는 예측 candidate만 사용. "
        "절대수치는 BraTS와 비교하지 말 것.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
