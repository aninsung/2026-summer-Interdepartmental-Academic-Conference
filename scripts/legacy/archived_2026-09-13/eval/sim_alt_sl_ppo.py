"""
교대 학습 시뮬레이션: SL Fix <-> PPO (메인 파이프라인 미변경)

Round:
  1) SL: candidate + local fix
  2) PPO: SL-fixed 마스크를 초기값으로 학습/평가
  3) PPO 이득 샘플을 bank에 넣고 SL 재학습

결과: results/sim_alt_sl_ppo.md
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
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.mask_refinement_env import MaskRefinementEnv
from src.utils.metrics import dice, hd95

OUT = ROOT / "results" / "sim_alt_sl_ppo.md"
H = W = 128
ROUNDS = 3


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


def gt_error_candidate(rough, gt, band=2):
    err = (rough > 0.5) != (gt > 0.5)
    m = rough > 0.5
    dil, er = m.copy(), m.copy()
    for _ in range(band):
        dil = binary_dilation(dil)
        er = binary_erosion(er)
    return (err | (dil ^ er)).astype(np.float32)


class DualHeadNet(nn.Module):
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


def train_sl(images, gts, roughs, probs, epochs=8, device="cpu", seed=0, net=None):
    torch.manual_seed(seed)
    x = np.stack([images, roughs, probs], 1).astype(np.float32)
    cand = np.stack([gt_error_candidate(r, g) for r, g in zip(roughs, gts)])[:, None].astype(np.float32)
    y = gts[:, None].astype(np.float32)
    rb = roughs[:, None].astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(cand),
            torch.from_numpy(y),
            torch.from_numpy(rb),
        ),
        batch_size=16,
        shuffle=True,
    )
    if net is None:
        net = DualHeadNet().to(device)
    else:
        net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    net.train()
    t0 = time.time()
    for _ in range(epochs):
        for xb, cb, yb, rbb in loader:
            xb, cb, yb, rbb = xb.to(device), cb.to(device), yb.to(device), rbb.to(device)
            opt.zero_grad(set_to_none=True)
            lc, lf = net(xb)
            w = cb * 0.8 + 0.2
            loss = (
                bce(lc, cb)
                + soft_dice(lc, cb)
                + F.binary_cross_entropy_with_logits(lf, yb, weight=w)
                + soft_dice(lf, yb)
                + 0.3 * ((torch.sigmoid(lf) - rbb).abs() * (1.0 - cb)).mean()
            )
            loss.backward()
            opt.step()
    return net, time.time() - t0


@torch.no_grad()
def sl_fix(net, images, roughs, probs, device="cpu", cand_thr=0.4):
    net.eval()
    x = torch.from_numpy(np.stack([images, roughs, probs], 1)).float().to(device)
    outs = []
    for i in range(0, len(x), 16):
        lc, lf = net(x[i : i + 16])
        cand = (torch.sigmoid(lc) > cand_thr).float()
        rb = x[i : i + 16, 1:2]
        blended = torch.where(cand > 0.5, torch.sigmoid(lf), rb)
        pred = (blended > 0.5).float()
        for j in range(pred.size(0)):
            if pred[j].sum() < 5:
                pred[j] = rb[j]
        outs.append(pred[:, 0].cpu().numpy())
    return np.concatenate(outs, 0)


def augment_sl_bank(bank, images, gts, probs, start_masks, final_masks, improved_idx):
    if not improved_idx:
        return bank
    extra = {
        "images": np.stack([images[i] for i in improved_idx]),
        "gts": np.stack([final_masks[i] for i in improved_idx]),
        "roughs": np.stack([start_masks[i] for i in improved_idx]),
        "probs": np.stack([probs[i] for i in improved_idx]),
    }
    if bank is None:
        return extra
    return {k: np.concatenate([bank[k], extra[k]], 0) for k in extra}


def train_sl_mixed(base_img, base_gt, base_r, base_p, bank, epochs, device, seed, net):
    imgs, gts, roughs, probs = [base_img], [base_gt], [base_r], [base_p]
    if bank is not None and len(bank["images"]) > 0:
        imgs.append(bank["images"])
        gts.append(bank["gts"])
        roughs.append(bank["roughs"])
        probs.append(bank["probs"])
    return train_sl(
        np.concatenate(imgs, 0),
        np.concatenate(gts, 0),
        np.concatenate(roughs, 0),
        np.concatenate(probs, 0),
        epochs=epochs,
        device=device,
        seed=seed,
        net=net,
    )


def train_eval_ppo_collect(
    mode, tr_img, tr_gt, tr_init, tr_p, ev_img, ev_gt, ev_init, ev_p, timesteps=2800, seed=0
):
    def make_train():
        return MaskRefinementEnv(
            images=tr_img,
            gt_masks=tr_gt,
            rough_masks=tr_init,
            uncertainty_maps=tr_p,
            max_steps=12,
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

    def rollout(images, gts, inits, probs):
        env = MaskRefinementEnv(
            images=images,
            gt_masks=gts,
            rough_masks=inits,
            uncertainty_maps=probs,
            max_steps=12,
            target_dsc=1.0,
            step_penalty=0.001,
            refinement_mode=mode,
            enable_stop=False,
        )
        finals, improved = [], []
        for i in range(len(images)):
            obs, _ = env.reset(seed=i)
            init_d = float(dice(env._current_mask, gts[i]))
            for _ in range(12):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
            fin = env._current_mask.copy()
            fin_d = float(dice(fin, gts[i]))
            finals.append(fin)
            if fin_d > init_d + 1e-4:
                improved.append(i)
        return np.stack(finals), improved

    ev_finals, ev_imp = rollout(ev_img, ev_gt, ev_init, ev_p)
    n_h = min(24, len(tr_img))
    tr_finals, tr_imp = rollout(tr_img[:n_h], tr_gt[:n_h], tr_init[:n_h], tr_p[:n_h])
    return ev_finals, sec, ev_imp, tr_finals, tr_imp, n_h


def run_mode(mode: str, seed: int = 42):
    print(f"\n=== {mode.upper()} ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_img, tr_gt, tr_r, tr_p = synth_batch(56, mode, seed)
    ev_img, ev_gt, ev_r, ev_p = synth_batch(28, mode, seed + 11)

    rough_s = score(ev_r, ev_gt)
    bank = None
    net = None
    rows = []
    t_sl = t_ppo = 0.0

    for rd in range(1, ROUNDS + 1):
        net, sec_sl = train_sl_mixed(
            tr_img, tr_gt, tr_r, tr_p, bank, epochs=8, device=device, seed=seed + rd, net=net
        )
        t_sl += sec_sl

        tr_fix = sl_fix(net, tr_img, tr_r, tr_p, device=device)
        ev_fix = sl_fix(net, ev_img, ev_r, ev_p, device=device)
        fix_s = score(ev_fix, ev_gt)

        ev_finals, sec_ppo, ev_imp, tr_finals, tr_imp, n_h = train_eval_ppo_collect(
            mode,
            tr_img,
            tr_gt,
            tr_fix,
            tr_p,
            ev_img,
            ev_gt,
            ev_fix,
            ev_p,
            timesteps=2800,
            seed=seed + 100 + rd,
        )
        t_ppo += sec_ppo
        hy_s = score(ev_finals, ev_gt)

        bank = augment_sl_bank(
            bank,
            tr_img[:n_h],
            tr_gt[:n_h],
            tr_p[:n_h],
            tr_fix[:n_h],
            tr_finals,
            tr_imp,
        )
        n_bank = 0 if bank is None else len(bank["images"])
        print(
            f"R{rd}: Fix={fix_s['dsc']:.4f} Hybrid={hy_s['dsc']:.4f} "
            f"imp={len(ev_imp)}/{len(ev_img)} bank={n_bank}",
            flush=True,
        )
        rows.append({"round": rd, "fix": fix_s, "hybrid": hy_s, "bank": n_bank})

    return {"mode": mode, "rough": rough_s, "rounds": rows, "t_sl": t_sl, "t_ppo": t_ppo}


def main():
    t0 = time.time()
    all_rows = [run_mode(m) for m in ("small", "medium", "large")]
    lines = [
        "# Alternating SL <-> PPO simulation",
        "",
        "Pipeline unchanged. Synthetic data. Rounds=3.",
        "",
        "Each round: SL Fix -> PPO on Fix inits -> success bank -> SL retrain",
        "",
        "| Size | Round | Fix DSC | Hybrid DSC | dHybrid vs Rough | bank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in all_rows:
        base = r["rough"]["dsc"]
        for row in r["rounds"]:
            lines.append(
                f"| {r['mode']} | {row['round']} | {row['fix']['dsc']:.4f} | "
                f"{row['hybrid']['dsc']:.4f} | {row['hybrid']['dsc']-base:+.4f} | {row['bank']} |"
            )
        f0, fL = r["rounds"][0]["fix"]["dsc"], r["rounds"][-1]["fix"]["dsc"]
        h0, hL = r["rounds"][0]["hybrid"]["dsc"], r["rounds"][-1]["hybrid"]["dsc"]
        lines.append(
            f"| {r['mode']} | summary | {f0:.4f}->{fL:.4f} | {h0:.4f}->{hL:.4f} | "
            f"hybrid_delta={hL-h0:+.4f} | sl={r['t_sl']:.0f}s ppo={r['t_ppo']:.0f}s |"
        )
    lines += ["", f"Total: {time.time()-t0:.1f}s", "", "Note: relative round trends only.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    sys.stdout.buffer.write((text + "\nWrote " + str(OUT) + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
