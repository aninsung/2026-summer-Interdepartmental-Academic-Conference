"""
Stage 3: Alternating SL Fix <-> PPO distillation.

Saves:
  checkpoints/sl_refiner_{small|medium|large}.pt
  checkpoints/ppo_{small|medium|large}.zip  (short PPO teacher)

Inference intended path: SL refiner only (PPO used to distill into SL).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.mask_refinement_env import MaskRefinementEnv
from src.models.sl_refiner import DualHeadRefiner, apply_sl_refiner
from src.utils.metrics import dice
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def gt_error_candidate(rough, gt, band=2):
    err = (rough > 0.5) != (gt > 0.5)
    m = rough > 0.5
    dil, er = m.copy(), m.copy()
    for _ in range(band):
        dil = binary_dilation(dil)
        er = binary_erosion(er)
    return (err | (dil ^ er)).astype(np.float32)


def soft_dice(logits, targets, eps=1e-5):
    p = torch.sigmoid(logits)
    num = 2 * (p * targets).sum((2, 3)) + eps
    den = p.sum((2, 3)) + targets.sum((2, 3)) + eps
    return 1 - (num / den).mean()


def _images_to_nchw(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3:
        return images[:, None].astype(np.float32)
    if images.ndim == 4:
        return images.astype(np.float32)
    raise ValueError(f"Unexpected images shape {images.shape}")


def train_sl(net, images, gts, roughs, probs, epochs, device, lr=1e-3):
    """Supervise against real GT only."""
    imgs = _images_to_nchw(images)
    x = np.concatenate(
        [imgs, roughs[:, None].astype(np.float32), probs[:, None].astype(np.float32)],
        axis=1,
    )
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
        num_workers=0,
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    net.train()
    for ep in range(epochs):
        losses = []
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
            losses.append(float(loss.detach()))
        log.info("SL epoch %d/%d loss=%.4f", ep + 1, epochs, float(np.mean(losses)))
    return net


def distill_sl_from_teacher(
    net, images, roughs, probs, teacher_masks, epochs, device, lr=5e-4
):
    """Separate distillation: Fix/Candidate targets = PPO teacher masks (not GT)."""
    if len(images) == 0:
        return net
    imgs = _images_to_nchw(images)
    x = np.concatenate(
        [imgs, roughs[:, None].astype(np.float32), probs[:, None].astype(np.float32)],
        axis=1,
    )
    cand = np.stack(
        [gt_error_candidate(r, t) for r, t in zip(roughs, teacher_masks)]
    )[:, None].astype(np.float32)
    y = teacher_masks[:, None].astype(np.float32)
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
        num_workers=0,
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    net.train()
    for ep in range(epochs):
        losses = []
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
            losses.append(float(loss.detach()))
        log.info("Distill epoch %d/%d loss=%.4f", ep + 1, epochs, float(np.mean(losses)))
    return net


@torch.no_grad()
def batch_sl_fix(net, images, roughs, probs, device):
    outs = []
    for i in range(len(images)):
        outs.append(
            apply_sl_refiner(net, images[i], roughs[i], probs[i], device, morph_small=False)
        )
    return np.stack(outs, 0)


def train_ppo_teacher(mode, images, gts, inits, probs, timesteps, seed, device_str="cpu"):
    def _fn():
        return MaskRefinementEnv(
            images=images,
            gt_masks=gts,
            rough_masks=inits,
            uncertainty_maps=probs,
            max_steps=15,
            target_dsc=1.0,
            step_penalty=0.001,
            refinement_mode=mode,
            enable_stop=True,
        )

    venv = DummyVecEnv([_fn])
    model = PPO(
        "CnnPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=3,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=dict(normalize_images=False, net_arch=[64, 64]),
        verbose=0,
        seed=seed,
        device=device_str,
    )
    model.learn(total_timesteps=timesteps)
    return model


def harvest_successes(model, mode, images, gts, inits, probs, max_n=64):
    n = min(max_n, len(images))
    env = MaskRefinementEnv(
        images=images[:n],
        gt_masks=gts[:n],
        rough_masks=inits[:n],
        uncertainty_maps=probs[:n],
        max_steps=15,
        target_dsc=1.0,
        step_penalty=0.001,
        refinement_mode=mode,
        enable_stop=True,
    )
    finals, improved = [], []
    for i in range(n):
        obs, _ = env.reset(seed=i)
        init_d = float(dice(env._current_mask, gts[i]))
        for _ in range(15):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        fin = env._current_mask.copy()
        if float(dice(fin, gts[i])) > init_d + 1e-4:
            improved.append(i)
            finals.append(fin)
        else:
            finals.append(inits[i].copy())
    return np.stack(finals), improved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refinement_mode", type=str, required=True, choices=["small", "medium", "large"])
    parser.add_argument("--model_type", type=str, default="caranet")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--max_train_patients", type=int, default=210)
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--sl_epochs", type=int, default=12)
    parser.add_argument("--distill_epochs", type=int, default=4)
    parser.add_argument("--ppo_timesteps", type=int, default=20000)
    parser.add_argument("--save_sl", type=str, default="")
    parser.add_argument("--save_ppo", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed, args.deterministic)
    mode = args.refinement_mode
    save_sl = args.save_sl or f"checkpoints/sl_refiner_{mode}.pt"
    save_ppo = args.save_ppo or f"checkpoints/ppo_{mode}.zip"

    sys.path.insert(0, str(ROOT / "scripts" / "train"))
    from train_agent import load_real_data
    from src.data.patient_split import load_or_create_patient_split

    model_type = {"small": "caranet", "medium": "unetplusplus", "large": "segresnet"}[mode]
    unet_map = {
        "caranet": "checkpoints/caranet_best.pt",
        "unetplusplus": "checkpoints/unetplusplus_best.pt",
        "segresnet": "checkpoints/segresnet_best.pt",
    }

    split = load_or_create_patient_split(
        args.train_root, args.max_train_patients, args.patient_split
    )
    train_ids = split["train"]

    log.info("Loading Stage2 rough masks for mode=%s (oracle expert, no synthetic mixup)", mode)
    images, gts, roughs, probs = load_real_data(
        train_root=args.train_root,
        modality=args.modality,
        target_size=128,
        max_patients=args.max_train_patients,
        unet_path=unet_map[model_type],
        model_type=model_type,
        refinement_mode=mode,
        patient_ids=train_ids,
        mixup=False,
        noise_seed=args.seed,
        oracle_expert=True,
    )
    img_ch = 1 if images.ndim == 3 else int(images.shape[1])
    sl_in_ch = img_ch + 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DualHeadRefiner(in_ch=sl_in_ch).to(device)
    log.info("DualHeadRefiner in_ch=%d (img_ch=%d + rough + prob)", sl_in_ch, img_ch)

    bank_img = bank_r = bank_p = bank_teacher = None

    for rd in range(1, args.rounds + 1):
        log.info("=== Round %d/%d SL on real GT ===", rd, args.rounds)
        net = train_sl(net, images, gts, roughs, probs, epochs=args.sl_epochs, device=device)

        if bank_teacher is not None and len(bank_teacher) > 0:
            log.info(
                "=== Round %d/%d distill from PPO teacher (%d samples) ===",
                rd, args.rounds, len(bank_teacher),
            )
            net = distill_sl_from_teacher(
                net,
                bank_img,
                bank_r,
                bank_p,
                bank_teacher,
                epochs=args.distill_epochs,
                device=device,
            )

        fixed = batch_sl_fix(net, images, roughs, probs, device)
        mean_d = float(np.mean([dice(fixed[i], gts[i]) for i in range(len(gts))]))
        log.info("Round %d SL-fix mean DSC vs GT (train class): %.4f", rd, mean_d)

        log.info("=== Round %d/%d PPO teacher (%d steps) ===", rd, args.rounds, args.ppo_timesteps)
        n_ppo = min(128, len(images))
        idx = np.random.default_rng(args.seed + rd).choice(len(images), n_ppo, replace=False)
        ppo = train_ppo_teacher(
            mode,
            images[idx],
            gts[idx],
            fixed[idx],
            probs[idx],
            timesteps=args.ppo_timesteps,
            seed=args.seed + rd,
            device_str="cpu",
        )
        finals, improved = harvest_successes(
            ppo, mode, images[idx], gts[idx], fixed[idx], probs[idx], max_n=n_ppo
        )
        log.info("PPO improved %d/%d", len(improved), n_ppo)
        if improved:
            bank_img = images[idx][improved]
            bank_r = fixed[idx][improved]
            bank_p = probs[idx][improved]
            bank_teacher = finals[improved]  # teacher only — never mixed as GT in train_sl

        os.makedirs(os.path.dirname(save_ppo) or ".", exist_ok=True)
        ppo_path = save_ppo[:-4] if save_ppo.endswith(".zip") else save_ppo
        ppo.save(ppo_path)

    if bank_teacher is not None and len(bank_teacher) > 0:
        log.info("Final distill from last PPO teacher bank (%d)", len(bank_teacher))
        net = distill_sl_from_teacher(
            net, bank_img, bank_r, bank_p, bank_teacher,
            epochs=args.distill_epochs, device=device,
        )

    os.makedirs(os.path.dirname(save_sl) or ".", exist_ok=True)
    torch.save(
        {"state_dict": net.state_dict(), "mode": mode, "in_ch": sl_in_ch, "img_ch": img_ch},
        save_sl,
    )
    log.info("Saved SL refiner -> %s", save_sl)
    log.info("Saved last PPO teacher -> %s.zip", ppo_path)


if __name__ == "__main__":
    main()
