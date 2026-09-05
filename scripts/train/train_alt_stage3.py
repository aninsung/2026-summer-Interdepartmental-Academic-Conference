"""
Stage 3: Alternating SL Fix <-> PPO distillation.

Saves:
  checkpoints/sl_refiner_{small|medium|large}.pt
  checkpoints/ppo_{small|medium|large}.zip  (PPO teacher; optional deploy zoom)

Deploy inference:
  --stage3_mode sl      → SL only (GT-free)
  --stage3_mode hybrid  → SL then zoom-PPO with gt_free last-mask (no GT cherry-pick)
  PPO is also distilled into SL when harvest finds improvements.
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
from scipy.ndimage import binary_dilation, binary_erosion, label as cc_label
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

# tee/에이전트 로그 비대화: tqdm 기본 OFF (TQDM_ENABLE=1 로 켤 수 있음)
from src.utils.progress import configure_quiet_logs
configure_quiet_logs()


def gt_error_candidate(rough, gt, band=2, shrink_focus=False):
    """Edit-candidate map. Medium/Large shrink_focus: rem_band FP/오류만 강조."""
    rough_b = rough > 0.5
    gt_b = gt > 0.5
    err = rough_b != gt_b
    dil, er = rough_b.copy(), rough_b.copy()
    for _ in range(band):
        dil = binary_dilation(dil)
        er = binary_erosion(er)
    if shrink_focus:
        rem = np.logical_and(rough_b, np.logical_not(er))
        fp = np.logical_and(rough_b, np.logical_not(gt_b))
        # 삭제할 외곽 FP + rem_band 오류 (추가/내부는 후보에서 약화)
        return np.logical_and(rem, np.logical_or(fp, err)).astype(np.float32)
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


def _dilate_iters(refinement_mode: str) -> int:
    return 2 if refinement_mode == "small" else 3


def _shrink_focus(refinement_mode: str) -> bool:
    return False


def _expand_focus(refinement_mode: str) -> bool:
    """Medium/Large: weight FN (under-seg expand) on boundary band."""
    return refinement_mode in ("medium", "large")


def _balanced_band_focus(refinement_mode: str) -> bool:
    """When expand_focus is on, skip balanced FP/FN mix."""
    return False


def _sl_boundary_mode(refinement_mode: str) -> str:
    """Align train apply with deploy: Medium/Large expand-only band."""
    if refinement_mode in ("medium", "large"):
        return "expand"
    return "replace"

def expand_to_components(
    images: np.ndarray,
    gts: np.ndarray,
    roughs: np.ndarray,
    probs: np.ndarray,
    refinement_mode: str = "medium",
    min_area: float = 5.0,
):
    """
    Match evaluate_pipeline Stage3 inputs:
      rough = connected component, prob = Expert/TTA-prob * dilated component.
    Targets use GT inside the dilated band (same spatial support as soft prob).
    """
    imgs = _images_to_nchw(images)
    dil_n = _dilate_iters(refinement_mode)
    struct = np.ones((3, 3), dtype=bool)
    out_img, out_gt, out_r, out_p = [], [], [], []
    for i in range(len(roughs)):
        rough = np.asarray(roughs[i], dtype=np.float32)
        prob = np.asarray(probs[i], dtype=np.float32)
        gt = np.asarray(gts[i], dtype=np.float32)
        img = imgs[i]
        lbl, nfeat = cc_label(rough > 0.2)
        for k in range(1, nfeat + 1):
            comp = (lbl == k).astype(np.float32)
            if float(comp.sum()) < min_area:
                continue
            dil = binary_dilation(comp.astype(bool), structure=struct, iterations=dil_n)
            dil_f = dil.astype(np.float32)
            soft = (prob * dil_f).astype(np.float32)
            if float(soft.sum()) == 0.0:
                soft = (rough * dil_f).astype(np.float32)
            out_img.append(img)
            out_r.append(comp)
            out_p.append(soft)
            out_gt.append((gt * dil_f).astype(np.float32))
    if not out_img:
        return (
            np.zeros((0,) + imgs.shape[1:], dtype=np.float32),
            np.zeros((0,) + roughs.shape[1:], dtype=np.float32),
            np.zeros((0,) + roughs.shape[1:], dtype=np.float32),
            np.zeros((0,) + roughs.shape[1:], dtype=np.float32),
        )
    return (
        np.stack(out_img, 0),
        np.stack(out_gt, 0),
        np.stack(out_r, 0),
        np.stack(out_p, 0),
    )


def train_sl(
    net, images, gts, roughs, probs, epochs, device, lr=1e-3, refinement_mode="medium", batch_size=128
):
    """Supervise against real GT on component-level inputs (deploy-aligned).

    Medium/Large: 경계 밴드 FP/FN 균형 가중 + replace band 추론과 정렬.
    """
    images, gts, roughs, probs = expand_to_components(
        images, gts, roughs, probs, refinement_mode=refinement_mode
    )
    if len(images) == 0:
        log.warning("SL: no components to train on")
        return net
    bs = max(1, min(int(batch_size), len(images)))
    focus = _shrink_focus(refinement_mode)
    expand_f = _expand_focus(refinement_mode)
    balanced = _balanced_band_focus(refinement_mode)
    log.info(
        "SL train components: %d (mode=%s, batch_size=%d, shrink_focus=%s, expand_focus=%s, balanced_band=%s)",
        len(images), refinement_mode, bs, focus, expand_f, balanced,
    )
    imgs = _images_to_nchw(images)
    x = np.concatenate(
        [imgs, roughs[:, None].astype(np.float32), probs[:, None].astype(np.float32)],
        axis=1,
    )
    cand = np.stack(
        [gt_error_candidate(r, g, shrink_focus=focus) for r, g in zip(roughs, gts)]
    )[:, None].astype(np.float32)
    rb = roughs[:, None].astype(np.float32)
    if focus:
        y = np.minimum(gts, roughs)[:, None].astype(np.float32)
    else:
        y = gts[:, None].astype(np.float32)
    if focus:
        w_np = []
        for r, g in zip(roughs, gts):
            rem = np.logical_and(r > 0.5, ~binary_erosion(r > 0.5, iterations=3))
            fp = np.logical_and(r > 0.5, g <= 0.5)
            fn = np.logical_and(r <= 0.5, g > 0.5)
            wt = np.ones_like(r, dtype=np.float32)
            wt = np.where(rem, wt * 2.0, wt)
            wt = np.where(np.logical_and(rem, fp), wt * 6.0, wt)
            wt = np.where(fn, wt * 0.15, wt)
            w_np.append(wt)
        pix_w = np.stack(w_np, 0)[:, None].astype(np.float32)
    elif expand_f:
        from src.envs.class_strategy import get_strategy
        band_n = max(1, int(get_strategy(refinement_mode).band_px))
        w_np = []
        for r, g in zip(roughs, gts):
            dil = binary_dilation(r > 0.5, iterations=band_n)
            add_band = np.logical_and(dil, r <= 0.5)
            fn = np.logical_and(r <= 0.5, g > 0.5)
            wt = np.ones_like(r, dtype=np.float32)
            wt = np.where(add_band, wt * 2.0, wt)
            wt = np.where(fn, wt * 5.0, wt)
            w_np.append(wt)
        pix_w = np.stack(w_np, 0)[:, None].astype(np.float32)
    elif balanced:
        # 경계 밴드 안 FP·FN 동등 강조 (과소/과분할 무관)
        w_np = []
        for r, g in zip(roughs, gts):
            rb_ = r > 0.5
            dil = binary_dilation(rb_, iterations=3)
            ero = binary_erosion(rb_, iterations=3)
            band = np.logical_xor(dil, ero)
            fp = np.logical_and(rb_, g <= 0.5)
            fn = np.logical_and(~rb_, g > 0.5)
            wt = np.ones_like(r, dtype=np.float32)
            wt = np.where(band, wt * 2.0, wt)
            wt = np.where(np.logical_and(band, fp), wt * 4.0, wt)
            wt = np.where(np.logical_and(band, fn), wt * 4.0, wt)
            w_np.append(wt)
        pix_w = np.stack(w_np, 0)[:, None].astype(np.float32)
    else:
        pix_w = np.ones_like(y, dtype=np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(cand),
            torch.from_numpy(y),
            torch.from_numpy(rb),
            torch.from_numpy(pix_w),
        ),
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    net.train()
    for ep in range(epochs):
        losses = []
        for xb, cb, yb, rbb, wb in loader:
            xb, cb, yb, rbb, wb = (
                xb.to(device), cb.to(device), yb.to(device), rbb.to(device), wb.to(device)
            )
            opt.zero_grad(set_to_none=True)
            lc, lf = net(xb)
            w = cb * 0.8 + 0.2
            sig = torch.sigmoid(lf)
            loss = (
                bce(lc, cb)
                + soft_dice(lc, cb)
                + F.binary_cross_entropy_with_logits(lf, yb, weight=w * wb)
                + soft_dice(lf, yb)
                + 0.5 * ((sig - rbb).abs() * (1.0 - cb)).mean()
            )
            if focus:
                # 명시적 anti-add: rough 밖 양성 예측 억제
                loss = loss + 0.75 * (sig * (1.0 - rbb)).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        log.info("SL epoch %d/%d loss=%.4f", ep + 1, epochs, float(np.mean(losses)))
    return net


def distill_sl_from_teacher(
    net,
    images,
    roughs,
    probs,
    teacher_masks,
    epochs,
    device,
    lr=5e-4,
    refinement_mode="medium",
    batch_size=128,
):
    """Distill PPO teacher on component-level inputs (same as deploy)."""
    if len(images) == 0:
        return net
    images, teacher_masks, roughs, probs = expand_to_components(
        images, teacher_masks, roughs, probs, refinement_mode=refinement_mode
    )
    if len(images) == 0:
        log.warning("Distill: no components")
        return net
    bs = max(1, min(int(batch_size), len(images)))
    focus = _shrink_focus(refinement_mode)
    log.info("Distill components: %d (batch_size=%d, shrink_focus=%s)", len(images), bs, focus)
    imgs = _images_to_nchw(images)
    x = np.concatenate(
        [imgs, roughs[:, None].astype(np.float32), probs[:, None].astype(np.float32)],
        axis=1,
    )
    cand = np.stack(
        [
            gt_error_candidate(r, t, shrink_focus=focus)
            for r, t in zip(roughs, teacher_masks)
        ]
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
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
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
def batch_sl_fix(net, images, roughs, probs, device, refinement_mode="medium"):
    """Component-wise SL then max-merge — same procedure as evaluate_pipeline."""
    from src.envs.class_strategy import get_strategy
    from src.models.sl_refiner import apply_sl_refiner

    dil_n = _dilate_iters(refinement_mode)
    struct = np.ones((3, 3), dtype=bool)
    morph_small = refinement_mode == "small"
    use_band = refinement_mode in ("medium", "large")
    boundary_mode = _sl_boundary_mode(refinement_mode)
    band_px = int(get_strategy(refinement_mode).band_px) if use_band else 0
    outs = []
    for i in range(len(images)):
        img = images[i]
        rough = np.asarray(roughs[i], dtype=np.float32)
        prob = np.asarray(probs[i], dtype=np.float32)
        lbl, nfeat = cc_label(rough > 0.2)
        final = np.zeros_like(rough, dtype=np.float32)
        for k in range(1, nfeat + 1):
            comp = (lbl == k).astype(np.float32)
            if float(comp.sum()) < 5:
                final = np.maximum(final, comp)
                continue
            dil = binary_dilation(comp.astype(bool), structure=struct, iterations=dil_n)
            soft = (prob * dil.astype(np.float32)).astype(np.float32)
            if float(soft.sum()) == 0.0:
                soft = (rough * dil.astype(np.float32)).astype(np.float32)
            refined = apply_sl_refiner(
                net,
                img,
                comp,
                soft,
                device,
                morph_small=morph_small,
                boundary_band_px=band_px,
                boundary_mode=boundary_mode,
                zoom_n_patches=0,
                zoom_patch=48,
                zoom_seed=(i * 1009 + k),
                cand_thr=(0.40 if use_band else 0.4),
            )
            if use_band and boundary_mode == "shrink":
                refined = np.minimum(refined, comp)
            elif use_band and boundary_mode == "expand":
                refined = np.maximum(refined, comp)
            final = np.maximum(final, refined.astype(np.float32))
        if float(final.sum()) == 0.0:
            final = rough.copy()
        outs.append(final)
    return np.stack(outs, 0)


def _ppo_seg_bias(mode: str) -> str:
    from src.envs.class_strategy import get_strategy
    return get_strategy(mode).seg_bias


def _class_ppo_kwargs(mode: str, local_action_band_px=None, local_action_uncert_floor=None):
    from src.envs.class_strategy import get_strategy
    s = get_strategy(mode)
    return dict(
        max_steps=s.max_steps,
        step_penalty=s.step_penalty,
        seg_bias=s.seg_bias,
        local_action_band_px=(s.band_px if local_action_band_px is None else local_action_band_px),
        local_action_uncert_floor=(
            s.uncert_floor if local_action_uncert_floor is None else local_action_uncert_floor
        ),
        use_zoom_obs=s.use_zoom_obs,
        zoom_size=s.zoom_size,
    )


def train_ppo_teacher(
    mode,
    images,
    gts,
    inits,
    probs,
    timesteps,
    seed,
    device_str=None,
    local_action_band_px=None,
    local_action_uncert_floor=None,
    log_every: int = 5000,
):
    if device_str is None:
        from src.utils.device import require_cuda_device
        device_str = str(require_cuda_device())
    elif not str(device_str).startswith("cuda"):
        raise RuntimeError(f"PPO teacher는 CUDA만 지원합니다 (got {device_str})")
    from src.envs.class_strategy import get_strategy
    from stable_baselines3.common.callbacks import BaseCallback

    strat = get_strategy(mode)
    scaled_steps = max(1000, int(round(timesteps * strat.ppo_timesteps_scale)))
    kw = _class_ppo_kwargs(mode, local_action_band_px, local_action_uncert_floor)

    class _PpoProgress(BaseCallback):
        def __init__(self, total: int, every: int):
            super().__init__()
            self.total = max(1, int(total))
            self.every = max(500, int(every))
            self._last = -1

        def _on_step(self) -> bool:
            n = int(self.num_timesteps)
            bucket = n // self.every
            if bucket == self._last:
                return True
            self._last = bucket
            pct = 100.0 * n / self.total
            ep = self.logger.name_to_value if self.logger is not None else {}
            rew = ep.get("rollout/ep_rew_mean")
            length = ep.get("rollout/ep_len_mean")
            extra = ""
            if rew is not None:
                extra += f" ep_rew={float(rew):.4f}"
            if length is not None:
                extra += f" ep_len={float(length):.1f}"
            log.info(
                "[%s] PPO progress %d/%d (%.1f%%)%s",
                mode, min(n, self.total), self.total, min(pct, 100.0), extra,
            )
            return True

        def _on_training_end(self) -> None:
            log.info("[%s] PPO progress done %d/%d (100%%)", mode, self.total, self.total)

    def _fn():
        return MaskRefinementEnv(
            images=images,
            gt_masks=gts,
            rough_masks=inits,
            uncertainty_maps=probs,
            target_dsc=1.0,
            refinement_mode=mode,
            enable_stop=True,
            device=device_str,
            **kw,
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
    log.info(
        "PPO teacher learn: mode=%s timesteps=%d (base=%d x scale=%.2f) band=%d zoom=%s "
        "(progress log every %d steps)",
        mode, scaled_steps, timesteps, strat.ppo_timesteps_scale,
        kw["local_action_band_px"], kw["use_zoom_obs"], log_every,
    )
    model.learn(
        total_timesteps=scaled_steps,
        callback=_PpoProgress(scaled_steps, log_every),
    )
    return model


def harvest_successes(
    model,
    mode,
    images,
    gts,
    inits,
    probs,
    max_n=64,
    local_action_band_px=None,
    local_action_uncert_floor=None,
):
    """Score PPO teacher with the same multi-patch path as deploy (GT OK for harvest only)."""
    from src.envs.class_strategy import get_strategy
    from src.envs.zoom_ppo_refine import (
        harvest_accept,
        predicted_boundary_band,
        refine_zoom_ppo,
    )
    from src.utils.metrics import hd95 as hd95_np
    from src.utils.device import require_cuda_device

    n = min(max_n, len(images))
    env_device = str(require_cuda_device())
    strat = get_strategy(mode)
    # Allow band overrides from CLI during harvest
    if local_action_band_px is not None or local_action_uncert_floor is not None:
        from dataclasses import replace
        strat = replace(
            strat,
            band_px=(strat.band_px if local_action_band_px is None else int(local_action_band_px)),
            uncert_floor=(
                strat.uncert_floor
                if local_action_uncert_floor is None
                else float(local_action_uncert_floor)
            ),
        )

    finals, improved = [], []
    deltas = []
    init_dscs = []
    best_dscs = []
    for i in range(n):
        init_mask = np.asarray(inits[i], dtype=np.float32)
        gt_i = np.asarray(gts[i], dtype=np.float32)
        img_i = images[i]
        prob_i = np.asarray(probs[i], dtype=np.float32)
        band = predicted_boundary_band(init_mask, strat.band_px)
        init_d = float(dice(init_mask, gt_i))
        init_b = float(
            dice(init_mask * band.astype(np.float32), gt_i * band.astype(np.float32))
        )
        init_hd = float(hd95_np(init_mask, gt_i))
        # Same multi-patch structure as deploy; GT best-of allowed only for teacher harvest.
        best = refine_zoom_ppo(
            model,
            img_i,
            gt_i,
            init_mask,
            prob_i,
            mode,
            device=env_device,
            enable_stop=True,
            strategy=strat,
            seed=i * 1009,
            gt_free=False,
        )
        best_d = float(dice(best, gt_i))
        best_b = float(
            dice(best * band.astype(np.float32), gt_i * band.astype(np.float32))
        )
        best_hd = float(hd95_np(best, gt_i))
        dlt = best_d - init_d
        deltas.append(dlt)
        init_dscs.append(init_d)
        best_dscs.append(best_d)
        if harvest_accept(init_d, best_d, init_b, best_b, init_hd, best_hd):
            improved.append(i)
            finals.append(best)
        else:
            finals.append(init_mask.copy())
    stats = {
        "n": n,
        "n_improved": len(improved),
        "init_dsc_mean": float(np.mean(init_dscs)) if init_dscs else 0.0,
        "best_dsc_mean": float(np.mean(best_dscs)) if best_dscs else 0.0,
        "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "delta_p50": float(np.percentile(deltas, 50)) if deltas else 0.0,
        "delta_p90": float(np.percentile(deltas, 90)) if deltas else 0.0,
        "delta_max": float(np.max(deltas)) if deltas else 0.0,
        "seg_bias": strat.seg_bias,
        "strategy": strat.name,
        "band_px": strat.band_px,
    }
    return np.stack(finals), improved, stats


def run_mode_training(*, mode, images, gts, roughs, probs, device, args, save_sl, save_ppo):
    """One class: alternating SL <-> PPO using already-loaded arrays."""
    img_ch = 1 if images.ndim == 3 else int(images.shape[1])
    sl_in_ch = img_ch + 2
    net = DualHeadRefiner(in_ch=sl_in_ch).to(device)
    log.info(
        "=== Stage3 mode=%s | n=%d | DualHeadRefiner in_ch=%d ===",
        mode, len(images), sl_in_ch,
    )

    bank_img = bank_r = bank_p = bank_teacher = None
    skip_ppo = args.ppo_timesteps <= 0
    ppo_path = None

    for rd in range(1, args.rounds + 1):
        log.info("=== [%s] Round %d/%d SL on real GT ===", mode, rd, args.rounds)
        net = train_sl(
            net, images, gts, roughs, probs,
            epochs=args.sl_epochs, device=device, refinement_mode=mode,
            batch_size=args.batch_size,
        )

        if bank_teacher is not None and len(bank_teacher) > 0:
            log.info(
                "=== [%s] Round %d/%d distill from PPO teacher (%d samples) ===",
                mode, rd, args.rounds, len(bank_teacher),
            )
            net = distill_sl_from_teacher(
                net,
                bank_img,
                bank_r,
                bank_p,
                bank_teacher,
                epochs=args.distill_epochs,
                device=device,
                refinement_mode=mode,
                batch_size=args.batch_size,
            )

        fixed = batch_sl_fix(net, images, roughs, probs, device, refinement_mode=mode)
        mean_d = float(np.mean([dice(fixed[i], gts[i]) for i in range(len(gts))]))
        mean_rough = float(np.mean([dice(roughs[i], gts[i]) for i in range(len(gts))]))
        log.info(
            "[%s] Round %d train-class DSC vs GT: rough=%.4f  SL-fix=%.4f  (headroom=%.4f)",
            mode, rd, mean_rough, mean_d, mean_d - mean_rough,
        )

        if skip_ppo:
            log.info("=== [%s] Round %d/%d PPO skipped (ppo_timesteps=%d) ===", mode, rd, args.rounds, args.ppo_timesteps)
            continue

        ppo_device = "cuda"
        n_ppo = min(128, len(images))
        idx = np.random.default_rng(args.seed + rd).choice(len(images), n_ppo, replace=False)
        if args.ppo_init == "sl_fixed":
            ppo_inits = fixed[idx]
            init_tag = "SL-fixed"
        else:
            ppo_inits = roughs[idx]
            init_tag = "Stage2-rough"
        init_mean = float(np.mean([dice(ppo_inits[j], gts[idx][j]) for j in range(n_ppo)]))
        from src.envs.class_strategy import get_strategy
        strat = get_strategy(mode)
        log.info(
            "=== [%s] Round %d/%d PPO teacher (base_steps=%d, device=%s, init=%s, "
            "subset_init_DSC=%.4f, strategy=%s band=%d zoom_patches=%d) ===",
            mode, rd, args.rounds, args.ppo_timesteps, ppo_device, init_tag,
            init_mean, strat.name, strat.band_px, strat.zoom_patches_infer,
        )
        ppo = train_ppo_teacher(
            mode,
            images[idx],
            gts[idx],
            ppo_inits,
            probs[idx],
            timesteps=args.ppo_timesteps,
            seed=args.seed + rd,
            device_str=ppo_device,
            local_action_band_px=args.local_action_band_px,
            local_action_uncert_floor=args.local_action_uncert_floor,
        )
        finals, improved, hstats = harvest_successes(
            ppo,
            mode,
            images[idx],
            gts[idx],
            ppo_inits,
            probs[idx],
            max_n=n_ppo,
            local_action_band_px=args.local_action_band_px,
            local_action_uncert_floor=args.local_action_uncert_floor,
        )
        log.info(
            "[%s] PPO harvest %d/%d (init=%s) | init_DSC=%.4f best=%.4f "
            "Δ mean=%.4f p50=%.4f p90=%.4f max=%.4f",
            mode, hstats["n_improved"], hstats["n"], init_tag,
            hstats["init_dsc_mean"], hstats["best_dsc_mean"],
            hstats["delta_mean"], hstats["delta_p50"], hstats["delta_p90"], hstats["delta_max"],
        )
        if improved:
            bank_img = images[idx][improved]
            bank_r = ppo_inits[improved]
            bank_p = probs[idx][improved]
            bank_teacher = finals[improved]
        else:
            log.warning(
                "[%s] PPO harvest 0/%d — distill bank empty, but keep training later rounds",
                mode, hstats["n"],
            )

        os.makedirs(os.path.dirname(save_ppo) or ".", exist_ok=True)
        ppo_path = save_ppo[:-4] if save_ppo.endswith(".zip") else save_ppo
        ppo.save(ppo_path)

    if bank_teacher is not None and len(bank_teacher) > 0:
        log.info("[%s] Final distill from last PPO teacher bank (%d)", mode, len(bank_teacher))
        net = distill_sl_from_teacher(
            net, bank_img, bank_r, bank_p, bank_teacher,
            epochs=args.distill_epochs, device=device, refinement_mode=mode,
            batch_size=args.batch_size,
        )

    os.makedirs(os.path.dirname(save_sl) or ".", exist_ok=True)
    img_ch = 1 if images.ndim == 3 else int(images.shape[1])
    torch.save(
        {"state_dict": net.state_dict(), "mode": mode, "in_ch": sl_in_ch, "img_ch": img_ch},
        save_sl,
    )
    log.info("[%s] Saved SL refiner -> %s", mode, save_sl)
    if ppo_path is not None:
        log.info("[%s] Saved last PPO teacher -> %s.zip", mode, ppo_path)
    elif args.ppo_timesteps <= 0:
        log.info("[%s] PPO teacher not saved (ppo_timesteps=0)", mode)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Alternating SL Fix <-> PPO distillation")
    parser.add_argument(
        "--refinement_mode",
        type=str,
        required=True,
        choices=["small", "medium", "large", "all"],
        help="'all' = BraTS/rough 1회 로드 후 small/medium/large 순차 학습",
    )
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
    parser.add_argument("--ppo_timesteps", type=int, default=40000)
    parser.add_argument(
        "--ppo_init",
        type=str,
        default="rough",
        choices=["rough", "sl_fixed"],
        help="PPO teacher 시작 마스크",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="SL/distill mini-batch size")
    parser.add_argument("--save_sl", type=str, default="")
    parser.add_argument("--save_ppo", type=str, default="")
    parser.add_argument(
        "--local_action_band_px",
        type=int,
        default=None,
        help="Override class strategy band px (default: small=5, medium=2, large=2)",
    )
    parser.add_argument(
        "--local_action_uncert_floor",
        type=float,
        default=None,
        help="Override class strategy uncertainty floor",
    )
    parser.add_argument(
        "--stage2_thresholds",
        type=str,
        default="0.85,0.92,0.70",
        help="Deploy-aligned Stage2 thresholds",
    )
    parser.add_argument("--cc_min_sizes", type=str, default="0,35,50")
    parser.add_argument("--stage2_erode_classes", type=str, default="1,2")
    parser.add_argument("--stage2_erode_px", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed, args.deterministic)
    sys.path.insert(0, str(ROOT / "scripts" / "train"))
    from train_agent import load_real_data, load_stage3_all_classes
    from src.data.patient_split import load_or_create_patient_split
    from src.utils.device import require_cuda_device

    unet_map = {
        "caranet": "checkpoints/caranet_best.pt",
        "unetplusplus": "checkpoints/unetplusplus_best.pt",
        "segresnet": "checkpoints/segresnet_best.pt",
    }
    mode_model = {"small": "caranet", "medium": "unetplusplus", "large": "segresnet"}

    split = load_or_create_patient_split(
        args.train_root, args.max_train_patients, args.patient_split
    )
    train_ids = split["train"]
    device = require_cuda_device()

    if args.refinement_mode == "all":
        modes = ["small", "medium", "large"]
        log.info("Stage3 ALL: BraTS/rough 1회 로드 후 클래스별 학습 (%s)", modes)
        bundle = load_stage3_all_classes(
            train_root=args.train_root,
            modality=args.modality,
            target_size=128,
            max_patients=args.max_train_patients,
            patient_ids=train_ids,
            noise_seed=args.seed,
            stage2_thresholds=args.stage2_thresholds,
            cc_min_sizes=args.cc_min_sizes,
            stage2_erode_classes=args.stage2_erode_classes,
            stage2_erode_px=args.stage2_erode_px,
        )
    else:
        modes = [args.refinement_mode]
        bundle = None

    for mode in modes:
        if bundle is not None:
            if mode not in bundle:
                log.warning("Skip mode=%s (no slices in bundle)", mode)
                continue
            images, gts, roughs, probs = bundle[mode]
        else:
            model_type = mode_model[mode]
            log.info(
                "Loading Stage2 rough masks for mode=%s "
                "(classifier filter + classifier Expert routing, no synthetic mixup)",
                mode,
            )
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
                oracle_expert=False,
                class_filter="classifier",
                stage2_thresholds=args.stage2_thresholds,
                cc_min_sizes=args.cc_min_sizes,
                stage2_erode_classes=args.stage2_erode_classes,
                stage2_erode_px=args.stage2_erode_px,
            )

        save_sl = (
            args.save_sl
            if (args.refinement_mode != "all" and args.save_sl)
            else f"checkpoints/sl_refiner_{mode}.pt"
        )
        save_ppo = (
            args.save_ppo
            if (args.refinement_mode != "all" and args.save_ppo)
            else f"checkpoints/ppo_{mode}.zip"
        )
        run_mode_training(
            mode=mode,
            images=images,
            gts=gts,
            roughs=roughs,
            probs=probs,
            device=device,
            args=args,
            save_sl=save_sl,
            save_ppo=save_ppo,
        )


if __name__ == "__main__":
    main()
