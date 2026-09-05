"""Multi-patch zoom-in PPO refinement for class-wise boundary correction."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion

from src.envs.class_strategy import ClassRefineStrategy, get_strategy
from src.envs.mask_refinement_env import MaskRefinementEnv
from src.utils.metrics import dice, hd95


def predicted_boundary_band(mask: np.ndarray, band_px: int) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float32) > 0.5
    if band_px <= 0 or not m.any():
        return np.zeros_like(m, dtype=bool)
    dil = binary_dilation(m, iterations=int(band_px))
    ero = binary_erosion(m, iterations=int(band_px))
    return np.logical_xor(dil, ero)


def band_mean_uncertainty(prob: np.ndarray, band: np.ndarray) -> float:
    if not np.any(band):
        return 0.0
    p = np.asarray(prob, dtype=np.float32)
    u = 1.0 - np.abs(2.0 * p - 1.0)
    return float(u[band].mean())


def sample_band_centers(
    mask: np.ndarray,
    prob: np.ndarray,
    band_px: int,
    k: int,
    rng: np.random.Generator,
    *,
    for_small: bool = False,
) -> List[Tuple[int, int]]:
    """Sample up to k centers on the predicted boundary band (uncertainty-weighted)."""
    m = np.asarray(mask, dtype=np.float32)
    if for_small or k <= 1:
        ys, xs = np.where(m > 0.5)
        if len(ys) == 0:
            return [(m.shape[0] // 2, m.shape[1] // 2)]
        return [(int(ys.mean()), int(xs.mean()))]

    band = predicted_boundary_band(m, band_px)
    ys, xs = np.where(band)
    if len(ys) == 0:
        ys, xs = np.where(m > 0.5)
        if len(ys) == 0:
            return [(m.shape[0] // 2, m.shape[1] // 2)]
        return [(int(ys.mean()), int(xs.mean()))]

    p = np.asarray(prob, dtype=np.float32)
    u = 1.0 - np.abs(2.0 * p[ys, xs] - 1.0)
    u = np.maximum(u, 1e-6)
    u = u / u.sum()
    n = min(int(k), len(ys))
    idx = rng.choice(len(ys), size=n, replace=False, p=u)
    return [(int(ys[i]), int(xs[i])) for i in idx]


def refine_zoom_ppo(
    agent,
    image: np.ndarray,
    gt: np.ndarray,
    init_mask: np.ndarray,
    prob_map: np.ndarray,
    refinement_mode: str,
    *,
    device: str = "cuda",
    enable_stop: bool = True,
    strategy: Optional[ClassRefineStrategy] = None,
    seed: int = 0,
    n_patches: Optional[int] = None,
    steps_per_patch: Optional[int] = None,
    gt_free: bool = True,
) -> np.ndarray:
    """
    Class-wise zoom-patch PPO:
      Small  — 1 lesion-centric crop
      Medium — many uncertainty-weighted boundary crops
      Large  — fewer crops; skip if band uncertainty below gate
    Edits are merged only inside band ∩ patch (env already enforces this).

    gt_free=True (deploy default):
      Use the **last** mask of each patch episode — no GT cherry-picking.
    gt_free=False (offline harvest / upper bound only):
      Keep best-of-N by GT DSC/boundary (not for deploy metrics).
    """
    strat = strategy or get_strategy(refinement_mode)
    if not strat.deploy_ppo:
        return np.asarray(init_mask, dtype=np.float32)

    init = np.asarray(init_mask, dtype=np.float32)
    prob = np.asarray(prob_map, dtype=np.float32)
    # Env still needs a GT tensor for internal reward channels; deploy must not
    # use it for mask selection (gt_free=True).
    gt_np = np.asarray(gt, dtype=np.float32) if gt is not None else np.zeros_like(init)
    band = predicted_boundary_band(init, strat.band_px)
    mean_u = band_mean_uncertainty(prob, band)
    if strat.ppo_gate_uncert > 0.0 and mean_u < strat.ppo_gate_uncert:
        return init.copy()

    k = int(strat.zoom_patches_infer if n_patches is None else n_patches)
    n_steps = int(strat.steps_per_patch if steps_per_patch is None else steps_per_patch)
    rng = np.random.default_rng(seed)
    centers = sample_band_centers(
        init,
        prob,
        strat.band_px,
        k,
        rng,
        for_small=(refinement_mode == "small"),
    )

    img = np.asarray(image, dtype=np.float32)
    out = init.copy()
    env = MaskRefinementEnv(
        images=img[None] if img.ndim == 2 else img[None],
        gt_masks=gt_np[None],
        rough_masks=out[None],
        uncertainty_maps=prob[None],
        max_steps=n_steps,
        target_dsc=1.0,
        step_penalty=strat.step_penalty,
        refinement_mode=refinement_mode,
        enable_stop=enable_stop,
        device=device,
        seg_bias=strat.seg_bias,
        local_action_band_px=strat.band_px,
        local_action_uncert_floor=strat.uncert_floor,
        use_zoom_obs=True,
        zoom_size=strat.zoom_size,
    )

    for pi, (cy, cx) in enumerate(centers):
        env.rough_masks = out[None].astype(np.float32)
        env._rough_t = torch.as_tensor(out[None], device=env.device, dtype=torch.float32)
        obs, _ = env.reset(seed=seed + pi, options={"zoom_center": (cy, cx)})
        cur = env._current_mask.copy()
        best = cur.copy()
        if not gt_free:
            best_d = float(dice(best, gt_np))
            best_b = float(
                dice(best * band.astype(np.float32), gt_np * band.astype(np.float32))
            )
        for _ in range(n_steps):
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            cur = env._current_mask.copy()
            if not gt_free:
                cur_d = float(info.get("dsc", dice(cur, gt_np)))
                cur_b = float(info.get("boundary_dsc", best_b))
                if (cur_d > best_d + 1e-4) or (cur_b > best_b + 1e-4):
                    best_d, best_b, best = cur_d, cur_b, cur
            else:
                # deploy: last mask wins (STOP just ends early with current mask)
                best = cur
            if term or trunc:
                break
        changed = (best > 0.5) != (out > 0.5)
        apply = changed & band
        out = out.copy()
        out[apply] = best[apply]

    return out.astype(np.float32)


def harvest_accept(
    init_dsc: float,
    best_dsc: float,
    init_bd: float,
    best_bd: float,
    init_hd: float,
    best_hd: float,
) -> bool:
    """Success if DSC, boundary DSC, or HD95 improves."""
    if best_dsc > init_dsc + 1e-4:
        return True
    if best_bd > init_bd + 1e-4:
        return True
    if best_hd + 1e-4 < init_hd:  # lower is better
        return True
    return False
