"""Class-wise boundary refine strategies (Small / Medium / Large)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ClassRefineStrategy:
    name: str
    # predicted-mask boundary band width (px)
    band_px: int
    # soft uncertainty floor inside band (1=ignore uncertainty)
    uncert_floor: float
    # PPO observes a zoom crop instead of full 128×128
    use_zoom_obs: bool
    zoom_size: int
    max_steps: int
    step_penalty: float
    # multiply base --ppo_timesteps
    ppo_timesteps_scale: float
    # inference: how many boundary patches to refine with PPO
    zoom_patches_infer: int
    steps_per_patch: int
    # reward mixing
    dsc_scale: float
    boundary_scale: float
    hd95_scale: float
    edit_cost_scale: float
    seg_bias: str
    # deploy: SL then optional zoom-PPO
    deploy_sl_first: bool
    deploy_ppo: bool
    # Large: run PPO only if mean band uncertainty >= gate (0=always)
    ppo_gate_uncert: float


STRATEGIES: Dict[str, ClassRefineStrategy] = {
    "small": ClassRefineStrategy(
        name="small",
        band_px=5,
        uncert_floor=0.20,
        use_zoom_obs=True,
        zoom_size=64,
        max_steps=20,
        step_penalty=0.001,
        ppo_timesteps_scale=1.5,
        zoom_patches_infer=1,
        steps_per_patch=15,
        dsc_scale=0.8,
        boundary_scale=1.3,
        hd95_scale=1.0,
        edit_cost_scale=1.0,
        seg_bias="none",
        deploy_sl_first=True,
        deploy_ppo=True,
        ppo_gate_uncert=0.0,
    ),
    "medium": ClassRefineStrategy(
        name="medium",
        # narrower band → smaller area jumps vs 1.2–1.4 gate
        band_px=2,
        uncert_floor=0.15,
        use_zoom_obs=True,
        zoom_size=64,
        max_steps=15,
        step_penalty=0.001,
        ppo_timesteps_scale=2.0,
        zoom_patches_infer=8,
        steps_per_patch=8,
        dsc_scale=0.55,
        boundary_scale=1.9,
        hd95_scale=1.0,
        edit_cost_scale=1.0,
        # under-seg expand bias (Stage2 thr/erode creates FN)
        seg_bias="under",
        deploy_sl_first=True,
        deploy_ppo=True,
        ppo_gate_uncert=0.0,
    ),
    "large": ClassRefineStrategy(
        name="large",
        band_px=2,
        uncert_floor=0.35,
        use_zoom_obs=True,
        zoom_size=64,
        max_steps=10,
        step_penalty=0.002,
        ppo_timesteps_scale=0.75,
        zoom_patches_infer=6,
        steps_per_patch=6,
        dsc_scale=0.7,
        boundary_scale=1.5,
        hd95_scale=1.2,
        edit_cost_scale=2.0,
        seg_bias="under",
        deploy_sl_first=True,
        deploy_ppo=True,
        ppo_gate_uncert=0.0,
    ),
}


def get_strategy(mode: str) -> ClassRefineStrategy:
    key = str(mode).lower().strip()
    if key not in STRATEGIES:
        raise ValueError(f"Unknown refinement mode {mode!r}; expected small|medium|large")
    return STRATEGIES[key]
