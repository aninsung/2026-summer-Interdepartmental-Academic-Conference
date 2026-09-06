"""
Mask boundary refinement RL env (Gymnasium).

Class-wise strategies (see src.envs.class_strategy):
  Small  — wide band + lesion-centric 64×64 zoom
  Medium — boundary band + uncertainty-weighted zoom crop
  Large  — thin band + zoom crop, heavier edit-cost

Action : 8-sector shrink/expand (+ optional STOP)
         SDF shift only inside predicted boundary band ∩ zoom window
Reward : fidelity (DSC / boundary DSC / HD95) + edit-cost
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.envs import torch_ops as tops
from src.envs.class_strategy import ClassRefineStrategy, get_strategy
from src.utils.metrics import apply_monotonic_dsc_gate
from src.utils.metrics import dice as _dice
from src.utils.metrics import hd95 as _hd95

__all__ = ["MaskRefinementEnv", "_dice", "_hd95", "apply_monotonic_dsc_gate"]


def _obs_image_array(img: np.ndarray) -> np.ndarray:
    """MRI → (C, H, W) float32. 단채널이면 C=1."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unexpected image ndim={arr.ndim}")


def _to_device(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


class MaskRefinementEnv(gym.Env):
    """
    뇌종양 마스크 경계 보정 RL 환경 (mask step on GPU when CUDA is available).

    Observation space: Box(0, 1, (C, H, W), float32) — returned as NumPy for SB3.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        images: np.ndarray,  # (N, H, W) or (N, C, H, W) float32
        gt_masks: np.ndarray,  # (N, H, W) float32
        rough_masks: np.ndarray,  # (N, H, W) float32
        uncertainty_maps: np.ndarray = None,  # (N, H, W) float32
        max_steps: int = 20,
        target_dsc: float = 0.88,
        step_penalty: float = 0.005,
        model_type: str = "unet",
        refinement_mode: str = "small",
        confidence_threshold: float = 0.85,
        enable_stop: bool = False,
        device: Optional[str | torch.device] = None,
        # "under"=과소분할 보정(FN↑), "over"=과대분할 억제(FP↓), None=모드별 기본
        seg_bias: Optional[str] = None,
        fn_penalty_ratio: float = 2.0,
        fp_penalty_ratio: float = 1.0,
        expand_prob_thr: float = 0.40,
        expand_intensity_ratio: float = 0.80,
        shrink_block_prob_thr: float = 0.50,
        # None → class strategy default
        local_action_band_px: Optional[int] = None,
        local_action_uncert_floor: Optional[float] = None,
        use_zoom_obs: Optional[bool] = None,
        zoom_size: Optional[int] = None,
        # 하위 호환 별칭
        asymmetric_fp_reward: Optional[bool] = None,
        expand_action_mask: Optional[bool] = None,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        assert images.shape[0] == gt_masks.shape[0] == rough_masks.shape[0]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.images = images
        self.gt_masks = gt_masks
        self.rough_masks = rough_masks

        self.refinement_mode = refinement_mode
        self.strategy: ClassRefineStrategy = get_strategy(refinement_mode)
        strat = self.strategy

        self.max_steps = int(max_steps)
        self.target_dsc = target_dsc
        self.step_penalty = float(step_penalty)
        self.model_type = model_type.lower()
        self.enable_stop = bool(enable_stop)

        self.local_action_band_px = (
            strat.band_px if local_action_band_px is None else max(0, int(local_action_band_px))
        )
        self.local_action_uncert_floor = float(
            np.clip(
                strat.uncert_floor if local_action_uncert_floor is None else local_action_uncert_floor,
                0.0,
                1.0,
            )
        )
        self.use_zoom_obs = bool(strat.use_zoom_obs if use_zoom_obs is None else use_zoom_obs)
        self.zoom_size = int(strat.zoom_size if zoom_size is None else zoom_size)
        self.dsc_scale = float(strat.dsc_scale)
        self.boundary_scale = float(strat.boundary_scale)
        self.hd95_scale = float(strat.hd95_scale)
        self.edit_cost_scale = float(strat.edit_cost_scale)

        med_large = refinement_mode in ("medium", "large")
        if seg_bias is None:
            seg_bias = strat.seg_bias
        seg_bias = str(seg_bias).lower()
        if seg_bias not in ("under", "over", "none"):
            raise ValueError("seg_bias must be 'under', 'over', or 'none'")
        self.seg_bias = seg_bias

        # 하위 호환: asymmetric_fp_reward / expand_action_mask 가 오면 over 쪽으로 해석
        if asymmetric_fp_reward is True and seg_bias == "under" and med_large:
            pass  # keep under unless explicitly over requested via seg_bias
        if expand_action_mask is False:
            self.use_action_mask = False
        else:
            self.use_action_mask = med_large and seg_bias in ("under", "over")

        self.fn_penalty_ratio = float(fn_penalty_ratio if seg_bias == "under" else 1.0)
        self.fp_penalty_ratio = float(
            fp_penalty_ratio if seg_bias == "under" else (2.0 if seg_bias == "over" else 1.0)
        )
        # under: expand 쉽게 / over: expand 엄격
        if seg_bias == "under":
            self.expand_prob_thr = float(expand_prob_thr)
            self.expand_intensity_ratio = float(expand_intensity_ratio)
        elif seg_bias == "over":
            self.expand_prob_thr = max(float(expand_prob_thr), 0.55)
            self.expand_intensity_ratio = max(float(expand_intensity_ratio), 0.85)
        else:
            self.expand_prob_thr = float(expand_prob_thr)
            self.expand_intensity_ratio = float(expand_intensity_ratio)
        self.shrink_block_prob_thr = float(shrink_block_prob_thr)
        self.asymmetric_reward = self.seg_bias in ("under", "over")

        N = images.shape[0]
        H, W = images.shape[-2:]
        self.H, self.W = H, W
        sample_img = _obs_image_array(images[0])
        self.img_ch = int(sample_img.shape[0])

        if images.ndim == 3:
            imgs_t = _to_device(images, self.device).unsqueeze(1)
        else:
            imgs_t = _to_device(images, self.device)
        self._images_t = imgs_t
        self._gt_t = _to_device(gt_masks, self.device)
        self._rough_t = _to_device(rough_masks, self.device)

        if uncertainty_maps is not None:
            self._prob_t = _to_device(uncertainty_maps, self.device)
            self.probability_maps = np.asarray(uncertainty_maps, dtype=np.float32)
        else:
            probs = []
            for i in range(N):
                probs.append(tops.gaussian_blur2d(self._rough_t[i], sigma=2.0))
            self._prob_t = torch.stack(probs, 0)
            self.probability_maps = self._prob_t.detach().cpu().numpy()

        edges = []
        for i in range(N):
            edges.append(tops.sobel_magnitude(self._images_t[i, 0]))
        self._edge_t = torch.stack(edges, 0)
        self.edge_maps = self._edge_t.detach().cpu().numpy()

        # Observation: zoom crop with extra channel (edge for small, candidate band otherwise)
        if self.use_zoom_obs:
            self._obs_ch = self.img_ch + 3
            z = self.zoom_size
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self._obs_ch, z, z), dtype=np.float32
            )
        elif self.refinement_mode == "small":
            self._obs_ch = self.img_ch + 3
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self._obs_ch, 64, 64), dtype=np.float32
            )
        else:
            self._obs_ch = self.img_ch + 2
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self._obs_ch, self.H, self.W), dtype=np.float32
            )

        if self.refinement_mode == "small":
            if self.enable_stop:
                low = np.array([-2.0] * 8 + [-2.0], dtype=np.float32)
                high = np.array([2.0] * 8 + [2.0], dtype=np.float32)
                self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(8,), dtype=np.float32)
        else:
            if self.enable_stop:
                self.action_space = spaces.MultiDiscrete([5] * 8 + [2])
            else:
                self.action_space = spaces.MultiDiscrete([5] * 8)

        self._idx = 0
        self._mask = torch.zeros(H, W, device=self.device, dtype=torch.float32)
        self._image = torch.zeros(self.img_ch, H, W, device=self.device, dtype=torch.float32)
        self._prob = torch.zeros(H, W, device=self.device, dtype=torch.float32)
        self._edge = torch.zeros(H, W, device=self.device, dtype=torch.float32)
        self._gt = torch.zeros(H, W, device=self.device, dtype=torch.float32)
        self._min_mask_limit = torch.zeros(H, W, device=self.device, dtype=torch.float32)
        self._max_mask_limit = torch.zeros(H, W, device=self.device, dtype=torch.float32)
        self._boundary_band = torch.zeros(H, W, device=self.device, dtype=torch.bool)
        self._action_band = torch.zeros(H, W, device=self.device, dtype=torch.bool)
        self._zoom_window = torch.ones(H, W, device=self.device, dtype=torch.bool)
        self._gt_dist_map = torch.zeros(H, W, device=self.device, dtype=torch.float32)
        self._ys = torch.arange(H, device=self.device, dtype=torch.float32)
        self._xs = torch.arange(W, device=self.device, dtype=torch.float32)
        self._YY, self._XX = torch.meshgrid(self._ys, self._xs, indexing="ij")
        self._step_count = 0
        self._prev_dsc = 0.0
        self._prev_boundary_dsc = 0.0
        self._prev_hd95 = 0.0
        self._prev_perimeter = 0.0
        self._initial_dsc = 0.0
        self._initial_boundary_dsc = 0.0
        self._initial_perimeter = 0.0
        self._zoom_cy = H // 2
        self._zoom_cx = W // 2
        self._lock_zoom_center = False

        # under-seg: expand 쪽을 조금 더 크게
        if self.seg_bias == "under" and med_large:
            self._shift_table = torch.tensor(
                [-0.6, -0.3, 0.0, 0.6, 1.2], device=self.device, dtype=torch.float32
            )
        else:
            self._shift_table = torch.tensor(
                [-1.0, -0.4, 0.0, 0.4, 1.0], device=self.device, dtype=torch.float32
            )

    @property
    def _current_mask(self) -> np.ndarray:
        return self._mask.detach().cpu().numpy()

    @_current_mask.setter
    def _current_mask(self, value: np.ndarray | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            self._mask = value.to(device=self.device, dtype=torch.float32)
        else:
            self._mask = _to_device(value, self.device)

    def _crop_zoom(self, obs: torch.Tensor, cy: int, cx: int) -> torch.Tensor:
        z = int(self.zoom_size)
        half = z // 2
        y1, y2 = max(0, cy - half), min(self.H, cy + half)
        x1, x2 = max(0, cx - half), min(self.W, cx + half)
        cropped = torch.zeros(self._obs_ch, z, z, device=self.device, dtype=torch.float32)
        pad_y1 = half - (cy - y1)
        pad_y2 = z - (half - (y2 - cy))
        pad_x1 = half - (cx - x1)
        pad_x2 = z - (half - (x2 - cx))
        cropped[:, pad_y1:pad_y2, pad_x1:pad_x2] = obs[:, y1:y2, x1:x2]
        return cropped

    def _update_zoom_window(self) -> None:
        z = int(self.zoom_size)
        half = z // 2
        cy, cx = int(self._zoom_cy), int(self._zoom_cx)
        self._zoom_window = (
            (self._YY >= (cy - half))
            & (self._YY < (cy + half))
            & (self._XX >= (cx - half))
            & (self._XX < (cx + half))
        )

    def _pick_zoom_center(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, int]:
        """Uncertainty-weighted point on action band; Small falls back to lesion centroid."""
        band = self._action_band
        if self.refinement_mode == "small" and not bool(band.any()):
            cy, cx = tops.largest_component_centroid(self._mask)
            return int(cy), int(cx)
        if not bool(band.any()):
            cy, cx = tops.largest_component_centroid(self._mask)
            return int(cy), int(cx)
        uncert = 1.0 - (2.0 * self._prob - 1.0).abs().clamp(0.0, 1.0)
        w = torch.where(band, uncert, torch.zeros_like(uncert))
        flat_w = w.reshape(-1)
        total = float(flat_w.sum().item())
        if total <= 1e-8:
            ys, xs = torch.where(band)
            mid = int(ys.numel()) // 2
            return int(ys[mid].item()), int(xs[mid].item())
        if rng is None:
            # deterministic: argmax uncertainty on band
            idx = int(torch.argmax(flat_w).item())
        else:
            p = (flat_w / flat_w.sum()).detach().cpu().numpy()
            idx = int(rng.choice(p.size, p=p))
        cy = idx // self.W
        cx = idx % self.W
        return cy, cx

    def _obs(self) -> np.ndarray:
        img = self._image
        cand = self._action_band.float()
        if self.use_zoom_obs:
            if self.refinement_mode == "small":
                extra = self._edge
            else:
                extra = cand
            obs = torch.cat(
                [img, self._mask[None], self._prob[None], extra[None]], dim=0
            )
            cropped = self._crop_zoom(obs, int(self._zoom_cy), int(self._zoom_cx))
            return cropped.detach().cpu().numpy().astype(np.float32)
        if self.refinement_mode == "small":
            obs = torch.cat(
                [img, self._mask[None], self._prob[None], self._edge[None]], dim=0
            )
            cy, cx = tops.largest_component_centroid(self._mask)
            return (
                self._crop_zoom(obs, int(cy), int(cx))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        obs = torch.cat([img, self._mask[None], self._prob[None]], dim=0)
        return obs.detach().cpu().numpy().astype(np.float32)

    def _sector_expand_allowed(self, sectors: torch.Tensor, sector_i: int) -> bool:
        """외곽에 종양 신호(prob/intensity)가 있을 때만 expand 허용."""
        mask_bool = self._mask > 0.5
        exterior = tops.binary_dilation(mask_bool, kernel=3, iterations=2) & (~mask_bool)
        sel = exterior & (sectors == sector_i)
        if not bool(sel.any()):
            return False
        mean_prob = float(self._prob[sel].mean().item())
        if mean_prob < self.expand_prob_thr:
            return False
        img0 = self._image[0]
        mean_I = float(img0[sel].mean().item())
        interior = mask_bool & (sectors == sector_i)
        if bool(interior.any()):
            ref_I = float(img0[interior].mean().item())
        elif bool(mask_bool.any()):
            ref_I = float(img0[mask_bool].mean().item())
        else:
            return False
        return mean_I >= ref_I * self.expand_intensity_ratio

    def _sector_shrink_allowed(self, sectors: torch.Tensor, sector_i: int) -> bool:
        """경계 rem_band 확률이 높으면(종양 가능성) shrink 금지 — 과소분할 악화 방지."""
        mask_bool = self._mask > 0.5
        rem = mask_bool & (~tops.binary_erosion(mask_bool, kernel=3, iterations=2))
        sel = rem & (sectors == sector_i)
        if not bool(sel.any()):
            return True
        mean_prob = float(self._prob[sel].mean().item())
        # 고확률 경계는 종양으로 보고 깎지 않음
        return mean_prob < self.shrink_block_prob_thr

    def _apply_action_mask(self, sector_action, sectors: torch.Tensor):
        """seg_bias에 따라 expand/shrink 마스킹. 반환: (action, n_expand_masked, n_shrink_masked)."""
        if not self.use_action_mask:
            return sector_action, 0, 0
        n_exp = 0
        n_shr = 0
        if self.refinement_mode == "small":
            shifts = np.asarray(sector_action, dtype=np.float32).copy()
            for i in range(8):
                v = float(shifts[i])
                if v > 0.1 and not self._sector_expand_allowed(sectors, i):
                    shifts[i] = 0.0
                    n_exp += 1
                elif v < -0.1 and self.seg_bias == "under" and not self._sector_shrink_allowed(sectors, i):
                    shifts[i] = 0.0
                    n_shr += 1
            return shifts, n_exp, n_shr

        acts = np.asarray(sector_action, dtype=np.int64).copy()
        for i in range(8):
            a = int(acts[i])
            if a >= 3:  # expand
                if not self._sector_expand_allowed(sectors, i):
                    acts[i] = 2
                    n_exp += 1
            elif a <= 1:  # shrink
                if self.seg_bias == "under" and not self._sector_shrink_allowed(sectors, i):
                    acts[i] = 2
                    n_shr += 1
                elif self.seg_bias == "over":
                    pass  # over: shrink 자유
        return acts, n_exp, n_shr

    def _predicted_action_band(self, mask_bool: torch.Tensor) -> torch.Tensor:
        """예측 마스크 경계 근방 밴드 (dilate ⊕ erode). GT 미사용 — 배포와 정렬."""
        n = self.local_action_band_px
        if n <= 0:
            return torch.ones_like(mask_bool, dtype=torch.bool)
        if not bool(mask_bool.any()):
            return torch.zeros_like(mask_bool, dtype=torch.bool)
        # iterations≈N px with 3×3 SE (same convention as SL boundary_band_px)
        dil = tops.binary_dilation(mask_bool, kernel=3, iterations=n)
        ero = tops.binary_erosion(mask_bool, kernel=3, iterations=n)
        return dil ^ ero

    def _local_shift_gate(self, action_band: torch.Tensor) -> torch.Tensor:
        """
        밴드 마스크 × 불확실성 soft gate.
        uncert = 1 - |2p-1| (p=0.5에서 1). floor로 확신 영역도 최소 강도 유지.
        """
        gate = action_band.float()
        floor = self.local_action_uncert_floor
        if floor >= 1.0 - 1e-6:
            return gate
        uncert = 1.0 - (2.0 * self._prob - 1.0).abs().clamp(0.0, 1.0)
        return gate * (floor + (1.0 - floor) * uncert)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._idx = seed % len(self.images)
        else:
            self._idx = int(self.np_random.integers(0, len(self.images)))

        self._image = self._images_t[self._idx].clone()
        self._mask = self._rough_t[self._idx].clone()
        self._prob = self._prob_t[self._idx].clone()
        self._edge = self._edge_t[self._idx].clone()
        self._gt = self._gt_t[self._idx].clone()
        self._step_count = 0

        rough_bool = self._mask > 0.5
        # under: 확장을 넓게 허용 / 과도한 내부 붕괴 방지
        if self.seg_bias == "under" and self.refinement_mode in ("medium", "large"):
            self._min_mask_limit = tops.binary_erosion(rough_bool, kernel=3, iterations=4).float()
            self._max_mask_limit = tops.binary_dilation(rough_bool, kernel=3, iterations=12).float()
        else:
            self._min_mask_limit = tops.binary_erosion(rough_bool, kernel=3, iterations=8).float()
            self._max_mask_limit = tops.binary_dilation(rough_bool, kernel=3, iterations=8).float()

        gt_bool = self._gt > 0.5
        dilated_gt = tops.binary_dilation(gt_bool, kernel=7, iterations=1)
        eroded_gt = tops.binary_erosion(gt_bool, kernel=7, iterations=1)
        self._boundary_band = dilated_gt ^ eroded_gt
        # Reward용 GT 밴드와 별개: 액션은 예측(rough) 경계 밴드만 사용
        self._action_band = self._predicted_action_band(self._mask > 0.5)

        opts = options or {}
        self._lock_zoom_center = False
        if "zoom_center" in opts and opts["zoom_center"] is not None:
            cy, cx = opts["zoom_center"]
            self._zoom_cy, self._zoom_cx = int(cy), int(cx)
            self._lock_zoom_center = True
        elif self.use_zoom_obs:
            if self.refinement_mode == "small":
                cy, cx = tops.largest_component_centroid(self._mask)
                self._zoom_cy, self._zoom_cx = int(cy), int(cx)
            else:
                # stochastic band sampling during training
                rng = np.random.default_rng(
                    int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
                )
                self._zoom_cy, self._zoom_cx = self._pick_zoom_center(rng)
        else:
            self._zoom_cy, self._zoom_cx = self.H // 2, self.W // 2
        self._update_zoom_window()

        self._prev_dsc = tops.dice(self._mask, self._gt)
        self._initial_dsc = self._prev_dsc
        self._prev_boundary_dsc = tops.dice(
            self._mask * self._boundary_band.float(),
            self._gt * self._boundary_band.float(),
        )
        self._initial_boundary_dsc = self._prev_boundary_dsc
        self._gt_dist_map = tops.distance_transform_edt(~gt_bool)
        self._prev_hd95 = tops.hd95(self._mask, self._gt, dist_b=self._gt_dist_map)
        
        self._prev_perimeter = tops.perimeter(self._mask)
        self._initial_perimeter = self._prev_perimeter

        return self._obs(), {
            "zoom_center": (int(self._zoom_cy), int(self._zoom_cx)),
            "band_px": self.local_action_band_px,
            "strategy": self.strategy.name,
        }

    def step(self, action):
        do_stop = False
        if self.enable_stop:
            action = np.asarray(action).reshape(-1)
            if self.refinement_mode == "small":
                do_stop = bool(float(action[8]) > 0.0)
                sector_action = action[:8]
            else:
                do_stop = bool(int(action[8]) == 1)
                sector_action = action[:8]
        else:
            sector_action = action

        if do_stop:
            cur_dsc = float(self._prev_dsc)
            if cur_dsc + 1e-6 < self._initial_dsc:
                reward = -5.0
            else:
                # Modest stop reward; bonus only for *improving* vs init (no absolute DSC jackpot).
                # Old +50 at DSC>=0.85 made "STOP immediately" dominate when init already ~0.86.
                delta = cur_dsc - float(self._initial_dsc)
                reward = 0.5
                if delta > 1e-4:
                    reward += float(min(5.0, 50.0 * delta))
            self._step_count += 1
            info = {
                "dsc": cur_dsc,
                "boundary_dsc": float(self._prev_boundary_dsc),
                "prev_dsc": cur_dsc,
                "prev_boundary_dsc": float(self._prev_boundary_dsc),
                "delta_dsc": 0.0,
                "delta_boundary": 0.0,
                "stopped": True,
            }
            return self._obs(), float(reward), True, False, info

        cy, cx = tops.largest_component_centroid(self._mask)
        angles = torch.atan2(self._YY - cy, self._XX - cx)
        sectors = ((angles + np.pi) / (2.0 * np.pi) * 8.0).long().clamp(0, 7)

        sector_action, n_expand_masked, n_shrink_masked = self._apply_action_mask(
            sector_action, sectors
        )

        sdf = tops.signed_distance(self._mask > 0.5)
        shift_map = torch.zeros_like(self._mask)
        num_non_keep = 0

        if self.refinement_mode == "small":
            shifts = torch.as_tensor(
                np.asarray(sector_action, dtype=np.float32), device=self.device
            )
            for i in range(8):
                shift_val = float(shifts[i].item())
                if abs(shift_val) > 0.1:
                    num_non_keep += 1
                shift_map[sectors == i] = shifts[i]
        else:
            acts = torch.as_tensor(
                np.asarray(sector_action, dtype=np.int64), device=self.device
            )
            for i in range(8):
                act = int(acts[i].item())
                shift_val = float(self._shift_table[act].item()) if 0 <= act < 5 else 0.0
                if act != 2:
                    num_non_keep += 1
                shift_map[sectors == i] = shift_val

        # 로컬화: 예측 경계 밴드 ∩ zoom window (+불확실성 soft gate)
        self._action_band = self._predicted_action_band(self._mask > 0.5)
        edit_region = self._action_band
        if self.use_zoom_obs:
            edit_region = edit_region & self._zoom_window
        shift_gate = self._local_shift_gate(edit_region)
        shift_map = shift_map * shift_gate
        band_frac = float(edit_region.float().mean().item())

        prev_bin = self._mask > 0.5
        new_mask = ((sdf + shift_map) >= 0.0).float()

        new_bool = new_mask > 0.5
        if int(new_bool.sum().item()) > 20:
            if self.seg_bias == "under" and self.refinement_mode in ("medium", "large"):
                # 과소분할 보정: closing으로 미세 FN/구멍 메움 후 opening
                new_bool = tops.binary_dilation(new_bool, kernel=3, iterations=1)
                new_bool = tops.binary_erosion(new_bool, kernel=3, iterations=1)
                new_bool = tops.binary_erosion(new_bool, kernel=3, iterations=1)
                new_bool = tops.binary_dilation(new_bool, kernel=3, iterations=1)
            elif self.seg_bias == "over" and self.refinement_mode in ("medium", "large"):
                # 과대분할 억제: opening만
                new_bool = tops.binary_erosion(new_bool, kernel=3, iterations=1)
                new_bool = tops.binary_dilation(new_bool, kernel=3, iterations=1)
            else:
                new_bool = tops.binary_dilation(new_bool, kernel=3, iterations=1)
                new_bool = tops.binary_erosion(new_bool, kernel=3, iterations=1)
                new_bool = tops.binary_erosion(new_bool, kernel=3, iterations=1)
                new_bool = tops.binary_dilation(new_bool, kernel=3, iterations=1)
        new_mask = new_bool.float()
        new_mask = torch.maximum(self._min_mask_limit, torch.minimum(new_mask, self._max_mask_limit))

        # SL boundary_band와 동일: 예측 경계 밴드 밖은 이전 마스크 유지 (모폴로지 스필오버 차단)
        if self.local_action_band_px > 0:
            prev_f = prev_bin.float()
            keep_region = self._action_band
            if self.use_zoom_obs:
                keep_region = keep_region & self._zoom_window
            new_mask = torch.where(keep_region, new_mask, prev_f)

        new_dsc = tops.dice(new_mask, self._gt)
        new_boundary_dsc = tops.dice(
            new_mask * self._boundary_band.float(),
            self._gt * self._boundary_band.float(),
        )
        delta_dsc = new_dsc - self._prev_dsc
        delta_boundary_dsc = new_boundary_dsc - self._prev_boundary_dsc

        prev_hd95 = self._prev_hd95
        curr_hd95 = tops.hd95(new_mask, self._gt, dist_b=self._gt_dist_map)
        delta_hd95 = prev_hd95 - curr_hd95
        self._prev_hd95 = curr_hd95
        
        curr_perimeter = tops.perimeter(new_mask)
        delta_perimeter = curr_perimeter - self._prev_perimeter

        tumor_area = max(1.0, float(self._gt.sum().item()))
        size_scale = max(0.5, min(3.0, 300.0 / tumor_area))

        # --- (1) Fidelity: 마스크 개선도 ---
        dsc_term = delta_dsc if delta_dsc >= 0 else delta_dsc * 2.0
        boundary_term = delta_boundary_dsc if delta_boundary_dsc >= 0 else delta_boundary_dsc * 2.0
        hd95_term = delta_hd95 if delta_hd95 >= 0 else delta_hd95 * 2.0
        if self.refinement_mode == "large":
            hd_w = 0.5
        elif self.refinement_mode == "medium":
            hd_w = 0.1
        else:
            hd_w = 0.2
        fidelity = (
            dsc_term * 20.0 * self.dsc_scale
            + boundary_term * 10.0 * self.boundary_scale
            + hd95_term * hd_w * self.hd95_scale
        ) * 30.0 * size_scale

        fp_inc = 0.0
        fn_inc = 0.0
        fn_dec = 0.0
        if self.asymmetric_reward:
            new_bin = new_mask > 0.5
            gt_bin = self._gt > 0.5
            prev_fp = int((prev_bin & (~gt_bin)).sum().item())
            new_fp = int((new_bin & (~gt_bin)).sum().item())
            prev_fn = int(((~prev_bin) & gt_bin).sum().item())
            new_fn = int(((~new_bin) & gt_bin).sum().item())
            fp_inc = float(max(0, new_fp - prev_fp))
            fn_inc = float(max(0, new_fn - prev_fn))
            fn_dec = float(max(0, prev_fn - new_fn))
            asym_pen = (
                self.fp_penalty_ratio * fp_inc + self.fn_penalty_ratio * fn_inc
            ) / tumor_area
            fidelity -= asym_pen * 30.0 * size_scale
            if self.seg_bias == "under" and fn_dec > 0:
                fidelity += (fn_dec / tumor_area) * 30.0 * size_scale * self.fn_penalty_ratio

        # --- (2) Edit-cost: 과도한 수정 / stopping 유도 ---
        edit_cost = num_non_keep * (self.step_penalty / 8.0)
        
        # 3단계 로드맵: Active Contour 곡률(Curvature) 페널티 결합
        # Perimeter가 늘어날수록 (경계선이 삐죽거릴수록) 페널티 부여
        if delta_perimeter > 0:
            # size_scale을 반영하여 큰 종양일수록 둘레 증가에 대한 페널티 비중을 적절히 조절
            edit_cost += (delta_perimeter / max(100.0, tumor_area**0.5)) * 0.5 * self.edit_cost_scale

        # 초기보다 나빠지면 강한 페널티 (over-correction)
        if new_dsc < self._initial_dsc:
            edit_cost += 5.0 * self.edit_cost_scale
        # 무개선 스텝: non-keep인데 DSC가 사실상 그대로면 추가 비용
        if num_non_keep > 0 and abs(delta_dsc) < 1e-4 and abs(delta_boundary_dsc) < 1e-4:
            edit_cost += 0.02 * num_non_keep * self.edit_cost_scale
        # 이미 충분히 좋으면 keep 유도
        keep_bonus = 0.0
        if num_non_keep == 0 and self._prev_dsc >= 0.85:
            keep_bonus = 0.05

        target_bonus = 0.0
        if not self.enable_stop:
            # Relative improvement bonus only (no absolute DSC jackpot).
            if new_dsc > self._initial_dsc + 1e-4:
                target_bonus = float(min(5.0, 50.0 * (new_dsc - self._initial_dsc)))

        reward = fidelity - edit_cost + keep_bonus + target_bonus

        old_prev_dsc = self._prev_dsc
        old_prev_boundary_dsc = self._prev_boundary_dsc

        self._mask = new_mask
        self._prev_dsc = new_dsc
        self._prev_boundary_dsc = new_boundary_dsc
        self._prev_perimeter = curr_perimeter
        self._step_count += 1

        terminated = bool(new_dsc >= self.target_dsc)
        truncated = bool(self._step_count >= self.max_steps)

        info = {
            "dsc": new_dsc,
            "boundary_dsc": new_boundary_dsc,
            "prev_dsc": old_prev_dsc,
            "prev_boundary_dsc": old_prev_boundary_dsc,
            "delta_dsc": delta_dsc,
            "delta_boundary": delta_boundary_dsc,
            "delta_hd95": delta_hd95,
            "stopped": False,
            "device": str(self.device),
            "fp_inc": fp_inc,
            "fn_inc": fn_inc,
            "fn_dec": fn_dec,
            "expand_masked": n_expand_masked,
            "shrink_masked": n_shrink_masked,
            "seg_bias": self.seg_bias,
            "fidelity": float(fidelity),
            "edit_cost": float(edit_cost),
            "keep_bonus": float(keep_bonus),
            "action_band_frac": band_frac,
            "local_action_band_px": self.local_action_band_px,
            "zoom_center": (int(self._zoom_cy), int(self._zoom_cx)),
            "strategy": self.strategy.name,
        }
        return self._obs(), float(reward), terminated, truncated, info
