"""
Step 2: RL 환경 설계 (Gymnasium 기반 커스텀 환경)

State  : [image(1,H,W), current_mask(1,H,W)] → 채널 concat → (2, H, W) 이미지
         CnnPolicy가 GPU에서 효율적으로 처리할 수 있는 형태
Action : 경계 픽셀을 기준으로 5-class 액션 (기존 3-class → 5-class 확장)
         0 = 강하게 수축 (erode 2px)
         1 = 약하게 수축 (erode 1px)
         2 = 유지 (keep)
         3 = 약하게 팽창 (dilate 1px)
         4 = 강하게 팽창 (dilate 2px)
Reward : Boundary-DSC 기반 보상 강화 + HD95 패널티
Episode: 최대 max_steps 스텝, DSC >= target_dsc 이면 조기 종료
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt


def _dice(a: np.ndarray, b: np.ndarray, smooth: float = 1e-5) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    return float((2.0 * (a * b).sum() + smooth) / (a.sum() + b.sum() + smooth))


def _hd95(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """HD95 계산 (두 경계 집합 간 95번째 백분위 거리). 빠른 근사 버전."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() or not b.any():
        return float(mask_a.shape[0])  # 최대 거리(이미지 높이)를 패널티로 반환

    dist_a = distance_transform_edt(~a)
    dist_b = distance_transform_edt(~b)
    d_ab = dist_b[a]
    d_ba = dist_a[b]
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """마스크 경계 픽셀 인덱스를 (2, N) 배열로 반환."""
    dilated = binary_dilation(mask, np.ones((3, 3)))
    eroded = binary_erosion(mask, np.ones((3, 3)))
    boundary = dilated.astype(bool) ^ eroded.astype(bool)
    return np.stack(np.where(boundary), axis=0)  # (2, N)


def _apply_action(mask: np.ndarray, action: int) -> np.ndarray:
    """
    5-class 액션 적용.
      0 = 강하게 수축 (erode 2px)
      1 = 약하게 수축 (erode 1px)
      2 = 유지
      3 = 약하게 팽창 (dilate 1px)
      4 = 강하게 팽창 (dilate 2px)
    """
    struct = np.ones((3, 3), dtype=bool)
    m = mask.astype(bool)
    if action == 0:
        m = binary_erosion(m, structure=struct, iterations=2)
    elif action == 1:
        m = binary_erosion(m, structure=struct, iterations=1)
    elif action == 2:
        pass  # 유지
    elif action == 3:
        m = binary_dilation(m, structure=struct, iterations=1)
    elif action == 4:
        m = binary_dilation(m, structure=struct, iterations=2)
    return m.astype(np.float32)


class MaskRefinementEnv(gym.Env):
    """
    뇌종양 마스크 경계 보정 RL 환경.

    Observation space: Box(0, 1, (3, H, W), float32)
        채널 0: 원본 MRI 이미지 (H,W)
        채널 1: 현재 마스크 (H,W)
        채널 2: Uncertainty Map (H,W)

    Action space: MultiDiscrete([5]*8)
        8개 방위 섹터에 대해 독립적으로 0=강수축, 1=약수축, 2=유지, 3=약팽창, 4=강팽창

    Reward: Boundary-DSC 향상분 중심 보상 + HD95 패널티 + 위상 최적화
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        images: np.ndarray,        # (N, H, W) float32
        gt_masks: np.ndarray,      # (N, H, W) float32
        rough_masks: np.ndarray,   # (N, H, W) float32
        uncertainty_maps: np.ndarray = None, # (N, H, W) float32
        max_steps: int = 20,
        target_dsc: float = 0.88,
        step_penalty: float = 0.005,
        model_type: str = "unet",
        refinement_mode: str = "small", # "small", "medium", "large"
    ):
        super().__init__()
        assert images.shape == gt_masks.shape == rough_masks.shape
        self.images = images
        self.gt_masks = gt_masks
        self.rough_masks = rough_masks
        if uncertainty_maps is not None:
            self.uncertainty_maps = uncertainty_maps
        else:
            self.uncertainty_maps = np.zeros_like(images)
            
        self.max_steps = max_steps
        self.target_dsc = target_dsc
        self.step_penalty = step_penalty
        self.model_type = model_type.lower()
        self.refinement_mode = refinement_mode

        N, H, W = images.shape
        self.H, self.W = H, W

        # (3, H, W) 이미지 형태로 관측 공간 정의 (small인 경우 64x64 Zoom-in)
        crop_size = 64 if self.refinement_mode == "small" else self.H
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, crop_size, crop_size),
            dtype=np.float32
        )
        
        # 8개 섹터, 각 섹터별 5개 행동 (수축/유지/팽창) - 통합 적용
        self.action_space = spaces.MultiDiscrete([5] * 8)

        self._idx = 0
        self._current_mask: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_image: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_uncertainty: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_gt: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._step_count = 0
        self._prev_dsc = 0.0
        self._prev_boundary_dsc = 0.0

    # ── 내부 유틸 ──────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        obs = np.stack([self._current_image, self._current_mask, self._current_uncertainty], axis=0).astype(np.float32)
        if self.refinement_mode == "small":
            y_indices, x_indices = np.where(self._current_mask > 0.5)
            if len(y_indices) > 0:
                cy, cx = int(y_indices.mean()), int(x_indices.mean())
            else:
                cy, cx = self.H // 2, self.W // 2
            
            half = 32
            y1, y2 = max(0, cy - half), min(self.H, cy + half)
            x1, x2 = max(0, cx - half), min(self.W, cx + half)
            
            cropped = np.zeros((3, 64, 64), dtype=np.float32)
            pad_y1 = half - (cy - y1)
            pad_y2 = 64 - (half - (y2 - cy))
            pad_x1 = half - (cx - x1)
            pad_x2 = 64 - (half - (x2 - cx))
            
            cropped[:, pad_y1:pad_y2, pad_x1:pad_x2] = obs[:, y1:y2, x1:x2]
            return cropped
        return obs

    # ── Gymnasium API ──────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._idx = seed % len(self.images)
        else:
            self._idx = self.np_random.integers(0, len(self.images))

        self._current_image = self.images[self._idx].copy()
        self._current_mask = self.rough_masks[self._idx].copy()
        self._current_uncertainty = self.uncertainty_maps[self._idx].copy()
        self._current_gt = self.gt_masks[self._idx].copy()
        self._step_count = 0
        
        # Boundary-Band calculation for reward
        struct = np.ones((7, 7), dtype=bool)
        dilated_gt = binary_dilation(self._current_gt.astype(bool), structure=struct)
        eroded_gt = binary_erosion(self._current_gt.astype(bool), structure=struct)
        self._boundary_band = dilated_gt ^ eroded_gt
        
        self._prev_dsc = _dice(self._current_mask, self._current_gt)
        self._prev_boundary_dsc = _dice(self._current_mask * self._boundary_band, self._current_gt * self._boundary_band)

        return self._obs(), {}

    def step(self, action):
        # 8개 섹터 개별 변형 (모든 모델 공통 적용)
        y_indices, x_indices = np.where(self._current_mask > 0.5)
        if len(y_indices) > 0:
            cy, cx = y_indices.mean(), x_indices.mean()
        else:
            cy, cx = self.H / 2.0, self.W / 2.0

        ys = np.arange(self.H)
        xs = np.arange(self.W)
        Y, X = np.meshgrid(ys, xs, indexing='ij')
        angles = np.arctan2(Y - cy, X - cx)  # [-pi, pi]
        sectors = ((angles + np.pi) / (2.0 * np.pi) * 8.0).astype(int)
        sectors = np.clip(sectors, 0, 7)

        struct = np.ones((3, 3), dtype=bool)
        m = self._current_mask.astype(bool)
        eroded_2 = binary_erosion(m, structure=struct, iterations=2)
        eroded_1 = binary_erosion(m, structure=struct, iterations=1)
        dilated_1 = binary_dilation(m, structure=struct, iterations=1)
        dilated_2 = binary_dilation(m, structure=struct, iterations=2)

        new_mask = self._current_mask.copy()
        action_penalty = 0.0
        
        for i in range(8):
            act = int(action[i])
            sector_pixels = (sectors == i)
            
            # Medium/Large: 보수적 조정 (큰 변형인 0, 4 선택 시 페널티 부여 및 작은 변형으로 강제)
            if self.refinement_mode in ["medium", "large"] and act in [0, 4]:
                action_penalty += 1.0
                act = 1 if act == 0 else 3
                
            if act == 0:
                new_mask[sector_pixels] = eroded_2[sector_pixels]
            elif act == 1:
                new_mask[sector_pixels] = eroded_1[sector_pixels]
            elif act == 2:
                pass  # 유지
            elif act == 3:
                new_mask[sector_pixels] = dilated_1[sector_pixels]
            elif act == 4:
                new_mask[sector_pixels] = dilated_2[sector_pixels]

        # ── 위상 보존 (Topological Constraints) ──
        # 구멍 메우기 및 불연속 섬 제거를 위한 Closing 후 Opening
        new_mask_bool = new_mask.astype(bool)
        new_mask_bool = binary_dilation(new_mask_bool, structure=struct, iterations=1)
        new_mask_bool = binary_erosion(new_mask_bool, structure=struct, iterations=1) # Closing
        new_mask_bool = binary_erosion(new_mask_bool, structure=struct, iterations=1)
        new_mask_bool = binary_dilation(new_mask_bool, structure=struct, iterations=1) # Opening
        new_mask = new_mask_bool.astype(np.float32)

        num_non_keep = sum([1 for a in action if int(a) != 2])
        step_cost = num_non_keep * (self.step_penalty / 8.0)
        is_keep_and_good = (num_non_keep == 0 and self._prev_dsc >= 0.85)

        new_dsc = _dice(new_mask, self._current_gt)
        new_boundary_dsc = _dice(new_mask * self._boundary_band, self._current_gt * self._boundary_band)

        # ── 보상: Shape Class (refinement_mode) 기반 고정 보상 ────────────────
        delta_dsc = new_dsc - self._prev_dsc
        delta_boundary_dsc = new_boundary_dsc - self._prev_boundary_dsc
        
        if self.refinement_mode == "large":
            # Large: HD95-Focused (보수적 조정 & 외곽 이상치 제거 집중)
            reward = delta_dsc * 10.0 + delta_boundary_dsc * 10.0
            hd95_val = _hd95(new_mask, self._current_gt)
            reward -= hd95_val * 0.05
            reward -= action_penalty * 0.5
            
        elif self.refinement_mode == "medium":
            # Medium: Conservative (경계 DSC 위주 및 액션 페널티 적용)
            reward = (delta_dsc * 0.3 + delta_boundary_dsc * 0.7) * 30.0 
            if delta_boundary_dsc < 0 or delta_dsc < 0:
                reward *= 2.0
            if self._step_count % 5 == 0 or num_non_keep == 0:
                reward -= _hd95(new_mask, self._current_gt) * 0.02
            reward -= action_penalty * 0.5
            
        else:
            # Small: Aggressive (전통적인 전역 및 경계 DSC 향상 위주)
            reward = (delta_dsc * 0.3 + delta_boundary_dsc * 0.7) * 30.0 
            if delta_boundary_dsc < 0 or delta_dsc < 0:
                reward *= 2.0
            if self._step_count % 5 == 0 or num_non_keep == 0:
                reward -= _hd95(new_mask, self._current_gt) * 0.01

        # 유지 보너스: 이미 좋은 마스크에서 유지 시 보상
        if is_keep_and_good:
            reward += 0.05
        else:
            reward -= step_cost

        self._current_mask = new_mask
        self._prev_dsc = new_dsc
        self._prev_boundary_dsc = new_boundary_dsc
        self._step_count += 1

        terminated = bool(new_dsc >= self.target_dsc)
        truncated = bool(self._step_count >= self.max_steps)

        info = {
            "dsc": new_dsc,
            "boundary_dsc": new_boundary_dsc,
            "prev_dsc": self._prev_dsc,
            "delta_dsc": delta_dsc,
            "delta_boundary": delta_boundary_dsc,
        }
        return self._obs(), reward, terminated, truncated, info
