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
Reward : ΔDSC × 10 - HD95 패널티 × 0.05 (보정 후 DSC 향상 + 경계 거리 패널티)
Episode: 최대 max_steps 스텝, DSC ≥ target_dsc 이면 조기 종료
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

    Observation space: Box(0, 1, (2, H, W), float32)
        채널 0: 원본 MRI 이미지 (H,W)
        채널 1: 현재 마스크 (H,W)

    Action space: Discrete(5)
        0=강수축, 1=약수축, 2=유지, 3=약팽창, 4=강팽창

    Reward: ΔDSC × 10 - step_penalty (비작동 행동 시 감산)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        images: np.ndarray,        # (N, H, W) float32
        gt_masks: np.ndarray,      # (N, H, W) float32
        rough_masks: np.ndarray,   # (N, H, W) float32
        max_steps: int = 20,
        target_dsc: float = 0.95,
        step_penalty: float = 0.01,
    ):
        super().__init__()
        assert images.shape == gt_masks.shape == rough_masks.shape
        self.images = images
        self.gt_masks = gt_masks
        self.rough_masks = rough_masks
        self.max_steps = max_steps
        self.target_dsc = target_dsc
        self.step_penalty = step_penalty

        N, H, W = images.shape
        self.H, self.W = H, W

        # CnnPolicy 호환: (C, H, W) 이미지 형태로 관측 공간 정의
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2, H, W), dtype=np.float32
        )
        # 5-class 행동 공간 (기존 3-class → 5-class 확장)
        self.action_space = spaces.Discrete(5)

        self._idx = 0
        self._current_mask: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_image: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_gt: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._step_count = 0
        self._prev_dsc = 0.0

    # ── 내부 유틸 ──────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        # (2, H, W) 이미지 형태 — CnnPolicy GPU 최적화
        return np.stack([self._current_image, self._current_mask], axis=0).astype(np.float32)

    # ── Gymnasium API ──────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._idx = seed % len(self.images)
        else:
            self._idx = self.np_random.integers(0, len(self.images))

        self._current_image = self.images[self._idx].copy()
        self._current_mask = self.rough_masks[self._idx].copy()
        self._current_gt = self.gt_masks[self._idx].copy()
        self._step_count = 0
        self._prev_dsc = _dice(self._current_mask, self._current_gt)

        return self._obs(), {}

    def step(self, action: int):
        prev_dsc = _dice(self._current_mask, self._current_gt)

        # 액션 적용
        new_mask = _apply_action(self._current_mask, int(action))
        new_dsc = _dice(new_mask, self._current_gt)

        # ── 보상: DSC 향상량 + 비작동 패널티 ────────────────────
        delta_dsc = new_dsc - prev_dsc
        reward = delta_dsc * 10.0
        if int(action) != 2:
            reward -= self.step_penalty

        self._current_mask = new_mask
        self._step_count += 1

        terminated = bool(new_dsc >= self.target_dsc)
        truncated = bool(self._step_count >= self.max_steps)

        info = {
            "dsc": new_dsc,
            "prev_dsc": prev_dsc,
            "delta_dsc": delta_dsc,
        }
        return self._obs(), reward, terminated, truncated, info
