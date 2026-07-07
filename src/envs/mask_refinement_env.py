"""
Step 2: RL 환경 설계 (Gymnasium 기반 커스텀 환경)

State  : [image(1,H,W), current_mask(1,H,W)] → 채널 concat → (2,H,W) 플랫 벡터
Action : 경계 픽셀을 기준으로 3-class 액션
         0 = 수축 (erode boundary)
         1 = 유지 (keep)
         2 = 팽창 (dilate boundary)
Reward : ΔDSC × 10  (보정 후 DSC - 보정 전 DSC)
Episode: 최대 max_steps 스텝, DSC ≥ target_dsc 이면 조기 종료
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import binary_erosion, binary_dilation


def _dice(a: np.ndarray, b: np.ndarray, smooth: float = 1e-5) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    return float((2.0 * (a * b).sum() + smooth) / (a.sum() + b.sum() + smooth))


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """마스크 경계 픽셀 인덱스를 (2, N) 배열로 반환."""
    dilated = binary_dilation(mask, np.ones((3, 3)))
    eroded = binary_erosion(mask, np.ones((3, 3)))
    boundary = dilated.astype(bool) ^ eroded.astype(bool)
    return np.stack(np.where(boundary), axis=0)  # (2, N)


def _apply_action(mask: np.ndarray, action: int) -> np.ndarray:
    """경계 픽셀에 액션 적용 (0=수축, 1=유지, 2=팽창)."""
    struct = np.ones((3, 3), dtype=bool)
    if action == 0:
        return binary_erosion(mask.astype(bool), structure=struct).astype(np.float32)
    elif action == 2:
        return binary_dilation(mask.astype(bool), structure=struct).astype(np.float32)
    else:  # action == 1
        return mask.copy()


class MaskRefinementEnv(gym.Env):
    """
    뇌종양 마스크 경계 보정 RL 환경.

    Observation space: Box(0, 1, (2*H*W,), float32)
        채널 0: 원본 MRI 이미지 (H,W)
        채널 1: 현재 마스크 (H,W)

    Action space: Discrete(3) — 0:수축, 1:유지, 2:팽창
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        images: np.ndarray,        # (N, H, W) float32
        gt_masks: np.ndarray,      # (N, H, W) float32
        rough_masks: np.ndarray,   # (N, H, W) float32
        max_steps: int = 20,
        target_dsc: float = 0.90,
    ):
        super().__init__()
        assert images.shape == gt_masks.shape == rough_masks.shape
        self.images = images
        self.gt_masks = gt_masks
        self.rough_masks = rough_masks
        self.max_steps = max_steps
        self.target_dsc = target_dsc

        N, H, W = images.shape
        self.H, self.W = H, W
        self.obs_dim = 2 * H * W

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0:erode, 1:keep, 2:dilate

        self._idx = 0
        self._current_mask: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_image: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_gt: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._step_count = 0

    # ── 내부 유틸 ──────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        obs = np.stack([self._current_image, self._current_mask], axis=0)  # (2,H,W)
        return obs.ravel().astype(np.float32)

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

        # Reward: DSC 향상량 (×10 스케일링)
        reward = (new_dsc - prev_dsc) * 10.0
        self._current_mask = new_mask
        self._step_count += 1

        terminated = bool(new_dsc >= self.target_dsc)
        truncated = bool(self._step_count >= self.max_steps)

        info = {
            "dsc": new_dsc,
            "prev_dsc": prev_dsc,
            "delta_dsc": new_dsc - prev_dsc,
        }
        return self._obs(), reward, terminated, truncated, info
