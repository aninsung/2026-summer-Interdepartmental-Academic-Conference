"""단일 백본용 바닐라 PPO 환경.

행동은 마스크 전체에 대한 하나다. 0은 1픽셀 수축, 1은 유지, 2는 1픽셀 팽창이다.
섹터, 경계 띠, FLAIR 가드, 위상 닫힘은 없다.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import binary_dilation, binary_erosion

from src.utils.metrics import dice as _dice


class VanillaRefineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        images: np.ndarray,
        gt_masks: np.ndarray,
        rough_masks: np.ndarray,
        probability_maps: np.ndarray,
        max_steps: int = 15,
    ):
        super().__init__()
        if images.ndim == 3:
            images = images[:, None]
        self.images = images.astype(np.float32, copy=False)
        self.gt_masks = gt_masks.astype(np.float32, copy=False)
        self.rough_masks = rough_masks.astype(np.float32, copy=False)
        self.probability_maps = probability_maps.astype(np.float32, copy=False)
        self.max_steps = max_steps
        self.n = len(self.gt_masks)
        channels = self.images.shape[1]
        height, width = self.images.shape[-2:]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(channels + 2, height, width), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self._index = 0
        self._step_count = 0
        self._mask = None
        self._prev_dsc = 0.0

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            self.images[self._index],
            self._mask[None],
            self.probability_maps[self._index][None],
        ], axis=0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._index = int(self.np_random.integers(0, self.n))
        self._mask = self.rough_masks[self._index].copy()
        self._step_count = 0
        self._prev_dsc = float(_dice(self._mask, self.gt_masks[self._index]))
        return self._obs(), {"dsc": self._prev_dsc}

    def step(self, action):
        action = int(action)
        if action == 0:
            self._mask = binary_erosion(self._mask > 0.5, iterations=1).astype(np.float32)
        elif action == 2:
            self._mask = binary_dilation(self._mask > 0.5, iterations=1).astype(np.float32)
        dsc = float(_dice(self._mask, self.gt_masks[self._index]))
        reward = dsc - self._prev_dsc
        self._prev_dsc = dsc
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        return self._obs(), float(reward), False, truncated, {"dsc": dsc}
