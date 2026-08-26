"""
Step 2: RL 환경 설계 (Gymnasium 기반 커스텀 환경)

State  : [image, current_mask, soft_probability, (optional edge)]
         - Medium/Large: (3, H, W) [Image, Current Mask, Soft Prob Map]
         - Small (Zoom-in): (4, 64, 64) [Zoomed Image, Mask, Prob, Edge Map]
Action : 8방위 섹터별 마스크 수축/팽창 조절
         - Medium/Large: MultiDiscrete([5]*8)
           0=강수축(-1.0px), 1=약수축(-0.4px), 2=유지, 3=약팽창(+0.4px), 4=강팽창(+1.0px)
         - Small: Box(-2.0, 2.0, shape=(8,)) 연속 픽셀 조절
Reward : Boundary-DSC 기반 보상 강화 + HD95(px 단위) 패널티 + 위상 최적화
Episode: 최대 max_steps 스텝, DSC >= target_dsc 이면 조기 종료
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt, gaussian_filter, sobel, label

from src.utils.metrics import dice as _dice, hd95 as _hd95, apply_monotonic_dsc_gate

__all__ = ["MaskRefinementEnv", "_dice", "_hd95", "apply_monotonic_dsc_gate"]


def _obs_image_slice(img: np.ndarray) -> np.ndarray:
    """다채널 MRI는 t1ce(첫 채널)만 관측에 사용."""
    if img.ndim == 3:
        return img[0].astype(np.float32)
    return img.astype(np.float32)


def _edge_map_from_image(img_2d: np.ndarray) -> np.ndarray:
    edge_x = sobel(img_2d, axis=0)
    edge_y = sobel(img_2d, axis=1)
    edge = np.sqrt(edge_x**2 + edge_y**2)
    e_min, e_max = edge.min(), edge.max()
    if e_max > e_min:
        edge = (edge - e_min) / (e_max - e_min)
    return edge.astype(np.float32)


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """마스크 경계 픽셀 인덱스를 (2, N) 배열로 반환."""
    dilated = binary_dilation(mask, np.ones((3, 3)))
    eroded = binary_erosion(mask, np.ones((3, 3)))
    boundary = dilated.astype(bool) ^ eroded.astype(bool)
    return np.stack(np.where(boundary), axis=0)  # (2, N)


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
        uncertainty_maps: np.ndarray = None, # (N, H, W) float32 (Soft Probability Maps)
        max_steps: int = 20,
        target_dsc: float = 0.88,
        step_penalty: float = 0.005,
        model_type: str = "unet",
        refinement_mode: str = "small", # "small", "medium", "large"
        confidence_threshold: float = 0.85, # RL-Refiner 진입을 결정하는 기준값
        edge_maps: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        assert images.shape[0] == gt_masks.shape[0] == rough_masks.shape[0]
        self.images = images
        self.gt_masks = gt_masks
        self.rough_masks = rough_masks
        
        # Soft Probability Maps 설정 (메모리 절약을 위해 copy 제거)
        if uncertainty_maps is not None:
            self.probability_maps = uncertainty_maps
        else:
            self.probability_maps = np.zeros_like(rough_masks)
            for i in range(len(rough_masks)):
                self.probability_maps[i] = gaussian_filter(rough_masks[i].astype(float), sigma=2.0)
            
        self.max_steps = max_steps
        self.target_dsc = target_dsc
        self.step_penalty = step_penalty
        self.model_type = model_type.lower()
        self.refinement_mode = refinement_mode

        N = images.shape[0]
        H, W = images.shape[-2:]
        self.H, self.W = H, W

        # 이미지 그래디언트 맵 (Sobel Edge Map) 외부에서 전달받아 공유하거나 없으면 계산
        if edge_maps is not None:
            self.edge_maps = edge_maps
        else:
            self.edge_maps = np.zeros((N, H, W), dtype=np.float32)
            for i in range(N):
                self.edge_maps[i] = _edge_map_from_image(_obs_image_slice(images[i]))

        # 관측 공간 정의 (small은 4채널 64x64 Zoom-in, 그 외는 기존 체크포인트와 호환되는 3채널 128x128)
        if self.refinement_mode == "small":
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(4, 64, 64),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(3, self.H, self.W),
                dtype=np.float32
            )
        
        # 8개 섹터, 각 섹터별 행동 정의 (small은 연속 행동 공간 Box, 그 외는 이산 MultiDiscrete)
        if self.refinement_mode == "small":
            self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(8,), dtype=np.float32)
        else:
            self.action_space = spaces.MultiDiscrete([5] * 8)

        self._idx = 0
        self._current_mask: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_image: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_prob: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_edge: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._current_gt: np.ndarray = np.zeros((H, W), dtype=np.float32)
        self._step_count = 0
        self._prev_dsc = 0.0
        self._prev_boundary_dsc = 0.0

    # ── 내부 유틸 ──────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        if self.refinement_mode == "small":
            obs = np.stack([self._current_image, self._current_mask, self._current_prob, self._current_edge], axis=0).astype(np.float32)
            
            # Find connected components to avoid center-of-mass falling in empty space between disconnected components
            lbl, num_features = label(self._current_mask > 0.5)
            if num_features > 0:
                component_sizes = [np.sum(lbl == k) for k in range(1, num_features + 1)]
                largest_k = np.argmax(component_sizes) + 1
                y_indices, x_indices = np.where(lbl == largest_k)
                cy, cx = int(y_indices.mean()), int(x_indices.mean())
            else:
                cy, cx = self.H // 2, self.W // 2
            
            half = 32
            y1, y2 = max(0, cy - half), min(self.H, cy + half)
            x1, x2 = max(0, cx - half), min(self.W, cx + half)
            
            cropped = np.zeros((4, 64, 64), dtype=np.float32)
            pad_y1 = half - (cy - y1)
            pad_y2 = 64 - (half - (y2 - cy))
            pad_x1 = half - (cx - x1)
            pad_x2 = 64 - (half - (x2 - cx))
            
            cropped[:, pad_y1:pad_y2, pad_x1:pad_x2] = obs[:, y1:y2, x1:x2]
            return cropped
        else:
            # 실제 Uncertainty(Probability) Map을 3번째 채널로 전달 (버그 수정: 기존 zeros → 실제 prob map)
            obs = np.stack([self._current_image, self._current_mask, self._current_prob], axis=0).astype(np.float32)
            return obs

    # ── Gymnasium API ──────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._idx = seed % len(self.images)
        else:
            self._idx = self.np_random.integers(0, len(self.images))

        img = self.images[self._idx]
        self._current_image = _obs_image_slice(img)
        self._current_mask = self.rough_masks[self._idx].copy()
        self._current_prob = self.probability_maps[self._idx].copy()
        self._current_edge = self.edge_maps[self._idx].copy()
        self._current_gt = self.gt_masks[self._idx].copy()
        self._step_count = 0
        
        # 초기 Rough 마스크 백업 및 허용 경계 제약용 마스크 사전 계산 (+-8 픽셀 범위로 대폭 완화)
        self._initial_rough_mask = self._current_mask.copy()
        struct_limit = np.ones((3, 3), dtype=bool)
        self._min_mask_limit = binary_erosion(self._initial_rough_mask.astype(bool), structure=struct_limit, iterations=8).astype(np.float32)
        self._max_mask_limit = binary_dilation(self._initial_rough_mask.astype(bool), structure=struct_limit, iterations=8).astype(np.float32)

        # Boundary-Band calculation for reward
        struct = np.ones((7, 7), dtype=bool)
        dilated_gt = binary_dilation(self._current_gt.astype(bool), structure=struct)
        eroded_gt = binary_erosion(self._current_gt.astype(bool), structure=struct)
        self._boundary_band = dilated_gt ^ eroded_gt
        
        self._prev_dsc = _dice(self._current_mask, self._current_gt)
        self._initial_dsc = self._prev_dsc
        self._prev_boundary_dsc = _dice(self._current_mask * self._boundary_band, self._current_gt * self._boundary_band)
        self._gt_dist_map = distance_transform_edt(~self._current_gt.astype(bool))
        self._prev_hd95 = _hd95(self._current_mask, self._current_gt, dist_b=self._gt_dist_map)

        return self._obs(), {}

    def step(self, action):
        # 8개 섹터 개별 변형 (SDF 기반 연속 미세 변형 적용)
        # 분리된 종양이 있을 때 질량 중심이 빈 공간에 놓이는 현상을 방지하기 위해 가장 큰 연결 요소의 중심 사용
        lbl, num_features = label(self._current_mask > 0.5)
        if num_features > 0:
            component_sizes = [np.sum(lbl == k) for k in range(1, num_features + 1)]
            largest_k = np.argmax(component_sizes) + 1
            y_indices, x_indices = np.where(lbl == largest_k)
            cy, cx = y_indices.mean(), x_indices.mean()
        else:
            cy, cx = self.H / 2.0, self.W / 2.0

        ys = np.arange(self.H)
        xs = np.arange(self.W)
        Y, X = np.meshgrid(ys, xs, indexing='ij')
        angles = np.arctan2(Y - cy, X - cx)  # [-pi, pi]
        sectors = ((angles + np.pi) / (2.0 * np.pi) * 8.0).astype(int)
        sectors = np.clip(sectors, 0, 7)

        # ── SDF (Signed Distance Field) 계산 ──
        m_bool = self._current_mask.astype(bool)
        if np.any(m_bool) and not np.all(m_bool):
            # sdf: 내부 양수, 외부 음수
            sdf = distance_transform_edt(m_bool) - distance_transform_edt(~m_bool)
        elif np.all(m_bool):
            sdf = np.ones_like(self._current_mask) * 999.0
        else:
            sdf = np.ones_like(self._current_mask) * -999.0

        # 행동 매핑: 각 섹터별 연속 픽셀 shift 매핑
        shift_map = np.zeros_like(self._current_mask)
        num_non_keep = 0
        mapped_actions = []

        for i in range(8):
            sector_pixels = (sectors == i)
            if self.refinement_mode == "small":
                # 연속 공간: action[i] 가 직접 픽셀 shift 거리로 사용됨 (예: [-2, 2] 범위)
                shift_val = float(action[i])
                mapped_actions.append(shift_val)
                # Keep 여부 판정 (실수값이므로 절대값 0.1 이하는 Keep으로 간주)
                if abs(shift_val) > 0.1:
                    num_non_keep += 1
            else:
                # 이산 공간: 기존 checkpoints 호환성 유지하면서 미세 SDF shift로 변환
                act = int(action[i])
                mapped_actions.append(act)
                # 0=강수축(-1.0px), 1=약수축(-0.4px), 2=유지(0.0px), 3=약팽창(0.4px), 4=강팽창(1.0px)
                # 고정밀 보정을 위해 기존 morphology(1px/3px)보다 폭을 줄여 오버슈트 방지
                mapping = {0: -1.0, 1: -0.4, 2: 0.0, 3: 0.4, 4: 1.0}
                shift_val = mapping.get(act, 0.0)
                if act != 2:
                    num_non_keep += 1
            shift_map[sector_pixels] = shift_val

        # SDF + shift_map >= 0 이면 새로운 마스크 영역으로 결정
        new_mask = (sdf + shift_map) >= 0.0
        new_mask = new_mask.astype(np.float32)

        # ── 위상 보존 (Topological Constraints) ──
        new_mask_bool = new_mask.astype(bool)
        if np.sum(new_mask_bool) > 20:
            struct = np.ones((3, 3), dtype=bool)
            new_mask_bool = binary_dilation(new_mask_bool, structure=struct, iterations=1)
            new_mask_bool = binary_erosion(new_mask_bool, structure=struct, iterations=1) # Closing
            new_mask_bool = binary_erosion(new_mask_bool, structure=struct, iterations=1)
            new_mask_bool = binary_dilation(new_mask_bool, structure=struct, iterations=1) # Opening
        new_mask = new_mask_bool.astype(np.float32)

        # ── 수색 영역 제한 (Boundary Band Constraint) ──
        new_mask = np.maximum(self._min_mask_limit, np.minimum(new_mask, self._max_mask_limit))

        step_cost = num_non_keep * (self.step_penalty / 8.0)
        
        new_dsc = _dice(new_mask, self._current_gt)
        new_boundary_dsc = _dice(new_mask * self._boundary_band, self._current_gt * self._boundary_band)

        delta_dsc = new_dsc - self._prev_dsc
        delta_boundary_dsc = new_boundary_dsc - self._prev_boundary_dsc
        
        # HD95 델타 계산
        prev_hd95 = self._prev_hd95
        curr_hd95 = _hd95(new_mask, self._current_gt, dist_b=self._gt_dist_map)
        delta_hd95 = prev_hd95 - curr_hd95
        self._prev_hd95 = curr_hd95
        
        # 종양 크기 가중치 계산
        tumor_area = max(1.0, float(np.sum(self._current_gt)))
        size_scale = max(0.5, min(3.0, 300.0 / tumor_area))
        
        # 감점 페널티 비대칭 적용
        dsc_weight = delta_dsc if delta_dsc >= 0 else delta_dsc * 2.0
        boundary_weight = delta_boundary_dsc if delta_boundary_dsc >= 0 else delta_boundary_dsc * 2.0
        hd95_weight = delta_hd95 if delta_hd95 >= 0 else delta_hd95 * 2.0

        # ── 타겟 달성 보너스 (Target Bonus) ──
        target_bonus = 0.0
        if self.refinement_mode == "small" and new_dsc >= 0.85:
            target_bonus = 50.0
        elif self.refinement_mode in ["medium", "large"] and new_dsc >= 0.95:
            target_bonus = 50.0

        # ── 고정밀 성능 유지 패널티 (Strict Monotonic Penalty) ──
        # 현재 DSC가 에피소드 초기 예측 성능(self._initial_dsc)보다 하락하는 행동을 하면 -5.0의 벌점을 부과
        if new_dsc < self._initial_dsc:
            drop_penalty = -5.0
        else:
            drop_penalty = 0.0

        if self.refinement_mode == "large":
            reward = (dsc_weight * 20.0 + boundary_weight * 10.0 + hd95_weight * 0.5) * 30.0 * size_scale + target_bonus
        elif self.refinement_mode == "medium":
            reward = (dsc_weight * 20.0 + boundary_weight * 10.0 + hd95_weight * 0.1) * 30.0 * size_scale + target_bonus
        else:
            reward = (dsc_weight * 20.0 + boundary_weight * 10.0 + hd95_weight * 0.2) * 30.0 * size_scale + target_bonus

        reward += drop_penalty - step_cost

        # 유지 보너스
        is_keep_and_good = (num_non_keep == 0 and self._prev_dsc >= 0.85)
        if is_keep_and_good:
            reward += 0.05

        old_prev_dsc = self._prev_dsc
        old_prev_boundary_dsc = self._prev_boundary_dsc

        self._current_mask = new_mask
        self._prev_dsc = new_dsc
        self._prev_boundary_dsc = new_boundary_dsc
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
        }
        return self._obs(), reward, terminated, truncated, info
