"""
Step 2: RL 환경 설계 (Gymnasium 기반 커스텀 환경)

State  : [image, current_mask, soft_probability, (optional edge)]
         - Medium/Large: (3, H, W) [Image, Current Mask, Soft Prob Map]
         - Small (Zoom-in): (4, 64, 64) [Zoomed Image, Mask, Prob, Edge Map]
Action : 8방위 섹터별 마스크 수축/팽창 조절
         - Medium/Large: MultiDiscrete([5]*8) (0=강수축 2px, 1=약수축 1px, 2=유지, 3=약팽창 1px, 4=강팽창 2px)
         - Small: Box(-2.0, 2.0, shape=(8,)) 연속적 픽셀 조절
Reward : Boundary-DSC 기반 보상 강화 + HD95(px 단위) 패널티 + 위상 최적화
Episode: 최대 max_steps 스텝, DSC >= target_dsc 이면 조기 종료
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt, gaussian_filter, sobel, label


def _dice(a: np.ndarray, b: np.ndarray, smooth: float = 1e-5) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    return float((2.0 * (a * b).sum() + smooth) / (a.sum() + b.sum() + smooth))


def _hd95(mask_a: np.ndarray, mask_b: np.ndarray, dist_b: Optional[np.ndarray] = None) -> float:
    """HD95 계산 (두 경계 집합 간 95번째 백분위 거리). 빠른 근사 버전."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not np.any(a) or not np.any(b):
        return min(30.0, float(mask_a.shape[0]))

    dist_a = distance_transform_edt(~a)
    if dist_b is None:
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
        uncertainty_maps: np.ndarray = None, # (N, H, W) float32 (Soft Probability Maps)
        max_steps: int = 20,
        target_dsc: float = 0.88,
        step_penalty: float = 0.005,
        model_type: str = "unet",
        refinement_mode: str = "small", # "small", "medium", "large"
    ):
        super().__init__()
        assert images.shape[0] == gt_masks.shape[0] == rough_masks.shape[0]
        self.images = images
        self.gt_masks = gt_masks
        self.rough_masks = rough_masks
        
        # Soft Probability Maps 설정 (없으면 rough_masks에 가우시안 블러 적용하여 시뮬레이션)
        if uncertainty_maps is not None:
            self.probability_maps = uncertainty_maps.copy()
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

        # 이미지 그래디언트 맵 (Sobel Edge Map) 미리 계산
        self.edge_maps = np.zeros((N, H, W), dtype=np.float32)
        for i in range(N):
            img = images[i]
            if img.ndim == 3:
                img = np.mean(img, axis=0)
            edge_x = sobel(img, axis=0)
            edge_y = sobel(img, axis=1)
            edge = np.sqrt(edge_x**2 + edge_y**2)
            e_min, e_max = edge.min(), edge.max()
            if e_max > e_min:
                edge = (edge - e_min) / (e_max - e_min)
            self.edge_maps[i] = edge

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
            # 기존 3채널 체크포인트 가중치와 호환되는 3개 채널 반환 (MRI, 마스크, 0-Uncertainty)
            obs = np.stack([self._current_image, self._current_mask, np.zeros_like(self._current_image)], axis=0).astype(np.float32)
            return obs

    # ── Gymnasium API ──────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._idx = seed % len(self.images)
        else:
            self._idx = self.np_random.integers(0, len(self.images))

        img = self.images[self._idx]
        self._current_image = np.mean(img, axis=0).astype(np.float32) if img.ndim == 3 else img.copy()
        self._current_mask = self.rough_masks[self._idx].copy()
        self._current_prob = self.probability_maps[self._idx].copy()
        self._current_edge = self.edge_maps[self._idx].copy()
        self._current_gt = self.gt_masks[self._idx].copy()
        self._step_count = 0
        
        # 초기 Rough 마스크 백업 및 허용 경계 제약용 마스크 사전 계산 (+-2 픽셀 범위)
        self._initial_rough_mask = self._current_mask.copy()
        struct_limit = np.ones((3, 3), dtype=bool)
        self._min_mask_limit = binary_erosion(self._initial_rough_mask.astype(bool), structure=struct_limit, iterations=2).astype(np.float32)
        self._max_mask_limit = binary_dilation(self._initial_rough_mask.astype(bool), structure=struct_limit, iterations=2).astype(np.float32)

        # Boundary-Band calculation for reward
        struct = np.ones((7, 7), dtype=bool)
        dilated_gt = binary_dilation(self._current_gt.astype(bool), structure=struct)
        eroded_gt = binary_erosion(self._current_gt.astype(bool), structure=struct)
        self._boundary_band = dilated_gt ^ eroded_gt
        
        self._prev_dsc = _dice(self._current_mask, self._current_gt)
        self._prev_boundary_dsc = _dice(self._current_mask * self._boundary_band, self._current_gt * self._boundary_band)
        self._gt_dist_map = distance_transform_edt(~self._current_gt.astype(bool))
        self._prev_hd95 = _hd95(self._current_mask, self._current_gt, dist_b=self._gt_dist_map)

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
        
        # 행동 매핑: 연속 공간인 경우 [-2, 2] 실수값을 반올림 후 [0, 4] 정수로 변환
        mapped_actions = []
        for i in range(8):
            if self.refinement_mode == "small":
                act_val = action[i]
                act = int(np.clip(np.round(act_val), -2, 2) + 2)
            else:
                act = int(action[i])
            mapped_actions.append(act)

        for i in range(8):
            act = mapped_actions[i]
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
        # 구멍 메우기 및 불연속 섬 제거를 위한 Closing 후 Opening (단, 20px 미만 미세 종양은 지워지지 않도록 보호)
        new_mask_bool = new_mask.astype(bool)
        if np.sum(new_mask_bool) > 20:
            new_mask_bool = binary_dilation(new_mask_bool, structure=struct, iterations=1)
            new_mask_bool = binary_erosion(new_mask_bool, structure=struct, iterations=1) # Closing
            new_mask_bool = binary_erosion(new_mask_bool, structure=struct, iterations=1)
            new_mask_bool = binary_dilation(new_mask_bool, structure=struct, iterations=1) # Opening
        new_mask = new_mask_bool.astype(np.float32)

        # ── 수색 영역 제한 (Boundary Band Constraint) ──
        # 초기 Rough 마스크 대비 +-2 픽셀 범위를 넘지 못하도록 클리핑
        new_mask = np.maximum(self._min_mask_limit, np.minimum(new_mask, self._max_mask_limit))

        num_non_keep = sum([1 for a in mapped_actions if a != 2])
        step_cost = num_non_keep * (self.step_penalty / 8.0)
        is_keep_and_good = (num_non_keep == 0 and self._prev_dsc >= 0.85)

        new_dsc = _dice(new_mask, self._current_gt)
        new_boundary_dsc = _dice(new_mask * self._boundary_band, self._current_gt * self._boundary_band)

        # ── 보상: 크기 비례 보상 (Size-normalized Reward) 및 Asymmetric Penalty ──
        delta_dsc = new_dsc - self._prev_dsc
        delta_boundary_dsc = new_boundary_dsc - self._prev_boundary_dsc
        
        # HD95 델타 계산 (캐싱된 gt_dist_map 활용하여 연산 속도 2배 향상)
        prev_hd95 = self._prev_hd95
        curr_hd95 = _hd95(new_mask, self._current_gt, dist_b=self._gt_dist_map)
        delta_hd95 = prev_hd95 - curr_hd95
        self._prev_hd95 = curr_hd95
        
        # 종양 크기 가중치 계산 (면적이 작을수록 보상 증폭, 최대 3배)
        tumor_area = max(1.0, float(np.sum(self._current_gt)))
        size_scale = max(0.5, min(3.0, 300.0 / tumor_area))
        
        # 감점 페널티 비대칭 적용 (하락 시 감점 2배)
        dsc_weight = delta_dsc if delta_dsc >= 0 else delta_dsc * 2.0
        boundary_weight = delta_boundary_dsc if delta_boundary_dsc >= 0 else delta_boundary_dsc * 2.0
        hd95_weight = delta_hd95 if delta_hd95 >= 0 else delta_hd95 * 2.0

        if self.refinement_mode == "large":
            # Large: HD95-Focused (보수적 조정 & 외곽 이상치 제거 집중)
            reward = (dsc_weight * 10.0 + boundary_weight * 10.0 + hd95_weight * 0.5) * size_scale
            reward -= curr_hd95 * 0.05
            reward -= action_penalty * 0.5
            
        elif self.refinement_mode == "medium":
            # Medium: Conservative (경계 DSC 위주 및 액션 페널티 적용)
            reward = (dsc_weight * 0.3 + boundary_weight * 0.7 + hd95_weight * 0.1) * 30.0 * size_scale
            if self._step_count % 5 == 0 or num_non_keep == 0:
                reward -= curr_hd95 * 0.02
            reward -= action_penalty * 0.5
            
        else:
            # Small: Aggressive (전통적인 전역 및 경계 DSC 향상 + HD95 직접 보상)
            reward = (dsc_weight * 0.3 + boundary_weight * 0.7 + hd95_weight * 0.2) * 30.0 * size_scale
            if self._step_count % 5 == 0 or num_non_keep == 0:
                reward -= curr_hd95 * 0.01

        # 유지 보너스: 이미 좋은 마스크에서 유지 시 보상
        if is_keep_and_good:
            reward += 0.05
        else:
            reward -= step_cost

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
