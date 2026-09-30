"""윤곽 진화 학습·평가용 데이터 준비.

Stage 2 Expert가 만든 확률맵과 마스크를 한 번 계산해 캐시하고, 연결요소마다
초기 윤곽과 목표 윤곽을 짝지어 학습 샘플로 만든다. 학습 스크립트와 평가
스크립트가 같은 전처리를 공유하도록 여기에 모아둔다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.contour import (
    align_target,
    component_contours,
    match_gt_component,
    nearest_boundary_targets,
)
from src.utils.metrics import gt_size_class
from src.utils.refinement_inputs import stage2_mask

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = (0.80, 0.80, 0.50)
DEFAULT_CC_SIZES = (0, 15, 25)


@dataclass
class SliceEntry:
    """한 슬라이스의 Stage 2 결과. 영상과 마스크는 메모리 절약을 위해 축소 저장한다."""

    image: np.ndarray       # (C, H, W) float16 — 중심 슬라이스 모달리티
    probability: np.ndarray  # (H, W) float16 — TTA 확률맵
    stage2: np.ndarray      # (H, W) uint8 — Stage 2 이진 마스크
    gt: np.ndarray          # (H, W) uint8
    route: int
    patient_id: str

    def model_input(self) -> np.ndarray:
        """윤곽 네트워크 입력: 영상 모달리티 + Stage 2 마스크 + 확률맵."""
        return np.concatenate([
            self.image.astype(np.float32),
            self.stage2.astype(np.float32)[None],
            self.probability.astype(np.float32)[None],
        ], axis=0)


def _tta_probability(pipeline, img_t: torch.Tensor, base_prob: torch.Tensor,
                     class_pred: torch.Tensor) -> np.ndarray:
    """평가 스크립트와 같은 좌우·상하 반전 TTA 평균."""
    with torch.no_grad():
        hf = torch.flip(pipeline(torch.flip(img_t, dims=[3]), true_class_preds=class_pred)[0], dims=[3])
        vf = torch.flip(pipeline(torch.flip(img_t, dims=[2]), true_class_preds=class_pred)[0], dims=[2])
        return ((base_prob + hf + vf) / 3.0).squeeze().cpu().numpy()


@torch.no_grad()
def build_stage2_entries(
    dataset,
    pipeline,
    device,
    classes: Sequence[int] = (1, 2),
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    cc_sizes: Sequence[int] = DEFAULT_CC_SIZES,
    use_tta: bool = True,
    log_every: int = 2000,
) -> List[SliceEntry]:
    """GT 면적으로 라우팅해 Stage 2 확률맵·마스크를 계산한다.

    라우팅을 GT 면적으로 고정하는 이유: 이 실험의 질문은 "윤곽 표현이 통하는가"이며,
    Stage 1 분류기(val acc 0.86)의 오라우팅 잡음을 섞으면 답이 흐려진다.
    """
    images, gt_masks, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    pids = getattr(dataset, "_sample_pids", [""] * len(images))
    allowed = set(int(c) for c in classes)

    entries: List[SliceEntry] = []
    for i in range(len(images)):
        gt_np = gt_masks[i]
        route = gt_size_class(gt_np)
        if route not in allowed:
            continue

        img_np = images_25d[i]
        img_t = (torch.from_numpy(img_np).unsqueeze(0) if img_np.ndim == 3
                 else torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)).to(device)
        route_t = torch.tensor([route], dtype=torch.long, device=device)
        base_prob, class_pred = pipeline(img_t, true_class_preds=route_t)

        prob_np = (_tta_probability(pipeline, img_t, base_prob, class_pred) if use_tta
                   else base_prob.squeeze().cpu().numpy())
        mask_np = stage2_mask(prob_np, route, list(thresholds), list(cc_sizes))

        entries.append(SliceEntry(
            image=images[i].astype(np.float16) if images[i].ndim == 3
            else images[i][None].astype(np.float16),
            probability=prob_np.astype(np.float16),
            stage2=(mask_np > 0.5).astype(np.uint8),
            gt=(gt_np > 0.5).astype(np.uint8),
            route=route,
            patient_id=pids[i] if i < len(pids) else "",
        ))
        if log_every and (i + 1) % log_every == 0:
            log.info("Stage 2 계산 %d/%d (수집 %d)", i + 1, len(images), len(entries))

    log.info("Stage 2 완료: 슬라이스 %d장 (클래스 %s)", len(entries), sorted(allowed))
    return entries


@dataclass
class ContourSample:
    slice_index: int
    init_points: np.ndarray    # (N, 2)
    target_points: np.ndarray  # (N, 2)
    gt_boundary: np.ndarray    # (N, 2) GT 외곽을 균일 재샘플한 점. 반대 방향 표면 거리용.


def build_contour_samples(
    entries: Sequence[SliceEntry],
    n_points: int = 128,
    min_area: int = 10,
    target_mode: str = "nearest",
) -> List[ContourSample]:
    """Stage 2 컴포넌트마다 초기 윤곽과 목표 윤곽을 만든다.

    target_mode
      nearest : 각 초기 점을 GT 윤곽의 최근접점으로 보낸다 (기본)
      arclength : GT 윤곽을 같은 점 수로 재샘플하고 순환 정렬한다
    GT와 전혀 겹치지 않는 컴포넌트(완전 오검출)는 학습 목표가 없으므로 버린다.
    """
    if target_mode not in {"nearest", "arclength"}:
        raise ValueError("target_mode must be nearest or arclength")
    from src.utils.contour import outer_contour

    samples: List[ContourSample] = []
    skipped_no_overlap = 0
    for idx, entry in enumerate(entries):
        contours, components, _ = component_contours(entry.stage2, n_points, min_area)
        for pts, comp in zip(contours, components):
            gt_comp = match_gt_component(comp, entry.gt)
            if gt_comp is None:
                skipped_no_overlap += 1
                continue
            if target_mode == "nearest":
                target = nearest_boundary_targets(pts, gt_comp)
            else:
                resampled = outer_contour(gt_comp, n_points)
                target = align_target(pts, resampled) if resampled is not None else None
            boundary = outer_contour(gt_comp, n_points)
            if target is None or boundary is None:
                continue
            samples.append(ContourSample(idx, pts.astype(np.float32),
                                         np.asarray(target, dtype=np.float32),
                                         boundary.astype(np.float32)))
    log.info("윤곽 샘플 %d개 (%s 목표, GT 미겹침으로 제외 %d개)",
             len(samples), target_mode, skipped_no_overlap)
    return samples


class ContourComponentDataset(Dataset):
    """컴포넌트 단위 학습 데이터셋."""

    def __init__(self, entries: Sequence[SliceEntry], samples: Sequence[ContourSample],
                 augment: bool = False):
        self.entries = entries
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        entry = self.entries[sample.slice_index]
        image = entry.model_input()
        init = sample.init_points.copy()
        target = sample.target_points.copy()
        boundary = sample.gt_boundary.copy()

        if self.augment:
            h, w = image.shape[-2:]
            if np.random.rand() < 0.5:  # 좌우 반전
                image = image[:, :, ::-1].copy()
                init[:, 1] = (w - 1) - init[:, 1]
                target[:, 1] = (w - 1) - target[:, 1]
                boundary[:, 1] = (w - 1) - boundary[:, 1]
            if np.random.rand() < 0.5:  # 상하 반전
                image = image[:, ::-1, :].copy()
                init[:, 0] = (h - 1) - init[:, 0]
                target[:, 0] = (h - 1) - target[:, 0]
                boundary[:, 0] = (h - 1) - boundary[:, 0]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "init_points": torch.from_numpy(init),
            "target_points": torch.from_numpy(target),
            "gt_boundary": torch.from_numpy(boundary),
        }
