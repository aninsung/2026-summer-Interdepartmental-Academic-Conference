"""마스크와 닫힌 윤곽(polygon) 사이를 오가는 공용 유틸리티.

윤곽 기반 분할(Deep Snake / MARL-MambaContour 계열)을 이 파이프라인에서 실험하려면
마스크를 점 시퀀스로 바꾸고 되돌리는 변환이 필요하다. 상한 측정과 학습·평가가
같은 변환을 쓰도록 여기에 모아둔다.

좌표 규약: 모든 점은 (row, col) 실수 좌표이며 픽셀 중심 기준이다.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def resample_closed_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
    """닫힌 윤곽을 호 길이 기준으로 n_points개 점으로 균일 재샘플."""
    pts = np.asarray(contour, dtype=np.float64)
    # find_contours는 닫힌 윤곽의 첫 점을 마지막에 반복한다. 중복을 없앤 뒤 명시적으로 닫는다.
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) == 0:
        raise ValueError("빈 윤곽은 재샘플할 수 없습니다.")
    if len(pts) < 3:
        return np.repeat(pts, n_points, axis=0)[:n_points]

    loop = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total <= 0:
        return np.repeat(pts[:1], n_points, axis=0)

    target = np.linspace(0.0, total, n_points, endpoint=False)
    rows = np.interp(target, arc, loop[:, 0])
    cols = np.interp(target, arc, loop[:, 1])
    return np.stack([rows, cols], axis=1)


def outer_contour(component: np.ndarray, n_points: int) -> Optional[np.ndarray]:
    """단일 컴포넌트의 외곽 윤곽을 n_points개 점으로 반환. 내부 구멍 윤곽은 버린다."""
    from skimage.measure import find_contours

    # 마스크가 배열 경계에 닿으면 윤곽이 열리므로 1픽셀 패딩 후 좌표를 되돌린다.
    padded = np.pad(np.asarray(component, dtype=np.float64), 1, mode="constant")
    contours = find_contours(padded, 0.5)
    if not contours:
        return None
    longest = max(contours, key=len)  # 가장 긴 윤곽이 외곽선
    return resample_closed_contour(longest, n_points) - 1.0


def rasterize(points_list: List[Optional[np.ndarray]], shape: Tuple[int, int]) -> np.ndarray:
    """다각형 목록을 이진 마스크로 채운다."""
    from skimage.draw import polygon as draw_polygon

    out = np.zeros(shape, dtype=np.float32)
    for pts in points_list:
        if pts is None or len(pts) < 3:
            continue
        rr, cc = draw_polygon(pts[:, 0], pts[:, 1], shape=shape)
        out[rr, cc] = 1.0
    return out


def component_contours(
    mask: np.ndarray,
    n_points: int,
    min_area: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """마스크의 연결요소마다 윤곽을 뽑는다.

    Returns
    -------
    contours : min_area 이상 컴포넌트의 윤곽 목록
    components : 위 윤곽에 대응하는 컴포넌트 이진 마스크 목록
    passthrough : min_area 미만이라 윤곽으로 다루지 않은 잔여 픽셀 마스크
    """
    from scipy.ndimage import label

    labeled, n = label(np.asarray(mask) > 0.5)
    contours: List[np.ndarray] = []
    components: List[np.ndarray] = []
    passthrough = np.zeros(mask.shape, dtype=np.float32)
    for k in range(1, n + 1):
        comp = labeled == k
        if int(comp.sum()) < min_area:
            passthrough[comp] = 1.0
            continue
        pts = outer_contour(comp.astype(np.float32), n_points)
        if pts is None:
            passthrough[comp] = 1.0
            continue
        contours.append(pts)
        components.append(comp.astype(np.float32))
    return contours, components, passthrough


def align_target(init_pts: np.ndarray, target_pts: np.ndarray) -> np.ndarray:
    """목표 윤곽을 순환 이동·방향 반전으로 초기 윤곽에 가장 잘 맞도록 정렬.

    윤곽점은 순서만 다르면 같은 도형이므로, 지도학습 목표를 만들 때
    점 대응을 먼저 맞춰야 회귀가 잘 정의된다.
    """
    init = np.asarray(init_pts, dtype=np.float64)
    n = len(init)
    best, best_cost = None, np.inf
    for candidate in (np.asarray(target_pts, dtype=np.float64),
                      np.asarray(target_pts, dtype=np.float64)[::-1]):
        # (n, n, 2): 각 순환 이동량별 후보 점 배치
        idx = (np.arange(n)[None, :] + np.arange(n)[:, None]) % n
        rolled = candidate[idx]
        cost = ((rolled - init[None]) ** 2).sum(axis=(1, 2))
        k = int(np.argmin(cost))
        if cost[k] < best_cost:
            best_cost = float(cost[k])
            best = rolled[k]
    return best


def nearest_boundary_targets(init_pts: np.ndarray, component: np.ndarray) -> Optional[np.ndarray]:
    """초기 윤곽점마다 GT 윤곽상의 가장 가까운 점을 목표로 반환한다.

    호 길이 재샘플로 점 순서를 맞추면 같은 경계 위를 옆으로 미끄러지는 목표가
    생겨 래스터화 결과가 거의 안 바뀐다. 최근접 경계점은 각 점이 경계를 향해
    수직으로 움직이도록 해 마스크 개선과 직접 연결된다.
    """
    from scipy.spatial import cKDTree
    from skimage.measure import find_contours

    padded = np.pad(np.asarray(component, dtype=np.float64), 1, mode="constant")
    contours = find_contours(padded, 0.5)
    if not contours:
        return None
    dense = np.vstack(contours) - 1.0
    _, nearest = cKDTree(dense).query(np.asarray(init_pts, dtype=np.float64), k=1)
    return dense[nearest].astype(np.float64)


def match_gt_component(component: np.ndarray, gt: np.ndarray) -> Optional[np.ndarray]:
    """초기 컴포넌트와 가장 많이 겹치는 GT 연결요소를 반환. 겹침이 없으면 None."""
    from scipy.ndimage import label

    labeled, n = label(np.asarray(gt) > 0.5)
    if n == 0:
        return None
    comp = np.asarray(component) > 0.5
    overlaps = np.array([int((comp & (labeled == k)).sum()) for k in range(1, n + 1)])
    if overlaps.max() == 0:
        return None
    return (labeled == int(np.argmax(overlaps)) + 1).astype(np.float32)
