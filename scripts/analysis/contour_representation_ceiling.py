"""윤곽 표현의 상한 측정 (학습 없음).

MARL-MambaContour류 윤곽 기반 분할을 이 데이터에 쓸 수 있는지 판단하려면,
먼저 "GT를 N점 닫힌 윤곽으로 표현하는 것만으로 얼마나 잃는가"를 알아야 한다.
이 스크립트는 GT 마스크만으로 그 상한을 재고, 손실을 세 요인으로 분해한다.

  poly   : 가장 큰 컴포넌트를 N점 다각형으로 근사할 때의 손실 (해당 컴포넌트 기준)
  single : 위 다각형 하나만 최종 마스크로 쓸 때의 손실 (컴포넌트 누락 포함)
  multi  : 컴포넌트마다 다각형 하나씩 쓸 때의 손실 (누락 없음, 구멍 메움만 남음)

single과 multi의 차이가 "분리된 구조" 비용이고, multi와 poly의 차이가 구멍 비용이다.

사용 예:
  python scripts/analysis/contour_representation_ceiling.py --max_patients 120 --n_points 128
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.contour import outer_contour, rasterize
from src.utils.metrics import dice, gt_size_class, hd95

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = {0: "small", 1: "medium", 2: "large"}


def analyze_slice(gt: np.ndarray, n_points: int) -> dict:
    from scipy.ndimage import binary_fill_holes, label

    shape = gt.shape
    binary = gt > 0.5
    labeled, n_comp = label(binary)
    if n_comp == 0:
        return {}

    areas = np.array([int((labeled == k).sum()) for k in range(1, n_comp + 1)])
    largest_k = int(np.argmax(areas)) + 1
    largest = (labeled == largest_k).astype(np.float32)

    filled = binary_fill_holes(binary)
    hole_pixels = int(filled.sum() - binary.sum())

    per_component = [outer_contour((labeled == k).astype(np.float32), n_points)
                     for k in range(1, n_comp + 1)]
    largest_poly = per_component[largest_k - 1]

    mask_single = rasterize([largest_poly], shape)
    mask_multi = rasterize(per_component, shape)

    return {
        "size_class": gt_size_class(gt),
        "area": float(binary.sum()),
        "n_components": int(n_comp),
        "n_tiny_components": int((areas < 10).sum()),
        "largest_area_frac": float(areas.max() / max(1, areas.sum())),
        "hole_pixels": hole_pixels,
        "has_hole": bool(hole_pixels > 0),
        # 다각형 근사 자체의 손실 (가장 큰 컴포넌트만 놓고 비교)
        "dsc_poly": dice(mask_single, largest),
        # 최종 마스크로서의 손실 (전체 GT 기준)
        "dsc_single": dice(mask_single, gt),
        "dsc_multi": dice(mask_multi, gt),
        "hd95_single": hd95(mask_single, gt),
        "hd95_multi": hd95(mask_multi, gt),
    }


def summarize(rows: list[dict]) -> dict:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["size_class"]].append(r)

    def block(items: list[dict]) -> dict:
        arr = lambda key: np.array([it[key] for it in items], dtype=np.float64)
        return {
            "n_slices": len(items),
            "mean_area": float(arr("area").mean()),
            "dsc_poly": float(arr("dsc_poly").mean()),
            "dsc_single": float(arr("dsc_single").mean()),
            "dsc_multi": float(arr("dsc_multi").mean()),
            "hd95_single": float(np.nanmean(arr("hd95_single"))),
            "hd95_multi": float(np.nanmean(arr("hd95_multi"))),
            "mean_components": float(arr("n_components").mean()),
            "frac_multi_component": float((arr("n_components") > 1).mean()),
            "mean_tiny_components": float(arr("n_tiny_components").mean()),
            "frac_with_hole": float(arr("has_hole").mean()),
            "mean_hole_pixels": float(arr("hole_pixels").mean()),
            "mean_largest_area_frac": float(arr("largest_area_frac").mean()),
        }

    summary = {"overall": block(rows)}
    for cls in sorted(by_class):
        summary[CLASS_NAMES[cls]] = block(by_class[cls])
    return summary


CLASS_ORDER = ["small", "medium", "large", "overall"]


def print_topology(summary: dict) -> None:
    keys = [k for k in CLASS_ORDER if k in summary]
    print("\n=== GT 위상 통계 (윤곽 표현의 난이도) ===\n")
    header = (f"{'클래스':<9}{'슬라이스':>9}{'평균면적':>10}{'컴포넌트':>10}"
              f"{'다중컴포넌트':>13}{'10px미만':>10}{'구멍있음':>10}{'구멍px':>9}")
    print(header)
    print("-" * 82)
    for k in keys:
        b = summary[k]
        print(f"{k:<9}{b['n_slices']:>9}{b['mean_area']:>10.0f}{b['mean_components']:>10.2f}"
              f"{b['frac_multi_component']:>13.3f}{b['mean_tiny_components']:>10.2f}"
              f"{b['frac_with_hole']:>10.3f}{b['mean_hole_pixels']:>9.1f}")


def print_report(summaries: dict[int, dict]) -> None:
    print("\n=== 윤곽 표현 상한 (GT만 사용, 학습 없음) ===")
    print("DSC poly   : 가장 큰 컴포넌트를 다각형으로 근사할 때만의 손실")
    print("DSC multi  : 컴포넌트마다 윤곽 하나 (구멍 메움 포함, 누락 없음)")
    print("DSC single : 슬라이스당 윤곽 하나 (분리된 구조 누락 포함)\n")

    for n_points in sorted(summaries):
        summary = summaries[n_points]
        keys = [k for k in CLASS_ORDER if k in summary]
        print(f"[N = {n_points} 점]")
        header = (f"{'클래스':<9}{'DSC poly':>10}{'DSC multi':>11}{'DSC single':>12}"
                  f"{'HD95 multi':>12}{'HD95 single':>13}")
        print(header)
        print("-" * len(header))
        for k in keys:
            b = summary[k]
            print(f"{k:<9}{b['dsc_poly']:>10.4f}{b['dsc_multi']:>11.4f}{b['dsc_single']:>12.4f}"
                  f"{b['hd95_multi']:>12.3f}{b['hd95_single']:>13.3f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="윤곽 표현 상한 측정 (GT만, 학습 없음)")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--max_patients", type=int, default=120)
    parser.add_argument("--n_points", type=str, default="128",
                        help="쉼표로 여러 값을 주면 한 번의 로드로 모두 비교한다 (예: 32,64,128)")
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--out", type=str, default="results/contour_ceiling.json")
    args = parser.parse_args()

    from src.data.brats2020_dataset import BraTS2020Dataset, _find_patient_dirs

    pids = [p.name for p in _find_patient_dirs(args.train_root)]
    if not pids:
        raise SystemExit(f"환자 폴더를 찾을 수 없습니다: {args.train_root}")
    rng = np.random.default_rng(42)
    if len(pids) > args.max_patients:
        pick = rng.choice(len(pids), size=args.max_patients, replace=False)
        pids = sorted(pids[i] for i in pick)
    log.info("환자 %d명으로 상한 측정", len(pids))

    os.environ.setdefault("BRATS_SKIP_CACHE_SAVE", "1")
    ds = BraTS2020Dataset(
        root_dir=args.train_root,
        modality="t1ce",  # GT만 쓰므로 모달리티 한 개로 로딩 비용을 줄인다
        target_size=args.target_size,
        patient_ids=pids,
        slice_selection="tumor",
        simulate_rough=False,
        num_workers=args.num_workers,
    )
    log.info("슬라이스 %d장 로드 완료. 윤곽 변환 시작", len(ds))

    point_counts = [int(v) for v in args.n_points.split(",") if v.strip()]
    summaries: dict[int, dict] = {}
    for n_points in point_counts:
        rows = []
        for i, sample in enumerate(ds._samples):
            row = analyze_slice(sample[1], n_points)
            if row:
                rows.append(row)
            if (i + 1) % 5000 == 0:
                log.info("N=%d: %d/%d 처리", n_points, i + 1, len(ds._samples))
        if not rows:
            raise SystemExit("분석할 슬라이스가 없습니다.")
        summaries[n_points] = summarize(rows)
        log.info("N=%d 완료 (%d 슬라이스)", n_points, len(rows))

    print_topology(summaries[point_counts[0]])
    print_report(summaries)

    payload = {
        "config": {
            "point_counts": point_counts,
            "target_size": args.target_size,
            "n_patients": len(pids),
            "n_slices": summaries[point_counts[0]]["overall"]["n_slices"],
        },
        "by_n_points": {str(k): v for k, v in summaries.items()},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("저장: %s", out_path)


if __name__ == "__main__":
    main()
