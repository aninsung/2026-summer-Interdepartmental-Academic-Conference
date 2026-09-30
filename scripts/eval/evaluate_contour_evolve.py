"""Stage 2 대비 지도학습 윤곽 진화 비교 평가.

같은 슬라이스·같은 Stage 2 확률맵을 공유하므로, 차이는 순수하게 "마스크를 그대로
쓸 것인가, 윤곽으로 다시 그릴 것인가"에서만 나온다. 세 가지를 같이 보고한다.

  stage2  : Expert 확률맵을 임계값·CC 필터로 이진화한 현행 기준
  contour : Stage 2 컴포넌트에서 초기화한 윤곽을 학습된 모델로 이동시킨 결과
  ceiling : 같은 컴포넌트의 GT 윤곽을 그대로 래스터화한 상한 (학습이 완벽할 때)

사용 예:
  BRATS_NUM_WORKERS=16 python scripts/eval/evaluate_contour_evolve.py \
      --split_role test --classes medium,large
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
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.contour_dataset import build_stage2_entries
from src.utils.contour import component_contours, match_gt_component, outer_contour, rasterize
from src.utils.evaluation_records import measured_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_INDEX = {"small": 0, "medium": 1, "large": 2}
CLASS_NAMES = {0: "small", 1: "medium", 2: "large"}
METHODS = ["stage2", "contour", "ceiling"]


@torch.no_grad()
def refine_slice(model, entry, device, n_points: int, min_area: int) -> tuple[np.ndarray, np.ndarray]:
    """한 슬라이스의 Stage 2 마스크를 윤곽으로 다시 그린다.

    min_area 미만 컴포넌트는 윤곽으로 다루기엔 점이 너무 적으므로 원본 픽셀을 유지한다.
    Returns (contour_mask, ceiling_mask)
    """
    shape = entry.stage2.shape
    contours, components, passthrough = component_contours(entry.stage2, n_points, min_area)
    if not contours:
        return passthrough.copy(), passthrough.copy()

    image = torch.from_numpy(entry.model_input()).unsqueeze(0).to(device)
    image = image.expand(len(contours), -1, -1, -1)
    init = torch.from_numpy(np.stack(contours).astype(np.float32)).to(device)
    pred = model(image, init)[-1].cpu().numpy()

    # 상한: 각 컴포넌트에 대응하는 GT 연결요소의 윤곽을 그대로 사용
    ceiling_polys = []
    for comp in components:
        gt_comp = match_gt_component(comp, entry.gt)
        target = outer_contour(gt_comp, n_points) if gt_comp is not None else None
        ceiling_polys.append(target)

    contour_mask = np.maximum(rasterize(list(pred), shape), passthrough)
    ceiling_mask = np.maximum(rasterize(ceiling_polys, shape), passthrough)
    return contour_mask, ceiling_mask


def summarize(rows: list[dict]) -> dict:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["route"]].append(r)

    def block(items: list[dict]) -> dict:
        out = {"n_slices": len(items)}
        for method in METHODS:
            dsc = np.array([it[method]["dsc"] for it in items], dtype=np.float64)
            hd = np.array([it[method]["hd95_surface_px"] for it in items
                           if it[method]["hd95_surface_px"] is not None], dtype=np.float64)
            prec = np.array([it[method]["precision"] for it in items], dtype=np.float64)
            rec = np.array([it[method]["recall"] for it in items], dtype=np.float64)
            out[method] = {
                "dsc": float(dsc.mean()),
                "hd95_surface_px": float(hd.mean()) if len(hd) else float("nan"),
                "hd95_defined": int(len(hd)),
                "precision": float(prec.mean()),
                "recall": float(rec.mean()),
            }
        improved = np.array([it["contour"]["dsc"] > it["stage2"]["dsc"] for it in items])
        out["frac_contour_better"] = float(improved.mean())
        return out

    summary = {"overall": block(rows)}
    for cls in sorted(by_class):
        summary[CLASS_NAMES[cls]] = block(by_class[cls])
    return summary


def print_report(summary: dict) -> None:
    order = ["medium", "large", "small", "overall"]
    keys = [k for k in order if k in summary]

    print("\n=== Stage 2 vs 지도학습 윤곽 진화 ===\n")
    header = (f"{'클래스':<9}{'슬라이스':>9}{'Stage2 DSC':>12}{'윤곽 DSC':>11}"
              f"{'상한 DSC':>11}{'윤곽이 나은 비율':>17}")
    print(header)
    print("-" * 74)
    for k in keys:
        b = summary[k]
        print(f"{k:<9}{b['n_slices']:>9}{b['stage2']['dsc']:>12.4f}"
              f"{b['contour']['dsc']:>11.4f}{b['ceiling']['dsc']:>11.4f}"
              f"{b['frac_contour_better']:>17.3f}")

    print(f"\n{'클래스':<9}{'Stage2 HD95':>13}{'윤곽 HD95':>12}{'상한 HD95':>12}"
          f"{'Stage2 P/R':>16}{'윤곽 P/R':>16}")
    print("-" * 82)
    for k in keys:
        b = summary[k]
        s, c, ce = b["stage2"], b["contour"], b["ceiling"]
        print(f"{k:<9}{s['hd95_surface_px']:>13.3f}{c['hd95_surface_px']:>12.3f}"
              f"{ce['hd95_surface_px']:>12.3f}"
              f"{s['precision']:>8.3f}/{s['recall']:<7.3f}"
              f"{c['precision']:>8.3f}/{c['recall']:<7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 대비 윤곽 진화 비교 평가")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--split_role", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--classes", type=str, default="medium,large")
    parser.add_argument("--apply_classes", type=str, default="",
                        help="윤곽을 적용할 클래스. 비우면 classes 전부. 나머지는 Stage 2를 유지")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/contour_evolve.pt")
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--out", type=str, default="results/contour_probe.json")
    args = parser.parse_args()

    classes = tuple(CLASS_INDEX[c.strip()] for c in args.classes.split(","))
    apply = (tuple(CLASS_INDEX[c.strip()] for c in args.apply_classes.split(","))
             if args.apply_classes else classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.patient_split, "r", encoding="utf-8") as f:
        split = json.load(f)
    pids = split[args.split_role]
    log.info("%s 환자 %d명", args.split_role, len(pids))

    from src.data.brats2020_dataset import BraTS2020Dataset
    dataset = BraTS2020Dataset(
        root_dir=args.train_root, modality=args.modality, target_size=args.target_size,
        patient_ids=pids, slice_selection="tumor", simulate_rough=False,
    )

    from src.models.dynamic_router import AdaptivePipeline
    images, _, _ = dataset.get_numpy_arrays()
    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch, strict_checkpoints=True)
    pipeline.eval()
    entries = build_stage2_entries(dataset, pipeline, device, classes, use_tta=not args.no_tta)
    del pipeline
    torch.cuda.empty_cache()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    from src.models.contour_evolve import build_contour_evolve
    model = build_contour_evolve(in_channels=cfg["in_channels"], n_iters=cfg["n_iters"],
                                 max_shift=cfg["max_shift"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("체크포인트 로드: %s (학습 시 val 컴포넌트 DSC %.4f)",
             args.checkpoint, ckpt.get("val_component_dsc", float("nan")))

    rows = []
    for i, entry in enumerate(entries):
        stage2 = measured_metrics(entry.stage2, entry.gt)
        if entry.route in apply:
            contour_mask, ceiling_mask = refine_slice(model, entry, device,
                                                      cfg["n_points"], cfg["min_area"])
            contour = measured_metrics(contour_mask, entry.gt)
            ceiling = measured_metrics(ceiling_mask, entry.gt)
        else:
            contour, ceiling = stage2, stage2
        rows.append({
            "route": entry.route,
            "patient_id": entry.patient_id,
            "stage2": stage2,
            "contour": contour,
            "ceiling": ceiling,
        })
        if (i + 1) % 1000 == 0:
            log.info("평가 %d/%d", i + 1, len(entries))

    summary = summarize(rows)
    summary["config"] = {
        "split_role": args.split_role, "classes": args.classes,
        "apply_classes": args.apply_classes or args.classes,
        "n_patients": len(pids), "n_slices": len(rows),
        "checkpoint": args.checkpoint, "contour_config": cfg,
        "tta": not args.no_tta,
    }
    print_report(summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("저장: %s", out_path)


if __name__ == "__main__":
    main()
