"""Stage 2 대비 경계 띠 보정 평가.

  stage2 : Expert 마스크
  raw    : 띠 안 켜기/끄기만 적용
  guard  : 같은 수정 중 FLAIR 상대 밝기를 통과한 것만 적용

사용 예:
  BRATS_NUM_WORKERS=16 python -u scripts/eval/evaluate_band_refine.py \
      --split_role test --classes medium --checkpoint checkpoints/band_refine_medium.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.contour_dataset import build_stage2_entries
from src.models.band_refine import build_band_refine, predict_pair
from src.utils.evaluation_records import measured_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_INDEX = {"small": 0, "medium": 1, "large": 2}
CLASS_NAMES = {0: "small", 1: "medium", 2: "large"}
METHODS = ["stage2", "raw", "guard"]


def summarize(rows: list[dict]) -> dict:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["route"]].append(row)

    def block(items: list[dict]) -> dict:
        out = {"n_slices": len(items)}
        for method in METHODS:
            dsc = np.array([it[method]["dsc"] for it in items], dtype=np.float64)
            hd = np.array([
                it[method]["hd95_surface_px"] for it in items
                if it[method]["hd95_surface_px"] is not None
            ], dtype=np.float64)
            prec = np.array([it[method]["precision"] for it in items], dtype=np.float64)
            rec = np.array([it[method]["recall"] for it in items], dtype=np.float64)
            out[method] = {
                "dsc": float(dsc.mean()),
                "hd95_surface_px": float(hd.mean()) if len(hd) else float("nan"),
                "hd95_defined": int(len(hd)),
                "precision": float(prec.mean()),
                "recall": float(rec.mean()),
            }
        changed = np.array([it["n_changed"] > 0 for it in items])
        better = np.array([it["guard"]["dsc"] > it["stage2"]["dsc"] for it in items])
        out["frac_guard_changed"] = float(changed.mean()) if len(items) else float("nan")
        out["frac_guard_dsc_better"] = float(better.mean()) if len(items) else float("nan")
        return out

    summary = {"overall": block(rows)}
    for cls in sorted(by_class):
        summary[CLASS_NAMES[cls]] = block(by_class[cls])
    return summary


def print_report(summary: dict) -> None:
    order = ["medium", "large", "small", "overall"]
    keys = [k for k in order if k in summary]
    print("\n=== Stage 2 vs 경계 띠 보정 ===\n")
    print(f"{'클래스':<9}{'슬라이스':>9}{'Stage2 DSC':>12}{'raw DSC':>10}{'guard DSC':>11}{'가드가 나은 비율':>16}")
    print("-" * 70)
    for key in keys:
        block = summary[key]
        print(f"{key:<9}{block['n_slices']:>9}{block['stage2']['dsc']:>12.4f}"
              f"{block['raw']['dsc']:>10.4f}{block['guard']['dsc']:>11.4f}"
              f"{block['frac_guard_dsc_better']:>16.3f}")
    print(f"\n{'클래스':<9}{'Stage2 HD95':>13}{'raw HD95':>11}{'guard HD95':>12}"
          f"{'Stage2 P/R':>16}{'guard P/R':>16}")
    print("-" * 78)
    for key in keys:
        block = summary[key]
        stage2, raw, guard = block["stage2"], block["raw"], block["guard"]
        print(f"{key:<9}{stage2['hd95_surface_px']:>13.3f}{raw['hd95_surface_px']:>11.3f}"
              f"{guard['hd95_surface_px']:>12.3f}"
              f"{stage2['precision']:>8.3f}/{stage2['recall']:<7.3f}"
              f"{guard['precision']:>8.3f}/{guard['recall']:<7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 대비 경계 띠 보정 평가")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--split_role", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--classes", type=str, default="medium")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/band_refine_medium.pt")
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--out", type=str, default="results/band_refine_medium.json")
    args = parser.parse_args()

    classes = tuple(CLASS_INDEX[c.strip()] for c in args.classes.split(","))
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
    model = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("체크포인트 로드: %s (val guard DSC %.4f / HD95 %.3f)",
             args.checkpoint, ckpt.get("val_guard_dsc", float("nan")),
             ckpt.get("val_guard_hd95", float("nan")))

    rows = []
    for entry in entries:
        raw, guarded = predict_pair(
            model, entry, device, cfg["prob_lo"], cfg["prob_hi"], cfg["radius"], cfg["flair_index"])
        rows.append({
            "route": entry.route,
            "patient_id": entry.patient_id,
            "stage2": measured_metrics(entry.stage2, entry.gt),
            "raw": measured_metrics(raw, entry.gt),
            "guard": measured_metrics(guarded, entry.gt),
            "n_changed": int((guarded.astype(bool) != (entry.stage2 > 0.5)).sum()),
        })

    summary = summarize(rows)
    summary["checkpoint"] = args.checkpoint
    summary["split_role"] = args.split_role
    summary["saved_without_gate"] = bool(ckpt.get("saved_without_gate", False))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print_report(summary)
    log.info("저장: %s", args.out)


if __name__ == "__main__":
    main()
