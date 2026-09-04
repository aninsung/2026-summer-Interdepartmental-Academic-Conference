"""Stage 2 expert assignment ablation.

기존 체크포인트를 재학습하지 않고, 같은 val 슬라이스에서
각 Expert 확률맵을 모두 만든 뒤 배정만 바꾼다.

- 3x3: CaraNet / UNet++ / SegResNet × Small / Medium / Large
- TRIO: 분류기 클래스대로 기존 배정 (S→CaraNet, M→UNet++, L→SegResNet)
- Swap M↔L: S→CaraNet, M→SegResNet, L→UNet++
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import DEFAULT_SPLIT_PATH, load_or_create_patient_split
from src.models.dynamic_router import AdaptivePipeline
from src.utils.metrics import dice, filter_small_components, hd95
from src.utils.zoom_crop import refine_with_zoom

CLASS_NAMES = ("Small", "Medium", "Large")
EXPERT_NAMES = ("CaraNet", "UNet++", "SegResNet")


def _binarize(prob, c, thr, cc_min, micro_area_floor, micro_thr_floor):
    mask = (prob > thr[c]).astype(np.float32)
    if c == 0 and np.sum(mask) < micro_area_floor:
        for t in np.arange(thr[c] - 0.05, micro_thr_floor - 1e-9, -0.05):
            mask = (prob > t).astype(np.float32)
            if np.sum(mask) >= micro_area_floor:
                break
    return filter_small_components(mask, cc_min[c])


def _summarize(dscs, hds, routed):
    routed = np.asarray(routed)
    out = {"n": int(len(dscs)), "dsc": float(np.mean(dscs)), "hd95": float(np.mean(hds)), "by_class": {}}
    for c, name in enumerate(CLASS_NAMES):
        idx = np.where(routed == c)[0]
        if len(idx) == 0:
            continue
        out["by_class"][name] = {
            "n": int(len(idx)),
            "dsc": float(np.mean(dscs[idx])),
            "hd95": float(np.mean(hds[idx])),
        }
    return out


def _print_summary(title, summary):
    print(f"\n{title}")
    print(f"  overall n={summary['n']}  DSC={summary['dsc']:.4f}  HD95={summary['hd95']:.4f}")
    for name, row in summary["by_class"].items():
        print(f"  {name:<8} n={row['n']:4d}  DSC={row['dsc']:.4f}  HD95={row['hd95']:.4f}")


@torch.no_grad()
def _expert_probs(pipeline, images_25d, device, batch_size):
    n, h, w = images_25d.shape[0], images_25d.shape[-2], images_25d.shape[-1]
    routed = np.zeros(n, dtype=np.int64)
    probs = {name: np.zeros((n, h, w), dtype=np.float32) for name in EXPERT_NAMES}
    experts = {
        "CaraNet": pipeline.expert_small,
        "UNet++": pipeline.expert_medium,
        "SegResNet": pipeline.expert_large,
    }

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(images_25d[start:end]).to(device)
        if batch.ndim == 3:
            batch = batch.unsqueeze(1)
        x_cls = pipeline._match_in_channels(batch, pipeline.classifier.conv1.weight.shape[1])
        _, class_preds = torch.max(pipeline.classifier(x_cls), 1)
        routed[start:end] = class_preds.cpu().numpy()

        for name, expert in experts.items():
            logits = pipeline._forward_expert(expert, batch)
            out = pipeline._to_wt_prob(logits)
            if name == "CaraNet" and pipeline.small_zoom:
                out = refine_with_zoom(
                    lambda z: pipeline._forward_expert(expert, z),
                    batch,
                    out,
                    patch_size=pipeline.small_zoom_patch,
                )
                if out.shape[1] != 1:
                    out = pipeline._to_wt_prob(out)
            probs[name][start:end] = out.squeeze(1).cpu().numpy()
        if (end % 256 == 0) or end == n:
            print(f"  forward {end}/{n}")
    return routed, probs


def parse_args():
    p = argparse.ArgumentParser(description="Medium↔Large expert swap (Stage 2 only)")
    p.add_argument("--train_root", type=str, default="src/data/archive")
    p.add_argument("--modality", type=str, default="t1ce+flair")
    p.add_argument("--max_patients", type=int, default=210)
    p.add_argument("--patient_split", type=str, default=DEFAULT_SPLIT_PATH)
    p.add_argument("--split_role", type=str, default="val")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--stage2_thresholds", type=str, default="0.80,0.80,0.50")
    p.add_argument("--cc_min_sizes", type=str, default="0,15,25")
    p.add_argument("--micro_area_floor", type=float, default=80.0)
    p.add_argument("--micro_thr_floor", type=float, default=0.15)
    p.add_argument("--out", type=str, default="results/expert_swap.json")
    return p.parse_args()


def main():
    args = parse_args()
    class_thr = [float(t) for t in args.stage2_thresholds.split(",")]
    cc_min = [int(t) for t in args.cc_min_sizes.split(",")]
    expert_thr = {"CaraNet": 0.80, "UNet++": 0.80, "SegResNet": 0.50}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split = load_or_create_patient_split(args.train_root, args.max_patients, args.patient_split)
    patient_ids = split[args.split_role]
    dataset = BraTS2020Dataset(
        root_dir=args.train_root,
        modality=args.modality,
        target_size=128,
        max_patients=None,
        patient_ids=patient_ids,
        simulate_rough=False,
    )
    images, gt_masks, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    print(f"Slices: {len(images)}")

    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch)
    pipeline.eval()

    print("Computing per-expert probability maps...")
    routed, probs = _expert_probs(pipeline, images_25d, device, args.batch_size)

    def score_assignment(pick_expert, thr_mode="class"):
        dscs, hds = [], []
        for i in range(len(images)):
            c = int(routed[i])
            name = pick_expert(c)
            if thr_mode == "class":
                thr = class_thr
            else:
                native = expert_thr[name]
                thr = [native, native, native]
                thr[c] = native
            pred = _binarize(
                probs[name][i], c, thr, cc_min, args.micro_area_floor, args.micro_thr_floor
            )
            dscs.append(dice(pred, gt_masks[i]))
            hds.append(hd95(pred, gt_masks[i]))
        return _summarize(np.asarray(dscs), np.asarray(hds), routed)

    trio = {
        0: "CaraNet",
        1: "UNet++",
        2: "SegResNet",
    }
    swap_ml = {
        0: "CaraNet",
        1: "SegResNet",
        2: "UNet++",
    }

    summaries = {
        "TRIO_class_thr": score_assignment(lambda c: trio[c], "class"),
        "swap_ML_class_thr": score_assignment(lambda c: swap_ml[c], "class"),
        "TRIO_expert_thr": score_assignment(lambda c: trio[c], "expert"),
        "swap_ML_expert_thr": score_assignment(lambda c: swap_ml[c], "expert"),
    }

    print("\n=== 3x3 Expert × size class (expert-native threshold) ===")
    grid = {}
    for ename in EXPERT_NAMES:
        grid[ename] = {}
        dscs = np.zeros(len(images), dtype=np.float64)
        hds = np.zeros(len(images), dtype=np.float64)
        native = expert_thr[ename]
        for i in range(len(images)):
            c = int(routed[i])
            thr = [native, native, native]
            pred = _binarize(
                probs[ename][i], c, thr, cc_min, args.micro_area_floor, args.micro_thr_floor
            )
            dscs[i] = dice(pred, gt_masks[i])
            hds[i] = hd95(pred, gt_masks[i])
        row = _summarize(dscs, hds, routed)
        grid[ename] = row
        print(f"{ename}: overall DSC={row['dsc']:.4f}")
        for cname, vals in row["by_class"].items():
            print(f"    {cname:<8} DSC={vals['dsc']:.4f}  HD95={vals['hd95']:.4f}")

    _print_summary("TRIO (class thresholds 0.80/0.80/0.50)", summaries["TRIO_class_thr"])
    _print_summary("Swap Medium↔Large (same class thresholds)", summaries["swap_ML_class_thr"])
    d1 = summaries["swap_ML_class_thr"]["dsc"] - summaries["TRIO_class_thr"]["dsc"]
    print(f"  Δ overall DSC (swap - TRIO, class thr): {d1:+.4f}")

    _print_summary("TRIO (expert-native thr)", summaries["TRIO_expert_thr"])
    _print_summary("Swap Medium↔Large (expert-native thr)", summaries["swap_ML_expert_thr"])
    d2 = summaries["swap_ML_expert_thr"]["dsc"] - summaries["TRIO_expert_thr"]["dsc"]
    print(f"  Δ overall DSC (swap - TRIO, expert thr): {d2:+.4f}")

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "note": "Inference-only swap of existing checkpoints. UNet++ was trained on Medium only; SegResNet on all sizes with ED/TC.",
        "n": len(images),
        "class_counts": {CLASS_NAMES[c]: int(np.sum(routed == c)) for c in range(3)},
        "summaries": summaries,
        "grid_expert_native_thr": grid,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
