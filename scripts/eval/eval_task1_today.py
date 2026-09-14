"""
Quick evaluation script for BraTS Task 1 sub-regions (ET, TC, WT)
on today's trained 3-Stage Adaptive Pipeline (CaraNet, UNet++, SegResNet).
"""

import sys
import os
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import load_or_create_patient_split
from src.models.dynamic_router import AdaptivePipeline
from src.utils.metrics import dice, hd95, precision, recall
from scipy.ndimage import distance_transform_edt


def evaluate_task1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval Task 1] Using device: {device}")

    # Load patient split
    split_path = "checkpoints/patient_split.json"
    with open(split_path, "r", encoding="utf-8") as f:
        p_split = json.load(f)
    val_pids = p_split["val"]
    print(f"[Eval Task 1] Loaded {len(val_pids)} validation patients.")

    # Load dataset
    dataset = BraTS2020Dataset(
        data_root="src/data/archive",
        modalities=["t1ce", "flair"],
        split_pids=val_pids,
        load_gt_regions=True,
    )
    print(f"[Eval Task 1] Total slices: {len(dataset)}")

    # Load Pipeline
    pipeline = AdaptivePipeline(device=device, in_channels=1).to(device)
    pipeline.eval()

    # Pre-allocate accumulators
    region_names = ["ET", "TC", "WT"]
    metrics = {
        "ET": {"dsc": [], "hd95": [], "prec": [], "rec": []},
        "TC": {"dsc": [], "hd95": [], "prec": [], "rec": []},
        "WT": {"dsc": [], "hd95": [], "prec": [], "rec": []},
    }

    # Process in batches
    batch_size = 256
    n_slices = len(dataset)
    images_25d = dataset.images_25d  # (N, 3, 128, 128)
    gt_regions_all = dataset.gt_task1_regions  # (N, 3, 128, 128)

    print(f"[Eval Task 1] Starting inference on {n_slices} slices (batch_size={batch_size})...")

    with torch.no_grad():
        for b_start in range(0, n_slices, batch_size):
            b_end = min(b_start + batch_size, n_slices)
            b_img = torch.from_numpy(images_25d[b_start:b_end]).to(device)

            # Forward pass through Stage 1 & Stage 2 experts
            # rough_mask: (B, 1, H, W), class_preds: (B,), region_probs: (B, 3, H, W)
            rough_masks, class_preds, region_probs = pipeline(b_img, return_regions=True)
            region_probs_np = region_probs.cpu().numpy()  # (B, 3, H, W)

            for bi in range(b_end - b_start):
                idx = b_start + bi
                pred_regions = (region_probs_np[bi] > 0.5).astype(np.float32)  # (3, H, W)
                gt_regions = gt_regions_all[idx]  # (3, H, W)

                for r_i, r_name in enumerate(region_names):
                    p_reg = pred_regions[r_i]
                    g_reg = gt_regions[r_i]

                    d_val = dice(p_reg, g_reg)
                    p_val = precision(p_reg, g_reg)
                    r_val = recall(p_reg, g_reg)

                    # Compute HD95
                    if np.any(g_reg > 0.5):
                        dist_gt = distance_transform_edt(~(g_reg > 0.5))
                        h_val = hd95(p_reg, g_reg, dist_b=dist_gt)
                    else:
                        h_val = 0.0 if not np.any(p_reg > 0.5) else 30.0

                    metrics[r_name]["dsc"].append(d_val)
                    metrics[r_name]["hd95"].append(h_val)
                    metrics[r_name]["prec"].append(p_val)
                    metrics[r_name]["rec"].append(r_val)

            if (b_end // 2000 != b_start // 2000) or (b_end == n_slices):
                print(f"  Processed {b_end}/{n_slices} ({b_end/n_slices*100:.1f}%)...")

    # Summarize results
    print("\n" + "=" * 80)
    print(" 🏥 오늘(Sep 14) 학습된 3-Stage Expert 파이프라인 WT / TC / ET 성능 리포트 ")
    print("=" * 80)
    print(f"검증 대상: 환자 {len(val_pids)}명 / 슬라이스 {n_slices}장 전수")
    print(f"{'하위 영역 (Region)':<20} | {'DSC (Dice Score)':<15} | {'HD95 (mm/px)':<15} | {'Precision':<15} | {'Recall':<15}")
    print("-" * 80)

    summary_json = {}
    for r_name in region_names:
        m = metrics[r_name]
        avg_dsc = float(np.mean(m["dsc"]))
        avg_hd95 = float(np.mean(m["hd95"]))
        avg_prec = float(np.mean(m["prec"]))
        avg_rec = float(np.mean(m["rec"]))

        summary_json[r_name] = {
            "dsc": avg_dsc,
            "hd95": avg_hd95,
            "precision": avg_prec,
            "recall": avg_rec,
        }

        print(f"{r_name:<20} | {avg_dsc:<15.4f} | {avg_hd95:<15.4f} | {avg_prec:<15.4f} | {avg_rec:<15.4f}")

    print("=" * 80)

    # Save to json file
    out_json_path = "results/task1_subregions_today.json"
    with open(out_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"✅ 저장 완료: {out_json_path}")


if __name__ == "__main__":
    evaluate_task1()
