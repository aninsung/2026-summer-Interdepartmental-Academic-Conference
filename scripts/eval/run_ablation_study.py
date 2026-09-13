"""Inference-component ablation on one fixed BraTS patient split/checkpoint set."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


VARIANTS = {
    "stage2_only_no_tta": ["--stage3_mode", "skip", "--no_tta"],
    "stage2_only_tta": ["--stage3_mode", "skip"],
    "ppo_no_tta_no_gates": [
        "--stage3_mode", "ppo", "--no_tta", "--cc_min_sizes", "0,0,0",
        "--area_gate_lo", "0", "--area_gate_hi", "1000000",
        "--area_gate_hi_medium", "1000000", "--area_gate_hi_large", "1000000",
    ],
    "ppo_no_tta_gated": ["--stage3_mode", "ppo", "--no_tta"],
    "fixed_segresnet_ppo": ["--stage3_mode", "ppo", "--fixed_route_class", "2"],
    "full": ["--stage3_mode", "ppo"],
}


def summarize(path: Path) -> dict[str, dict[str, float]]:
    data = np.load(path, allow_pickle=True)
    names = [str(x) for x in data["task1_regions"]]
    return {
        name: {
            "stage2_dsc": float(np.mean(data["task1_init_dsc"][i])),
            "final_dsc": float(np.mean(data["task1_final_dsc"][i])),
            "stage2_hd95": float(np.mean(data["task1_init_hd95"][i])),
            "final_hd95": float(np.mean(data["task1_final_hd95"][i])),
        }
        for i, name in enumerate(names)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_patients", type=int, default=1251)
    parser.add_argument("--patient_split", default="checkpoints/patient_split.json")
    parser.add_argument("--modality", default="t1ce+flair")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", default="results/ablation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    common = [
        sys.executable, "scripts/eval/evaluate_pipeline.py",
        "--max_patients", str(args.max_patients),
        "--patient_split", args.patient_split,
        "--split_role", "val", "--deploy_mode", "--skip_plot", "--skip_stats",
        "--modality", args.modality, "--eval_batch_size", str(args.batch_size),
        "--stage3_skip_classes", "", "--boundary_band_px", "2",
        "--boundary_band_mode", "expand", "--boundary_band_classes", "1,2",
        "--stage2_thresholds", "0.70,0.75,0.50", "--task1_thresholds", "0.50,0.50,0.50",
        "--cc_min_sizes", "0,15,25", "--stage2_erode_classes", "", "--stage2_erode_px", "0",
        "--area_gate_lo", "0.85", "--area_gate_hi", "1.2",
        "--area_gate_hi_medium", "1.5", "--area_gate_hi_large", "1.35",
    ]
    for name, flags in VARIANTS.items():
        metrics = out_dir / f"{name}.npz"
        cmd = common + ["--metrics_out", str(metrics)] + flags
        print(f"[ablation] {name}", flush=True)
        subprocess.run(cmd, check=True)
        results[name] = summarize(metrics)

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "region", "stage2_dsc", "final_dsc", "stage2_hd95", "final_hd95"])
        for variant, regions in results.items():
            for region, values in regions.items():
                writer.writerow([variant, region, *(values[k] for k in ("stage2_dsc", "final_dsc", "stage2_hd95", "final_hd95"))])
    print(f"[ablation] saved {out_dir / 'summary.json'} and {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
