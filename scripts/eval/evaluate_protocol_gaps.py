"""평가 프로토콜에서 빠져 있던 항목.

- 종양 비율 0.2% 미만 슬라이스의 거짓 양성
- 두 방법 모두 예측과 정답이 비어 있지 않은 공통 슬라이스만의 HD95. 둘 다 빈 단면의 거리 0은 넣지 않는다.
- 슬라이스를 z 순으로 쌓은 환자 단위 3D Dice
- HD95 px * 1.9 = mm

환자 분할은 seed 42의 개발 400명을 뺀 851명이다. 학습 시드만 가중치 폴더로 구분한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.eval.evaluate_band_ppo_locked import bootstrap_ci, tta_probability
from src.models.band_refine import build_band_refine, refine_batch
from src.utils.evaluation_records import measured_metrics
from src.utils.metrics import dice
from src.utils.refinement_inputs import stage2_mask

PX_TO_MM = 1.9
EMPTY_RATIO = 0.002
PAPER_THR = [0.80, 0.80, 0.50]
CC_SIZES = [0, 15, 25]


def patient_mean(rows, key, keep):
    grouped = defaultdict(list)
    for row, ok in zip(rows, keep):
        if ok and row.get(key) is not None and np.isfinite(row[key]):
            grouped[row["patient_id"]].append(float(row[key]))
    if not grouped:
        return {"n_patients": 0, "mean": float("nan"), "ci95": None}
    values = np.array([float(np.mean(v)) for v in grouped.values()], dtype=float)
    return {
        "n_patients": int(len(values)),
        "mean": float(values.mean()),
        "ci95": bootstrap_ci(values, 42),
    }


@torch.no_grad()
def large_tta(pipeline, image: torch.Tensor) -> torch.Tensor:
    route = torch.full((image.shape[0],), 2, device=image.device, dtype=torch.long)
    base, _ = pipeline(image, true_class_preds=route)
    horizontal = torch.flip(pipeline(torch.flip(image, dims=[3]), true_class_preds=route)[0], dims=[3])
    vertical = torch.flip(pipeline(torch.flip(image, dims=[2]), true_class_preds=route)[0], dims=[2])
    return (base + horizontal + vertical) / 3.0


def empty_report(flags_empty, flags_fp) -> dict:
    n = int(flags_empty.sum())
    fp = int((flags_empty & flags_fp).sum())
    return {
        "n_slices_below_0.2pct": n,
        "n_false_positive": fp,
        "false_positive_rate": float(fp / n) if n else float("nan"),
    }


def hd_block(rows, key, common):
    vals = [row[key] for row, ok in zip(rows, common) if ok and row[key] is not None]
    if not vals:
        return {"n_slices": 0, "mean_px": float("nan"), "mean_mm": float("nan")}
    mean_px = float(np.mean(vals))
    return {"n_slices": len(vals), "mean_px": mean_px, "mean_mm": mean_px * PX_TO_MM}


def main() -> None:
    parser = argparse.ArgumentParser(description="빈 슬라이스, 공통 HD95, 3D Dice, mm")
    parser.add_argument("--train_root", default="src/data/archive")
    parser.add_argument("--patient_split", default="checkpoints/patient_split.json")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed_label", type=int, default=42)
    parser.add_argument("--out", default="results/protocol_gaps/seed_42.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.patient_split, encoding="utf-8") as f:
        split = json.load(f)
    from src.data.patient_split import list_patient_ids
    used = set(split["train"]) | set(split["val"]) | set(split["test"])
    held = sorted(set(list_patient_ids(args.train_root)) - used)
    from src.data.brats2020_dataset import BraTS2020Dataset
    dataset = BraTS2020Dataset(
        args.train_root, modality="t1ce+flair", target_size=128,
        patient_ids=held, slice_selection="all", simulate_rough=False, num_workers=16,
    )
    print(f"held {len(held)} slices {len(dataset)}", flush=True)

    from src.models.dynamic_router import AdaptivePipeline
    pipeline = AdaptivePipeline(
        device, in_channels=2, strict_checkpoints=True, checkpoint_dir=args.checkpoint_dir,
    ).eval()
    ckpt_path = os.path.join(args.checkpoint_dir, "band_ppo.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(ckpt["model"])
    actor.eval()

    groups = defaultdict(list)
    for index, pid in enumerate(dataset._sample_pids):
        groups[pid].append(index)
    rows = []
    methods = ("stage2", "ppo", "large")
    dice3d = {name: [] for name in methods}
    seen = 0
    for patient_index, pid in enumerate(held):
        indices = groups.get(pid, [])
        if not indices:
            continue
        images = np.stack([dataset._samples[i][0] for i in indices])
        gts = np.stack([dataset._samples[i][1] for i in indices])
        vol = np.stack([dataset._samples[i][4] for i in indices])
        zs = [int(dataset._sample_zs[i]) for i in indices]
        for i in indices:
            dataset._samples[i] = None
        plane = {name: [None] * len(indices) for name in ("gt", *methods)}
        for start in range(0, len(indices), args.batch_size):
            stop = min(start + args.batch_size, len(indices))
            image_25d = torch.from_numpy(np.ascontiguousarray(vol[start:stop])).to(device)
            center = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
            with torch.no_grad():
                base, route = pipeline(image_25d)
                prob = tta_probability(pipeline, image_25d, base, route)
                large_prob = large_tta(pipeline, image_25d)
            prob_np = prob.squeeze(1).detach().cpu().numpy()
            large_np = large_prob.squeeze(1).detach().cpu().numpy()
            route_np = route.detach().cpu().numpy()
            stage2_np = np.stack([
                stage2_mask(prob_np[offset], int(route_np[offset]), PAPER_THR, CC_SIZES)
                for offset in range(stop - start)
            ]).astype(np.float32)
            large_mask = np.stack([
                stage2_mask(large_np[offset], 2, PAPER_THR, CC_SIZES)
                for offset in range(stop - start)
            ]).astype(np.float32)
            flair = center[:, int(cfg["flair_index"]):int(cfg["flair_index"]) + 1]
            refined = refine_batch(
                actor, center, torch.from_numpy(prob_np).unsqueeze(1).to(device),
                torch.from_numpy(stage2_np).unsqueeze(1).to(device), flair,
                int(cfg["n_steps"]), float(cfg["prob_lo"]), float(cfg["prob_hi"]),
                int(cfg["radius"]), guard_last=True,
            ).detach().cpu().numpy()[:, 0]
            for offset, index in enumerate(range(start, stop)):
                gt = (gts[index] > 0.5).astype(np.float32)
                pred = {
                    "stage2": (stage2_np[offset] > 0.5).astype(np.float32),
                    "ppo": (refined[offset] > 0.5).astype(np.float32),
                    "large": (large_mask[offset] > 0.5).astype(np.float32),
                }
                metrics = {name: measured_metrics(mask, gt) for name, mask in pred.items()}
                ratio = float(gt.mean())
                row = {
                    "patient_id": pid,
                    "z": zs[index],
                    "gt_ratio": ratio,
                    "empty": ratio < EMPTY_RATIO,
                }
                for name, metric in metrics.items():
                    row[f"{name}_dsc"] = metric["dsc"]
                    row[f"{name}_hd95"] = metric["hd95_surface_px"]
                    row[f"{name}_fp"] = bool(row["empty"] and pred[name].any())
                    # 둘 다 비면 measured_metrics가 HD95를 0으로 둔다. 표면 거리가 아니므로 제외한다.
                    row[f"{name}_defined"] = metric["predicted_pixels"] > 0 and metric["gt_pixels"] > 0
                rows.append(row)
                plane["gt"][index] = gt
                for name in methods:
                    plane[name][index] = pred[name]
        order = np.argsort(zs)
        z0, z1 = min(zs), max(zs)
        height, width = plane["gt"][0].shape
        gt_vol = np.zeros((z1 - z0 + 1, height, width), dtype=np.uint8)
        pred_vol = {name: np.zeros_like(gt_vol) for name in methods}
        for index in order:
            slot = zs[index] - z0
            gt_vol[slot] = plane["gt"][index] > 0.5
            for name in methods:
                pred_vol[name][slot] = plane[name][index] > 0.5
        for name in methods:
            dice3d[name].append(dice(pred_vol[name], gt_vol))
        seen += len(indices)
        if patient_index % 20 == 0:
            print(f"추론 {seen}/{len(dataset)} {pid}", flush=True)
    dataset._samples.clear()

    del pipeline, actor
    torch.cuda.empty_cache()

    tumor = np.array([not row["empty"] for row in rows])
    summary = {
        "seed": args.seed_label,
        "checkpoint_dir": args.checkpoint_dir,
        "n_patients": len(held),
        "n_slices": len(rows),
        "n_tumor_slices": int(tumor.sum()),
        "n_below_0.2pct": int((~tumor).sum()),
        "px_to_mm": PX_TO_MM,
        "tumor_slice_patient_dsc": {},
        "empty_slice_false_positive": {},
        "hd95_defined_slices": {},
        "hd95_common_slices": {},
        "dice_3d": {},
    }
    for name in methods:
        summary["tumor_slice_patient_dsc"][name] = patient_mean(rows, f"{name}_dsc", tumor)
        summary["empty_slice_false_positive"][name] = empty_report(
            ~tumor, np.array([row[f"{name}_fp"] for row in rows]),
        )
        defined = np.array([row[f"{name}_defined"] for row in rows])
        block = hd_block(rows, f"{name}_hd95", defined)
        block["patient_mean_px"] = patient_mean(rows, f"{name}_hd95", defined)
        block["patient_mean_mm"] = None if block["patient_mean_px"]["mean"] != block["patient_mean_px"]["mean"] else {
            **block["patient_mean_px"],
            "mean": block["patient_mean_px"]["mean"] * PX_TO_MM,
            "ci95": None if block["patient_mean_px"]["ci95"] is None else [v * PX_TO_MM for v in block["patient_mean_px"]["ci95"]],
        }
        summary["hd95_defined_slices"][name] = block

    def common_mask(names):
        keep = np.ones(len(rows), dtype=bool)
        for name in names:
            keep &= np.array([row[f"{name}_defined"] for row in rows])
        return keep

    for label, names in (
        ("stage2_and_ppo", ("stage2", "ppo")),
        ("stage2_ppo_large", ("stage2", "ppo", "large")),
    ):
        keep = common_mask(names)
        summary["hd95_common_slices"][label] = {
            "n_slices": int(keep.sum()),
            "methods": {
                name: {
                    **hd_block(rows, f"{name}_hd95", keep),
                    "patient_mean_px": patient_mean(rows, f"{name}_hd95", keep),
                }
                for name in names
            },
        }
        for name in names:
            px = summary["hd95_common_slices"][label]["methods"][name]["patient_mean_px"]
            summary["hd95_common_slices"][label]["methods"][name]["patient_mean_mm"] = {
                **px,
                "mean": px["mean"] * PX_TO_MM,
                "ci95": None if px["ci95"] is None else [v * PX_TO_MM for v in px["ci95"]],
            }

    for name in methods:
        values = np.array(dice3d[name], dtype=float)
        summary["dice_3d"][name] = {
            "n_patients": int(len(values)),
            "mean": float(values.mean()),
            "ci95": bootstrap_ci(values, 42),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps({
        "seed": args.seed_label,
        "tumor_dsc": {k: v["mean"] for k, v in summary["tumor_slice_patient_dsc"].items()},
        "empty_fp": {k: v["false_positive_rate"] for k, v in summary["empty_slice_false_positive"].items()},
        "dice_3d": {k: v["mean"] for k, v in summary["dice_3d"].items()},
        "common_hd95_mm": {
            k: v["patient_mean_mm"]["mean"]
            for k, v in summary["hd95_common_slices"]["stage2_and_ppo"]["methods"].items()
        },
    }, indent=2), flush=True)
    print(f"저장 {args.out}", flush=True)


if __name__ == "__main__":
    main()
