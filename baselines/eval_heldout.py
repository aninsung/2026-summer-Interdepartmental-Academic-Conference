"""개발 400명을 뺀 851명에서 베이스라인을 환자 평균으로 평가한다.

임계값은 검증 60명에서만 고른다. 851명 점수로 임계값을 고르지 않는다.
DSC/HD95 정의는 scripts/eval/evaluate_band_ppo_locked.py 와 같다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.eval_baseline import load_baseline
from src.data.patient_split import list_patient_ids
from src.utils.evaluation_records import measured_metrics
from src.utils.metrics import gt_size_class

CLASS_NAMES = {0: "small", 1: "medium", 2: "large"}


def held_out_ids(root: str, split_path: str) -> list[str]:
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    used = set(split["train"]) | set(split["val"]) | set(split["test"])
    return sorted(set(list_patient_ids(root)) - used)


def bootstrap_ci(values: np.ndarray, seed: int = 42, repeats: int = 2000) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(repeats)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def patient_means(rows: list[dict], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["patient_id"], []).append(row[key])
    return {pid: float(np.mean(vals)) for pid, vals in grouped.items() if vals}


def pack(values: dict[str, float], seed: int) -> dict:
    ids = sorted(values)
    arr = np.array([values[i] for i in ids], dtype=float)
    return {
        "n_patients": len(ids),
        "mean": float(arr.mean()) if len(ids) else float("nan"),
        "ci95": bootstrap_ci(arr, seed),
    }


@torch.no_grad()
def predict_rows(model, dataset, device, batch_size: int, threshold: float) -> list[dict]:
    images, gts, _ = dataset.get_numpy_arrays()
    pids = list(dataset._sample_pids)
    rows = []
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        batch = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
        prob = torch.sigmoid(model(batch).float()).squeeze(1).cpu().numpy()
        for offset, index in enumerate(range(start, stop)):
            pred = (prob[offset] > threshold).astype(np.float32)
            metrics = measured_metrics(pred, gts[index])
            rows.append({
                "patient_id": pids[index],
                "gt_class": CLASS_NAMES[gt_size_class(gts[index])],
                "dsc": metrics["dsc"],
                "hd95": metrics["hd95_surface_px"],
            })
    return rows


def summarize(rows: list[dict], seed: int) -> dict:
    def block(class_name: str | None) -> dict:
        picked = rows if class_name is None else [r for r in rows if r["gt_class"] == class_name]
        dsc = pack(patient_means(picked, "dsc"), seed)
        hd_rows = [{**r, "hd": r["hd95"]} for r in picked if r["hd95"] is not None]
        hd = pack(patient_means(hd_rows, "hd"), seed + 1)
        return {"n_slices": len(picked), "dsc": dsc, "hd95": hd}

    summary = {"overall": block(None)}
    summary["by_gt_class"] = {name: block(name) for name in ("small", "medium", "large")}
    return summary


def val_threshold(model, root, split_path, modality, target_size, device, batch_size: int) -> float:
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    from src.data.brats2020_dataset import BraTS2020Dataset
    dataset = BraTS2020Dataset(
        root_dir=root, modality=modality, target_size=target_size,
        patient_ids=split["val"], slice_selection="tumor", simulate_rough=False, num_workers=4,
    )
    images, gts, _ = dataset.get_numpy_arrays()
    probs = []
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        batch = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
        probs.append(torch.sigmoid(model(batch).float()).squeeze(1).cpu().numpy())
    prob = np.concatenate(probs, axis=0)
    best_thr, best_dsc = 0.5, -1.0
    for thr in np.arange(0.30, 0.91, 0.05):
        scores = [measured_metrics((prob[i] > thr).astype(np.float32), gts[i])["dsc"] for i in range(len(gts))]
        dsc = float(np.mean(scores))
        if dsc > best_dsc:
            best_thr, best_dsc = round(float(thr), 2), dsc
    return best_thr


def main() -> None:
    parser = argparse.ArgumentParser(description="851명 환자 평균 베이스라인 평가")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_root", default="src/data/archive")
    parser.add_argument("--patient_split", default="checkpoints/patient_split.json")
    parser.add_argument("--modality", default="t1ce+flair")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, method, target_size = load_baseline(args.checkpoint, device)
    threshold = args.threshold
    if threshold is None:
        threshold = val_threshold(
            model, args.train_root, args.patient_split, args.modality, target_size, device, args.batch_size,
        )
        print(f"검증 60명에서 고른 임계값 {threshold:.2f}", flush=True)

    held = held_out_ids(args.train_root, args.patient_split)
    from src.data.brats2020_dataset import BraTS2020Dataset
    dataset = BraTS2020Dataset(
        root_dir=args.train_root, modality=args.modality, target_size=target_size,
        patient_ids=held, slice_selection="tumor", simulate_rough=False, num_workers=8,
    )
    rows = predict_rows(model, dataset, device, args.batch_size, threshold)
    summary = summarize(rows, args.seed)
    summary.update({
        "method": method,
        "checkpoint": args.checkpoint,
        "threshold": threshold,
        "n_patients_held_out": len(held),
        "n_slices": len(rows),
        "aggregation": "equal-weight patient means of 2D tumor-slice metrics; HD95 in resized pixels",
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    overall = summary["overall"]
    print(
        f"{method} thr={threshold:.2f} patients={overall['dsc']['n_patients']} "
        f"DSC={overall['dsc']['mean']:.4f} HD95={overall['hd95']['mean']:.3f}",
        flush=True,
    )
    print(f"저장: {args.out}", flush=True)


if __name__ == "__main__":
    main()
