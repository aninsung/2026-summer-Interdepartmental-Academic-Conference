"""임계값을 검증 60명으로 다시 고른 Stage 2 대조.

검증 60명에서 분류기 라우팅 기준 크기별 임계값 세 개를 환자 평균 DSC로 함께 고른다.
851명에서 고정 Stage 2, 재선택 Stage 2, 각각에 경계 띠 PPO를 얹은 결과를 환자 짝으로 비교한다.
PPO 가중치는 고정 임계값 마스크로 학습했다.

사용 예:
  python -u scripts/eval/evaluate_stage2_retuned.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.eval.evaluate_band_ppo_locked import (
    CC_SIZES,
    CLASS_NAMES,
    THRESHOLDS,
    bootstrap_ci,
    held_out_ids,
    tta_probability,
)
from src.models.band_refine import build_band_refine, refine_batch
from src.utils.evaluation_records import measured_metrics
from src.utils.metrics import dice, gt_size_class
from src.utils.refinement_inputs import stage2_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)
VARIANTS = ("stage2_fixed", "stage2_tuned", "ppo_fixed", "ppo_tuned")
CONTRASTS = (
    ("ppo_fixed", "stage2_tuned"),
    ("ppo_tuned", "stage2_tuned"),
    ("stage2_tuned", "stage2_fixed"),
    ("ppo_fixed", "stage2_fixed"),
)


def load_dataset(root: str, patient_ids: list[str]):
    from src.data.brats2020_dataset import BraTS2020Dataset
    dataset = BraTS2020Dataset(
        root_dir=root, modality="t1ce+flair", target_size=128,
        patient_ids=patient_ids, slice_selection="tumor", simulate_rough=False,
    )
    images, gts, _ = dataset.get_numpy_arrays()
    return images, dataset.get_numpy_25d_arrays(), gts, list(dataset._sample_pids)


def predict(pipeline, images_25d: np.ndarray, start: int, stop: int, device):
    image_25d = torch.from_numpy(np.ascontiguousarray(images_25d[start:stop])).to(device)
    with torch.no_grad():
        base, route = pipeline(image_25d)
        prob = tta_probability(pipeline, image_25d, base, route)
    return prob.squeeze(1).detach().cpu().numpy(), route.detach().cpu().numpy().astype(int)


def select_thresholds(pipeline, root: str, val_ids: list[str], batch_size: int, device) -> tuple[list[float], float]:
    _, images_25d, gts, pids = load_dataset(root, val_ids)
    n = len(gts)
    table = np.zeros((n, len(GRID)), dtype=np.float64)
    routes = np.zeros(n, dtype=int)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        prob_np, route_np = predict(pipeline, images_25d, start, stop, device)
        for offset, index in enumerate(range(start, stop)):
            cls = int(route_np[offset])
            routes[index] = cls
            for g, thr in enumerate(GRID):
                thresholds = list(THRESHOLDS)
                thresholds[cls] = float(thr)
                table[index, g] = dice(stage2_mask(prob_np[offset], cls, thresholds, CC_SIZES), gts[index])

    _, patient_index = np.unique(pids, return_inverse=True)
    counts = np.bincount(patient_index)
    rows = np.arange(n)
    best, best_score = None, -1.0
    for combo in itertools.product(range(len(GRID)), repeat=3):
        slice_dsc = table[rows, np.asarray(combo)[routes]]
        score = float((np.bincount(patient_index, weights=slice_dsc) / counts).mean())
        if score > best_score:
            best, best_score = combo, score
    chosen = [float(GRID[i]) for i in best]
    fixed_idx = [int(np.argmin(np.abs(GRID - t))) for t in THRESHOLDS]
    fixed_score = float((np.bincount(patient_index, weights=table[rows, np.asarray(fixed_idx)[routes]]) / counts).mean())
    log.info("검증 60명 %d장: 고정 %s DSC %.4f → 재선택 %s DSC %.4f", n, THRESHOLDS, fixed_score, chosen, best_score)
    for cls, i in enumerate(best):
        if i in (0, len(GRID) - 1):
            log.warning("%s 임계값 %.2f 가 탐색 범위 끝입니다", CLASS_NAMES[cls], GRID[i])
    return chosen, best_score


def patient_mean(rows: list[dict], key: str, class_name: str | None) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if class_name is not None and row["gt_class"] != class_name:
            continue
        if row[key] is None:
            continue
        grouped.setdefault(row["patient_id"], []).append(row[key])
    return {pid: float(np.mean(v)) for pid, v in grouped.items()}


def block(rows: list[dict], metric: str, class_name: str | None, seed: int) -> dict:
    maps = {v: patient_mean(rows, f"{v}_{metric}", class_name) for v in VARIANTS}
    out = {}
    for v in VARIANTS:
        arr = np.array(list(maps[v].values()), dtype=float)
        out[v] = {"n_patients": len(arr), "mean": float(arr.mean()), "ci95": bootstrap_ci(arr, seed)}
    for a, b in CONTRASTS:
        ids = sorted(set(maps[a]) & set(maps[b]))
        delta = np.array([maps[a][i] - maps[b][i] for i in ids], dtype=float)
        out[f"{a}-{b}"] = {"n_patients": len(ids), "mean": float(delta.mean()), "ci95": bootstrap_ci(delta, seed)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="검증 60명 임계값 재선택 Stage 2 대조")
    parser.add_argument("--train_root", default="src/data/archive")
    parser.add_argument("--patient_split", default="checkpoints/patient_split.json")
    parser.add_argument("--checkpoint", default="checkpoints/band_ppo.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/stage2_retuned.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.patient_split, encoding="utf-8") as f:
        split = json.load(f)
    held = held_out_ids(args.train_root, args.patient_split)

    from src.models.dynamic_router import AdaptivePipeline
    pipeline = AdaptivePipeline(device, in_channels=2, strict_checkpoints=True)
    pipeline.eval()
    tuned, val_score = select_thresholds(pipeline, args.train_root, split["val"], args.batch_size, device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(ckpt["model"])
    actor.eval()

    images, images_25d, gts, pids = load_dataset(args.train_root, held)
    log.info("확정 평가 %d명 %d장", len(held), len(gts))

    def refine(center, prob_np, masks):
        return refine_batch(
            actor, center,
            torch.from_numpy(np.ascontiguousarray(prob_np)).unsqueeze(1).to(device),
            torch.from_numpy(np.stack(masks).astype(np.float32)).unsqueeze(1).to(device),
            center[:, cfg["flair_index"]:cfg["flair_index"] + 1],
            int(cfg["n_steps"]), float(cfg["prob_lo"]), float(cfg["prob_hi"]), int(cfg["radius"]),
            guard_last=True,
        ).detach().cpu().numpy()[:, 0]

    rows = []
    for start in range(0, len(gts), args.batch_size):
        stop = min(start + args.batch_size, len(gts))
        prob_np, route_np = predict(pipeline, images_25d, start, stop, device)
        fixed = [stage2_mask(prob_np[i], int(route_np[i]), THRESHOLDS, CC_SIZES) for i in range(stop - start)]
        retuned = [stage2_mask(prob_np[i], int(route_np[i]), tuned, CC_SIZES) for i in range(stop - start)]
        center = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
        outputs = {
            "stage2_fixed": fixed,
            "stage2_tuned": retuned,
            "ppo_fixed": refine(center, prob_np, fixed),
            "ppo_tuned": refine(center, prob_np, retuned),
        }
        for local, index in enumerate(range(start, stop)):
            row = {"patient_id": pids[index], "gt_class": CLASS_NAMES[gt_size_class(gts[index])]}
            for name, masks in outputs.items():
                m = measured_metrics(masks[local], gts[index])
                row[f"{name}_dsc"] = m["dsc"]
                row[f"{name}_hd95"] = m["hd95_surface_px"]
                row[f"{name}_p"] = m["precision"]
                row[f"{name}_r"] = m["recall"]
            rows.append(row)
        if start % (args.batch_size * 250) == 0:
            log.info("평가 %d/%d", stop, len(gts))

    summary = {
        "thresholds_fixed": THRESHOLDS,
        "thresholds_tuned": tuned,
        "threshold_grid": [float(g) for g in GRID],
        "val_patient_mean_dsc_tuned": val_score,
        "n_patients_held_out": len(held),
        "n_slices": len(rows),
        "dsc": {name or "overall": block(rows, "dsc", name, args.seed) for name in (None, "small", "medium", "large")},
        "precision": {"overall": block(rows, "p", None, args.seed)},
        "recall": {"overall": block(rows, "r", None, args.seed)},
        "note": "PPO weights were trained on fixed-threshold Stage 2 masks.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print(f"\n임계값 고정 {THRESHOLDS}  재선택 {tuned}")
    print(f"{'구간':<9}" + "".join(f"{v:>14}" for v in VARIANTS))
    for name, b in summary["dsc"].items():
        print(f"{name:<9}" + "".join(f"{b[v]['mean']:>14.4f}" for v in VARIANTS))
    for a, b in CONTRASTS:
        d = summary["dsc"]["overall"][f"{a}-{b}"]
        print(f"{a} - {b}: {d['mean']:+.4f} CI95 {d['ci95']}")
    for key in ("precision", "recall"):
        b = summary[key]["overall"]
        print(key, " ".join(f"{v}={b[v]['mean']:.4f}" for v in VARIANTS))
    log.info("저장: %s", args.out)


if __name__ == "__main__":
    main()
