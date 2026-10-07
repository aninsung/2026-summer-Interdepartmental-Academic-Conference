"""U-Net만과 U-Net+바닐라 PPO를 미사용 851명에서 비교한다.

집계는 종양 슬라이스 DSC의 환자 평균이다. 초기 마스크 임계값은 학습과 같은 0.5다.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.eval.evaluate_band_ppo_locked import bootstrap_ci, held_out_ids
from scripts.eval.plot_single_backbone_grid import build_model, predict_prob
from src.data.brats2020_dataset import BraTS2020Dataset
from src.utils.evaluation_records import measured_metrics
from src.utils.metrics import gt_size_class

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MAX_STEPS = 15
BATCH = 32


def apply_action(mask: np.ndarray, action: int) -> np.ndarray:
    if action == 0:
        return binary_erosion(mask > 0.5, iterations=1).astype(np.float32)
    if action == 2:
        return binary_dilation(mask > 0.5, iterations=1).astype(np.float32)
    return (mask > 0.5).astype(np.float32)


def refine(agent, images: np.ndarray, prob: np.ndarray, rough: np.ndarray) -> np.ndarray:
    mask = rough.copy()
    for _ in range(MAX_STEPS):
        out = np.empty_like(mask)
        for start in range(0, len(mask), BATCH):
            stop = min(start + BATCH, len(mask))
            obs = np.concatenate([
                images[start:stop],
                mask[start:stop, None],
                prob[start:stop, None],
            ], axis=1).astype(np.float32)
            actions, _ = agent.predict(obs, deterministic=True)
            for offset, action in enumerate(np.atleast_1d(actions)):
                out[start + offset] = apply_action(mask[start + offset], int(action))
        mask = out
    return mask


def patient_mean(rows: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        value = row[key]
        if value is None:
            continue
        grouped[row["patient_id"]].append(value)
    values = np.array([float(np.mean(v)) for v in grouped.values()], dtype=float)
    if len(values) == 0:
        return {"n_patients": 0, "mean": None, "ci95": None}
    return {"n_patients": int(len(values)), "mean": float(values.mean()), "ci95": bootstrap_ci(values)}


def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    held = held_out_ids("src/data/archive", "checkpoints/patient_split.json")
    dataset = BraTS2020Dataset(
        root_dir="src/data/archive", modality="t1ce+flair", target_size=128,
        patient_ids=held, slice_selection="tumor", simulate_rough=False, num_workers=8,
    )
    print(f"held {len(held)} slices {len(dataset)}", flush=True)
    groups = defaultdict(list)
    for index, pid in enumerate(dataset._sample_pids):
        groups[pid].append(index)

    sample = dataset._samples[groups[held[0]][0]][0]
    model = build_model("unet", sample.shape[0], device)
    model.load_state_dict(torch.load("checkpoints/single_backbone/unet.pt", map_location=device, weights_only=True))
    model.eval()
    agent = PPO.load("checkpoints/single_backbone/ppo_unet_vanilla", device=str(device))

    rows = []
    for patient_index, pid in enumerate(held):
        indices = groups.get(pid, [])
        if not indices:
            continue
        images = np.stack([dataset._samples[i][0] for i in indices]).astype(np.float32)
        gts = np.stack([dataset._samples[i][1] for i in indices]).astype(np.float32)
        for i in indices:
            dataset._samples[i] = None
        prob = predict_prob(model, images, device, batch_size=BATCH)
        rough = (prob > 0.5).astype(np.float32)
        refined = refine(agent, images, prob, rough)
        for index in range(len(indices)):
            gt = gts[index]
            base = measured_metrics(rough[index], gt)
            ppo = measured_metrics(refined[index], gt)
            rows.append({
                "patient_id": pid,
                "gt_class": gt_size_class(gt),
                "unet_dsc": base["dsc"],
                "ppo_dsc": ppo["dsc"],
                "unet_hd95": base["hd95_surface_px"] if base["predicted_pixels"] > 0 and base["gt_pixels"] > 0 else None,
                "ppo_hd95": ppo["hd95_surface_px"] if ppo["predicted_pixels"] > 0 and ppo["gt_pixels"] > 0 else None,
            })
        if (patient_index + 1) % 20 == 0:
            print(f"평가 {patient_index + 1}/{len(held)} {pid}", flush=True)
    dataset._samples.clear()

    paired = []
    by_patient = defaultdict(lambda: {"unet": [], "ppo": []})
    for row in rows:
        by_patient[row["patient_id"]]["unet"].append(row["unet_dsc"])
        by_patient[row["patient_id"]]["ppo"].append(row["ppo_dsc"])
    for bucket in by_patient.values():
        paired.append(float(np.mean(bucket["ppo"]) - np.mean(bucket["unet"])))
    paired = np.array(paired, dtype=float)

    by_class = {}
    for name, code in (("small", 0), ("medium", 1), ("large", 2)):
        subset = [row for row in rows if row["gt_class"] == code]
        by_class[name] = {
            "unet": patient_mean(subset, "unet_dsc"),
            "vanilla_ppo": patient_mean(subset, "ppo_dsc"),
        }
    summary = {
        "split": "개발 400명을 뺀 851명, 종양 슬라이스 DSC의 환자 평균",
        "threshold": 0.5,
        "max_steps": MAX_STEPS,
        "n_slices": len(rows),
        "unet": patient_mean(rows, "unet_dsc"),
        "vanilla_ppo": patient_mean(rows, "ppo_dsc"),
        "paired_ppo_minus_unet": {
            "n_patients": int(len(paired)),
            "mean": float(paired.mean()),
            "ci95": bootstrap_ci(paired),
        },
        "hd95_defined_slices": {
            "unet": patient_mean(rows, "unet_hd95"),
            "vanilla_ppo": patient_mean(rows, "ppo_hd95"),
        },
        "by_gt_class": by_class,
    }
    dest = "results/vanilla_unet_851.json"
    os.makedirs("results", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"저장 {dest}", flush=True)


if __name__ == "__main__":
    main()
