"""클래스 공통 경계 띠 PPO 평가.

클래스마다 Stage 2와 비교하고, DSC와 HD95가 둘 다 나은 클래스만 PPO를 쓴다.
그 혼합 결과의 전체 DSC가 Stage 2 이상이고 전체 HD95가 더 낮으면 채택이다.

사용 예:
  BRATS_NUM_WORKERS=16 python -u scripts/eval/evaluate_band_ppo.py \
      --split_role test --checkpoint checkpoints/band_ppo.pt
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
from src.models.band_refine import build_band_refine, refine_batch
from src.utils.evaluation_records import measured_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_INDEX = {"small": 0, "medium": 1, "large": 2}
CLASS_NAMES = {0: "small", 1: "medium", 2: "large"}


def _block(rows: list[dict], key: str) -> dict:
    dsc = np.array([row[key]["dsc"] for row in rows], dtype=np.float64)
    hd = [row[key]["hd95_surface_px"] for row in rows if row[key]["hd95_surface_px"] is not None]
    prec = np.array([row[key]["precision"] for row in rows], dtype=np.float64)
    rec = np.array([row[key]["recall"] for row in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "dsc": float(dsc.mean()) if len(rows) else float("nan"),
        "hd95_surface_px": float(np.mean(hd)) if hd else float("nan"),
        "precision": float(prec.mean()) if len(rows) else float("nan"),
        "recall": float(rec.mean()) if len(rows) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="경계 띠 PPO 평가")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--split_role", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/band_ppo.pt")
    parser.add_argument("--eval_batch", type=int, default=32)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--out", type=str, default="results/band_ppo.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.patient_split, "r", encoding="utf-8") as f:
        split = json.load(f)
    from src.data.brats2020_dataset import BraTS2020Dataset
    dataset = BraTS2020Dataset(
        root_dir=args.train_root, modality=args.modality, target_size=args.target_size,
        patient_ids=split[args.split_role], slice_selection="tumor", simulate_rough=False,
    )
    from src.models.dynamic_router import AdaptivePipeline
    images, _, _ = dataset.get_numpy_arrays()
    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch, strict_checkpoints=True)
    pipeline.eval()
    entries = build_stage2_entries(dataset, pipeline, device, (0, 1, 2), use_tta=not args.no_tta)
    del pipeline
    torch.cuda.empty_cache()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(ckpt["model"])
    actor.eval()

    rows = []
    for start in range(0, len(entries), args.eval_batch):
        chunk = entries[start:start + args.eval_batch]
        image = torch.stack([
            torch.from_numpy(np.asarray(e.image, dtype=np.float32)) for e in chunk
        ]).to(device)
        prob = torch.stack([
            torch.from_numpy(np.asarray(e.probability, dtype=np.float32)) for e in chunk
        ]).unsqueeze(1).to(device)
        mask = torch.stack([
            torch.from_numpy(np.asarray(e.stage2, dtype=np.float32)) for e in chunk
        ]).unsqueeze(1).to(device)
        flair = image[:, cfg["flair_index"]:cfg["flair_index"] + 1]
        pred = refine_batch(
            actor, image, prob, mask, flair, cfg["n_steps"],
            cfg["prob_lo"], cfg["prob_hi"], cfg["radius"], guard_last=True,
        ).detach().cpu().numpy()
        for index, entry in enumerate(chunk):
            rows.append({
                "route": entry.route,
                "patient_id": entry.patient_id,
                "stage2": measured_metrics(entry.stage2, entry.gt),
                "ppo": measured_metrics(pred[index, 0], entry.gt),
            })

    by_class = defaultdict(list)
    for row in rows:
        by_class[row["route"]].append(row)

    summary = {"classes": {}, "accepted": []}
    for route in sorted(by_class):
        name = CLASS_NAMES[route]
        stage2 = _block(by_class[route], "stage2")
        ppo = _block(by_class[route], "ppo")
        accept = ppo["dsc"] >= stage2["dsc"] and ppo["hd95_surface_px"] < stage2["hd95_surface_px"]
        summary["classes"][name] = {"stage2": stage2, "ppo": ppo, "accept": accept}
        if accept:
            summary["accepted"].append(name)

    accepted = {CLASS_INDEX[name] for name in summary["accepted"]}
    mixed_rows = []
    for row in rows:
        chosen = row["ppo"] if row["route"] in accepted else row["stage2"]
        mixed_rows.append({"mixed": chosen, "stage2": row["stage2"]})
    summary["overall_stage2"] = _block([{"stage2": r["stage2"]} for r in rows], "stage2")
    summary["overall_ppo"] = _block(rows, "ppo")
    summary["overall_mixed"] = _block(mixed_rows, "mixed")
    mixed = summary["overall_mixed"]
    base = summary["overall_stage2"]
    summary["promote"] = bool(
        mixed["dsc"] >= base["dsc"] and mixed["hd95_surface_px"] < base["hd95_surface_px"]
    )
    summary["checkpoint"] = args.checkpoint
    summary["split_role"] = args.split_role
    summary["saved_without_gate"] = bool(ckpt.get("saved_without_gate", False))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print("\n=== 클래스 공통 경계 띠 PPO ===\n")
    print(f"{'클래스':<9}{'n':>7}{'Stage2 DSC':>12}{'PPO DSC':>10}{'Stage2 HD95':>13}{'PPO HD95':>10}{'채택':>6}")
    for name in ("small", "medium", "large"):
        if name not in summary["classes"]:
            continue
        item = summary["classes"][name]
        stage2, ppo = item["stage2"], item["ppo"]
        print(f"{name:<9}{stage2['n']:>7}{stage2['dsc']:>12.4f}{ppo['dsc']:>10.4f}"
              f"{stage2['hd95_surface_px']:>13.3f}{ppo['hd95_surface_px']:>10.3f}"
              f"{'예' if item['accept'] else '아니오':>6}")
    print(f"\n혼합 DSC {mixed['dsc']:.4f} (Stage2 {base['dsc']:.4f})  "
          f"HD95 {mixed['hd95_surface_px']:.3f} (Stage2 {base['hd95_surface_px']:.3f})")
    print("Stage 3 교체" if summary["promote"] else "Stage 3 유지")
    log.info("저장: %s", args.out)


if __name__ == "__main__":
    main()
