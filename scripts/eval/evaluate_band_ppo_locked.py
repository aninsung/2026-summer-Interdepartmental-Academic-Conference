"""1251명 풀의 확정 평가.

방법 선택과 학습에 쓴 400명(train/val/test)은 빼다. 남은 환자만 분류기 라우팅으로
Stage 2와 경계 띠 PPO를 비교한다. 집계는 환자 평균이고 bootstrap은 환자 단위다.
임계값, 스텝 수, 가중치는 바꾸지 않는다.

사용 예:
  BRATS_NUM_WORKERS=16 python -u scripts/eval/evaluate_band_ppo_locked.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.patient_split import list_patient_ids
from src.models.band_refine import build_band_refine, refine_batch
from src.utils.evaluation_records import measured_metrics
from src.utils.metrics import gt_size_class
from src.utils.refinement_inputs import stage2_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = {0: "small", 1: "medium", 2: "large"}
THRESHOLDS = [0.80, 0.80, 0.50]
CC_SIZES = [0, 15, 25]


def bootstrap_ci(values: np.ndarray, seed: int = 42, repeats: int = 2000) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(repeats)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def held_out_ids(root: str, split_path: str) -> list[str]:
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    used = set(split["train"]) | set(split["val"]) | set(split["test"])
    held = sorted(set(list_patient_ids(root)) - used)
    if set(held) & used:
        raise SystemExit("확정 평가 환자가 개발 분할과 겹칩니다.")
    return held


def tta_probability(pipeline, image: torch.Tensor, base: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        horizontal = torch.flip(pipeline(torch.flip(image, dims=[3]), true_class_preds=route)[0], dims=[3])
        vertical = torch.flip(pipeline(torch.flip(image, dims=[2]), true_class_preds=route)[0], dims=[2])
        return (base + horizontal + vertical) / 3.0


def patient_means(rows: list[dict], key: str, class_of) -> dict[str, np.ndarray]:
    """환자마다 해당 키의 슬라이스 평균. class_of가 None이면 전체."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if class_of is not None and class_of(row) is None:
            continue
        grouped.setdefault(row["patient_id"], []).append(row[key])
    return {pid: float(np.mean(vals)) for pid, vals in grouped.items() if vals}


def pack_metric(stage2: dict[str, float], ppo: dict[str, float], seed: int) -> dict:
    ids = sorted(set(stage2) & set(ppo))
    base = np.array([stage2[i] for i in ids], dtype=float)
    pred = np.array([ppo[i] for i in ids], dtype=float)
    delta = pred - base
    return {
        "n_patients": len(ids),
        "stage2_mean": float(base.mean()) if len(ids) else float("nan"),
        "ppo_mean": float(pred.mean()) if len(ids) else float("nan"),
        "stage2_ci95": bootstrap_ci(base, seed),
        "ppo_ci95": bootstrap_ci(pred, seed),
        "delta_mean": float(delta.mean()) if len(ids) else float("nan"),
        "delta_ci95": bootstrap_ci(delta, seed),
    }


def summarize(rows: list[dict], seed: int) -> dict:
    def dsc_map(source: str, class_name: str | None, field: str):
        picked = rows if class_name is None else [r for r in rows if r[field] == class_name]
        return patient_means(picked, f"{source}_dsc", None)

    def hd_map(source: str, class_name: str | None, field: str):
        picked = []
        for row in rows:
            if class_name is not None and row[field] != class_name:
                continue
            if row[f"{source}_hd95"] is None:
                continue
            picked.append({**row, "hd": row[f"{source}_hd95"]})
        return patient_means(picked, "hd", None)

    def block(class_name: str | None, field: str) -> dict:
        dsc = pack_metric(dsc_map("stage2", class_name, field), dsc_map("ppo", class_name, field), seed)
        hd_stage = hd_map("stage2", class_name, field)
        hd_ppo = hd_map("ppo", class_name, field)
        hd = pack_metric(hd_stage, hd_ppo, seed + 1)
        accept = bool(
            dsc["n_patients"] > 0
            and dsc["ppo_mean"] >= dsc["stage2_mean"]
            and hd["ppo_mean"] < hd["stage2_mean"]
        )
        return {"dsc": dsc, "hd95": hd, "accept": accept}

    summary = {
        "overall": block(None, "gt_class"),
        "by_gt_class": {name: block(name, "gt_class") for name in ("small", "medium", "large")},
        "by_predicted_class": {name: block(name, "pred_class") for name in ("small", "medium", "large")},
    }
    summary["routing_slice_accuracy"] = float(np.mean([
        row["pred_class"] == row["gt_class"] for row in rows
    ])) if rows else float("nan")
    summary["paper_ok"] = bool(
        summary["overall"]["accept"]
        and all(summary["by_gt_class"][name]["accept"] for name in ("small", "medium", "large"))
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="1251명 풀의 미사용 환자 확정 평가")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/band_ppo.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/band_ppo_locked.json")
    parser.add_argument("--stage2_thresholds", type=str, default="0.80,0.80,0.50")
    args = parser.parse_args()
    thresholds = [float(t) for t in args.stage2_thresholds.split(",")]
    if len(thresholds) != 3:
        raise SystemExit("--stage2_thresholds 는 소형,중형,대형 3개 값이어야 합니다.")
    log.info("Stage 2 임계값 %s", thresholds)

    if not os.path.exists("checkpoints/shape_classifier_best.pt"):
        raise SystemExit("분류기 가중치가 없습니다: checkpoints/shape_classifier_best.pt")
    held = held_out_ids(args.train_root, args.patient_split)
    log.info("확정 평가 환자 %d명 (개발 400명 제외)", len(held))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.data.brats2020_dataset import BraTS2020Dataset
    dataset = BraTS2020Dataset(
        root_dir=args.train_root, modality="t1ce+flair", target_size=128,
        patient_ids=held, slice_selection="tumor", simulate_rough=False,
    )
    images, gt_masks, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    pids = list(dataset._sample_pids)
    log.info("종양 슬라이스 %d장", len(images))

    from src.models.dynamic_router import AdaptivePipeline
    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch, strict_checkpoints=True)
    pipeline.eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(ckpt["model"])
    actor.eval()

    rows = []
    for start in range(0, len(images), args.batch_size):
        stop = min(start + args.batch_size, len(images))
        image_25d = torch.from_numpy(np.ascontiguousarray(images_25d[start:stop])).to(device)
        with torch.no_grad():
            base, route = pipeline(image_25d)
            prob = tta_probability(pipeline, image_25d, base, route)
        prob_np = prob.squeeze(1).detach().cpu().numpy()
        route_np = route.detach().cpu().numpy()
        masks = []
        meta = []
        for offset, index in enumerate(range(start, stop)):
            cls = int(route_np[offset])
            mask = stage2_mask(prob_np[offset], cls, thresholds, CC_SIZES)
            masks.append((mask > 0.5).astype(np.float32))
            meta.append((index, cls))
        center = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
        mask_t = torch.from_numpy(np.stack(masks)).unsqueeze(1).to(device)
        prob_t = torch.from_numpy(np.ascontiguousarray(prob_np)).unsqueeze(1).to(device)
        flair = center[:, cfg["flair_index"]:cfg["flair_index"] + 1]
        refined = refine_batch(
            actor, center, prob_t, mask_t, flair, int(cfg["n_steps"]),
            float(cfg["prob_lo"]), float(cfg["prob_hi"]), int(cfg["radius"]), guard_last=True,
        ).detach().cpu().numpy()
        for local, (index, cls) in enumerate(meta):
            gt = gt_masks[index]
            stage2 = measured_metrics(masks[local], gt)
            pred = measured_metrics(refined[local, 0], gt)
            rows.append({
                "patient_id": pids[index],
                "gt_class": CLASS_NAMES[gt_size_class(gt)],
                "pred_class": CLASS_NAMES[cls],
                "stage2_dsc": stage2["dsc"],
                "ppo_dsc": pred["dsc"],
                "stage2_hd95": stage2["hd95_surface_px"],
                "ppo_hd95": pred["hd95_surface_px"],
            })
        if start % (args.batch_size * 50) == 0:
            log.info("평가 %d/%d", stop, len(images))

    del pipeline
    torch.cuda.empty_cache()
    summary = summarize(rows, args.seed)
    summary["n_patients_held_out"] = len(held)
    summary["n_slices"] = len(rows)
    summary["development_split"] = args.patient_split
    summary["development_excluded"] = 400
    summary["routing"] = "shape_classifier"
    summary["checkpoint"] = args.checkpoint
    summary["stage2_thresholds"] = thresholds
    summary["aggregation"] = "equal-weight patient means of 2D tumor-slice metrics; HD95 in resized pixels"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print_report(summary)
    log.info("저장: %s", args.out)


def print_report(summary: dict) -> None:
    print("\n=== 1251명 풀, 개발 400명 제외, 분류기 라우팅, 환자 평균 ===\n")
    print(f"{'구분':<16}{'환자':>6}{'Stage2 DSC':>12}{'PPO DSC':>10}{'Stage2 HD95':>13}{'PPO HD95':>10}")

    def line(name: str, block: dict) -> None:
        dsc, hd = block["dsc"], block["hd95"]
        print(f"{name:<16}{dsc['n_patients']:>6}{dsc['stage2_mean']:>12.4f}{dsc['ppo_mean']:>10.4f}"
              f"{hd['stage2_mean']:>13.3f}{hd['ppo_mean']:>10.3f}")

    line("overall", summary["overall"])
    for name in ("small", "medium", "large"):
        line(f"gt_{name}", summary["by_gt_class"][name])
    overall = summary["overall"]
    print(f"\n분류기 슬라이스 정확도 {summary['routing_slice_accuracy']:.4f}")
    print(f"DSC 차이 {overall['dsc']['delta_mean']:+.4f} CI95 {overall['dsc']['delta_ci95']}")
    print(f"HD95 차이 {overall['hd95']['delta_mean']:+.3f} CI95 {overall['hd95']['delta_ci95']}")
    print("논문 확정 표로 사용" if summary["paper_ok"] else "논문 확정 표로 쓰지 않음")


if __name__ == "__main__":
    main()
