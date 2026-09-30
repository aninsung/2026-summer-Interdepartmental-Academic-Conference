"""현재 경계 띠 PPO와 재학습한 KAIST, NVAUTO를 같은 슬라이스 3×6으로 그린다.

행 이름은 기존 비교 그림과 같이 TRIO, KAIST, NVAUTO다.
TRIO 칸은 분류기 라우팅과 임계값 0.80/0.80/0.50의 경계 띠 PPO다.
슬라이스는 개발 400명을 뺀 환자에서 고르고, 각 크기 구간의 평균 DSC에 가까운 장이다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baselines.eval_baseline import load_baseline
from baselines.eval_heldout import val_threshold
from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import list_patient_ids
from src.models.band_refine import build_band_refine, refine_batch
from src.models.dynamic_router import AdaptivePipeline
from src.utils.metrics import dice, gt_size_class
from src.utils.refinement_inputs import stage2_mask


THRESHOLDS = [0.80, 0.80, 0.50]
CC_SIZES = [0, 15, 25]
CLASS_NAMES = {0: "Small", 1: "Medium", 2: "Large"}


def held_out_ids(root: str, split_path: str) -> list[str]:
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    used = set(split["train"]) | set(split["val"]) | set(split["test"])
    return sorted(set(list_patient_ids(root)) - used)


def tta_probability(pipeline, image: torch.Tensor, base: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        horizontal = torch.flip(pipeline(torch.flip(image, dims=[3]), true_class_preds=route)[0], dims=[3])
        vertical = torch.flip(pipeline(torch.flip(image, dims=[2]), true_class_preds=route)[0], dims=[2])
        return (base + horizontal + vertical) / 3.0
ROW_TITLES = ["TRIO", "KAIST", "NVAUTO"]
COLORS = {"pipe": "#00bcd4", "kaist": "#2b7fd4", "nvauto": "#e06c2a"}


def crop_box(gt: np.ndarray, margin: int = 15):
    ys, xs = np.where(gt > 0.5)
    if len(ys) == 0:
        return 0, gt.shape[1] - 1, 0, gt.shape[0] - 1
    return (
        max(0, int(xs.min()) - margin),
        min(gt.shape[1] - 1, int(xs.max()) + margin),
        max(0, int(ys.min()) - margin),
        min(gt.shape[0] - 1, int(ys.max()) + margin),
    )


def render(rows, out_path: str) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(22, 12.5))
    fig.suptitle(
        "Representative slices near class-mean DSC  ·  dashed green = GT",
        fontsize=22,
        fontweight="bold",
        color="black",
        y=0.98,
    )
    keys = ["pipe", "kaist", "nvauto"]
    dkeys = ["d_pipe", "d_kaist", "d_nvauto"]
    colors = [COLORS["pipe"], COLORS["kaist"], COLORS["nvauto"]]
    for col, sample in enumerate(rows):
        img, gt = sample["img"], sample["gt"]
        x0, x1, y0, y1 = crop_box(gt)
        sample_i = 1 if col % 2 == 0 else 2
        for row in range(3):
            ax = axes[row, col]
            mask = sample[keys[row]]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask > 0.5] = [*matplotlib.colors.to_rgb(colors[row]), 0.42]
            ax.imshow(overlay)
            ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.3, linestyles="--")
            ax.contour(mask, levels=[0.5], colors=[colors[row]], linewidths=1.6)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    f"{CLASS_NAMES[sample['gt_c']]} {sample_i}",
                    fontsize=20,
                    fontweight="bold",
                    color="black",
                    pad=10,
                )
            ax.text(
                0.5,
                -0.06,
                f"DSC={sample[dkeys[row]]:.3f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
                color="black",
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.6),
            )
            if col == 0:
                ax.set_ylabel(
                    ROW_TITLES[row],
                    fontsize=22,
                    fontweight="bold",
                    color="black",
                    labelpad=14,
                )
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"저장: {out_path}", flush=True)


def choose(gt_cls, d_pipe, d_k, d_n, pids, n_per: int = 2) -> list[int]:
    chosen = []
    for cls in (0, 1, 2):
        idx = np.where(gt_cls == cls)[0]
        target = np.array([
            float(d_pipe[idx].mean()),
            float(d_k[idx].mean()),
            float(d_n[idx].mean()),
        ])
        dist = (
            np.abs(d_pipe[idx] - target[0])
            + np.abs(d_k[idx] - target[1])
            + np.abs(d_n[idx] - target[2])
        )
        order = idx[np.argsort(dist)]
        picked, used = [], set()
        for index in order:
            pid = pids[int(index)]
            if pid in used:
                continue
            picked.append(int(index))
            used.add(pid)
            if len(picked) >= n_per:
                break
        chosen.extend(picked)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", default="src/data/archive")
    parser.add_argument("--patient_split", default="checkpoints/patient_split.json")
    parser.add_argument("--modality", default="t1ce+flair")
    parser.add_argument("--kaist", default="baselines/checkpoints/kaist_best.pt")
    parser.add_argument("--nvauto", default="baselines/checkpoints/nvauto_best.pt")
    parser.add_argument("--band", default="checkpoints/band_ppo.pt")
    parser.add_argument("--sample_patients", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", default="results/method_comparison_current_3x6.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    held = held_out_ids(args.train_root, args.patient_split)
    rng = np.random.default_rng(args.seed)
    sample = sorted(rng.choice(held, size=min(args.sample_patients, len(held)), replace=False).tolist())
    print(f"held-out {len(held)}명 중 {len(sample)}명", flush=True)

    kaist, _, _ = load_baseline(args.kaist, device)
    nvauto, _, _ = load_baseline(args.nvauto, device)
    thr_k = val_threshold(kaist, args.train_root, args.patient_split, args.modality, 128, device, 16)
    thr_n = val_threshold(nvauto, args.train_root, args.patient_split, args.modality, 128, device, 16)
    print(f"검증 임계값 KAIST {thr_k:.2f}  NVAUTO {thr_n:.2f}", flush=True)

    dataset = BraTS2020Dataset(
        root_dir=args.train_root,
        modality=args.modality,
        target_size=128,
        patient_ids=sample,
        slice_selection="tumor",
        simulate_rough=False,
        num_workers=4,
    )
    images, gts, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    pids = list(dataset._sample_pids)
    n = len(gts)
    gt_cls = np.array([gt_size_class(g) for g in gts], dtype=np.int32)
    print(f"슬라이스 {n}  small {(gt_cls==0).sum()} medium {(gt_cls==1).sum()} large {(gt_cls==2).sum()}", flush=True)

    d_k = np.zeros(n, dtype=np.float32)
    d_n = np.zeros(n, dtype=np.float32)
    pred_k = np.zeros((n, 128, 128), dtype=np.float32)
    pred_n = np.zeros((n, 128, 128), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, 16):
            stop = min(start + 16, n)
            batch = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
            pk = torch.sigmoid(kaist(batch).float()).squeeze(1)
            pn = torch.sigmoid(nvauto(batch).float()).squeeze(1)
            pred_k[start:stop] = (pk > thr_k).cpu().numpy()
            pred_n[start:stop] = (pn > thr_n).cpu().numpy()
    for i in range(n):
        d_k[i] = dice(pred_k[i], gts[i])
        d_n[i] = dice(pred_n[i], gts[i])
    del kaist, nvauto
    torch.cuda.empty_cache()

    in_ch = images_25d.shape[1]
    pipeline = AdaptivePipeline(device, in_channels=in_ch, strict_checkpoints=True)
    pipeline.eval()
    ckpt = torch.load(args.band, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(ckpt["model"])
    actor.eval()

    d_pipe = np.zeros(n, dtype=np.float32)
    pred_pipe = np.zeros((n, 128, 128), dtype=np.float32)
    for start in range(0, n, args.batch_size):
        stop = min(start + args.batch_size, n)
        image_25d = torch.from_numpy(np.ascontiguousarray(images_25d[start:stop])).to(device)
        with torch.no_grad():
            base, route = pipeline(image_25d)
            prob = tta_probability(pipeline, image_25d, base, route)
        prob_np = prob.squeeze(1).detach().cpu().numpy()
        route_np = route.detach().cpu().numpy()
        masks = [
            stage2_mask(prob_np[offset], int(route_np[offset]), THRESHOLDS, CC_SIZES)
            for offset in range(stop - start)
        ]
        center = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
        mask_t = torch.from_numpy(np.stack(masks).astype(np.float32)).unsqueeze(1).to(device)
        prob_t = torch.from_numpy(np.ascontiguousarray(prob_np)).unsqueeze(1).to(device)
        flair = center[:, cfg["flair_index"]:cfg["flair_index"] + 1]
        refined = refine_batch(
            actor, center, prob_t, mask_t, flair, int(cfg["n_steps"]),
            float(cfg["prob_lo"]), float(cfg["prob_hi"]), int(cfg["radius"]), guard_last=True,
        ).detach().cpu().numpy()
        pred_pipe[start:stop] = refined[:, 0]
        if start % (args.batch_size * 40) == 0:
            print(f"TRIO {stop}/{n}", flush=True)
    for i in range(n):
        d_pipe[i] = dice(pred_pipe[i], gts[i])

    chosen = choose(gt_cls, d_pipe, d_k, d_n, pids)
    print("선택", chosen, flush=True)
    rows = []
    for index in chosen:
        center = images[index]
        rows.append({
            "img": center[0] if center.ndim == 3 else center,
            "gt": gts[index],
            "gt_c": int(gt_cls[index]),
            "pipe": pred_pipe[index],
            "kaist": pred_k[index],
            "nvauto": pred_n[index],
            "d_pipe": float(d_pipe[index]),
            "d_kaist": float(d_k[index]),
            "d_nvauto": float(d_n[index]),
        })
        print(
            f"{CLASS_NAMES[int(gt_cls[index])]} pid={pids[index]} "
            f"TRIO {d_pipe[index]:.3f} KAIST {d_k[index]:.3f} NVAUTO {d_n[index]:.3f}",
            flush=True,
        )
    render(rows, args.out)


if __name__ == "__main__":
    main()
