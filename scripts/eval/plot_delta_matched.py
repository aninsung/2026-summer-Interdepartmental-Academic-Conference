"""851명 표의 클래스별 DSC·HD95 변화와 가까운 슬라이스를 그린다."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.brats2020_dataset import BraTS2020Dataset
from src.models.band_refine import build_band_refine, refine_batch
from src.models.dynamic_router import AdaptivePipeline
from src.utils.evaluation_records import measured_metrics
from src.utils.refinement_inputs import stage2_mask

OUT = "results/band_ppo_delta_matched"
PICKS = [
    ("small", "BraTS2021_01290", 96),
    ("small", "BraTS2021_00567", 93),
    ("medium", "BraTS2021_00249", 42),
    ("medium", "BraTS2021_01227", 118),
    ("large", "BraTS2021_00512", 75),
    ("large", "BraTS2021_01385", 79),
]
TITLES = {"small": "Small", "medium": "Medium", "large": "Large"}
ROUTE_NAMES = {0: "CaraNet", 1: "UNet++", 2: "SegResNet"}
THRESHOLDS = [0.80, 0.80, 0.50]
CC_SIZES = [0, 15, 25]


def tta(pipeline, image, base, route):
    with torch.no_grad():
        horizontal = torch.flip(pipeline(torch.flip(image, dims=[3]), true_class_preds=route)[0], dims=[3])
        vertical = torch.flip(pipeline(torch.flip(image, dims=[2]), true_class_preds=route)[0], dims=[2])
        return (base + horizontal + vertical) / 3.0


def crop_box(gt, margin=15):
    ys, xs = np.where(gt > 0.5)
    if len(ys) == 0:
        return 0, gt.shape[1] - 1, 0, gt.shape[0] - 1
    ymin = max(0, int(ys.min()) - margin)
    ymax = min(gt.shape[0] - 1, int(ys.max()) + margin)
    xmin = max(0, int(xs.min()) - margin)
    xmax = min(gt.shape[1] - 1, int(xs.max()) + margin)
    return xmin, xmax, ymin, ymax


def main():
    os.makedirs(OUT, exist_ok=True)
    pids = sorted({pid for _, pid, _ in PICKS})
    dataset = BraTS2020Dataset(
        root_dir="src/data/archive", modality="t1ce+flair", target_size=128,
        patient_ids=pids, slice_selection="tumor", simulate_rough=False, num_workers=1,
    )
    index = {(pid, int(z)): i for i, (pid, z) in enumerate(zip(dataset._sample_pids, dataset._sample_zs))}
    missing = [(pid, z) for _, pid, z in PICKS if (pid, z) not in index]
    if missing:
        raise SystemExit(f"missing slices {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, gt_masks, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch, strict_checkpoints=True)
    pipeline.eval()
    ckpt = torch.load("checkpoints/band_ppo.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(ckpt["model"])
    actor.eval()

    samples = []
    for cls, pid, z in PICKS:
        i = index[(pid, z)]
        image_25d = torch.from_numpy(np.ascontiguousarray(images_25d[i:i + 1])).to(device)
        with torch.no_grad():
            base, route = pipeline(image_25d)
            prob = tta(pipeline, image_25d, base, route)
        route_id = int(route.item())
        prob_np = prob.squeeze().detach().cpu().numpy()
        mask = (stage2_mask(prob_np, route_id, THRESHOLDS, CC_SIZES) > 0.5).astype(np.float32)
        center = torch.from_numpy(np.ascontiguousarray(images[i:i + 1])).to(device)
        mask_t = torch.from_numpy(mask).view(1, 1, *mask.shape).to(device)
        prob_t = torch.from_numpy(np.ascontiguousarray(prob_np)).view(1, 1, *prob_np.shape).to(device)
        flair = center[:, cfg["flair_index"]:cfg["flair_index"] + 1]
        with torch.no_grad():
            refined = refine_batch(
                actor, center, prob_t, mask_t, flair, int(cfg["n_steps"]),
                float(cfg["prob_lo"]), float(cfg["prob_hi"]), int(cfg["radius"]), guard_last=True,
            )[0, 0].detach().cpu().numpy()
        gt = gt_masks[i]
        stage2 = measured_metrics(mask, gt)
        pred = measured_metrics(refined, gt)
        samples.append({
            "cls": cls, "pid": pid, "z": z, "route": route_id,
            "img": images[i, 0] if images.ndim == 4 else images[i],
            "gt": gt, "rough": mask, "final": refined,
            "init_dsc": stage2["dsc"], "fin_dsc": pred["dsc"],
            "init_hd": stage2["hd95_surface_px"], "fin_hd": pred["hd95_surface_px"],
        })
        print(
            f"{cls} {pid} z={z} DSC {stage2['dsc']:.4f}->{pred['dsc']:.4f} "
            f"({pred['dsc'] - stage2['dsc']:+.4f}) "
            f"HD {stage2['hd95_surface_px']:.3f}->{pred['hd95_surface_px']:.3f} "
            f"({pred['hd95_surface_px'] - stage2['hd95_surface_px']:+.3f})"
        )

    fig, axes = plt.subplots(3, len(samples), figsize=(4.4 * len(samples), 14.5))
    for col, sample in enumerate(samples):
        xmin, xmax, ymin, ymax = crop_box(sample["gt"])
        delta_dsc = sample["fin_dsc"] - sample["init_dsc"]
        delta_hd = sample["fin_hd"] - sample["init_hd"]
        axes[0, col].imshow(sample["img"], cmap="gray", vmin=0, vmax=1)
        axes[0, col].contour(sample["gt"], levels=[0.5], colors="lime", linewidths=1.6)
        axes[0, col].set_title(
            f"{TITLES[sample['cls']]} · {ROUTE_NAMES[sample['route']]}\nSample {col % 2 + 1}",
            fontsize=20, fontweight="bold", pad=10,
        )
        axes[1, col].imshow(sample["img"], cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*sample["rough"].shape, 4))
        overlay[sample["rough"] > 0.5] = [1, 0, 0, 0.40]
        axes[1, col].imshow(overlay)
        axes[1, col].contour(sample["rough"], levels=[0.5], colors="red", linewidths=1.4)
        axes[1, col].contour(sample["gt"], levels=[0.5], colors="lime", linewidths=1.0, linestyles="--")
        axes[1, col].set_xlabel(
            f"DSC={sample['init_dsc']:.3f}\nHD95={sample['init_hd']:.2f} px",
            fontsize=18, fontweight="bold", labelpad=8,
        )
        axes[2, col].imshow(sample["img"], cmap="gray", vmin=0, vmax=1)
        axes[2, col].contour(sample["final"], levels=[0.5], colors="cyan", linewidths=1.6)
        axes[2, col].contour(sample["gt"], levels=[0.5], colors="lime", linewidths=1.0, linestyles="--")
        axes[2, col].set_xlabel(
            f"DSC={sample['fin_dsc']:.3f} (Δ={delta_dsc:+.3f})\n"
            f"HD95={sample['fin_hd']:.2f} px (Δ={delta_hd:+.3f})",
            fontsize=18, fontweight="bold", labelpad=8,
        )
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_xlim(xmin, xmax)
            axes[row, col].set_ylim(ymax, ymin)

    fig.text(0.008, 0.80, "MRI + GT", va="center", rotation="vertical", fontsize=22, fontweight="bold", color="green")
    fig.text(0.008, 0.51, "Stage 2", va="center", rotation="vertical", fontsize=22, fontweight="bold", color="red")
    fig.text(0.008, 0.20, "Stage 3", va="center", rotation="vertical", fontsize=22, fontweight="bold", color="darkcyan")
    plt.tight_layout(rect=[0.05, 0.02, 1, 0.98])
    fig.subplots_adjust(hspace=0.38, wspace=0.18)
    path = os.path.join(OUT, "delta_matched_comparison.png")
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close()
    records = []
    for sample in samples:
        records.append({
            key: (None if value is None else float(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in sample.items() if key not in {"img", "gt", "rough", "final"}
        })
    with open(os.path.join(OUT, "selected.json"), "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
