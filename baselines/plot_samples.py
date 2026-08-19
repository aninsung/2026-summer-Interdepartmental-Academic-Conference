"""두 베이스라인의 분할 결과를 정성 비교 그림으로 만든다.

표본 선택 기준이 중요하다. 개선 폭이 가장 큰 슬라이스를 고르면 최선의 사례만
보여 주게 되므로, 여기서는 크기 구간별로 **DSC 중앙값 부근**을 고른다.
따라서 아래 그림은 "대표적인 성능"에 해당한다. `--pick worst` 로 하위 사례도
함께 확인할 수 있다.

사용 예시
    python baselines/plot_samples.py
    python baselines/plot_samples.py --pick worst --threshold 0.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.common import check_split_compatibility, simple_collate
from baselines.eval_baseline import load_baseline
from src.utils.metrics import dice, gt_size_class

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LABELS = {"kaist": "KAIST nnU-Net (1st)", "nvauto": "NVAUTO SegResNet (2nd)"}
COLORS = {"kaist": "#2b7fd4", "nvauto": "#e06c2a"}
CLASS_TITLES = {0: "Small", 1: "Medium", 2: "Large"}


def crop_box(gt: np.ndarray, margin: int = 15):
    ys, xs = np.where(gt > 0.5)
    if len(ys) == 0:
        return 0, gt.shape[1] - 1, 0, gt.shape[0] - 1
    return (
        max(0, xs.min() - margin),
        min(gt.shape[1] - 1, xs.max() + margin),
        max(0, ys.min() - margin),
        min(gt.shape[0] - 1, ys.max() + margin),
    )


@torch.no_grad()
def score_all(models: dict, loader: DataLoader, device, threshold: float):
    """슬라이스별 DSC와 크기 클래스를 모아 둔다(그림용 표본 선택에 사용)."""
    per_method: dict[str, list[float]] = {m: [] for m in models}
    classes: list[int] = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["gt_mask"].numpy()
        preds = {
            name: (torch.sigmoid(m(img).float()).cpu().numpy() > threshold).astype(np.float32)
            for name, m in models.items()
        }
        for i in range(gt.shape[0]):
            classes.append(gt_size_class(gt[i, 0]))
            for name in models:
                per_method[name].append(dice(preds[name][i, 0], gt[i, 0]))
    return per_method, np.array(classes)


def select_indices(per_method: dict, classes: np.ndarray, pick: str, n_per_class: int):
    """크기 구간별로 표본 인덱스를 고른다. 기본은 중앙값 부근."""
    mean_dsc = np.mean([np.array(v) for v in per_method.values()], axis=0)
    chosen: dict[int, list[int]] = {}
    for c in (0, 1, 2):
        idx = np.where(classes == c)[0]
        if len(idx) == 0:
            chosen[c] = []
            continue
        order = idx[np.argsort(mean_dsc[idx])]
        if pick == "worst":
            chosen[c] = list(order[:n_per_class])
        elif pick == "best":
            chosen[c] = list(order[::-1][:n_per_class])
        else:  # median
            mid = len(order) // 2
            start = max(0, mid - n_per_class // 2)
            chosen[c] = list(order[start : start + n_per_class])
    return chosen


@torch.no_grad()
def render(models, dataset, chosen, device, threshold, out_dir, pick):
    cols = [(c, i) for c in (0, 1, 2) for i in chosen[c]]
    if not cols:
        raise SystemExit("그릴 표본이 없습니다.")

    n_rows = 1 + len(models)
    fig, axes = plt.subplots(n_rows, len(cols), figsize=(3.1 * len(cols), 3.1 * n_rows))
    if len(cols) == 1:
        axes = axes[:, None]

    for col, (c, idx) in enumerate(cols):
        sample = dataset[idx]
        img_t = sample["image"].unsqueeze(0).to(device)
        gt = sample["gt_mask"].numpy()[0]
        img = sample["image"].numpy()[0]
        x0, x1, y0, y1 = crop_box(gt)

        ax = axes[0, col]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=2.0)
        ax.set_title(f"{CLASS_TITLES[c]}  (GT area={int(gt.sum())}px)", fontsize=11, fontweight="bold")

        for row, (name, model) in enumerate(models.items(), start=1):
            prob = torch.sigmoid(model(img_t).float()).cpu().numpy()[0, 0]
            pred = (prob > threshold).astype(np.float32)
            d = dice(pred, gt)

            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            overlay = np.zeros((*pred.shape, 4))
            rgb = matplotlib.colors.to_rgb(COLORS[name])
            overlay[pred > 0.5] = [*rgb, 0.45]
            ax.imshow(overlay)
            ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
            ax.set_xlabel(f"DSC={d:.3f}", fontsize=11, fontweight="bold")

        for row in range(n_rows):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_xlim(x0, x1)
            axes[row, col].set_ylim(y1, y0)

    fig.text(0.005, 1 - 0.5 / n_rows, "MRI + GT", va="center", rotation="vertical",
             fontsize=13, fontweight="bold", color="green")
    for row, name in enumerate(models, start=1):
        fig.text(0.005, 1 - (row + 0.5) / n_rows, LABELS[name], va="center",
                 rotation="vertical", fontsize=12, fontweight="bold", color=COLORS[name])

    pick_desc = {"median": "median DSC (representative)", "worst": "lowest DSC",
                 "best": "highest DSC"}[pick]
    fig.suptitle(
        f"BraTS 2021 top-2 baselines (2D adaptation) — samples at {pick_desc}, "
        f"threshold={threshold}\ndashed green = ground truth contour",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"baseline_samples_{pick}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("저장: %s", path)


def main():
    p = argparse.ArgumentParser(description="베이스라인 정성 비교 그림")
    p.add_argument("--checkpoints", type=str, nargs="+",
                   default=["baselines/checkpoints/kaist_best.pt",
                            "baselines/checkpoints/nvauto_best.pt"])
    p.add_argument("--train_root", type=str, default="src/data/archive")
    p.add_argument("--modality", type=str, default="t1ce+flair")
    p.add_argument("--max_patients", type=int, default=210)
    p.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pick", type=str, default="median", choices=["median", "worst", "best"])
    p.add_argument("--n_per_class", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="baselines/results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for ckpt in args.checkpoints:
        model, method, _ = load_baseline(ckpt, device)
        models[method] = model

    check_split_compatibility(args.train_root, args.max_patients, args.patient_split)

    from src.data.patient_split import load_split_brats_datasets

    _, val_ds = load_split_brats_datasets(
        train_root=args.train_root,
        modality=args.modality,
        target_size=128,
        max_patients=args.max_patients,
        patient_split=args.patient_split,
        refinement_mode=None,
        simulate_rough=False,
    )
    log.info("검증 슬라이스 %d개", len(val_ds))

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=simple_collate)

    per_method, classes = score_all(models, loader, device, args.threshold)
    for name, vals in per_method.items():
        log.info("%s 평균 DSC: %.4f", name, float(np.mean(vals)))

    chosen = select_indices(per_method, classes, args.pick, args.n_per_class)
    log.info("선택된 인덱스: %s", {CLASS_TITLES[c]: v for c, v in chosen.items()})
    render(models, val_ds, chosen, device, args.threshold, args.out_dir, args.pick)


if __name__ == "__main__":
    main()
