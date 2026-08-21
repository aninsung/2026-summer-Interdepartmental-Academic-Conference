"""베이스라인 평가 결과를 그림으로 만든다.

eval_baseline.py 가 저장한 JSON 만 읽으므로 재추론이 필요 없다.
논문/발표에 바로 쓸 수 있도록 라벨은 영문으로 둔다(한글 폰트 의존 제거).

사용 예시
    python baselines/plot_results.py
    python baselines/plot_results.py --methods kaist nvauto --out_dir baselines/results
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LABELS = {
    "kaist": "KAIST Extending nnU-Net (1st)",
    "nvauto": "NVAUTO SegResNet + RR (2nd)",
}
COLORS = {"kaist": "#3b6ea5", "nvauto": "#c4622d"}
SIZE_ORDER = ("Overall", "Small", "Medium", "Large")


def load(method: str, results_dir: str) -> dict:
    path = os.path.join(results_dir, f"{method}_metrics.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_by_size(data: dict[str, dict], out_path: str) -> None:
    """크기 구간별 DSC / Precision / Recall / HD95 막대 그래프."""
    metrics = [
        ("dsc", "Dice Similarity Coefficient", None),
        ("precision", "Precision", None),
        ("recall", "Recall", None),
        ("hd95", "HD95 (lower is better)", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    width = 0.38
    x = np.arange(len(SIZE_ORDER))

    for ax, (key, title, _) in zip(axes.ravel(), metrics):
        for i, (method, payload) in enumerate(data.items()):
            vals = [payload["results"][s][key] for s in SIZE_ORDER]
            offset = (i - (len(data) - 1) / 2) * width
            bars = ax.bar(
                x + offset, vals, width, label=LABELS.get(method, method),
                color=COLORS.get(method), edgecolor="white", linewidth=0.8,
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{s}\n(n={list(data.values())[0]['results'][s]['n']})" for s in SIZE_ORDER],
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if key == "hd95":
            ax.set_ylabel("pixels")
        else:
            ax.set_ylim(0.7, 1.02)

    axes[0, 0].legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "BraTS 2021 Top-2 Methods (2D adaptation) — Validation 42 patients / 2,373 slices\n"
        "binarization threshold = 0.5, identical patient split and metric implementation",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("저장: %s", out_path)


def plot_sweep(data: dict[str, dict], out_path: str) -> None:
    """임계값에 따른 DSC / Precision / Recall 곡선."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for key, ax, title in zip(
        ("dsc", "precision", "recall"), axes, ("DSC", "Precision", "Recall")
    ):
        for method, payload in data.items():
            sweep = payload.get("sweep")
            if not sweep:
                continue
            thrs = sorted(float(t) for t in sweep)
            vals = [sweep[f"{t}"][key] for t in thrs]
            ax.plot(
                thrs, vals, marker="o", markersize=4,
                label=LABELS.get(method, method), color=COLORS.get(method),
            )
            if key == "dsc":
                best = payload.get("best_threshold")
                if best is not None:
                    ax.axvline(best, color=COLORS.get(method), linestyle=":", alpha=0.7)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("binarization threshold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Threshold sensitivity — both baselines are already well calibrated "
        "(optimal threshold gains only +0.0009 / +0.0001 DSC)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("저장: %s", out_path)


def main():
    p = argparse.ArgumentParser(description="베이스라인 결과 시각화")
    p.add_argument("--methods", type=str, nargs="+", default=["kaist", "nvauto"])
    p.add_argument("--results_dir", type=str, default="baselines/results")
    p.add_argument("--out_dir", type=str, default="baselines/results")
    args = p.parse_args()

    data = {}
    for m in args.methods:
        try:
            data[m] = load(m, args.results_dir)
        except FileNotFoundError:
            log.warning("결과 파일이 없어 건너뜁니다: %s", m)
    if not data:
        raise SystemExit("읽을 결과 파일이 없습니다. eval_baseline.py 를 먼저 실행하세요.")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_by_size(data, os.path.join(args.out_dir, "baseline_by_size.png"))
    plot_sweep(data, os.path.join(args.out_dir, "baseline_threshold_sweep.png"))


if __name__ == "__main__":
    main()
