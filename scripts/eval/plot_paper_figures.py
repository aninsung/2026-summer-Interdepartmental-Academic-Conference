"""논문 그림 1·2.

그림 1의 DSC와 HD95는 본문 표 3과 같다.
그림 2에는 결과 표와 설명 문장을 넣지 않는다.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_FONT = "/tmp/fonts/NotoSansKR.otf"
if os.path.isfile(_FONT):
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS = os.path.join(ROOT, "results")


def _box(ax, x, y, w, h, fc, ec, title, body):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.4,
    ))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1c2430")
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
            fontsize=8.5, color="#34404c", linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color="#3d4754", shrinkA=2, shrinkB=2,
    ))


def draw_overview(path: str) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 4.6), dpi=150)
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 46)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    stages = [
        (2, "#e8eef6", "#3d5a80", "Input", "T1ce + FLAIR\n128 x 128"),
        (27, "#edf1f7", "#536b8a", "크기 분류", "ResNet-18\nsmall / medium / large"),
        (52, "#eef6f1", "#2d6a4f", "Experts", "CaraNet / UNet++\nSegResNet"),
        (77, "#fdecef", "#9b2335", "Band PPO", "5 steps\nband pixels only"),
        (102, "#fff6dc", "#9a7209", "Locked eval", "851 patients\nslice DSC, patient mean"),
    ]
    for x, fc, ec, title, body in stages:
        _box(ax, x, 24, 20, 16, fc, ec, title, body)
    for x in (22, 47, 72, 97):
        _arrow(ax, x, 32, x + 5, 32)

    _box(
        ax, 27, 4, 70, 14, "#f7f8fa", "#d5d9e0",
        "DSC  0.8337  →  0.8604    (+0.0267)",
        "HD95  4.888  →  4.609 px",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_internals(path: str) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.2), dpi=150)
    ax.set_xlim(0, 126)
    ax.set_ylim(0, 62)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(2, 59, "Inference", fontsize=12, fontweight="bold", color="#1c2430", va="center")
    infer = [
        (2, "#e8eef6", "#3d5a80", "Observation", "T1ce, FLAIR\nmask, probability"),
        (28, "#eef6f1", "#2d6a4f", "Band net", "width 32\n3 logits / pixel"),
        (54, "#fff6dc", "#9a7209", "Edit band", "[0.35, 0.65]\nor boundary ±2 px"),
        (80, "#fdecef", "#9b2335", "Last step", "FLAIR guard\non logits"),
        (106, "#fde8d8", "#c45c26", "Update", "argmax\n5 steps"),
    ]
    for x, fc, ec, title, body in infer:
        _box(ax, x, 40, 20, 14, fc, ec, title, body)
    for x in (22, 48, 74, 100):
        _arrow(ax, x, 47, x + 6, 47)

    ax.text(2, 32, "Training", fontsize=12, fontweight="bold", color="#1c2430", va="center")
    train = [
        (2, "#e8eef6", "#3d5a80", "Rollout", "batch 16\n5 steps"),
        (28, "#fde8d8", "#c45c26", "Reward", "GT match ±1\n+ 0.25 ΔHD95"),
        (54, "#eee8f7", "#5c4d7a", "PPO", "GAE on band\nclip 0.1"),
        (80, "#eef6f1", "#2d6a4f", "Update", "AdamW\nactor 1e-5"),
        (106, "#fff6dc", "#9a7209", "Select", "macro DSC, HD95\non validation"),
    ]
    for x, fc, ec, title, body in train:
        _box(ax, x, 8, 20, 14, fc, ec, title, body)
    for x in (22, 48, 74, 100):
        _arrow(ax, x, 15, x + 6, 15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    overview = os.path.join(RESULTS, "pipeline_overview.jpg")
    internals = os.path.join(RESULTS, "ppo_internal_route.jpg")
    draw_overview(overview)
    draw_internals(internals)
    draw_internals(os.path.join(RESULTS, "band_ppo_agent_internals.png"))
    draw_internals(os.path.join(RESULTS, "그림1.png"))
    print(overview)
    print(internals)


if __name__ == "__main__":
    main()
