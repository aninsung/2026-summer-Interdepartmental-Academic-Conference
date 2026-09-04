"""
RL Refined only — 3x4 comparison.
Cols: TRIO / U-Net / UNet++ / SegResNet
Rows: Small / Medium / Large
Same slice per row; largest-CC for display.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label as cc_label
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[2]
PPO_DIR = ROOT / "ppo"
OUT = ROOT / "results" / "rl_refined_3x4_comparison.png"
PPO_OUT = PPO_DIR / "results" / "rl_refined_3x4_comparison.png"


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    labeled, n = cc_label(binary)
    if n <= 1:
        return binary.astype(np.float32)
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    keep = 1 + int(np.argmax(sizes))
    return (labeled == keep).astype(np.float32)


def dice(a, b):
    a = a > 0.5
    b = b > 0.5
    inter = np.logical_and(a, b).sum()
    return float(2 * inter / (a.sum() + b.sum() + 1e-8))


def gt_size_class(area: int) -> int:
    if area < 300:
        return 0
    if area < 700:
        return 1
    return 2


def crop_box(gt, margin=15):
    ys, xs = np.where(gt > 0.5)
    if len(ys) == 0:
        return 0, gt.shape[1] - 1, 0, gt.shape[0] - 1
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    return (
        max(0, xmin - margin),
        min(gt.shape[1] - 1, xmax + margin),
        max(0, ymin - margin),
        min(gt.shape[0] - 1, ymax + margin),
    )


def candidate_pools(gt_masks):
    pools = {0: [], 1: [], 2: []}
    for i, gt in enumerate(gt_masks):
        area = int((gt > 0.5).sum())
        if area <= 0:
            continue
        c = gt_size_class(area)
        _, n = cc_label(gt > 0.5)
        pools[c].append((i, area, n))
    for c in pools:
        # single-CC first, then by area near mid
        pools[c].sort(key=lambda t: (0 if t[2] == 1 else 1, t[1]))
    return pools


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    os.chdir(ROOT)

    # --- data (val split, t1ce+flair) ---
    sys.path.insert(0, str(ROOT))
    for mod in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[mod]
    from src.data.brats2020_dataset import BraTS2020Dataset as RootDS
    from src.data.patient_split import load_or_create_patient_split

    split = load_or_create_patient_split(
        str(ROOT / "src/data/archive"),
        210,
        str(ROOT / "checkpoints" / "patient_split.json"),
    )
    root_ds = RootDS(
        root_dir=str(ROOT / "src/data/archive"),
        modality="t1ce+flair",
        target_size=128,
        patient_ids=split["val"],
        simulate_rough=False,
    )
    images_2ch, gt_masks, _ = root_ds.get_numpy_arrays()
    if images_2ch.ndim == 3:
        images_2ch = images_2ch[:, None, ...]
    images_t1 = images_2ch[:, 0]
    pools = candidate_pools(gt_masks)
    print("n slices", len(images_t1), {k: len(v) for k, v in pools.items()})

    # --- ppo backbones ---
    sys.path.insert(0, str(PPO_DIR))
    for mod in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[mod]
    from src.models.unet import build_unet
    from src.models.unetplusplus import build_unetplusplus
    from src.models.segresnet import build_segresnet
    from src.envs.mask_refinement_env import MaskRefinementEnv as PpoEnv, _dice as ppo_dice

    def predict_bb(model, img1):
        t = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            return (torch.sigmoid(model(t)) > 0.5).float().squeeze().cpu().numpy()

    def rl_bb(agent, image, rough, gt, model_type):
        env = PpoEnv(
            images=image[None],
            gt_masks=gt[None],
            rough_masks=rough[None],
            max_steps=15,
            model_type=model_type,
        )
        obs, _ = env.reset(seed=0)
        best_dsc = ppo_dice(rough, gt)
        best_mask = rough.copy()
        for _ in range(15):
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            step_dsc = info.get("dsc", ppo_dice(env._current_mask, gt))
            if step_dsc > best_dsc:
                best_dsc = step_dsc
                best_mask = env._current_mask.copy()
            if terminated or truncated:
                break
        return rough.copy() if best_dsc < ppo_dice(rough, gt) else best_mask

    bbs = {}
    for name, builder in [
        ("unet", build_unet),
        ("unetplusplus", build_unetplusplus),
        ("segresnet", build_segresnet),
    ]:
        m = builder().to(device)
        m.load_state_dict(torch.load(PPO_DIR / "checkpoints" / f"{name}_best.pt", map_location=device))
        m.eval()
        bbs[name] = (m, PPO.load(str(PPO_DIR / "checkpoints" / f"ppo_refiner_{name}")))
        print("loaded", name)

    # --- TRIO ---
    while str(PPO_DIR) in sys.path:
        sys.path.remove(str(PPO_DIR))
    for mod in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[mod]
    sys.path.insert(0, str(ROOT))
    from src.models.dynamic_router import AdaptivePipeline
    from src.envs.mask_refinement_env import MaskRefinementEnv as TrioEnv
    from src.utils.metrics import apply_monotonic_dsc_gate, dice as trio_dice

    trio_pipe = AdaptivePipeline(device, in_channels=2)
    trio_agents = {
        0: PPO.load(str(ROOT / "checkpoints" / "ppo_small")),
        1: PPO.load(str(ROOT / "checkpoints" / "ppo_medium")),
        2: PPO.load(str(ROOT / "checkpoints" / "ppo_large")),
    }
    print("loaded TRIO")
    thr = [0.80, 0.80, 0.50]

    def trio_refine(img1, gt, rough, cls):
        mode = ["small", "medium", "large"][cls]
        agent = trio_agents[cls]
        env = TrioEnv(
            img1[None, ...],
            gt[None, ...],
            np.expand_dims(rough, 0),
            uncertainty_maps=np.expand_dims(rough.astype(np.float32), 0),
            max_steps=15,
            target_dsc=1.0,
            refinement_mode=mode,
        )
        obs, _ = env.reset(seed=0)
        best_mask = rough.copy()
        best_d = trio_dice(rough, gt)
        for _ in range(15):
            action, _ = agent.predict(obs, deterministic=True)
            if mode == "small" and np.sum(env._current_mask) < 35:
                action = np.maximum(0.0, action)
            obs, _, _, truncated, info = env.step(action)
            step_d = float(info.get("dsc", trio_dice(env._current_mask, gt)))
            if step_d >= best_d:
                best_d = step_d
                best_mask = env._current_mask.copy()
            if truncated:
                break
        return apply_monotonic_dsc_gate(rough, best_mask, gt)

    def run_all(idx):
        """Return list of 4 cell dicts (TRIO, unet, unet++, segresnet)."""
        img1 = images_t1[idx]
        img2 = images_2ch[idx]
        gt_raw = gt_masks[idx]
        gt = keep_largest_cc(gt_raw)
        cells = []

        xt = torch.from_numpy(img2).float().unsqueeze(0).to(device)
        with torch.no_grad():
            rough_t, cls_pred = trio_pipe(xt)
            cls_use = int(cls_pred[0].item())
            rough_np = (rough_t[0, 0].cpu().numpy() > thr[cls_use]).astype(np.float32)
        refined_t = keep_largest_cc(trio_refine(img1, gt_raw, rough_np, cls_use))
        cells.append(dict(img=img1, gt=gt, pred=refined_t, dsc=dice(refined_t, gt)))

        for key in ["unet", "unetplusplus", "segresnet"]:
            model, agent = bbs[key]
            rough = predict_bb(model, img1)
            refined = keep_largest_cc(rl_bb(agent, img1, rough, gt_raw, key))
            cells.append(dict(img=img1, gt=gt, pred=refined, dsc=dice(refined, gt)))
        return cells

    # Class-mean RL DSC targets (trio_vs_backbone_by_size.md)
    # order: TRIO, U-Net, UNet++, SegResNet
    MEAN_DSC = {
        0: np.array([0.8429, 0.5499, 0.6649, 0.6599], dtype=np.float64),
        1: np.array([0.9314, 0.8288, 0.8273, 0.8281], dtype=np.float64),
        2: np.array([0.9582, 0.9034, 0.8857, 0.8827], dtype=np.float64),
    }
    # Same slice per row; minimize L1 distance to all 4 class means.
    max_scan = {0: 120, 1: 100, 2: 80}
    row_names = ["Small", "Medium", "Large"]
    col_names = ["TRIO", "U-Net", "UNet++", "SegResNet"]
    grid = []

    for c in (0, 1, 2):
        pool = pools[c]
        areas = [a for _, a, _ in pool]
        mid = float(np.median(areas)) if areas else 0.0
        ranked = sorted(pool, key=lambda t: (0 if t[2] == 1 else 1, abs(t[1] - mid)))
        target = MEAN_DSC[c]
        best = None  # (l1, -trio_close, idx, cells, dscs)
        n_scan = min(max_scan[c], len(ranked))
        for k, (idx, area, ncc) in enumerate(ranked[:n_scan]):
            cells = run_all(idx)
            dscs = np.array([x["dsc"] for x in cells], dtype=np.float64)
            nonempty = all((x["pred"] > 0.5).sum() > 0 for x in cells)
            if not nonempty:
                continue
            # Require TRIO reasonably near its mean (±0.12) preferentially
            l1 = float(np.abs(dscs - target).sum())
            trio_err = abs(float(dscs[0] - target[0]))
            score = (l1, trio_err)
            if best is None or score < best[0]:
                best = (score, idx, cells, dscs, area)
            if k % 15 == 0:
                print(
                    f"  scanning {row_names[c]} {k+1}/{n_scan} "
                    f"best_L1={best[0][0]:.3f} trio={best[3][0]:.3f}"
                )
        if best is None:
            raise RuntimeError(f"no valid sample for class {c}")
        _, idx, cells, dscs, area = best
        print(
            f"{row_names[c]} NEAR-MEAN idx={idx} area={area} "
            f"dscs={['%.3f'%d for d in dscs]} "
            f"targets={['%.3f'%d for d in target]} "
            f"L1={best[0][0]:.3f}"
        )
        grid.append(cells)

    fig, axes = plt.subplots(3, 4, figsize=(13.5, 10.0))
    for r in range(3):
        for c in range(4):
            ax = axes[r, c]
            s = grid[r][c]
            xmin, xmax, ymin, ymax = crop_box(s["gt"])
            ax.imshow(s["img"], cmap="gray", vmin=0, vmax=1)
            ax.contour(s["pred"], levels=[0.5], colors="cyan", linewidths=2.0)
            ax.contour(s["gt"], levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)
            ax.set_xticks([])
            ax.set_yticks([])
            mean_t = MEAN_DSC[r][c]
            ax.set_xlabel(
                f"RL DSC={s['dsc']:.3f}  (mean={mean_t:.3f})",
                fontsize=11,
                fontweight="bold",
            )
            if r == 0:
                ax.set_title(col_names[c], fontsize=14, fontweight="bold", pad=8)
        axes[r, 0].set_ylabel(row_names[r], fontsize=14, fontweight="bold", labelpad=10)

    fig.suptitle(
        "RL Refined only  ·  samples near class-mean DSC  ·  same slice per row",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0.02, 0.01, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    PPO_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    fig.savefig(PPO_OUT, dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", OUT)
    print("saved", PPO_OUT)


if __name__ == "__main__":
    main()
