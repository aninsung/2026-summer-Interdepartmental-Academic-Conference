"""
subregion_pipeline/scripts/evaluate_subregion_pipeline.py
----------------------------------------------------------
Stage 4: Subregion-Guided (ET / TC / WT) Dynamic Routing 파이프라인 최종 성능 검증 및 시각화
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import json
from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from subregion_pipeline.src.subregion_dataset import SubregionBraTSDataset
from subregion_pipeline.src.subregion_router import SubregionAdaptivePipeline
from src.envs.mask_refinement_env import MaskRefinementEnv, _dice, _hd95

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def evaluate_subregion_pipeline(
    train_root: str = "src/data/archive",
    max_patients: int = 20,
    output_dir: str = "results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    log.info(f"Loading Test Patients ({max_patients}) for Subregion Pipeline Benchmark...")

    ds = SubregionBraTSDataset(root_dir=train_root, target_size=128, max_patients=max_patients)
    if len(ds) == 0:
        raise ValueError("Evaluation dataset is empty.")

    router = SubregionAdaptivePipeline(
        classifier_path="subregion_pipeline/checkpoints/subregion_classifier.pt",
        expert_et_path="subregion_pipeline/checkpoints/expert_et_best.pt",
        expert_tc_path="subregion_pipeline/checkpoints/expert_tc_best.pt",
        expert_wt_path="subregion_pipeline/checkpoints/expert_wt_best.pt",
        device=device
    )

    agents = {}
    for mode_name, class_idx in zip(["et", "tc", "wt"], [0, 1, 2]):
        agent_p = f"subregion_pipeline/checkpoints/ppo_refiner_{mode_name}.zip"
        if os.path.exists(agent_p):
            agents[class_idx] = PPO.load(agent_p, device=device)
            log.info(f"Loaded Subregion PPO Agent: {agent_p}")
        else:
            log.warning(f"PPO Agent missing for class {mode_name}: {agent_p}")
            agents[class_idx] = None

    class_init_dsc = {0: [], 1: [], 2: []}
    class_fin_dsc  = {0: [], 1: [], 2: []}
    class_fin_hd95 = {0: [], 1: [], 2: []}

    pipeline_samples = {}

    log.info(f"Starting Evaluation across {len(ds)} slices...")
    for i in range(len(ds)):
        s = ds.samples[i]
        img_np = s["image"]
        c_target = s["subregion_label"]

        if c_target == 0:
            gt_np = s["gt_et"][0]
        elif c_target == 1:
            gt_np = s["gt_tc"][0]
        else:
            gt_np = s["gt_wt"][0]

        img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
        rough_bin, rough_prob, c_pred = router.predict_mask(img_t, class_override=c_target)

        init_dsc = _dice(rough_bin, gt_np)
        agent = agents.get(c_target, None)
        mode_param = "small" if c_target == 0 else ("medium" if c_target == 1 else "large")

        if agent is not None:
            env = MaskRefinementEnv(
                np.expand_dims(img_np, 0),
                np.expand_dims(gt_np, 0),
                np.expand_dims(rough_bin, 0),
                uncertainty_maps=np.expand_dims(rough_prob, 0),
                max_steps=5,
                refinement_mode=mode_param
            )
            obs, _ = env.reset(seed=0)
            for _ in range(5):
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, _, _, _ = env.step(action)

            refined_mask = env._current_mask
            refined_probs = rough_prob[refined_mask > 0.2]
            avg_conf = np.mean(rough_prob[rough_bin > 0.2]) if np.sum(rough_bin > 0.2) > 0 else 0.0
            ref_conf = np.mean(refined_probs) if len(refined_probs) > 0 else 0.0

            if ref_conf < avg_conf:
                final_mask = rough_bin
            else:
                final_mask = refined_mask
        else:
            final_mask = rough_bin

        fin_dsc = _dice(final_mask, gt_np)
        fin_hd95 = _hd95(final_mask, gt_np)

        class_init_dsc[c_target].append(init_dsc)
        class_fin_dsc[c_target].append(fin_dsc)
        class_fin_hd95[c_target].append(fin_hd95)

        if c_target not in pipeline_samples:
            pipeline_samples[c_target] = []
        if len(pipeline_samples[c_target]) < 2:
            pipeline_samples[c_target].append({
                "img": img_np[0],
                "gt": gt_np,
                "rough": rough_bin,
                "final": final_mask,
                "init_dsc": init_dsc,
                "fin_dsc": fin_dsc,
            })

    # Summary
    all_init = [val for l in class_init_dsc.values() for val in l]
    all_fin  = [val for l in class_fin_dsc.values() for val in l]
    all_hd   = [val for l in class_fin_hd95.values() for val in l]

    log.info("\n=== ⚙️ BraTS Subregion-Guided (ET / TC / WT) Pipeline Benchmark Results ===")
    log.info(f"Total Slices Evaluated: {len(ds)}")
    log.info(f"Overall Initial DSC: {np.mean(all_init):.4f}")
    log.info(f"Overall Final DSC:   {np.mean(all_fin):.4f}")
    log.info(f"Overall Final HD95:  {np.mean(all_hd):.4f} px")

    names = {0: "ET (Enhancing Tumor / Attention U-Net)", 1: "TC (Tumor Core / UNet++)", 2: "WT (Whole Tumor / SegResNet)"}
    metrics_summary = {}
    for c in [0, 1, 2]:
        if len(class_init_dsc[c]) > 0:
            i_avg = float(np.mean(class_init_dsc[c]))
            f_avg = float(np.mean(class_fin_dsc[c]))
            h_avg = float(np.mean(class_fin_hd95[c]))
            log.info(f"[{names[c]}] count: {len(class_init_dsc[c])} | Initial DSC: {i_avg:.4f} -> Final DSC: {f_avg:.4f} | HD95: {h_avg:.4f} px")
            metrics_summary[names[c]] = {"count": len(class_init_dsc[c]), "init_dsc": i_avg, "final_dsc": f_avg, "hd95": h_avg}

    # Save metrics json
    with open(os.path.join(output_dir, "subregion_pipeline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    # Plot Subregion Comparison Figure
    _plot_subregion_results(pipeline_samples, output_dir=output_dir)


def _plot_subregion_results(pipeline_samples: dict, output_dir: str = "results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_list = []
    class_labels = []
    names = {0: "ET (Enhancing Tumor)", 1: "TC (Tumor Core)", 2: "WT (Whole Tumor)"}

    for c in [0, 1, 2]:
        if c in pipeline_samples:
            for idx, s in enumerate(pipeline_samples[c]):
                sample_list.append(s)
                class_labels.append(f"{names[c]}\nSample {idx+1}")

    if not sample_list:
        return

    n_cols = len(sample_list)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 9.5))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, s in enumerate(sample_list):
        img = s["img"]
        gt = s["gt"]
        rough = s["rough"]
        final = s["final"]

        y_idx, x_idx = np.where(gt > 0.5)
        if len(y_idx) > 0:
            ymin, ymax = y_idx.min(), y_idx.max()
            xmin, xmax = x_idx.min(), x_idx.max()
            margin = 15
            ymin = max(0, ymin - margin)
            ymax = min(gt.shape[0] - 1, ymax + margin)
            xmin = max(0, xmin - margin)
            xmax = min(gt.shape[1] - 1, xmax + margin)
        else:
            ymin, ymax = 0, gt.shape[0] - 1
            xmin, xmax = 0, gt.shape[1] - 1

        # Row 0: Original MRI + GT
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].contour(gt, levels=[0.5], colors="lime", linewidths=2.0)
        axes[0, col].set_title(class_labels[col], fontsize=13, fontweight="bold", pad=8)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Row 1: Stage 2 Expert Rough Mask
        axes[1, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[1, col].contour(rough, levels=[0.5], colors="red", linewidths=2.0)
        axes[1, col].contour(gt, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
        axes[1, col].set_xlabel(f"Initial DSC={s['init_dsc']:.3f}", fontsize=12, fontweight="bold")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        # Row 2: Stage 3 PPO Refined Mask
        axes[2, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[2, col].contour(final, levels=[0.5], colors="cyan", linewidths=2.0)
        axes[2, col].contour(gt, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
        axes[2, col].set_xlabel(f"Final DSC={s['fin_dsc']:.3f}", fontsize=12, fontweight="bold")
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

        for r_idx in range(3):
            axes[r_idx, col].set_xlim(xmin, xmax)
            axes[r_idx, col].set_ylim(ymax, ymin)

    fig.text(0.01, 0.78, "MRI + GT", va="center", rotation="vertical", fontsize=14, fontweight="bold", color="lime")
    fig.text(0.01, 0.50, "Stage 2 Expert", va="center", rotation="vertical", fontsize=14, fontweight="bold", color="red")
    fig.text(0.01, 0.22, "Stage 3 PPO Refinement", va="center", rotation="vertical", fontsize=14, fontweight="bold", color="cyan")

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    save_path = os.path.join(output_dir, "subregion_pipeline_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"🖼️ Subregion Pipeline Comparison Image saved cleanly: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--max_patients", type=int, default=20)
    args = parser.parse_args()

    evaluate_subregion_pipeline(train_root=args.train_root, max_patients=args.max_patients)
