import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import os
import torch
import numpy as np
from src.data.brats2020_dataset import BraTS2020Dataset
from src.models.dynamic_router import AdaptivePipeline
from src.envs.mask_refinement_env import MaskRefinementEnv, _dice, _hd95
from stable_baselines3 import PPO

def main():
    import argparse
    parser = argparse.ArgumentParser(description="3-Stage Dynamic Routing Pipeline Evaluation")
    parser.add_argument("--train_root", type=str, default="src/data/archive", help="데이터셋 경로")
    parser.add_argument("--modality", type=str, default="t1ce+flair", help="MRI 모달리티 ('t1ce', 't1ce+flair' 등)")
    parser.add_argument("--max_patients", type=int, default=20, help="평가 환자 수")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Dataset Load (Test size)
    dataset = BraTS2020Dataset(root_dir=args.train_root, modality=args.modality, target_size=128, max_patients=args.max_patients, simulate_rough=False)
    
    # Extract arrays
    images, gt_masks, _ = dataset.get_numpy_arrays()
    
    # 2. Stage 1 & 2: Dynamic Router
    print("Loading 3-Stage Pipeline Models...")
    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch)
    
    # 3. Stage 3: RL Refiner (Multi-Agent)
    print("Loading PPO Refiners...")
    agents = {}
    agent_name_map = {
        "small": "ppo_refiner_attention_unet.zip",
        "medium": "ppo_refiner_unetplusplus.zip",
        "large": "ppo_refiner_segresnet.zip"
    }
    for mode, class_idx in zip(["small", "medium", "large"], [0, 1, 2]):
        agent_path = f"checkpoints/{agent_name_map[mode]}"
        if not os.path.exists(agent_path):
            agent_path = f"checkpoints/ppo_{mode}.zip"
            
        if os.path.exists(agent_path):
            print(f"Loading PPO Agent: {agent_path}")
            agents[class_idx] = PPO.load(agent_path, device=device)
        else:
            print(f"[Warning] PPO Agent not found: {agent_path}. S3 Refinement will be skipped for class {class_idx}.")
            agents[class_idx] = None
        
    initial_dsc_list = []
    final_dsc_list = []
    final_hd95_list = []
    
    class_initial_dsc = {0: [], 1: [], 2: []}
    class_final_dsc = {0: [], 1: [], 2: []}
    class_final_hd95 = {0: [], 1: [], 2: []}
    
    class_counts = {0:0, 1:0, 2:0}
    pipeline_samples = {}
    
    small_active_init, small_active_fin, small_active_hd = [], [], []
    small_micro_init, small_micro_fin, small_micro_hd = [], [], []
    
    print("\nStarting Evaluation...")
    for i in range(len(images)):
        img_np = images[i]
        gt_np = gt_masks[i]
        
        if img_np.ndim == 2:
            img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        else:
            img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
        
        # Stage 1 & 2
        with torch.no_grad():
            rough_mask_t, class_pred = pipeline(img_t)
            
        c = class_pred.item()
        class_counts[c] += 1
        
        rough_prob_np = rough_mask_t.squeeze().cpu().numpy()
        
        # Adaptive Thresholding & Morphological Halo Trimming for Micro Fragments (<50px)
        main_area_50 = np.sum(rough_prob_np > 0.4)
        if c == 0 and main_area_50 < 60:
            thresh = 0.38
            rough_mask_np = (rough_prob_np > thresh).astype(np.float32)
            from scipy.ndimage import binary_erosion
            if np.sum(rough_mask_np) > 25:
                rough_mask_np = binary_erosion(rough_mask_np.astype(bool), structure=np.ones((3, 3))).astype(np.float32)
        else:
            thresh = 0.5
            rough_mask_np = (rough_prob_np > thresh).astype(np.float32)
            
        init_dsc = _dice(rough_mask_np, gt_np)
        initial_dsc_list.append(init_dsc)
        
        # Stage 3: Component-wise Independent Refinement
        from scipy.ndimage import label as sp_label
        lbl, num_feats = sp_label(rough_mask_np > 0.2)
        
        # 10px 이상인 유효 Component만 추출
        valid_comp_indices = [k for k in range(1, num_feats + 1) if np.sum(lbl == k) >= 10]
        
        if len(valid_comp_indices) > 1:
            refined_components = np.zeros_like(rough_mask_np)
            for k in valid_comp_indices:
                comp_mask_k = (lbl == k).astype(np.float32)
                comp_area = float(np.sum(comp_mask_k))
                
                # Component-level Size Routing
                if comp_area < 300:
                    ck = 0
                elif comp_area < 700:
                    ck = 1
                else:
                    ck = 2
                    
                agent_k = agents[ck]
                ref_mode_k = {0: "small", 1: "medium", 2: "large"}[ck]
                
                if agent_k is not None:
                    # GT-Free Confidence Guard: >0.90일 경우 RL 보정 Skip하여 Initial DSC 100% 보존
                    comp_selected_probs = rough_prob_np[comp_mask_k > 0.2]
                    comp_confidence = np.mean(comp_selected_probs) if len(comp_selected_probs) > 0 else 0.0
                    
                    if comp_confidence > 0.90:
                        refined_components = np.maximum(refined_components, comp_mask_k)
                    else:
                        env_k = MaskRefinementEnv(
                            images[i:i+1],
                            gt_masks[i:i+1],
                            np.expand_dims(comp_mask_k, 0),
                            uncertainty_maps=np.expand_dims(rough_prob_np * comp_mask_k, 0),
                            max_steps=3,
                            refinement_mode=ref_mode_k
                        )
                        obs_k, _ = env_k.reset(seed=0)
                        for _ in range(3):
                            action_k, _ = agent_k.predict(obs_k, deterministic=True)
                            obs_k, _, _, _, _ = env_k.step(action_k)
                        
                        # Probability Fallback Gate
                        refined_k_mask = env_k._current_mask
                        refined_comp_selected_probs = rough_prob_np[refined_k_mask > 0.2]
                        refined_comp_confidence = np.mean(refined_comp_selected_probs) if len(refined_comp_selected_probs) > 0 else 0.0
                        
                        if refined_comp_confidence < comp_confidence:
                            refined_components = np.maximum(refined_components, comp_mask_k)
                        else:
                            refined_components = np.maximum(refined_components, refined_k_mask)
                else:
                    refined_components = np.maximum(refined_components, comp_mask_k)
            final_mask_np = refined_components
        else:
            final_mask_np = rough_mask_np
            agent = agents[c]
            refinement_mode = {0: "small", 1: "medium", 2: "large"}[c]
            
            if agent is not None:
                selected_probs = rough_prob_np[rough_mask_np > 0.2]
                avg_confidence = np.mean(selected_probs) if len(selected_probs) > 0 else 0.0
                
                # Active RL Refinement with 3% Tolerance Fallback Gate
                env = MaskRefinementEnv(
                    images[i:i+1], 
                    gt_masks[i:i+1], 
                    np.expand_dims(rough_mask_np, 0), 
                    uncertainty_maps=np.expand_dims(rough_prob_np, 0),
                    max_steps=5, 
                    refinement_mode=refinement_mode
                )
                obs, _ = env.reset(seed=0)
                for _ in range(5):
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, _, _, _, _ = env.step(action)
                
                # Probability Fallback Gate: 보정 후 신뢰도가 3%p 이상 급락한 경우에만 안전 복원
                refined_mask_np = env._current_mask
                refined_selected_probs = rough_prob_np[refined_mask_np > 0.2]
                refined_confidence = np.mean(refined_selected_probs) if len(refined_selected_probs) > 0 else 0.0
                
                if refined_confidence < (avg_confidence - 0.03):
                    final_mask_np = rough_mask_np
                else:
                    final_mask_np = refined_mask_np

        fin_dsc = _dice(final_mask_np, gt_np)
        fin_hd95 = _hd95(final_mask_np, gt_np)
        
        final_dsc_list.append(fin_dsc)
        final_hd95_list.append(fin_hd95)
        
        class_initial_dsc[c].append(init_dsc)
        class_final_dsc[c].append(fin_dsc)
        class_final_hd95[c].append(fin_hd95)
        
        # Stratification for Small Class
        if c == 0:
            gt_area = np.sum(gt_np)
            if gt_area >= 50:
                small_active_init.append(init_dsc)
                small_active_fin.append(fin_dsc)
                small_active_hd.append(fin_hd95)
            else:
                small_micro_init.append(init_dsc)
                small_micro_fin.append(fin_dsc)
                small_micro_hd.append(fin_hd95)
        
        # Save representative samples per class for visualization
        if c not in pipeline_samples:
            pipeline_samples[c] = []
        if len(pipeline_samples[c]) < 2:
            pipeline_samples[c].append({
                "img": img_np[0] if img_np.ndim == 3 else img_np,
                "gt": gt_np,
                "rough": rough_mask_np,
                "final": final_mask_np,
                "init_dsc": init_dsc,
                "fin_dsc": fin_dsc,
            })

        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(images)} slices...")
            
    print("\n--- Pipeline Evaluation Results (Pure RL - GT Free) ---")
    print(f"Total Slices Evaluated: {len(images)}")
    print(f"Class Distribution: Small: {class_counts[0]}, Medium: {class_counts[1]}, Large: {class_counts[2]}")
    print(f"Average Initial DSC (Stage 2): {np.mean(initial_dsc_list):.4f}")
    print(f"Average Final DSC (Stage 3 Pure RL):   {np.mean(final_dsc_list):.4f}")
    print(f"Average Final HD95 (px):              {np.mean(final_hd95_list):.4f}")
    
    print("\n--- Class-wise Performance Breakdown ---")
    names = {0: "Small (Attention U-Net)", 1: "Medium (UNet++)", 2: "Large (SegResNet)"}
    for c in [0, 1, 2]:
        if len(class_initial_dsc[c]) > 0:
            init_avg = np.mean(class_initial_dsc[c])
            fin_avg = np.mean(class_final_dsc[c])
            hd_avg = np.mean(class_final_hd95[c])
            print(f"[{names[c]}] count: {len(class_initial_dsc[c])} | Initial DSC: {init_avg:.4f} -> Final DSC: {fin_avg:.4f} | HD95 (px): {hd_avg:.4f}")
            
    print("\n--- Stratified Analysis for Small Class ---")
    if len(small_active_init) > 0:
        print(f"[Small - Active Tumor (>=50px)] count: {len(small_active_init)} | Initial DSC: {np.mean(small_active_init):.4f} -> Final DSC: {np.mean(small_active_fin):.4f} | HD95 (px): {np.mean(small_active_hd):.4f}")
    if len(small_micro_init) > 0:
        print(f"[Small - Micro Boundary Fragment (<50px)] count: {len(small_micro_init)} | Initial DSC: {np.mean(small_micro_init):.4f} -> Final DSC: {np.mean(small_micro_fin):.4f} | HD95 (px): {np.mean(small_micro_hd):.4f}")
    
    # 🖼️ 3-Stage Dynamic Routing 파이프라인 샘플 시각화 저장
    _plot_pipeline_results(pipeline_samples, output_dir="results")


def _plot_pipeline_results(pipeline_samples: dict, output_dir: str = "results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    sample_list = []
    class_labels = []
    for c in [0, 1, 2]:
        if c in pipeline_samples:
            for idx, s in enumerate(pipeline_samples[c]):
                sample_list.append(s)
                c_name = "Small (Attention U-Net)" if c == 0 else ("Medium (UNet++)" if c == 1 else "Large (SegResNet)")
                class_labels.append(f"{c_name}\nSample {idx+1}")

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
        init_dsc = s["init_dsc"]
        fin_dsc = s["fin_dsc"]

        y_indices, x_indices = np.where(gt > 0.5)
        if len(y_indices) > 0:
            ymin, ymax = y_indices.min(), y_indices.max()
            xmin, xmax = x_indices.min(), x_indices.max()
            margin = 15
            ymin = max(0, ymin - margin)
            ymax = min(gt.shape[0] - 1, ymax + margin)
            xmin = max(0, xmin - margin)
            xmax = min(gt.shape[1] - 1, xmax + margin)
        else:
            ymin, ymax = 0, gt.shape[0] - 1
            xmin, xmax = 0, gt.shape[1] - 1

        # Row 0: Original MRI + Ground Truth
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].contour(gt, levels=[0.5], colors="lime", linewidths=2.0)
        axes[0, col].set_title(class_labels[col], fontsize=13, fontweight="bold", pad=8)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Row 1: Stage 2 Rough Mask
        axes[1, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[1, col].contour(rough, levels=[0.5], colors="red", linewidths=2.0)
        axes[1, col].contour(gt, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
        axes[1, col].set_xlabel(f"Initial DSC={init_dsc:.3f}", fontsize=12, fontweight="bold")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        # Row 2: Stage 3 Pure RL Final Mask
        axes[2, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[2, col].contour(final, levels=[0.5], colors="cyan", linewidths=2.0)
        axes[2, col].contour(gt, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
        axes[2, col].set_xlabel(f"Final DSC={fin_dsc:.3f}", fontsize=12, fontweight="bold")
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

        for row_idx in range(3):
            axes[row_idx, col].set_xlim(xmin, xmax)
            axes[row_idx, col].set_ylim(ymax, ymin)

    # Row headers
    fig.text(0.01, 0.78, "MRI + GT", va="center", rotation="vertical", fontsize=14, fontweight="bold", color="lime")
    fig.text(0.01, 0.50, "Stage 2 Rough", va="center", rotation="vertical", fontsize=14, fontweight="bold", color="red")
    fig.text(0.01, 0.22, "Stage 3 RL Refined", va="center", rotation="vertical", fontsize=14, fontweight="bold", color="cyan")

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    save_path = os.path.join(output_dir, "pipeline_sample_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"2026-08-17 09:57:24 [INFO] 🖼️ 3-Stage Dynamic Routing 파이프라인 시각화 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
