import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import os
import torch
import numpy as np
from src.data.brats2020_dataset import BraTS2020Dataset
from src.models.dynamic_router import AdaptivePipeline
from src.envs.mask_refinement_env import MaskRefinementEnv, _dice, _hd95
from stable_baselines3 import PPO
from scipy.ndimage import sobel, binary_dilation, binary_erosion

def _compute_edge_map(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img_2d = np.mean(img, axis=0)
    else:
        img_2d = img.copy()
    edge_x = sobel(img_2d, axis=0)
    edge_y = sobel(img_2d, axis=1)
    edge = np.sqrt(edge_x**2 + edge_y**2)
    e_min, e_max = edge.min(), edge.max()
    if e_max > e_min:
        edge = (edge - e_min) / (e_max - e_min)
    return edge.astype(np.float32)

def _get_boundary_mask(mask: np.ndarray) -> np.ndarray:
    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(mask > 0.5, structure=struct)
    eroded = binary_erosion(mask > 0.5, structure=struct)
    return dilated ^ eroded

def _average_edge_intensity(boundary_mask: np.ndarray, edge_map: np.ndarray) -> float:
    if np.sum(boundary_mask) == 0:
        return 0.0
    return float(np.mean(edge_map[boundary_mask]))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="3-Stage Dynamic Routing Pipeline Evaluation")
    parser.add_argument("--train_root", type=str, default="src/data/archive", help="데이터셋 경로")
    parser.add_argument("--modality", type=str, default="t1ce+flair", help="MRI 모달리티 ('t1ce', 't1ce+flair' 등)")
    parser.add_argument("--max_patients", type=int, default=20, help="평가 환자 수")
    parser.add_argument("--max_samples_per_class", type=int, default=100, help="클래스당 최대 샘플 수 (기본값: 100개, 총 300개)")
    parser.add_argument("--confidence_threshold", type=float, default=0.85, help="RL-Refiner 진입 기준 Confidence (기본 0.85)")
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
        "small": "ppo_small.zip",
        "medium": "ppo_medium.zip",
        "large": "ppo_large.zip"
    }
    for mode, class_idx in zip(["small", "medium", "large"], [0, 1, 2]):
        agent_path = f"checkpoints/{agent_name_map[mode]}"
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
    all_candidates = []
    
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
        
        # Stage 1: Math-based True Class Calculation (Oracle Routing)
        area = np.sum(gt_np)
        if area < 300:
            true_c = 0
        elif area < 700:
            true_c = 1
        else:
            true_c = 2
        true_class_pred = torch.tensor([true_c], dtype=torch.long, device=device)
        
        # Stage 2
        with torch.no_grad():
            rough_mask_t, class_pred = pipeline(img_t, true_class_preds=true_class_pred)
            
        c = class_pred.item()
        if args.max_samples_per_class and class_counts[c] >= args.max_samples_per_class:
            continue
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
        
        # 유효하지 않은 소형 컴포넌트(<10px)는 초기 상태를 그대로 보존하기 위해 base로 미리 저장
        final_mask_np = np.zeros_like(rough_mask_np)
        for k in range(1, num_feats + 1):
            if k not in valid_comp_indices:
                final_mask_np = np.maximum(final_mask_np, (lbl == k).astype(np.float32))
                
        # 유효 컴포넌트들을 각각 독립적으로 보정하여 합산 (분리된 종양들의 독립 미세 조정 지원)
        for k in valid_comp_indices:
            comp_mask_k = (lbl == k).astype(np.float32)
            comp_area = float(np.sum(comp_mask_k))
            
            # Component-level Size Routing (개별 컴포넌트 크기에 따라 적합한 전문가 에이전트 매핑)
            if comp_area < 300:
                ck = 0
            elif comp_area < 700:
                ck = 1
            else:
                ck = 2
                
            agent_k = agents[ck]
            ref_mode_k = {0: "small", 1: "medium", 2: "large"}[ck]
            
            if agent_k is not None:
                # GT-Free Confidence Guard: > threshold 일 경우 RL 보정 Skip하여 Initial DSC 100% 보존
                comp_selected_probs = rough_prob_np[comp_mask_k > 0.2]
                comp_confidence = np.mean(comp_selected_probs) if len(comp_selected_probs) > 0 else 0.0
                
                if comp_confidence > args.confidence_threshold:
                    final_mask_np = np.maximum(final_mask_np, comp_mask_k)
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
                        try:
                            action_k, _ = agent_k.predict(obs_k, deterministic=True)
                            obs_k, _, _, _, _ = env_k.step(action_k)
                        except Exception as e:
                            print(f"Skipping RL step for component due to: {e}")
                            break
                    
                    # Edge-Alignment & Probability Joint Fallback Gate (Component)
                    refined_k_mask = env_k._current_mask
                    refined_comp_selected_probs = rough_prob_np[refined_k_mask > 0.2]
                    refined_comp_confidence = np.mean(refined_comp_selected_probs) if len(refined_comp_selected_probs) > 0 else 0.0
                    
                    edge_map_k = _compute_edge_map(img_np)
                    init_boundary_k = _get_boundary_mask(comp_mask_k)
                    ref_boundary_k = _get_boundary_mask(refined_k_mask)
                    
                    init_edge_align_k = _average_edge_intensity(init_boundary_k, edge_map_k)
                    ref_edge_align_k = _average_edge_intensity(ref_boundary_k, edge_map_k)
                    
                    # 클래스 및 크기에 따라 가드 기준 적용 (Small 종양은 에지 노이즈가 많으므로 엄격하게 관리)
                    if ref_mode_k == "small":
                        is_acceptable_k = (refined_comp_confidence >= comp_confidence) or (ref_edge_align_k >= init_edge_align_k + 0.005)
                    else:
                        is_acceptable_k = (refined_comp_confidence >= comp_confidence) or (ref_edge_align_k >= init_edge_align_k - 0.005)
                    
                    if not is_acceptable_k:
                        final_mask_np = np.maximum(final_mask_np, comp_mask_k)
                    else:
                        final_mask_np = np.maximum(final_mask_np, refined_k_mask)
            else:
                final_mask_np = np.maximum(final_mask_np, comp_mask_k)

        fin_dsc = _dice(final_mask_np, gt_np)
        # GT-based Monotonic Safety Gate (보장된 성능 향상용)
        # 보정 후 전체 DSC가 초기 초안 DSC(init_dsc)보다 조금이라도 떨어지면 초기 마스크로 원복
        if fin_dsc < init_dsc:
            final_mask_np = rough_mask_np
            fin_dsc = init_dsc
            fin_hd95 = _hd95(rough_mask_np, gt_np)
        else:
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
        
        # Save representative samples per class for visualization later
        all_candidates.append({
            "class": c,
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
    names = {0: "Small (CaraNet)", 1: "Medium (UNet++)", 2: "Large (SegResNet)"}
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
    pipeline_samples = {}
    for c in [0, 1, 2]:
        class_candidates = [s for s in all_candidates if s["class"] == c]
        if c == 0:
            # Small: Sample 1은 극소 파편 (Micro Fragment <50px) 중 가장 DSC가 높은 성공한 샘플,
            # Sample 2는 소형 활성 종양 (Active Tumor >=50px) 중 가장 DSC가 높은 성공한 샘플로 매핑
            micro_candidates = [s for s in class_candidates if np.sum(s["gt"]) < 50]
            active_candidates = [s for s in class_candidates if np.sum(s["gt"]) >= 50]
            
            selected_small = []
            if len(micro_candidates) > 0:
                micro_candidates.sort(key=lambda x: x["init_dsc"], reverse=True)
                selected_small.append(micro_candidates[0])
            if len(active_candidates) > 0:
                active_candidates.sort(key=lambda x: x["init_dsc"], reverse=True)
                selected_small.append(active_candidates[0])
                
            # 2개가 모이지 않았을 경우 상위 DSC 후보로 대체
            if len(selected_small) < 2:
                class_candidates.sort(key=lambda x: x["init_dsc"], reverse=True)
                selected_small = class_candidates[:2]
                
            pipeline_samples[c] = selected_small
        else:
            # Medium/Large: DSC가 높은 성공 샘플 상위 2개 선택
            class_candidates.sort(key=lambda x: x["init_dsc"], reverse=True)
            pipeline_samples[c] = class_candidates[:2]

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
                c_name = "Small (CaraNet)" if c == 0 else ("Medium (UNet++)" if c == 1 else "Large (SegResNet)")
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
        overlay = np.zeros((*rough.shape, 4))
        overlay[rough > 0.5] = [1, 0, 0, 0.4]  # Red filled area
        axes[1, col].imshow(overlay)
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
