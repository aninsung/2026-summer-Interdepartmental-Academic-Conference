import os
import torch
import numpy as np
from src.data.brats2020_dataset import BraTS2020Dataset
from src.models.dynamic_router import AdaptivePipeline
from src.envs.mask_refinement_env import MaskRefinementEnv, _dice, _hd95
from stable_baselines3 import PPO

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Dataset Load (Test size)
    dataset = BraTS2020Dataset(root_dir='src/data/archive', modality='t1ce', target_size=128, max_patients=20, simulate_rough=False)
    
    # Extract arrays
    images, gt_masks, _ = dataset.get_numpy_arrays()
    
    # 2. Stage 1 & 2: Dynamic Router
    print("Loading 3-Stage Pipeline Models...")
    pipeline = AdaptivePipeline(device)
    
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
            print(f"Warning: No PPO agent found at {agent_path}")
            agents[class_idx] = None
        
    initial_dsc_list = []
    final_dsc_list = []
    final_hd95_list = []
    
    class_initial_dsc = {0: [], 1: [], 2: []}
    class_final_dsc = {0: [], 1: [], 2: []}
    class_final_hd95 = {0: [], 1: [], 2: []}
    
    class_counts = {0:0, 1:0, 2:0}
    
    small_active_init, small_active_fin, small_active_hd = [], [], []
    small_micro_init, small_micro_fin, small_micro_hd = [], [], []
    
    print("\nStarting Evaluation...")
    for i in range(len(images)):
        img_np = images[i]
        gt_np = gt_masks[i]
        
        img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        
        # Stage 1 & 2
        with torch.no_grad():
            rough_mask_t, class_pred = pipeline(img_t)
            
        c = class_pred.item()
        class_counts[c] += 1
        
        rough_prob_np = rough_mask_t.squeeze().cpu().numpy()
        rough_mask_np = (rough_prob_np > 0.5).astype(np.float32)
        
        init_dsc = _dice(rough_mask_np, gt_np)
        initial_dsc_list.append(init_dsc)
        
        # Stage 3: Component-wise Independent Refinement (둘 이상 분리형 병변 개별 크롭 & 독립 보정)
        from scipy.ndimage import label as sp_label
        lbl, num_feats = sp_label(rough_mask_np > 0.5)
        
        # 15px 이상인 유효 Component만 추출
        valid_comp_indices = [k for k in range(1, num_feats + 1) if np.sum(lbl == k) >= 15]
        
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
                    # GT-Free Confidence check
                    comp_selected_probs = rough_prob_np[comp_mask_k > 0.5]
                    comp_confidence = np.mean(comp_selected_probs) if len(comp_selected_probs) > 0 else 0.0
                    
                    if comp_confidence > 0.95:
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
                        refined_comp_selected_probs = rough_prob_np[refined_k_mask > 0.5]
                        refined_comp_confidence = np.mean(refined_comp_selected_probs) if len(refined_comp_selected_probs) > 0 else 0.0
                        
                        if refined_comp_confidence < comp_confidence - 0.02:
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
                selected_probs = rough_prob_np[rough_mask_np > 0.5]
                avg_confidence = np.mean(selected_probs) if len(selected_probs) > 0 else 0.0
                
                if avg_confidence > 0.95:
                    final_mask_np = rough_mask_np
                else:
                    env = MaskRefinementEnv(
                        images[i:i+1], 
                        gt_masks[i:i+1], 
                        np.expand_dims(rough_mask_np, 0), 
                        uncertainty_maps=np.expand_dims(rough_prob_np, 0),
                        max_steps=3, 
                        refinement_mode=refinement_mode
                    )
                    obs, _ = env.reset(seed=0)
                    for _ in range(3):
                        action, _ = agent.predict(obs, deterministic=True)
                        obs, _, _, _, _ = env.step(action)
                    
                    # Probability Fallback Gate
                    refined_mask_np = env._current_mask
                    refined_selected_probs = rough_prob_np[refined_mask_np > 0.5]
                    refined_confidence = np.mean(refined_selected_probs) if len(refined_selected_probs) > 0 else 0.0
                    
                    if refined_confidence < avg_confidence - 0.02:
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
    
if __name__ == "__main__":
    main()
