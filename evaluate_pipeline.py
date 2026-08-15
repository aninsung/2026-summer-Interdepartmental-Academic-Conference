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
    dataset = BraTS2020Dataset(root_dir='src/data/archive/BraTS2021_Training_Data', modality='t1ce', target_size=128, max_patients=20, simulate_rough=False)
    
    # Extract arrays
    images, gt_masks, _ = dataset.get_numpy_arrays()
    
    # 2. Stage 1 & 2: Dynamic Router
    print("Loading 3-Stage Pipeline Models...")
    pipeline = AdaptivePipeline(device)
    
    # 3. Stage 3: RL Refiner (Multi-Agent)
    print("Loading PPO Refiners...")
    agents = {}
    for mode, class_idx in zip(["small", "medium", "large"], [0, 1, 2]):
        agent_path = f"checkpoints/ppo_{mode}.zip"
        if os.path.exists(agent_path):
            agents[class_idx] = PPO.load(agent_path, device=device)
        else:
            print(f"Warning: No PPO agent found at {agent_path}")
            agents[class_idx] = None
        
    initial_dsc_list = []
    final_dsc_list = []
    final_hd95_list = []
    
    class_counts = {0:0, 1:0, 2:0}
    
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
        
        rough_mask_np = rough_mask_t.squeeze().cpu().numpy()
        rough_mask_np = (rough_mask_np > 0.5).astype(np.float32)
        
        init_dsc = _dice(rough_mask_np, gt_np)
        initial_dsc_list.append(init_dsc)
        
        # Stage 3
        final_mask_np = rough_mask_np
        agent = agents[c]
        refinement_mode = {0: "small", 1: "medium", 2: "large"}[c]
        
        if agent is not None:
            # We must simulate the environment step for this image
            env = MaskRefinementEnv(
                images[i:i+1], 
                gt_masks[i:i+1], 
                np.expand_dims(rough_mask_np, 0), 
                max_steps=10, 
                refinement_mode=refinement_mode
            )
            obs, _ = env.reset(seed=0)
            for _ in range(10):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            final_mask_np = env._current_mask
            
        fin_dsc = _dice(final_mask_np, gt_np)
        fin_hd95 = _hd95(final_mask_np, gt_np)
        
        final_dsc_list.append(fin_dsc)
        final_hd95_list.append(fin_hd95)
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(images)} slices...")
            
    print("\n--- Pipeline Evaluation Results ---")
    print(f"Total Slices Evaluated: {len(images)}")
    print(f"Class Distribution: Small: {class_counts[0]}, Medium: {class_counts[1]}, Large: {class_counts[2]}")
    print(f"Average Initial DSC (Stage 2): {np.mean(initial_dsc_list):.4f}")
    print(f"Average Final DSC (Stage 3):   {np.mean(final_dsc_list):.4f}")
    print(f"Average Final HD95:            {np.mean(final_hd95_list):.4f}")
    
if __name__ == "__main__":
    main()
