"""
subregion_pipeline/scripts/train_subregion_ppo.py
--------------------------------------------------
Stage 3: Subregion-Guided PPO Refiner 에이전트 학습 스크립트
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from subregion_pipeline.src.subregion_dataset import SubregionBraTSDataset
from subregion_pipeline.src.subregion_router import SubregionAdaptivePipeline
from src.envs.mask_refinement_env import MaskRefinementEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_subregion_ppo(
    subregion_mode: str = "et", # "et", "tc", "wt"
    train_root: str = "src/data/archive",
    total_timesteps: int = 40000,
    save_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if save_path is None:
        save_path = f"subregion_pipeline/checkpoints/ppo_refiner_{subregion_mode}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    log.info(f"Loading Dataset for Subregion PPO [{subregion_mode.upper()}] Refiner...")
    ds = SubregionBraTSDataset(root_dir=train_root, target_size=128, max_patients=100)

    target_map = {"et": 0, "tc": 1, "wt": 2}
    target_idx = target_map[subregion_mode]

    # Initialize Router
    router = SubregionAdaptivePipeline(
        classifier_path="subregion_pipeline/checkpoints/subregion_classifier.pt",
        expert_et_path="subregion_pipeline/checkpoints/expert_et_best.pt",
        expert_tc_path="subregion_pipeline/checkpoints/expert_tc_best.pt",
        expert_wt_path="subregion_pipeline/checkpoints/expert_wt_best.pt",
        device=device
    )

    # Collect predictions for specified subregion
    filtered_imgs = []
    filtered_gts  = []
    filtered_rough = []
    filtered_probs = []

    for i in range(len(ds)):
        s = ds.samples[i]
        if target_idx == 0:
            gt = s["gt_et"][0]
        elif target_idx == 1:
            gt = s["gt_tc"][0]
        else:
            gt = s["gt_wt"][0]

        if gt.sum() >= 20:
            img = s["image"]
            img_t = torch.from_numpy(img).unsqueeze(0).to(device)
            rough_bin, rough_prob, _ = router.predict_mask(img_t, class_override=target_idx)

            filtered_imgs.append(img)
            filtered_gts.append(gt)
            filtered_rough.append(rough_bin)
            filtered_probs.append(rough_prob)

    if len(filtered_imgs) == 0:
        raise ValueError(f"No samples found for subregion mode {subregion_mode}")

    imgs_arr   = np.stack(filtered_imgs, axis=0)
    gts_arr    = np.stack(filtered_gts, axis=0)
    rough_arr  = np.stack(filtered_rough, axis=0)
    probs_arr  = np.stack(filtered_probs, axis=0)

    log.info(f"[{subregion_mode.upper()} PPO] Filtered RL Samples: {len(imgs_arr)}")

    mode_param = "small" if subregion_mode == "et" else ("medium" if subregion_mode == "tc" else "large")
    env = MaskRefinementEnv(
        images=imgs_arr,
        gt_masks=gts_arr,
        rough_masks=rough_arr,
        uncertainty_maps=probs_arr,
        max_steps=20,
        refinement_mode=mode_param
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        verbose=1,
        device=device
    )

    log.info(f"PPO Training started for [{subregion_mode.upper()}] (total_timesteps={total_timesteps})...")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    log.info(f"PPO Model saved cleanly to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subregion", type=str, default="et", choices=["et", "tc", "wt"])
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--total_timesteps", type=int, default=40000)
    args = parser.parse_args()

    train_subregion_ppo(subregion_mode=args.subregion, train_root=args.train_root, total_timesteps=args.total_timesteps)
