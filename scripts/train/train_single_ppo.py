"""단일 백본의 예측을 초기 마스크로 삼아, 크기 구분 없는 PPO 하나를 학습한다."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.eval.plot_single_backbone_grid import build_model, predict_prob
from src.data.patient_split import load_or_create_patient_split
from src.data.brats2020_dataset import BraTS2020Dataset
from src.envs.mask_refinement_env import MaskRefinementEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--train_root", default="src/data/archive")
    parser.add_argument("--patient_split", default="checkpoints/patient_split.json")
    parser.add_argument("--max_train_patients", type=int, default=400)
    parser.add_argument("--modality", default="t1ce+flair")
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_or_create_patient_split(args.train_root, args.max_train_patients, args.patient_split)
    dataset = BraTS2020Dataset(
        root_dir=args.train_root, modality=args.modality, target_size=128,
        patient_ids=list(split["train"]), slice_selection="tumor", simulate_rough=False, num_workers=4,
    )
    images, gts, _ = dataset.get_numpy_arrays()
    model = build_model(args.model_type, images.shape[1], device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    prob = predict_prob(model, images, device)
    rough = (prob > 0.5).astype(np.float32)
    print(f"{args.model_type} 학습 슬라이스 {len(gts)}  rough DSC 준비 완료", flush=True)

    def make_env():
        return MaskRefinementEnv(
            images=images, gt_masks=gts, rough_masks=rough, uncertainty_maps=prob,
            max_steps=args.max_steps, model_type=args.model_type, refinement_mode="medium",
        )

    # 스텝마다 CPU에서 거리 변환을 하므로, 같은 10만 스텝을 환경 8개가 나눠 모은다.
    env = SubprocVecEnv([make_env for _ in range(8)], start_method="fork")
    agent = PPO(
        policy="CnnPolicy", env=env, n_steps=512, batch_size=64, n_epochs=4,
        learning_rate=3e-4, ent_coef=0.01, verbose=0, device=str(device), seed=args.seed,
        policy_kwargs=dict(net_arch=[256, 256], normalize_images=False),
    )
    agent.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    agent.save(args.save_path)
    print(f"저장: {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
