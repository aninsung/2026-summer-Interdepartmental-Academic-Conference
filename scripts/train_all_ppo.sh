#!/bin/bash
set -e

echo "Starting PPO Retraining (Small, Medium, Large) with Curvature Penalty..."

echo "1. Training Small Expert..."
python3 scripts/train/train_agent.py --refinement_mode small --total_timesteps 200000 --save_path checkpoints/ppo_refiner_small --log_path logs/ppo/small --train_root src/data --max_train_patients 125

echo "2. Training Medium Expert..."
python3 scripts/train/train_agent.py --refinement_mode medium --total_timesteps 200000 --save_path checkpoints/ppo_refiner_medium --log_path logs/ppo/medium --train_root src/data --max_train_patients 125

echo "3. Training Large Expert..."
python3 scripts/train/train_agent.py --refinement_mode large --total_timesteps 200000 --save_path checkpoints/ppo_refiner_large --log_path logs/ppo/large --train_root src/data --max_train_patients 125

echo "All PPO agents retrained successfully!"
