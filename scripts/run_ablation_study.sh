#!/bin/bash
echo "Starting Ablation Study 3: Max Steps = 64"
python scripts/train/train_boundary_3d.py --config configs/ablation_max_steps_64.json --stage all

echo "Starting Ablation Study 3: Max Steps = 1024"
python scripts/train/train_boundary_3d.py --config configs/ablation_max_steps_1024.json --stage all

echo "Ablation Studies Completed!"
