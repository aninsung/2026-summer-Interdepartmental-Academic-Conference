Active Pipeline Entrypoints
==========================

Use `run_pipeline.py` from the repository root to launch the full 4-stage pipeline.

Current 4-Stage Flow:
1. `scripts/train/train_shape_classifier.py` (Stage 1: Shape Classifier, ResNet-18 P2)
2. `scripts/train/train_stage2_all.py` (Stage 2: Experts - CaraNet 2.5D, UNet++, SegResNet)
3. `scripts/train/train_ppo_mask_refiner.py` (Stage 3: Alternating SL / PPO Refiner)
4. `scripts/eval/evaluate_pipeline.py` (Stage 4: Evaluation - BraTS Multi-Region ET/TC/WT)

Required Helper Scripts:
- `scripts/train/train_caranet.py`
- `scripts/train/train_unetplusplus.py`
- `scripts/train/train_segresnet.py`
- `scripts/eval/paired_stats.py`
