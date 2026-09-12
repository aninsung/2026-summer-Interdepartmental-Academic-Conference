Active pipeline entrypoints
==========================

Use `run_pipeline.py` from the repository root.

Current 4-stage flow:

1. `scripts/train/train_shape_classifier.py`
2. `scripts/train/train_stage2_all.py`
3. `scripts/train/train_ppo_mask_refiner.py`
4. `scripts/eval/evaluate_pipeline.py`

Required helper scripts:

- `scripts/train/train_caranet.py`
- `scripts/train/train_unetplusplus.py`
- `scripts/train/train_segresnet.py`
- `scripts/train/train_agent.py`
- `scripts/eval/paired_stats.py`

Stage 3 is a single PPO mask refiner:

- checkpoint: `checkpoints/ppo_mask_refiner.zip`
- metadata: `checkpoints/ppo_mask_refiner.json`
- no `ppo_small.zip`, `ppo_medium.zip`, or `ppo_large.zip` in the active pipeline

Legacy / experiment-only files
==============================

Files under `scripts/legacy/` may reference older class-wise PPO checkpoints,
paper-generation utilities, standalone simulations, or obsolete comparison
experiments. They are kept for reproducibility, but `run_pipeline.py` does not
call them.
