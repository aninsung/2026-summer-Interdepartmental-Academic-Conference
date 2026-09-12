"""Backward-compatible imports for the renamed PPO mask refiner env."""
from src.envs.ppo_mask_refinement_env import ClassConditionedFeatures, MODES, PPOMaskRefinementEnv

UnifiedMaskRefinementEnv = PPOMaskRefinementEnv
