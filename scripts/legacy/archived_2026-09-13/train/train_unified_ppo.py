"""Train one PPO across size classes with bounded host memory."""
import gc
import json
import logging
import os
from pathlib import Path
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.envs.unified_mask_refinement_env import UnifiedMaskRefinementEnv, ClassConditionedFeatures, MODES
from src.data.patient_split import load_or_create_patient_split
from src.utils.metrics import dice, hd95
from scripts.train.train_agent import load_stage3_all_classes

class MixedEnv(gym.Env):
    def __init__(self, banks, device):
        self.envs = {}
        for mode in MODES:
            self.envs[mode] = {}
            for bucket, b in banks[mode].items():
                if b:
                    self.envs[mode][bucket] = UnifiedMaskRefinementEnv(
                        images=b[0], gt_masks=b[1], rough_masks=b[2],
                        uncertainty_maps=b[3], refinement_mode=mode,
                        device=device, target_dsc=1.0)
        first_mode = next(m for m in MODES if self.envs[m])
        first_bucket = next(iter(self.envs[first_mode]))
        self.observation_space = self.envs[first_mode][first_bucket].observation_space
        self.action_space = self.envs[first_mode][first_bucket].action_space
        self.current = self.envs[first_mode][first_bucket]
        self.episode = 0
        self.phase = 0
        self.buckets = ('fp', 'fn', 'mixed')

    def set_phase(self, phase):
        self.phase = int(np.clip(phase, 0, 3))

    def _bucket_weights(self, mode):
        available = [b for b in self.buckets if b in self.envs[mode]]
        if self.phase == 0:
            base = {'fp': .50, 'fn': .25, 'mixed': .25}
        elif self.phase == 1:
            base = {'fp': .25, 'fn': .50, 'mixed': .25}
        elif self.phase == 2:
            base = {'fp': 1 / 3, 'fn': 1 / 3, 'mixed': 1 / 3}
        else:
            counts = {b: len(self.envs[mode][b].images) for b in available}
            total = max(1, sum(counts.values()))
            base = {b: counts[b] / total for b in available}
        weights = np.asarray([base.get(b, 0.0) for b in available], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(len(available), dtype=np.float64)
        return available, weights / weights.sum()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mode = MODES[self.episode % 3]
        available, weights = self._bucket_weights(mode)
        bucket = available[int(self.np_random.choice(len(available), p=weights))]
        self.current = self.envs[mode][bucket]
        self.episode += 1
        obs, info = self.current.reset(seed=int(self.np_random.integers(0, 2**31-1)))
        info.update({'mode': mode, 'error_bucket': bucket, 'curriculum_phase': self.phase})
        return obs, info

    def step(self, action):
        return self.current.step(action)

def _error_bucket(gt, rough):
    gt_b = gt > .5
    rough_b = rough > .5
    fp = int((rough_b & (~gt_b)).sum())
    fn = int(((~rough_b) & gt_b).sum())
    if fp + fn == 0:
        return None
    if fp > fn * 1.25:
        return 'fp'
    if fn > fp * 1.25:
        return 'fn'
    return 'mixed'

def _empty_bucket_banks():
    return {m: {b: [] for b in ('fp', 'fn', 'mixed')} for m in MODES}

def _empty_priorities():
    return {m: {b: np.empty(0, dtype=np.float32) for b in ('fp', 'fn', 'mixed')} for m in MODES}

def _add_to_banks(banks, priorities, mode, arrays, rng, cap):
    if len(arrays[0]) == 0:
        return
    labels = np.asarray([_error_bucket(g, r) for g, r in zip(arrays[1], arrays[2])], dtype=object)
    for bucket in ('fp', 'fn', 'mixed'):
        sel = labels == bucket
        if not bool(sel.any()):
            continue
        new_arrays = [a[sel] for a in arrays]
        keys = rng.random(len(new_arrays[0])).astype(np.float32)
        if banks[mode][bucket]:
            candidates = [np.concatenate((old, new)) for old, new in zip(banks[mode][bucket], new_arrays)]
            keys = np.concatenate((priorities[mode][bucket], keys))
        else:
            candidates = new_arrays
        keep = np.argsort(keys)[:cap]
        banks[mode][bucket] = [a[keep].copy() for a in candidates]
        priorities[mode][bucket] = keys[keep]

def _load_balanced_banks(args, ids, rng, samples_per_bucket):
    banks = _empty_bucket_banks()
    priorities = _empty_priorities()
    for offset in range(0, len(ids), args.patient_chunk):
        chunk = ids[offset:offset + args.patient_chunk]
        bundle = load_stage3_all_classes(train_root=args.train_root, modality=args.modality,
            target_size=128, max_patients=None, patient_ids=chunk, noise_seed=args.seed)
        for mode, arrays in bundle.items():
            _add_to_banks(banks, priorities, mode, arrays, rng, samples_per_bucket)
            del arrays
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
        retained = {m: {b: len(priorities[m][b]) for b in ('fp', 'fn', 'mixed')} for m in MODES}
        logging.info('Prepared patients %d/%d; retained %s', min(offset + len(chunk), len(ids)), len(ids), retained)
    return banks

def _evaluate_model(model, banks, device, steps):
    from src.envs.zoom_ppo_refine import refine_zoom_ppo
    rows = []
    for mode in MODES:
        merged = None
        for bucket in ('fp', 'fn', 'mixed'):
            b = banks[mode][bucket]
            if not b:
                continue
            merged = b if merged is None else [np.concatenate((x, y)) for x, y in zip(merged, b)]
        if not merged:
            continue
        rough_d, ppo_d, rough_fp, ppo_fp, rough_fn, ppo_fn, rough_h, ppo_h = ([] for _ in range(8))
        for i in range(len(merged[0])):
            gt = merged[1][i] > .5
            rough = merged[2][i] > .5
            refined = refine_zoom_ppo(
                model, merged[0][i], gt, rough, merged[3][i], mode,
                device=device, enable_stop=False, seed=i,
                steps_per_patch=steps, gt_free=True,
            ) > .5
            rough_d.append(dice(rough, gt)); ppo_d.append(dice(refined, gt))
            rough_fp.append(int((rough & (~gt)).sum())); ppo_fp.append(int((refined & (~gt)).sum()))
            rough_fn.append(int(((~rough) & gt).sum())); ppo_fn.append(int(((~refined) & gt).sum()))
            rough_h.append(hd95(rough, gt)); ppo_h.append(hd95(refined, gt))
        rows.append({
            'mode': mode, 'n': len(merged[0]),
            'rough_dice': float(np.mean(rough_d)), 'ppo_dice': float(np.mean(ppo_d)),
            'delta_dice': float(np.mean(ppo_d) - np.mean(rough_d)),
            'rough_fp': float(np.mean(rough_fp)), 'ppo_fp': float(np.mean(ppo_fp)),
            'rough_fn': float(np.mean(rough_fn)), 'ppo_fn': float(np.mean(ppo_fn)),
            'rough_hd95': float(np.mean(rough_h)), 'ppo_hd95': float(np.mean(ppo_h)),
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', default='src/data/archive')
    parser.add_argument('--patient_split', default='checkpoints/patient_split.json')
    parser.add_argument('--max_train_patients', type=int, default=1251)
    parser.add_argument('--modality', default='t1ce+flair')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ppo_timesteps', type=int, default=10000)
    parser.add_argument('--patient_chunk', type=int, default=16)
    parser.add_argument('--samples_per_mode', type=int, default=128)
    parser.add_argument('--eval_samples_per_mode', type=int, default=48)
    parser.add_argument('--eval_steps', type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    os.environ.setdefault('BRATS_NUM_WORKERS', '4')
    split = load_or_create_patient_split(args.train_root, args.max_train_patients, args.patient_split)
    rng = np.random.default_rng(args.seed)
    ids = split['train']
    samples_per_bucket = max(1, int(np.ceil(args.samples_per_mode / 3)))
    banks = _load_balanced_banks(args, ids, rng, samples_per_bucket)
    if any(not any(banks[m].values()) for m in MODES):
        raise RuntimeError('Training split must supply all three size classes')
    env = MixedEnv(banks, 'cuda')
    class Progress(BaseCallback):
        def __init__(self, total_timesteps, init_ent=0.01, final_ent=0.001):
            super().__init__()
            self.total_timesteps = max(1, total_timesteps)
            self.init_ent = init_ent
            self.final_ent = final_ent

        def _on_step(self):
            progress = min(1.0, self.num_timesteps / self.total_timesteps)
            phase = min(3, int(progress * 4))
            self.training_env.env_method('set_phase', phase)
            # Cosine decay: 0.01 -> 0.001
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            current_ent = float(self.final_ent + (self.init_ent - self.final_ent) * cosine_decay)
            self.model.ent_coef = current_ent
            if self.num_timesteps % 500 == 0:
                logging.info('Unified PPO %d/%d phase=%d ent_coef=%.5f', self.num_timesteps, self.total_timesteps, phase, current_ent)
            return True

    model = PPO('MultiInputPolicy', env, n_steps=512, batch_size=256, n_epochs=3,
        policy_kwargs={'features_extractor_class':ClassConditionedFeatures, 'normalize_images':False},
        ent_coef=0.01, seed=args.seed, device='cuda', verbose=1)
    model.learn(args.ppo_timesteps, callback=Progress(args.ppo_timesteps))
    model.unified_metadata = {
        'policy_type': 'unified_class_conditioned',
        'action_dim': 8,
        'action_bins': 5,
        'mode_ids': dict(zip(MODES, range(3))),
        'enable_stop': False,
        'thresholds': {'small': 0.80, 'medium': 0.80, 'large': 0.70},
        'strategy_config': {
            'small': {'band_px': 5, 'uncert_floor': 0.10, 'action_rule': 'allow_all'},
            'medium': {'band_px': 2, 'uncert_floor': 0.25, 'action_rule': 'boundary_centered'},
            'large': {'band_px': 2, 'uncert_floor': 0.35, 'action_rule': 'conservative_remove_generous_add'}
        },
        'reward_spec': {
            'w_d': 20.0,
            'w_b': 10.0,
            'base_w': 15.0,
            'w_h': 1.0,
            'w_e': 0.5
        },
        'num_timesteps': int(model.num_timesteps)
    }
    model.save('checkpoints/ppo_unified.zip')
    Path('checkpoints/ppo_unified.json').write_text(json.dumps(model.unified_metadata, indent=2))
    logging.info('Saved unified PPO: %d actual steps', model.num_timesteps)
    if args.eval_samples_per_mode > 0 and split.get('val'):
        eval_args = argparse.Namespace(**vars(args))
        eval_args.samples_per_mode = args.eval_samples_per_mode
        eval_rng = np.random.default_rng(args.seed + 999)
        eval_banks = _load_balanced_banks(eval_args, split['val'], eval_rng,
            max(1, int(np.ceil(args.eval_samples_per_mode / 3))))
        rows = _evaluate_model(model, eval_banks, 'cuda', args.eval_steps)
        Path('results').mkdir(exist_ok=True)
        Path('results/unified_ppo_eval.json').write_text(json.dumps(rows, indent=2))
        for r in rows:
            logging.info('VAL %s n=%d dice %.4f -> %.4f Δ=%+.4f | FP %.1f -> %.1f | FN %.1f -> %.1f | HD95 %.2f -> %.2f',
                r['mode'], r['n'], r['rough_dice'], r['ppo_dice'], r['delta_dice'],
                r['rough_fp'], r['ppo_fp'], r['rough_fn'], r['ppo_fn'],
                r['rough_hd95'], r['ppo_hd95'])

if __name__ == '__main__':
    main()
