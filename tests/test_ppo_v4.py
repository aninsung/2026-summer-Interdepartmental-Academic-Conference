from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.envs.mask_refinement_env import MaskRefinementEnv, V4_SECTORS


class PPOV4Tests(unittest.TestCase):
    def make_env(self, gt=None, mode='medium', prob=.5):
        rough = np.zeros((128, 128), dtype=np.float32)
        rough[48:80, 48:80] = 1
        image = np.zeros((1, 2, 128, 128), dtype=np.float32)
        image[0, 1] = .7
        return MaskRefinementEnv(image, (rough if gt is None else gt)[None], rough[None],
                                 uncertainty_maps=np.full((1, 128, 128), prob, dtype=np.float32),
                                 max_steps=15, refinement_mode=mode, refinement_profile='ppo_v4')

    def test_spaces_use_all_modalities_and_finer_sectors(self):
        for mode in ('small', 'medium', 'large'):
            env = self.make_env(mode=mode)
            obs, _ = env.reset(seed=0, options={'direction': 'expand'})
            side = 64 if mode == 'small' else 128
            self.assertEqual(obs.shape, (7, side, side))
            self.assertEqual(env.action_space.shape[0], V4_SECTORS[mode])
            self.assertTrue(np.allclose(obs[1][obs[1] > 0], .7))
            self.assertTrue(np.all(obs[-1] == 1.))

    def test_direction_constraint_blocks_opposite_moves(self):
        env = self.make_env()
        env.reset(seed=0, options={'direction': 'expand'})
        area = env._current_mask.sum()
        env.step(np.zeros(env.n_sectors, dtype=int))
        self.assertEqual(env._current_mask.sum(), area)
        env.reset(seed=0, options={'direction': 'shrink'})
        env.step(np.full(env.n_sectors, 4))
        self.assertEqual(env._current_mask.sum(), area)
        env.step(np.zeros(env.n_sectors, dtype=int))
        self.assertLess(env._current_mask.sum(), area)

    def test_confident_sectors_are_frozen(self):
        env = self.make_env(prob=.95)
        env.reset(seed=0, options={'direction': 'expand'})
        self.assertFalse(env._editable.any())
        original = env._current_mask.copy()
        env.step(np.full(env.n_sectors, 4))
        np.testing.assert_array_equal(env._current_mask, original)

    def test_reward_rewards_dsc_and_hd95_improvement(self):
        gt = np.zeros((128, 128), dtype=np.float32); gt[46:82, 46:82] = 1
        env = self.make_env(gt)
        env.reset(seed=0, options={'direction': 'expand'})
        _, reward, _, _, info = env.step(np.full(env.n_sectors, 4))
        self.assertGreater(reward, 0)
        self.assertLess(info['hd95'], info['initial_hd95'])
        env.reset(seed=0, options={'direction': 'shrink'})
        _, reward, _, _, _ = env.step(np.zeros(env.n_sectors, dtype=int))
        self.assertLess(reward, 0)

    def test_selective_inference_and_short_training(self):
        from stable_baselines3 import PPO
        from scripts.train.train_agent import SelectiveGainCallback
        from scripts.eval.evaluate_pipeline import _selective_refine
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            env = self.make_env(mode='small')
            with tempfile.TemporaryDirectory() as temp:
                model = PPO('CnnPolicy', env, n_steps=2, batch_size=2, n_epochs=1, device='cpu',
                            policy_kwargs={'normalize_images': False, 'net_arch': [16]}, seed=42)
                model.refinement_profile = 'ppo_v4'
                callback = SelectiveGainCallback(env.images, env.gt_masks, env.rough_masks,
                                                 env.probability_maps, 'small', temp, eval_freq=4, max_samples=1)
                model.learn(total_timesteps=4, callback=callback)
                loaded = PPO.load(Path(temp)/'best_model.zip', device='cpu')
                self.assertEqual(loaded.refinement_profile, 'ppo_v4')
                out = _selective_refine(loaded, env.images[0], env.rough_masks[0],
                                        env.probability_maps[0], 'small', clip_shrink=True)
                self.assertEqual(out.shape, (128, 128))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
