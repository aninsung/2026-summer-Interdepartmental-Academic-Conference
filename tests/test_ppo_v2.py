from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.envs.mask_refinement_env import MaskRefinementEnv
from src.utils.refinement_inputs import component_inputs


class PPOV2Tests(unittest.TestCase):
    def make_env(self, gt=None, mode='medium'):
        rough = np.zeros((128, 128), dtype=np.float32)
        rough[48:80, 48:80] = 1
        return MaskRefinementEnv(np.zeros((1, 2, 128, 128), dtype=np.float32),
                                 (rough if gt is None else gt)[None], rough[None],
                                 uncertainty_maps=np.full((1, 128, 128), .5, dtype=np.float32),
                                 max_steps=15, refinement_mode=mode, refinement_profile='ppo_v2')

    def test_keep_preserves_good_mask_without_reward_farming(self):
        env = self.make_env()
        env.reset(seed=0)
        original = env._current_mask.copy()
        for _ in range(15):
            _, reward, terminated, _, _ = env.step(np.full(8, 2))
            np.testing.assert_array_equal(env._current_mask, original)
            self.assertEqual(reward, 0)
            self.assertFalse(terminated)

    def test_shrink_and_accumulated_weak_expansion_change_boundary(self):
        env = self.make_env(); env.reset(seed=0)
        area = env._current_mask.sum()
        env.step(np.zeros(8, dtype=int))
        self.assertLess(env._current_mask.sum(), area)
        env.reset(seed=0)
        env.step(np.full(8, 3))
        self.assertEqual(env._current_mask.sum(), area)
        env.step(np.full(8, 3))
        self.assertGreater(env._current_mask.sum(), area)

    def test_improvement_reward_has_correct_sign(self):
        gt = np.zeros((128, 128), dtype=np.float32); gt[47:81, 47:81] = 1
        env = self.make_env(gt); env.reset(seed=0)
        _, reward, _, _, info = env.step(np.full(8, 4))
        self.assertGreater(reward, 0)
        self.assertGreater(info['delta_from_initial'], 0)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.zeros(8, dtype=int))
        self.assertLess(reward, 0)
        self.assertLess(info['delta_from_initial'], 0)

    def test_probability_evidence_outside_initial_is_preserved(self):
        rough = np.zeros((32, 32), dtype=np.float32); rough[10:20, 10:20] = 1
        p = np.full_like(rough, .4); p[10:20, 10:20] = .9
        _, route, initial, probability = next(component_inputs(rough, p, .8))
        self.assertEqual(route, 0)
        self.assertEqual(initial[9, 15], 0)
        self.assertGreater(probability[9, 15], 0)
        self.assertEqual(probability[0, 0], 0)

    def test_short_real_ppo_training_saves_profile_and_raw_gain_selection(self):
        from stable_baselines3 import PPO
        from scripts.train.train_agent import RefinementGainCallback
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            env = self.make_env(mode='small')
            with tempfile.TemporaryDirectory() as temp:
                model = PPO('CnnPolicy', env, n_steps=2, batch_size=2, n_epochs=1, device='cpu',
                            policy_kwargs={'normalize_images': False, 'net_arch': [16]}, seed=42)
                model.refinement_profile = 'ppo_v2'
                callback = RefinementGainCallback(env.images, env.gt_masks, env.rough_masks,
                                                  env.probability_maps, 'small', temp, eval_freq=4, max_samples=1)
                model.learn(total_timesteps=4, callback=callback)
                loaded = PPO.load(Path(temp)/'best_model.zip', device='cpu')
                self.assertEqual(loaded.refinement_profile, 'ppo_v2')
                self.assertTrue((Path(temp)/'selection.json').exists())
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
