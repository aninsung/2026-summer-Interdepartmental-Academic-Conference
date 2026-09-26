"""V3 mechanics and wiring; synthetic fixtures are not performance evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from src.envs.mask_refinement_env import MaskRefinementEnv
from src.utils.metrics import dice
from scripts.eval.evaluate_pipeline import _refine_with_ppo, refinement_candidates


class ConstantAgent:
    refinement_profile = 'ppo_v3'

    def __init__(self, action):
        self.action = action

    def predict(self, obs, deterministic=True):
        return self.action, None


class V3Tests(unittest.TestCase):
    def env(self, mode='medium', rough=None, gt=None):
        if rough is None:
            rough = np.zeros((64, 64), np.float32)
            rough[20:44, 20:44] = 1
        images = np.zeros((1, 2, *rough.shape), np.float32)
        images[:, 1] = .7
        return MaskRefinementEnv(images, (rough if gt is None else gt)[None], rough[None],
                                 uncertainty_maps=np.full((1, *rough.shape), .5, np.float32),
                                 max_steps=15, refinement_profile='ppo_v3', refinement_mode=mode)

    def test_observes_flair_accumulated_sdf_time_and_constraints(self):
        weak, keep = self.env(), self.env()
        obs, _ = weak.reset(seed=0); keep.reset(seed=0)
        self.assertEqual(obs.shape, (8, 64, 64))
        self.assertTrue(weak.observation_space.contains(obs))
        np.testing.assert_allclose(obs[1], .7)
        a, *_ = weak.step(np.full(8, 3))
        b, *_ = keep.step(np.full(8, 2))
        np.testing.assert_array_equal(a[2], b[2])
        self.assertFalse(np.array_equal(a[4], b[4]))
        np.testing.assert_allclose(a[5], 14/15)
        weak.step(np.full(8, 3)); keep.step(np.full(8, 3))
        self.assertGreater(weak._current_mask.sum(), keep._current_mask.sum())

    def test_small_false_positive_can_be_removed_in_training_and_inference(self):
        mask = np.zeros((64, 64), np.float32); mask[28:33, 28:33] = 1
        env = self.env('small', mask, np.zeros_like(mask))
        env.reset(seed=0)
        for _ in range(15):
            obs, reward, _, _, info = env.step(np.full(8, -2.))
        self.assertEqual(env._current_mask.sum(), 0)
        self.assertAlmostEqual(info['dsc'], 1.)
        self.assertTrue(env.observation_space.contains(obs))
        agent = ConstantAgent(np.full(8, -2.))
        result = _refine_with_ppo(agent, env.images[0], mask, env.probability_maps[0], 'small', clip_shrink=True)
        self.assertEqual(result.sum(), 0)
        masks, calls, _ = refinement_candidates({0: agent}, env.images[0], mask,
                                                env.probability_maps[0], 0, .8)
        self.assertEqual(masks['ppo_raw'].sum(), 0)
        self.assertEqual(calls, 1)

    def test_keep_preserves_holes_and_disconnected_components_without_closing(self):
        mask = np.zeros((64, 64), np.float32)
        mask[10:25, 10:25] = 1; mask[16, 16] = 0; mask[45:48, 45:48] = 1
        env = self.env('small', mask)
        env.reset(seed=0)
        agent = ConstantAgent(np.zeros(8))
        masks, _, _ = refinement_candidates({0: agent}, env.images[0], mask,
                                            env.probability_maps[0], 0, .8)
        np.testing.assert_array_equal(masks['ppo_raw'], mask)
        np.testing.assert_array_equal(masks['augmentation'], mask)
        for _ in range(15):
            _, reward, _, _, _ = env.step(np.zeros(8))
            self.assertEqual(reward, 0.)

    def test_reward_uses_exact_whole_slice_delta(self):
        env = self.env(); obs, _ = env.reset(seed=0)
        previous = dice(env._current_mask, env._current_gt)
        _, reward, _, _, info = env.step(np.full(8, 4))
        expected = dice(env._current_mask, env._current_gt) - previous
        self.assertAlmostEqual(reward, 100 * expected - env.step_penalty)
        self.assertAlmostEqual(info['delta_dsc'], expected)

    def test_validation_selection_weights_patients_not_slices(self):
        from scripts.train.train_agent import WholeSliceGainCallback
        ones = np.ones((64, 64), np.float32); zeros = np.zeros_like(ones)
        roughs = np.stack([zeros, zeros, ones])
        candidates = np.stack([ones, ones, zeros])
        with tempfile.TemporaryDirectory() as tmp:
            cb = WholeSliceGainCallback(np.zeros((3, 2, 64, 64), np.float32),
                                        np.stack([ones]*3), roughs, candidates,
                                        ['a', 'a', 'b'], 'small', tmp, 1)
            cb.model = SimpleNamespace(logger=Mock(), save=Mock())
            with patch('scripts.eval.evaluate_pipeline._refine_with_ppo',
                       side_effect=lambda agent, image, rough, probability, mode: probability):
                cb._evaluate()
            report = json.loads((Path(tmp)/'selection.json').read_text())
            self.assertAlmostEqual(report['patient_mean_delta_dsc'], 0.)
            self.assertEqual(report['patients'], 2)
            self.assertFalse(report['improves_baseline'])

    def test_short_ppo_train_save_reload_v3_observation(self):
        from stable_baselines3 import PPO
        from scripts.train.train_agent import WholeSliceGainCallback
        previous = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            env = self.env('small')
            with tempfile.TemporaryDirectory() as tmp:
                model = PPO('CnnPolicy', env, n_steps=2, batch_size=2, n_epochs=1,
                            device='cpu', seed=42,
                            policy_kwargs={'normalize_images': False, 'net_arch': [16]})
                model.refinement_profile = 'ppo_v3'
                cb = WholeSliceGainCallback(env.images, env.gt_masks, env.rough_masks,
                                            env.probability_maps, ['patient'], 'small', tmp, 4)
                model.learn(total_timesteps=4, callback=cb)
                loaded = PPO.load(Path(tmp)/'best_model.zip', device='cpu')
                self.assertEqual(loaded.refinement_profile, 'ppo_v3')
                self.assertEqual(loaded.observation_space.shape, (8, 64, 64))
                self.assertTrue((Path(tmp)/'selection.json').exists())
        finally:
            torch.set_num_threads(previous)

    def test_v3_training_uses_all_slices_full_gt_and_predicted_routes(self):
        from scripts.train.train_agent import load_real_data
        images = np.zeros((3, 2, 64, 64), np.float32)
        gts = np.zeros((3, 64, 64), np.float32); gts[1, 3:60, 3:60] = 1
        ds = Mock(); ds.get_numpy_arrays.return_value = (images, gts, gts)
        ds.get_numpy_25d_arrays.return_value = np.zeros((3, 6, 64, 64), np.float32)
        ds._sample_pids = ['a', 'b', 'c']
        def predict(x, true_class_preds=None):
            probability = torch.full((len(x), 1, 64, 64), .9)
            return probability, torch.zeros(len(x), dtype=torch.long)
        metadata = {}
        with patch('src.data.brats2020_dataset.BraTS2020Dataset', return_value=ds) as factory, \
             patch('src.models.dynamic_router.AdaptivePipeline', return_value=predict), \
             patch('scripts.train.train_agent.os.path.exists', return_value=True), \
             patch('torch.cuda.is_available', return_value=False):
            _, gt, masks, probabilities = load_real_data('root', 't1ce+flair', 64, 3,
                patient_ids=['a', 'b', 'c'], refinement_mode='small',
                refinement_profile='ppo_v3', sample_metadata=metadata)
        self.assertEqual(factory.call_args.kwargs['slice_selection'], 'all')
        np.testing.assert_array_equal(gt, gts)
        self.assertEqual(metadata['patient_ids'], ['a', 'b', 'c'])
        self.assertEqual(masks.shape, gt.shape)
        self.assertEqual(probabilities.shape, gt.shape)

    def test_runner_forwards_v3_compare_and_keeps_test_untouched(self):
        import run_pipeline
        args = ['run_pipeline.py', '--config', 'configs/ppo_brats_v3.yaml',
                '--skip_classifier', '--skip_experts', '--skip_agents',
                '--eval_mode', 'compare', '--no_plots', '--output_dir', 'results/v3_test_fixture']
        with patch('sys.argv', args), \
             patch('src.data.patient_split.load_or_create_patient_split', return_value={}), \
             patch.object(run_pipeline, 'run_command') as run:
            run_pipeline.main()
        cmd = run.call_args.args[0]
        for key, value in [('refinement_profile', 'ppo_v3'), ('agent_dir', 'checkpoints/ppo_v3'),
                           ('split_role', 'val'), ('eval_mode', 'compare'), ('slice_selection', 'all')]:
            self.assertEqual(cmd[cmd.index('--'+key)+1], value)
        self.assertIn('--no_plots', cmd)

    def test_v3_compare_records_empty_output_without_guard_reversion(self):
        import contextlib
        import io
        import os
        from scripts.eval import evaluate_pipeline as ev
        from src.utils.evaluation_records import read_records
        images = np.zeros((1, 2, 64, 64), np.float32)
        mask = np.zeros((1, 64, 64), np.float32); mask[:, 28:33, 28:33] = 1
        ds = Mock(); ds.__len__ = Mock(return_value=1)
        ds._sample_pids = ['val']; ds._sample_zs = [10]
        ds.get_numpy_arrays.return_value = (images, np.zeros_like(mask), mask)
        ds.get_numpy_25d_arrays.return_value = np.zeros((1, 6, 64, 64), np.float32)
        pipeline = Mock(return_value=(torch.from_numpy(mask[:, None] * .99), torch.tensor([0])))
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(tmp); Path('checkpoints/ppo_v3').mkdir(parents=True)
                for name in ['shape_classifier', 'caranet', 'unetplusplus', 'segresnet']:
                    Path('checkpoints', name+'_best.pt').write_bytes(b'fixture')
                for name in ['small', 'medium', 'large']:
                    Path('checkpoints/ppo_v3', 'ppo_'+name+'.zip').write_bytes(b'fixture')
                argv = ['eval', '--refinement_profile', 'ppo_v3', '--slice_selection', 'all',
                        '--eval_mode', 'compare', '--output_dir', 'output', '--no_plots']
                with patch('sys.argv', argv), patch.object(ev, 'BraTS2020Dataset', return_value=ds), \
                     patch.object(ev, 'AdaptivePipeline', return_value=pipeline), \
                     patch.object(ev, 'load_or_create_patient_split', return_value={'val': ['val']}), \
                     patch.object(ev.PPO, 'load', return_value=ConstantAgent(np.full(8, -2.))):
                    ev.main()
                meta, records = read_records('output')
                self.assertTrue(meta['complete'])
                self.assertEqual(meta['ppo_unit'], 'slice')
                self.assertEqual(records[0]['ppo_slice_calls'], 1)
                self.assertEqual(records[0]['ppo_component_calls'], 0)
                self.assertEqual(records[0]['methods']['ppo_raw']['predicted_pixels'], 0)
                self.assertEqual(records[0]['methods']['ppo_raw']['dsc'], 1.)
                self.assertNotIn('heuristic', records[0]['methods'])
            finally:
                os.chdir(previous)

    def test_v3_observation_and_actions_do_not_depend_on_gt(self):
        a = self.env()
        b = self.env(gt=np.zeros((64, 64), np.float32))
        obs_a, _ = a.reset(seed=0); obs_b, _ = b.reset(seed=0)
        np.testing.assert_array_equal(obs_a, obs_b)
        for _ in range(15):
            obs_a, _, _, _, _ = a.step(np.full(8, 3))
            obs_b, _, _, _, _ = b.step(np.full(8, 3))
            np.testing.assert_array_equal(obs_a, obs_b)
            np.testing.assert_array_equal(a._current_mask, b._current_mask)


if __name__ == '__main__':
    unittest.main()
