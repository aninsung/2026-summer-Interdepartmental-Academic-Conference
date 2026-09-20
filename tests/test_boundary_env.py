"""Regression tests for policy-only transitions, oracle rewards and cleanup."""
import copy
import unittest

import numpy as np

from src.boundary.env import BoundaryEnv
from src.boundary.metrics import quality
from src.boundary.postprocessing import remove_small_components


def case_with_island():
    shape = (16, 16, 16)
    target = np.zeros(shape, np.uint8)
    target[5:11, 5:11, 5:11] = 2
    target[6:10, 6:10, 6:10] = 1
    target[7:9, 7:9, 7:9] = 3
    initial = target.copy()
    initial[1, 1, 1] = 2
    logits = np.full((4,) + shape, -.25, np.float32)
    np.put_along_axis(logits, initial[None].astype(np.int64), .25, axis=0)
    return dict(patient="island_fixture", image=np.zeros_like(logits),
                logits=logits, target=target, spacing=(1., 1., 1.))


class BoundaryRewardTests(unittest.TestCase):
    def test_keep_final_step_is_identity_in_training_and_label_free_inference(self):
        case = case_with_island()
        no_label = {key: value for key, value in case.items() if key != "target"}
        for reward_enabled, data in ((True, case), (False, no_label)):
            with self.subTest(reward_enabled=reward_enabled):
                env = BoundaryEnv([data], patch_size=16, max_steps=1,
                                  reward_enabled=reward_enabled)
                env.reset(seed=1)
                initial = env.mask.copy()
                _, reward, done, _, info = env.step(6)
                self.assertTrue(done)
                self.assertEqual(reward, 0.)
                self.assertEqual(info["changed_voxels"], 0)
                np.testing.assert_array_equal(env.mask, initial)
                self.assertEqual(env.mask[1, 1, 1], 2)
                self.assertEqual("delta_dsc" in info, reward_enabled)

    def test_oracle_matches_every_actual_action_at_initial_and_final_step(self):
        for prefix in ((), (0, 3)):
            env = BoundaryEnv([case_with_island()], patch_size=12, max_steps=3)
            env.reset(seed=3)
            for action in prefix:
                env.step(action)
            before = copy.deepcopy(env.__dict__)
            gains, changed = env.action_gains()
            for key in ("initial", "logits", "mask", "edited", "band"):
                np.testing.assert_array_equal(getattr(env, key), before[key])
            self.assertEqual(env.current_metrics, before["current_metrics"])
            self.assertEqual((env.t, env.done), (before["t"], before["done"]))
            self.assertEqual(env.np_random.bit_generator.state,
                             before["_np_random"].bit_generator.state)
            for action in range(7):
                actual = copy.deepcopy(env)
                _, reward, _, _, info = actual.step(action)
                self.assertAlmostEqual(float(gains[action]), reward, places=4)
                self.assertEqual(changed[action], info["changed_voxels"])
                self.assertAlmostEqual(reward, info["dsc_reward"] + info["hd95_reward"])

    def test_asymmetric_return_equals_quality_delta_minus_trajectory_penalty(self):
        env = BoundaryEnv([case_with_island()], patch_size=12, max_steps=3,
                          hd95_weight=.5, hd95_scale_mm=5., dsc_drop_penalty=3.)
        env.reset(seed=1)
        initial_quality = quality(env.current_metrics, env.hd95_weight, env.hd95_scale_mm)
        total, extra_penalty, dsc_drop_sum = 0., 0., 0.
        for action in (0, 3, 5):
            _, reward, _, _, info = env.step(action)
            total += reward
            extra_penalty += info["dsc_drop_extra_penalty"]
            dsc_drop_sum += max(-info["delta_dsc"], 0.)
        final_quality = quality(env.current_metrics, env.hd95_weight, env.hd95_scale_mm)
        self.assertGreater(dsc_drop_sum, 0.)
        self.assertAlmostEqual(extra_penalty,
                               env.reward_scale * (env.dsc_drop_penalty - 1.) * dsc_drop_sum)
        self.assertAlmostEqual(total,
                               env.reward_scale * (final_quality - initial_quality) - extra_penalty)

    def test_dsc_penalty_mitigates_but_does_not_guarantee_preservation(self):
        env = BoundaryEnv([case_with_island()], hd95_weight=.5, hd95_scale_mm=5.)
        before = {"mean": {"dice": .9, "hd95_mm": 5.}}
        after = {"mean": {"dice": .89, "hd95_mm": 4.6}}
        self.assertAlmostEqual(env._asymmetric_reward(before, after), 1.)

    def test_explicit_cleanup_preserves_input_and_nested_labels(self):
        initial = case_with_island()["logits"].argmax(0).astype(np.uint8)
        before = initial.copy()
        cleaned = remove_small_components(initial)
        np.testing.assert_array_equal(initial, before)
        self.assertFalse(np.shares_memory(initial, cleaned))
        self.assertEqual(cleaned[1, 1, 1], 0)
        self.assertEqual(np.count_nonzero(initial != cleaned), 1)
        np.testing.assert_array_equal(cleaned[5:11, 5:11, 5:11],
                                      initial[5:11, 5:11, 5:11])
        self.assertEqual(cleaned.dtype, initial.dtype)

    def test_cleanup_threshold_and_largest_component_rule(self):
        mask = np.zeros((12, 12, 12), np.uint8)
        mask[1:4, 1:4, 1:4] = 2  # Largest, 27 voxels.
        mask[8:10, 8:10, 8:10] = 3  # Eight voxels.
        np.testing.assert_array_equal(remove_small_components(mask, min_voxels=8), mask)
        self.assertEqual(np.count_nonzero(remove_small_components(mask, min_voxels=9)), 27)
        self.assertEqual(np.count_nonzero(remove_small_components(mask, min_voxels=100)), 27)
        empty = np.zeros_like(mask)
        np.testing.assert_array_equal(remove_small_components(empty), empty)


if __name__ == "__main__":
    unittest.main()
