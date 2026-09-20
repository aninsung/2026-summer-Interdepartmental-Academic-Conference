"""Scientific-contract tests; synthetic fixtures are not training results."""
import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from src.boundary.data import validate_split
from src.boundary.env import BoundaryEnv
from src.boundary.metrics import quality, region_metrics


def fixture():
    shape = (24, 24, 24)
    target = np.zeros(shape, np.uint8)
    target[5:19, 5:19, 5:19] = 2
    target[8:16, 8:16, 8:16] = 1
    target[10:14, 10:14, 10:14] = 3
    initial = target.copy()
    initial[4:5, 5:19, 5:19] = 2
    logits = np.full((4,) + shape, -.25, np.float32)
    np.put_along_axis(logits, initial[None].astype(np.int64), .25, axis=0)
    return dict(patient="synthetic_contract_test", image=np.zeros((4,) + shape, np.float32),
                target=target, logits=logits, spacing=[1., 1., 2.])


class BoundaryTests(unittest.TestCase):
    def test_identity_empty_and_spacing(self):
        target = fixture()["target"]
        exact = region_metrics(target, target, (1, 1, 2))
        self.assertEqual(exact["mean"]["dice"], 1.)
        self.assertEqual(exact["mean"]["hd95_mm"], 0.)
        empty = np.zeros_like(target)
        self.assertEqual(region_metrics(empty, empty, (1, 1, 2))["mean"]["dice"], 1.)
        missed = region_metrics(empty, target, (1, 1, 2), empty_distance_mm=99)
        self.assertEqual(missed["mean"]["hd95_mm"], 99.)
        shifted = np.roll(target, 1, axis=2)
        a = region_metrics(shifted, target, (1, 1, 1))
        b = region_metrics(shifted, target, (2, 2, 2))
        self.assertAlmostEqual(b["mean"]["hd95_mm"], 2 * a["mean"]["hd95_mm"])

    def test_no_ground_truth_in_observation_candidates_or_termination(self):
        case = fixture()
        other = dict(case, target=np.zeros_like(case["target"]))
        no_label = {k: v for k, v in case.items() if k != "target"}
        environments = [BoundaryEnv([c], patch_size=16, max_steps=3, reward_enabled=False)
                        for c in (case, other, no_label)]
        observations = [e.reset(seed=7)[0] for e in environments]
        for obs in observations[1:]:
            np.testing.assert_array_equal(obs, observations[0])
        for action in (0, 3, 6):
            transitions = [e.step(action) for e in environments]
            for transition in transitions[1:]:
                np.testing.assert_array_equal(transition[0], transitions[0][0])
                self.assertEqual(transition[2:4], transitions[0][2:4])

    def test_keep_and_oracle_restore_and_band(self):
        env = BoundaryEnv([fixture()], patch_size=16, max_steps=3)
        env.reset(seed=1)
        original, logits, edited = env.mask.copy(), env.logits.copy(), env.edited.copy()
        gains, _ = env.action_gains()
        self.assertEqual(gains[6], 0.)
        np.testing.assert_array_equal(env.mask, original)
        np.testing.assert_array_equal(env.logits, logits)
        np.testing.assert_array_equal(env.edited, edited)
        _, reward, _, _, _ = env.step(6)
        self.assertEqual(reward, 0.)
        np.testing.assert_array_equal(env.mask, original)
        env.step(0)
        np.testing.assert_array_equal(env.mask[~env.band], original[~env.band])

    def test_symmetric_reward_telescopes_and_hierarchy(self):
        env = BoundaryEnv([fixture()], patch_size=16, max_steps=3, dsc_drop_penalty=1.)
        env.reset(seed=1)
        before = quality(env.current_metrics)
        total = sum(env.step(a)[1] for a in (0, 3, 5))
        self.assertAlmostEqual(total, env.reward_scale * (quality(env.current_metrics) - before))
        et, tc, wt = env.mask == 3, np.isin(env.mask, (1, 3)), env.mask > 0
        self.assertFalse(np.any(et & ~tc))
        self.assertFalse(np.any(tc & ~wt))
        self.assertTrue(env.done)
        with self.assertRaises(RuntimeError):
            env.step(6)

    def test_gym_contract_and_empty_prediction(self):
        env = BoundaryEnv([fixture()], patch_size=16, max_steps=3)
        check_env(env, warn=False)
        case = fixture()
        case["logits"][:] = 0
        empty = BoundaryEnv([case], patch_size=16)
        obs, _ = empty.reset()
        self.assertTrue(empty.observation_space.contains(obs))
        _, _, done, truncated, _ = empty.step(0)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertFalse(empty.mask.any())

    def test_patient_overlap_rejected(self):
        with self.assertRaises(ValueError):
            validate_split(dict(backbone_train=["a"], rl_train=["a"], val=["b"], test=["c"]))

    def test_paired_summary_quadrants(self):
        from scripts.train.train_boundary_3d import paired_summary
        def dummy_record(et_dice, et_hd95):
            m = {"dice": et_dice, "hd95_mm": et_hd95}
            return {
                "initial": {"ET": {"dice": 0.5, "hd95_mm": 10.0}, "TC": m, "WT": m, "mean": m},
                "ppo": {"ET": {"dice": et_dice, "hd95_mm": et_hd95}, "TC": m, "WT": m, "mean": m}
            }
        records = [
            dummy_record(0.6, 8.0),
            dummy_record(0.6, 12.0),
            dummy_record(0.4, 8.0),
            dummy_record(0.4, 12.0)
        ]
        summary = paired_summary(records, "ppo", seed=42)
        et = summary["ET"]
        self.assertEqual(et["both_strictly_improved"], 1)
        self.assertEqual(et["dice_improved_hd95_worsened"], 1)
        self.assertEqual(et["hd95_improved_dice_worsened"], 1)
        self.assertEqual(et["both_worsened_or_unchanged"], 1)


    def test_restore_full_mask(self):
        from scripts.infer_boundary_3d import restore_full_mask
        mask_cropped = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_cropped[2:5, 2:5, 2:5] = 3
        case_info = {
            "cropped_shape": [10, 10, 10],
            "original_shape": [20, 20, 20],
            "crop": (slice(5, 15), slice(5, 15), slice(5, 15))
        }
        full = restore_full_mask(mask_cropped, case_info)
        self.assertEqual(full.shape, (20, 20, 20))
        # Internal class 3 should be converted to BraTS label 4
        self.assertEqual(full[7, 7, 7], 4)
        self.assertEqual(full[0, 0, 0], 0)


if __name__ == "__main__":
    unittest.main()

