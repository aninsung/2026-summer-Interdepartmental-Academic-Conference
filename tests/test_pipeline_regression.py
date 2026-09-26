"""Dependency-light regression checks for inference guards and runner wiring."""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluation_helpers(**overrides):
    tree = ast.parse((ROOT / 'scripts/eval/evaluate_pipeline.py').read_text())
    names = {'_gt_free_accept', '_guard_refinement', '_refine_with_ppo'}
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = {'np': np, **overrides}
    exec(compile(functions, '<evaluation helpers>', 'exec'), namespace)
    return namespace


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.edge = Mock(return_value=True)
        self.oracle = Mock(side_effect=lambda rough, refined, gt: rough)
        self.h = evaluation_helpers(_edge_accept=self.edge, apply_monotonic_dsc_gate=self.oracle)
        self.rough = np.ones((4, 4), dtype=np.float32)

    def test_default_never_calls_oracle(self):
        candidate = self.rough.copy()
        candidate[0] = 0
        out = self.h['_guard_refinement'](None, self.rough, candidate)
        np.testing.assert_array_equal(out, candidate)
        self.oracle.assert_not_called()

    def test_area_and_edge_rejections(self):
        for candidate in (np.zeros_like(self.rough), self.rough.copy()):
            self.edge.return_value = False
            out = self.h['_guard_refinement'](None, self.rough, candidate)
            np.testing.assert_array_equal(out, self.rough)
        self.oracle.assert_not_called()

    def test_explicit_oracle(self):
        self.h['_guard_refinement'](None, self.rough, self.rough, oracle_gt=self.rough)
        self.oracle.assert_called_once()

    def test_rollout_uses_final_step_without_gt_selection(self):
        env = Mock()
        env._current_mask = self.rough.copy()
        env.reset.return_value = (None, {})
        def step(action):
            env._current_mask = env._current_mask * 0.9
            return None, 0, False, False, {}
        env.step.side_effect = step
        factory = Mock(return_value=env)
        agent = Mock()
        agent.predict.return_value = (np.zeros(8), None)
        helper = evaluation_helpers(MaskRefinementEnv=factory)['_refine_with_ppo']
        out = helper(agent, self.rough, self.rough, self.rough, 'small', n_steps=3)
        np.testing.assert_allclose(out, self.rough * 0.9**3)
        self.assertEqual(env.step.call_count, 3)
        self.assertEqual(factory.call_args.kwargs['target_dsc'], float('inf'))
        agent.predict.side_effect = ValueError('bad policy')
        with self.assertRaises(ValueError):
            helper(agent, self.rough, self.rough, self.rough, 'small')


class RunnerTests(unittest.TestCase):
    def test_eval_settings_forwarded(self):
        import run_pipeline
        split_module = types.ModuleType('src.data.patient_split')
        split_module.load_or_create_patient_split = Mock(return_value={'train': ['a'], 'val': ['b'], 'seed': 42})
        for extra in ([], ['--allow_oracle_gate', '--disable_edge_gate', '--no_deterministic']):
            argv = ['run_pipeline.py', '--skip_classifier', '--skip_experts', '--skip_agents', *extra]
            with patch.dict(sys.modules, {'src.data.patient_split': split_module}), patch.object(sys, 'argv', argv), patch.object(run_pipeline, 'run_command') as run:
                run_pipeline.main()
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--stage2_thresholds') + 1], '0.80,0.80,0.50')
            self.assertEqual('--allow_oracle_gate' in cmd, bool(extra))
            self.assertIn('--no_deterministic' if extra else '--deterministic', cmd)


if __name__ == '__main__':
    unittest.main()
