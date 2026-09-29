import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from scripts.eval import evaluate_pipeline as ev
from src.utils.evaluation_records import read_records
from src.utils.quality_gate import FEATURE_NAMES


class EvaluationModeTests(unittest.TestCase):
    def test_modes_record_real_outputs_and_oracle_opt_in(self):
        image = np.zeros((1, 2, 16, 16), dtype=np.float32)
        gt = np.zeros((1, 16, 16), dtype=np.float32); gt[:, 4:12, 4:12] = 1
        dataset = Mock()
        dataset.__len__ = Mock(return_value=1)
        dataset._sample_pids = ['heldout']; dataset._sample_zs = [7]
        dataset.get_numpy_arrays.return_value = (image, gt, gt)
        dataset.get_numpy_25d_arrays.return_value = np.zeros((1, 6, 16, 16), dtype=np.float32)
        probability = torch.from_numpy(gt[:, None] * .98)
        pipeline = Mock(return_value=(probability, torch.tensor([0])))
        names = ['shape_classifier_best.pt', 'caranet_best.pt', 'unetplusplus_best.pt', 'segresnet_best.pt',
                 'ppo_small.zip', 'ppo_medium.zip', 'ppo_large.zip']
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(temp)
                Path('checkpoints').mkdir()
                for name in names:
                    Path('checkpoints', name).write_bytes(b'test fixture, never loaded')
                artifact = dict(schema_version=1, features=FEATURE_NAMES, mean=[0.]*12, scale=[1.]*12,
                                weights=[0.]*12, bias=0., threshold=1., fit_patients=['fit'], calibration_patients=['cal'],
                                checkpoint_hashes={n: hashlib.sha256(b'test fixture, never loaded').hexdigest() for n in names},
                                evaluation_config=dict(stage2_thresholds='0.80,0.80,0.50', cc_min_sizes='0,15,25', modality='t1ce+flair',
                                                       slice_selection='tumor', confidence_threshold=None, small_confidence_threshold=None,
                                                       micro_area_floor=80., micro_thr_floor=.15, refinement_profile="legacy"))
                Path('gate.json').write_text(json.dumps(artifact))
                for mode, extra in [('stage2', []), ('augmentation', []), ('ppo_raw', []), ('heuristic', []),
                                    ('compare', []), ('oracle', []), ('quality', ['--quality_gate', 'gate.json']),
                                    ('compare', ['--quality_gate', 'gate.json', '--allow_oracle_gate'])]:
                    directory = f'run{len(list(Path(temp).glob("run*")))}'
                    argv = ['evaluate', '--eval_mode', mode, '--output_dir', directory, '--no_plots', '--save_masks', *extra]
                    with patch('sys.argv', argv), patch.object(ev, 'BraTS2020Dataset', return_value=dataset), \
                         patch.object(ev, 'load_or_create_patient_split', return_value={'val': ['heldout']}), \
                         patch.object(ev, 'AdaptivePipeline', return_value=pipeline), \
                         patch.object(ev.PPO, 'load', return_value=Mock(refinement_profile='legacy')), \
                         patch.object(ev, '_refine_with_ppo', side_effect=lambda agent, img, initial, *a, **kw: np.zeros_like(initial)) as refine:
                        ev.main()
                    meta, records = read_records(directory)
                    self.assertTrue(meta['complete'])
                    self.assertEqual(records[0]['slice_z'], 7)
                    self.assertEqual('oracle' in records[0]['methods'], mode == 'oracle' or '--allow_oracle_gate' in extra)
                    if mode in {'stage2', 'augmentation'}:
                        refine.assert_not_called()
                    if 'ppo_raw' in records[0]['methods']:
                        self.assertGreaterEqual(records[0]['methods']['ppo_raw']['dsc'], 0.0)
                    if 'heuristic' in records[0]['methods']:
                        self.assertGreaterEqual(records[0]['methods']['heuristic']['dsc'], 0.0)
                    if mode == 'quality':
                        self.assertFalse(records[0]['quality_accepted'])
                    self.assertTrue((Path(directory)/records[0]['mask_file']).exists())
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
