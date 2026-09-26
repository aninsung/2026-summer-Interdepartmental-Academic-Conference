import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.utils.evaluation_records import measured_metrics, RecordWriter, read_records
from src.utils.quality_gate import pair_features, QualityGate, FEATURE_NAMES
from scripts.eval.patient_level_stats import summarize
from scripts.train.train_quality_gate import fit_gate


def record(patient, z=0, initial=.8, final=.85):
    base = dict(dsc=initial, hd95_surface_px=1., hd95_one_empty=False,
                gt_pixels=10, predicted_pixels=10, false_positive_on_empty=False)
    return dict(patient_id=patient, slice_z=z, sample_index=z,
                methods={'stage2': base, 'ppo_raw': dict(base, dsc=final)},
                quality_features=[final-initial] + [0.]*(len(FEATURE_NAMES)-1),
                quality_accepted=None, shared_inference_seconds=.1)


class MeasuredEvaluationTests(unittest.TestCase):
    def test_surface_distance_and_empty_cases(self):
        a = np.zeros((16, 16)); a[4:8, 4:8] = 1
        b = np.zeros_like(a); b[4:8, 6:10] = 1
        self.assertEqual(measured_metrics(a, a)['hd95_surface_px'], 0)
        self.assertEqual(measured_metrics(a, b)['hd95_surface_px'], 2)
        self.assertIsNone(measured_metrics(a, np.zeros_like(a))['hd95_surface_px'])
        self.assertTrue(measured_metrics(a, np.zeros_like(a))['false_positive_on_empty'])
        self.assertEqual(measured_metrics(a*0, a*0)['dsc'], 1)

    def test_patient_weighting_and_negative_delta(self):
        records = [record('one', i, .8, .9) for i in range(10)] + [record('two', 0, .8, .6)]
        result = summarize(records)['methods']['ppo_raw']
        self.assertAlmostEqual(result['paired_delta_dsc'], -.05)
        self.assertEqual(result['harmed_patient_rate'], .5)
        self.assertAlmostEqual(result['worst_patient_delta'], -.2)

    def test_record_roundtrip_incomplete_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run'
            writer = RecordWriter(path, {'config': {}})
            writer.append(record('p'))
            with self.assertRaises(ValueError):
                read_records(path)
            writer.finish()
            self.assertEqual(len(read_records(path)[1]), 1)
            writer.append(record('p'))
            with self.assertRaises(ValueError):
                read_records(path)
            with self.assertRaises(FileExistsError):
                RecordWriter(path, {})

    def test_features_are_finite_and_do_not_take_gt(self):
        image = np.zeros((2, 16, 16))
        p = np.full((16, 16), .5)
        rough = np.zeros_like(p); rough[5:10, 5:10] = 1
        features = pair_features(image, p, p, rough, rough)
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[3], 0)

    def test_fit_calibration_isolation_and_overlap_rejection(self):
        records = [record(str(i), initial=.8, final=.9 if i % 2 else .7) for i in range(12)]
        metadata = {'config': {'split_role': 'train', 'oracle_routing': False}, 'checkpoint_hashes': {}}
        artifact = fit_gate(metadata, records)
        self.assertFalse(set(artifact['fit_patients']) & set(artifact['calibration_patients']))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'gate.json'; path.write_text(json.dumps(artifact))
            gate = QualityGate.load(path)
            gate.check_patients(['unseen'])
            with self.assertRaises(ValueError):
                gate.check_patients(['0'])
            self.assertTrue(np.isfinite(gate.predict_delta(np.zeros(len(FEATURE_NAMES)))))
        metadata['config']['split_role'] = 'val'
        with self.assertRaises(ValueError):
            fit_gate(metadata, records)

    def test_all_harmful_calibration_rejects_all(self):
        records = [record(str(i), initial=.8, final=.6) for i in range(8)]
        artifact = fit_gate({'config': {'split_role': 'train', 'oracle_routing': False}, 'checkpoint_hashes': {}}, records)
        self.assertEqual(artifact['calibration']['slice_acceptance_rate'], 0)
        self.assertFalse(QualityGate(artifact).accept(np.asarray(records[0]['quality_features'])))

    def test_all_slice_selection_ignores_gt(self):
        import nibabel as nib
        from src.data.brats2020_dataset import BraTS2020Dataset
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'BraTS2021_00001'; p.mkdir()
            image = np.ones((16, 16, 3), dtype=np.float32)
            gt = np.zeros_like(image); gt[2:9, 2:9, 1] = 1
            for name, data in [('t1ce', image), ('seg', gt)]:
                nib.save(nib.Nifti1Image(data, np.eye(4)), p/f'BraTS2021_00001_{name}.nii.gz')
            with patch('src.data.brats2020_dataset._select_slices', side_effect=AssertionError('GT selector called')):
                ds = BraTS2020Dataset(temp, target_size=16, simulate_rough=False, slice_selection='all')
            self.assertEqual(ds._sample_zs, [0, 1, 2])
            self.assertEqual(len(ds), 3)


if __name__ == '__main__':
    unittest.main()
