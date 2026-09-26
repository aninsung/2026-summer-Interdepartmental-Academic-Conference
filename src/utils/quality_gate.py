"""GT-free pairwise features and a frozen, calibrated linear quality gate."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion, sobel

FEATURE_NAMES = [
    'rough_area', 'candidate_area', 'area_log_ratio', 'changed_fraction',
    'rough_probability', 'candidate_probability', 'rough_boundary_entropy',
    'candidate_boundary_entropy', 'rough_flip_disagreement',
    'candidate_flip_disagreement', 'rough_edge_alignment', 'candidate_edge_alignment',
]


def pair_features(image, probability, augmented_probability, rough, candidate):
    """No GT, class labels, Dice, or reward enters this feature vector."""
    image = np.asarray(image, dtype=float)
    image = image.mean(axis=0) if image.ndim == 3 else image
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    entropy = -(p * np.log(p) + (1-p) * np.log(1-p))
    disagreement = np.abs(probability - augmented_probability)
    edge = np.hypot(sobel(image, axis=0), sobel(image, axis=1))
    edge /= max(float(edge.max()), 1e-6)
    r, c = rough > .5, candidate > .5
    def mean(values, region):
        return float(values[region].mean()) if region.any() else 0.0
    rb, cb = r & ~binary_erosion(r), c & ~binary_erosion(c)
    out = [r.mean(), c.mean(), np.log((c.sum()+1)/(r.sum()+1)), np.mean(r != c),
           mean(p, r), mean(p, c), mean(entropy, rb), mean(entropy, cb),
           mean(disagreement, rb), mean(disagreement, cb), mean(edge, rb), mean(edge, cb)]
    if not np.isfinite(out).all():
        raise ValueError('Non-finite quality features')
    return np.asarray(out, dtype=float)


@dataclass
class QualityGate:
    artifact: dict

    @classmethod
    def load(cls, path):
        artifact = json.loads(Path(path).read_text())
        if artifact.get('schema_version') != 1 or artifact.get('features') != FEATURE_NAMES:
            raise ValueError('Unsupported quality gate schema/features')
        n = len(FEATURE_NAMES)
        for key in ('mean', 'scale', 'weights'):
            value = np.asarray(artifact[key])
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid quality gate {key}')
        if np.any(np.asarray(artifact['scale']) <= 0):
            raise ValueError('Invalid feature scale')
        if not np.isfinite([artifact['bias'], artifact['threshold']]).all():
            raise ValueError('Invalid gate threshold/bias')
        return cls(artifact)

    def predict_delta(self, features):
        a = self.artifact
        return float(((features - np.asarray(a['mean'])) / np.asarray(a['scale'])) @ np.asarray(a['weights']) + a['bias'])

    def accept(self, features):
        return self.predict_delta(features) > self.artifact['threshold']

    def check_patients(self, patient_ids):
        used = set(self.artifact['fit_patients']) | set(self.artifact['calibration_patients'])
        overlap = used & set(patient_ids)
        if overlap:
            raise ValueError(f'Quality gate evaluation overlaps fit/calibration patients: {sorted(overlap)}')
