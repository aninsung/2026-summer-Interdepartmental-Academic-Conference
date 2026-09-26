"""Fit a patient-balanced ridge model; calibrate acceptance on disjoint patients."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.evaluation_records import read_records
from src.utils.quality_gate import FEATURE_NAMES


def fit_gate(metadata, records, seed=42, calibration_fraction=.25, ridge=1.0, max_harm_rate=.1):
    if metadata['config']['split_role'] != 'train' or metadata['config']['oracle_routing']:
        raise ValueError('Use classifier-routed train records only; held-out evaluation must remain untouched')
    if not 0 < calibration_fraction < 1 or ridge <= 0 or not 0 <= max_harm_rate <= 1:
        raise ValueError('Invalid calibration fraction, ridge, or harm rate')
    if any('stage2' not in r['methods'] or 'ppo_raw' not in r['methods'] for r in records):
        raise ValueError('Both stage2 and ppo_raw measurements are required; use --eval_mode compare')
    patients = sorted({r['patient_id'] for r in records})
    if len(patients) < 4:
        raise ValueError('At least four patients are required for disjoint fit/calibration')
    rng = np.random.default_rng(seed)
    patients = list(rng.permutation(patients))
    n_cal = max(2, min(len(patients)-2, round(len(patients)*calibration_fraction)))
    calibration = set(patients[:n_cal])
    fit = set(patients[n_cal:])
    x = np.asarray([r['quality_features'] for r in records], dtype=float)
    if x.shape != (len(records), len(FEATURE_NAMES)) or not np.isfinite(x).all():
        raise ValueError('Missing/nonfinite quality features; generate compare records with PPO')
    y = np.asarray([r['methods']['ppo_raw']['dsc'] - r['methods']['stage2']['dsc'] for r in records])
    ids = np.asarray([r['patient_id'] for r in records])
    tr, ca = np.isin(ids, list(fit)), np.isin(ids, list(calibration))
    weights = np.asarray([1 / np.sum(ids[tr] == p) for p in ids[tr]])
    mean = np.average(x[tr], axis=0, weights=weights)
    scale = np.sqrt(np.average((x[tr]-mean)**2, axis=0, weights=weights))
    scale = np.maximum(scale, 1e-6)
    z = np.column_stack([(x-mean)/scale, np.ones(len(x))])
    penalty = np.eye(z.shape[1]) * ridge
    penalty[-1, -1] = 0
    coef = np.linalg.solve(z[tr].T @ (weights[:, None]*z[tr]) + penalty, z[tr].T @ (weights*y[tr]))
    scores = z[ca] @ coef
    # The same area guard is used in inference and calibration.
    area_ok = np.asarray([r['methods']['ppo_raw']['predicted_pixels'] >= 1 and
                          .2*max(1, r['methods']['stage2']['predicted_pixels']) <= r['methods']['ppo_raw']['predicted_pixels'] <=
                          4*max(1, r['methods']['stage2']['predicted_pixels']) for r in records])[ca]
    best_threshold = max(0.0, float(scores.max()) + 1e-9)
    best_gain, best_harm, best_coverage = 0.0, 0.0, 0.0
    for threshold in np.unique(np.r_[0., np.maximum(0., scores)]):
        accept = (scores > threshold) & area_ok
        gains = np.asarray([np.mean(np.where(accept[ids[ca] == p], y[ca][ids[ca] == p], 0)) for p in sorted(calibration)])
        # Fraction of calibration patients whose mean DSC would decrease.
        harm = float(np.mean(gains < -1e-6))
        gain = float(gains.mean())
        if harm <= max_harm_rate and gain > best_gain:
            best_threshold, best_gain, best_harm, best_coverage = float(threshold), gain, harm, float(accept.mean())
    return dict(schema_version=1, features=FEATURE_NAMES, mean=mean.tolist(), scale=scale.tolist(),
                weights=coef[:-1].tolist(), bias=float(coef[-1]), threshold=best_threshold,
                fit_patients=sorted(fit), calibration_patients=sorted(calibration), seed=seed,
                checkpoint_hashes=metadata['checkpoint_hashes'], evaluation_config=metadata['config'],
                calibration=dict(patient_mean_delta_dsc=best_gain, patient_harm_rate=best_harm,
                                 slice_acceptance_rate=best_coverage, max_harm_rate=max_harm_rate),
                target='candidate DSC minus Stage 2 DSC',
                limitation='Calibration statistics are not independent test performance; no DSC guarantee')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, help='Completed train-split compare evaluation directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--calibration_fraction', type=float, default=.25)
    parser.add_argument('--ridge', type=float, default=1.)
    parser.add_argument('--max_harm_rate', type=float, default=.1)
    args = parser.parse_args()
    meta, records = read_records(args.records)
    artifact = fit_gate(meta, records, args.seed, args.calibration_fraction, args.ridge, args.max_harm_rate)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(artifact, f, indent=2, allow_nan=False)
    print(json.dumps(artifact['calibration'], indent=2))
    print('Saved frozen gate:', path)


if __name__ == '__main__':
    main()
