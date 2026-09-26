"""Patient-paired analysis of measured records (no synthetic patient scores)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.evaluation_records import read_records


def bootstrap_ci(values, seed=42, repeats=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(repeats)]
    return np.percentile(means, [2.5, 97.5]).tolist()


def summarize(records, baseline='stage2', seed=42):
    patients = sorted({r['patient_id'] for r in records})
    methods = set(records[0]['methods'])
    if baseline not in methods or any(set(r['methods']) != methods for r in records):
        raise ValueError('All slices must contain exactly the same methods and paired baseline')
    report = {'n_patients': len(patients), 'n_slices': len(records),
              'aggregation': 'equal-weight patient means of 2D slice metrics; not 3D volume Dice', 'methods': {}}
    means = {}
    for method in sorted(methods):
        means[method] = np.asarray([np.mean([r['methods'][method]['dsc'] for r in records if r['patient_id'] == p]) for p in patients])
    for method in sorted(methods):
        values = means[method]
        delta = values - means[baseline]
        patient_hd = []
        for p in patients:
            distances = [r['methods'][method]['hd95_surface_px'] for r in records if r['patient_id'] == p and r['methods'][method]['hd95_surface_px'] is not None]
            if distances:
                patient_hd.append(float(np.mean(distances)))
        metrics = [r['methods'][method] for r in records]
        empty_gt = [m for m in metrics if m['gt_pixels'] == 0]
        # GT-based initial-quality bins are retrospective analysis only.
        strata = {}
        for label, lo, hi in [('poor', 0., .8), ('medium', .8, .9), ('good', .9, 1.000001)]:
            selected = (means[baseline] >= lo) & (means[baseline] < hi)
            strata[label] = {'patients': int(selected.sum()), 'delta_dsc': float(delta[selected].mean()) if selected.any() else None}
        report['methods'][method] = dict(
            mean_dsc=float(values.mean()), median_dsc=float(np.median(values)),
            mean_dsc_ci95=bootstrap_ci(values, seed), paired_delta_dsc=float(delta.mean()),
            paired_delta_ci95=bootstrap_ci(delta, seed), harmed_patient_rate=float(np.mean(delta < -1e-6)),
            worst_patient_delta=float(delta.min()), large_drop_patient_rate=float(np.mean(delta < -.02)),
            mean_hd95_surface_px_defined_only=float(np.mean(patient_hd)) if patient_hd else None,
            hd95_defined_patient_count=len(patient_hd), hd95_one_empty_slices=sum(m['hd95_one_empty'] for m in metrics),
            empty_gt_slices=len(empty_gt), false_positive_empty_slice_rate=(sum(m['false_positive_on_empty'] for m in empty_gt)/len(empty_gt) if empty_gt else None),
            initial_quality_strata=strata,
            per_patient={p: {'dsc': float(v), 'delta_dsc': float(d)} for p, v, d in zip(patients, values, delta)},
        )
    accepted = [r['quality_accepted'] for r in records if r.get('quality_accepted') is not None]
    report['quality_acceptance_rate'] = float(np.mean(accepted)) if accepted else None
    report['shared_inference_seconds'] = float(sum(r['shared_inference_seconds'] for r in records))
    report['timing_note'] = 'Shared candidate generation time; not per-method latency. Run modes separately for timing.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--baseline', default='stage2')
    parser.add_argument('--output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    meta, records = read_records(args.records)
    report = summarize(records, args.baseline, args.seed)
    report['evaluation_metadata'] = meta
    output = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(output)
    print(output)


if __name__ == '__main__':
    main()
