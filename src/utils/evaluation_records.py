"""Measured evaluation records; surface distances in resized image pixels."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from src.utils.metrics import dice, precision, recall


def measured_metrics(pred, gt):
    p, g = pred > .5, gt > .5
    one_empty = bool(p.any() != g.any())
    if one_empty:
        hd = None  # Undefined distance: do not hide failures behind an arbitrary cap.
    elif not p.any():
        hd = 0.0
    else:
        ps, gs = p & ~binary_erosion(p), g & ~binary_erosion(g)
        distances = np.concatenate([distance_transform_edt(~gs)[ps], distance_transform_edt(~ps)[gs]])
        hd = float(np.percentile(distances, 95))
    return dict(dsc=dice(p, g), hd95_surface_px=hd, hd95_one_empty=one_empty,
                precision=precision(p, g), recall=recall(p, g),
                predicted_pixels=int(p.sum()), gt_pixels=int(g.sum()),
                false_positive_on_empty=bool(not g.any() and p.any()))


class RecordWriter:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, schema_version=1, complete=False)
        self._metadata()
        self.path = self.directory / 'slices.jsonl'

    def _metadata(self):
        (self.directory / 'metadata.json').write_text(json.dumps(self.metadata, indent=2, allow_nan=False))

    def append(self, record, masks=None):
        if masks is not None:
            name = f"mask_{record['sample_index']:06d}.npz"
            np.savez_compressed(self.directory / name, **masks)
            record['mask_file'] = name
        with self.path.open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')

    def finish(self):
        self.metadata['complete'] = True
        self._metadata()


def read_records(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text())
    if meta.get('schema_version') != 1 or not meta.get('complete'):
        raise ValueError('Evaluation is incomplete or unsupported')
    records = [json.loads(line) for line in (directory / 'slices.jsonl').read_text().splitlines() if line.strip()]
    keys = [(r['patient_id'], r['slice_z']) for r in records]
    if not records or len(set(keys)) != len(keys):
        raise ValueError('Empty evaluation or duplicate patient/slice IDs')
    return meta, records
