"""Four-modality 3D data and explicit disjoint backbone / RL / val / test IDs."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def make_split(root, seed=20260920):
    ids = sorted(p.name for p in Path(root).glob("BraTS2021_*")
                 if p.is_dir() and (p / f"{p.name}_seg.nii.gz").exists())
    if len(ids) < 12:
        raise ValueError("At least 12 complete patients are required")
    ids = np.random.default_rng(seed).permutation(ids).tolist()
    ntest, nval = max(1, round(len(ids) * .15)), max(1, round(len(ids) * .15))
    train = ids[ntest + nval:]
    nrl = max(1, round(len(train) * .25))
    split = dict(test=ids[:ntest], val=ids[ntest:ntest + nval],
                 rl_train=train[:nrl], backbone_train=train[nrl:])
    validate_split(split)
    return split


def validate_split(split):
    seen = set()
    for key in ("backbone_train", "rl_train", "val", "test"):
        ids = split[key]
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Empty, duplicate or overlapping patient IDs: {key}")
        seen.update(ids)


def load_case(root, patient, resize=None):
    directory = Path(root) / patient
    images, reference = [], None
    for modality in ("t1", "t1ce", "t2", "flair"):
        nii = nib.load(directory / f"{patient}_{modality}.nii.gz")
        if reference is None:
            reference = nii
        elif nii.shape != reference.shape or not np.allclose(nii.affine, reference.affine):
            raise ValueError(f"Misaligned modalities for {patient}")
        images.append(nii.get_fdata(dtype=np.float32))
    image = np.stack(images)
    seg = nib.load(directory / f"{patient}_seg.nii.gz")
    if seg.shape != reference.shape or not np.allclose(seg.affine, reference.affine):
        raise ValueError(f"Misaligned segmentation for {patient}")
    target = np.asarray(seg.dataobj, dtype=np.uint8)
    if not set(np.unique(target)).issubset({0, 1, 2, 4}):
        raise ValueError(f"Unexpected BraTS labels for {patient}")
    target[target == 4] = 3
    # Foreground cropping uses MRI only, including at validation/inference.
    foreground = np.any(image != 0, axis=0)
    coords = np.where(foreground)
    if not coords[0].size:
        raise ValueError(f"Empty MRI: {patient}")
    slices = tuple(slice(max(0, int(c.min()) - 2), min(s, int(c.max()) + 3))
                   for c, s in zip(coords, target.shape))
    original_shape = target.shape
    spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    empty_distance = float(np.linalg.norm(np.asarray(original_shape) * spacing))
    image, target = image[(slice(None),) + slices].copy(), target[slices].copy()
    cropped_shape = target.shape
    for channel in image:
        valid = channel != 0
        if valid.any():
            channel[valid] = np.clip((channel[valid] - channel[valid].mean()) /
                                     max(float(channel[valid].std()), 1e-6), -5, 5)
    if resize is not None:
        shape = (int(resize),) * 3
        image = F.interpolate(torch.from_numpy(image)[None], size=shape,
                              mode="trilinear", align_corners=False)[0].numpy()
        target = F.interpolate(torch.from_numpy(target.astype(np.float32))[None, None],
                               size=shape, mode="nearest-exact")[0, 0].numpy().astype(np.uint8)
        spacing = spacing * np.asarray(cropped_shape) / np.asarray(shape)
    metadata = dict(patient=patient, spacing=spacing.tolist(), original_shape=list(original_shape),
                    crop=[[s.start, s.stop] for s in slices], cropped_shape=list(cropped_shape),
                    evaluation_grid="resampled_pilot" if resize else "native_cropped",
                    empty_distance_mm=empty_distance)
    return dict(image=image.astype(np.float32), target=target, **metadata)


def save_case(path, case):
    arrays = {k: case[k] for k in ("image", "target", "logits") if k in case}
    metadata = {k: v for k, v in case.items() if k not in arrays}
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata))


def read_case(path):
    with np.load(path, allow_pickle=False) as data:
        case = json.loads(str(data["metadata"]))
        case.update({k: data[k].astype(np.float32) if k != "target" else data[k]
                     for k in ("image", "target", "logits") if k in data})
    return case
