"""Restore internal cropped/resampled class masks to the original BraTS grid."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _shape(value, name):
    if len(value) != 3 or any(isinstance(v, (bool, np.bool_)) or int(v) != v or v <= 0 for v in value):
        raise ValueError(f"{name} must contain three positive integer dimensions")
    return tuple(int(v) for v in value)


def restore_full_mask(mask_3d, case_info):
    """Nearest-exact restoration; accepts serialized bounds or Python slices.

    Input labels are internal 0/1/2/3. Output labels are BraTS 0/1/2/4.
    Bounds are checked before assigning into the full image, so stale crop
    metadata cannot silently shift a mask or wrap negative array indices.
    """
    mask = np.asarray(mask_3d)
    if mask.ndim != 3 or any(n <= 0 for n in mask.shape):
        raise ValueError("Mask must be a nonempty three-dimensional array")
    if not np.isin(mask, (0, 1, 2, 3)).all():
        raise ValueError("Expected internal class labels 0/1/2/3")
    original_shape = _shape(case_info["original_shape"], "original_shape")
    cropped_shape = _shape(case_info["cropped_shape"], "cropped_shape")
    crop = case_info["crop"]
    if len(crop) != 3:
        raise ValueError("crop must contain three axis bounds")
    slices = []
    for bounds, full, cropped in zip(crop, original_shape, cropped_shape):
        if isinstance(bounds, slice):
            if bounds.step not in (None, 1):
                raise ValueError("Crop slices must have unit stride")
            start, stop = bounds.start, bounds.stop
        else:
            if len(bounds) != 2:
                raise ValueError("Crop bounds must be [start, stop]")
            start, stop = bounds
        if (start is None or stop is None or int(start) != start or int(stop) != stop
                or not 0 <= start < stop <= full or stop - start != cropped):
            raise ValueError("Crop bounds disagree with original/cropped shapes")
        slices.append(slice(int(start), int(stop)))
    if mask.shape != cropped_shape:
        tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None]
        mask = F.interpolate(tensor, size=cropped_shape, mode="nearest-exact")[0, 0].numpy()
    full_mask = np.zeros(original_shape, dtype=np.uint8)
    full_mask[tuple(slices)] = mask.astype(np.uint8)
    full_mask[full_mask == 3] = 4
    return full_mask
