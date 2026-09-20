"""Explicit label-preserving postprocessing, independent of PPO and its rewards."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def remove_small_components(mask, min_voxels=10):
    """Return a copy with small non-largest 6-connected WT components removed.

    WT is every nonzero class. The largest component is always retained, matching
    the former environment cleanup, and surviving voxels keep their class labels.
    The threshold is measured in voxels, not physical volume. This function never
    modifies its input and does not use a reference segmentation.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError("Expected a 3D exclusive-class mask")
    if not isinstance(min_voxels, (int, np.integer)) or min_voxels < 1:
        raise ValueError("min_voxels must be a positive integer")
    result = mask.copy()
    labeled, count = ndi.label(mask != 0, structure=ndi.generate_binary_structure(3, 1))
    if count <= 1:
        return result
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    remove = sizes < min_voxels
    remove[0] = False
    remove[int(sizes.argmax())] = False
    result[remove[labeled]] = 0
    return result
