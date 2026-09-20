"""Patient-level ET/TC/WT metrics on exclusive BraTS classes 0, 1, 2, 3.

Class 3 is the remapped BraTS enhancing-tumour label 4. Distances are in mm.
An empty pair has Dice=1 and distances=0; a one-sided empty region has Dice=0
and a fixed physical image diagonal penalty. These cases are never dropped.
"""
from __future__ import annotations

import numpy as np
from surface_distance import metrics as surface


REGIONS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}


def region_metrics(pred, target, spacing, tolerance_mm=2.0, empty_distance_mm=None):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("Expected matching 3D masks")
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing must contain three positive values")
    if empty_distance_mm is None:
        empty_distance_mm = float(np.linalg.norm(np.asarray(pred.shape) * spacing))
    result = {}
    for name, classes in REGIONS.items():
        p, y = np.isin(pred, classes), np.isin(target, classes)
        npred, ngt = int(p.sum()), int(y.sum())
        if not npred or not ngt:
            both_empty = not npred and not ngt
            result[name] = dict(dice=float(both_empty), surface_dice=float(both_empty),
                                hd95_mm=0.0 if both_empty else float(empty_distance_mm),
                                assd_mm=0.0 if both_empty else float(empty_distance_mm),
                                pred_voxels=npred, gt_voxels=ngt,
                                empty_case="both" if both_empty else "one")
            continue
        distances = surface.compute_surface_distances(y, p, spacing)
        gt_area, pred_area = distances["surfel_areas_gt"], distances["surfel_areas_pred"]
        assd = ((distances["distances_gt_to_pred"] * gt_area).sum()
                + (distances["distances_pred_to_gt"] * pred_area).sum()) / (gt_area.sum() + pred_area.sum())
        result[name] = dict(
            dice=float(2 * np.count_nonzero(p & y) / (npred + ngt)),
            hd95_mm=float(surface.compute_robust_hausdorff(distances, 95)),
            assd_mm=float(assd),
            surface_dice=float(surface.compute_surface_dice_at_tolerance(distances, tolerance_mm)),
            pred_voxels=npred, gt_voxels=ngt, empty_case="neither",
        )
    result["mean"] = {key: float(np.mean([result[r][key] for r in REGIONS]))
                      for key in ("dice", "hd95_mm", "assd_mm", "surface_dice")}
    return result


def quality(metrics, hd95_weight=0.2, hd95_scale_mm=20.0):
    """A gain of 1 mm HD95 equals 0.01 Dice at the default weights."""
    return metrics["mean"]["dice"] - hd95_weight * metrics["mean"]["hd95_mm"] / hd95_scale_mm
