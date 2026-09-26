"""Shared, GT-free Stage 2 and component inputs for PPO v2 training/inference."""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label, sobel
from src.utils.metrics import filter_small_components

def energy_input(image, probability, mask):
    image = np.asarray(image, dtype=np.float32)
    image = image.mean(axis=0) if image.ndim == 3 else image
    image = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
    return np.stack([image, np.asarray(probability, dtype=np.float32),
                     np.asarray(mask, dtype=np.float32),
                     np.asarray(probability, dtype=np.float32) * (np.asarray(mask, dtype=np.float32) > .5)], axis=0)



def stage2_mask(probability, route, thresholds, cc_sizes, micro_area_floor=80., micro_thr_floor=.15):
    mask = (probability > thresholds[route]).astype(np.float32)
    if route == 0 and mask.sum() < micro_area_floor:
        for threshold in np.arange(thresholds[route]-.05, micro_thr_floor-1e-9, -.05):
            mask = (probability > threshold).astype(np.float32)
            if mask.sum() >= micro_area_floor:
                break
    return filter_small_components(mask, cc_sizes[route])


def component_inputs(rough, augmented_probability, threshold):
    labels, count = label(rough > .5)
    for k in range(1, count+1):
        comp = (labels == k).astype(np.float32)
        area = float(comp.sum())
        route = 0 if area < 300 else (1 if area < 700 else 2)
        if area < 5:
            yield comp, route, None, None
            continue
        region = binary_dilation(comp, np.ones((3, 3)), iterations=2 if route == 0 else 3)
        threshold_k = .30 if route == 0 and area < 50 else threshold
        initial = ((augmented_probability > threshold_k) * region).astype(np.float32)
        if not initial.any():
            initial = comp.copy()
        # Preserve probability evidence outside the initial mask in the full reachable band.
        support = binary_dilation(initial, np.ones((3, 3)), iterations=8)
        probability = augmented_probability * support
        yield comp, route, initial, probability.astype(np.float32)


def boundary_energy(image, original_probability, tta_probability, component, model=None, device=None):
    """GT-free local error score for deciding whether PPO is worth running."""
    if model is not None:
        import torch
        x = torch.from_numpy(energy_input(image, tta_probability, component)).unsqueeze(0)
        with torch.no_grad():
            score = model(x.to(device)).squeeze().detach().cpu().numpy()
        mask = component > .5
        boundary = binary_dilation(mask, np.ones((3, 3), dtype=bool)) ^ binary_erosion(mask, np.ones((3, 3), dtype=bool))
        return float(score[boundary].mean()) if boundary.any() else float(score.mean())
    image = np.asarray(image, dtype=np.float32)
    image = image.mean(axis=0) if image.ndim == 3 else image
    p = np.clip(np.asarray(tta_probability, dtype=np.float32), 1e-5, 1 - 1e-5)
    q = np.clip(np.asarray(original_probability, dtype=np.float32), 1e-5, 1 - 1e-5)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)
    disagreement = np.abs(p - q)
    edge = np.hypot(sobel(image, 0), sobel(image, 1))
    edge = edge / max(float(edge.max()), 1e-6)
    mask = component > 0.5
    boundary = binary_dilation(mask, np.ones((3, 3), dtype=bool)) ^ binary_erosion(mask, np.ones((3, 3), dtype=bool))
    if not boundary.any():
        return 1.0
    return float(np.mean(0.45 * entropy[boundary] + 0.35 * disagreement[boundary]
                        + 0.20 * (1.0 - edge[boundary])))
