"""GPU (PyTorch) replacements for SciPy morphology / EDT used by MaskRefinementEnv."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def binary_dilation(mask: torch.Tensor, kernel: int = 3, iterations: int = 1) -> torch.Tensor:
    """Binary dilation with square ones structuring element (odd kernel)."""
    x = mask.float().view(1, 1, *mask.shape[-2:])
    for _ in range(max(1, iterations)):
        x = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return x.view(*mask.shape) > 0.5


@torch.no_grad()
def binary_erosion(mask: torch.Tensor, kernel: int = 3, iterations: int = 1) -> torch.Tensor:
    return ~binary_dilation(~mask, kernel=kernel, iterations=iterations)


@torch.no_grad()
def distance_transform_edt(mask: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """
    Euclidean distance transform matching scipy.ndimage.distance_transform_edt.

    For True (foreground) pixels: distance to nearest False (background).
    Background pixels are 0. If mask is all-True or all-False, returns zeros.
    """
    m = mask.bool()
    H, W = m.shape
    device = m.device
    out = torch.zeros(H, W, device=device, dtype=torch.float32)
    if not bool(m.any()) or bool(m.all()):
        return out

    bg_y, bg_x = torch.where(~m)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    min_d2 = torch.full((H, W), float("inf"), device=device)
    for i in range(0, int(bg_y.numel()), chunk):
        by = bg_y[i : i + chunk].float()
        bx = bg_x[i : i + chunk].float()
        d2 = (yy.unsqueeze(-1) - by) ** 2 + (xx.unsqueeze(-1) - bx) ** 2
        min_d2 = torch.minimum(min_d2, d2.min(dim=-1).values)
    return torch.where(m, torch.sqrt(min_d2), out)


@torch.no_grad()
def signed_distance(mask: torch.Tensor) -> torch.Tensor:
    """Inside positive, outside negative (same convention as the NumPy env)."""
    m = mask.bool()
    if bool(m.all()):
        return torch.full(m.shape, 999.0, device=m.device, dtype=torch.float32)
    if not bool(m.any()):
        return torch.full(m.shape, -999.0, device=m.device, dtype=torch.float32)
    return distance_transform_edt(m) - distance_transform_edt(~m)


@torch.no_grad()
def label_components(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    4-connected component labels (matches scipy.ndimage.label default).
    Returns (labels HxW int64, num_features). Background is 0; components are 1..K.
    """
    m = mask.bool()
    device = m.device
    H, W = m.shape
    if not bool(m.any()):
        return torch.zeros(H, W, device=device, dtype=torch.long), 0

    flat_ids = torch.arange(H * W, device=device, dtype=torch.long).view(H, W)
    labels = torch.where(m, flat_ids, torch.full_like(flat_ids, fill_value=H * W)).float()
    big = float(H * W)
    for _ in range(H + W):
        nbr = labels.clone()
        nbr[1:, :] = torch.minimum(nbr[1:, :], labels[:-1, :])
        nbr[:-1, :] = torch.minimum(nbr[:-1, :], labels[1:, :])
        nbr[:, 1:] = torch.minimum(nbr[:, 1:], labels[:, :-1])
        nbr[:, :-1] = torch.minimum(nbr[:, :-1], labels[:, 1:])
        nbr = torch.where(m, nbr, torch.full_like(nbr, big))
        if torch.equal(nbr, labels):
            break
        labels = nbr
    labels = torch.where(m, labels.long(), torch.zeros(H, W, device=device, dtype=torch.long))

    uniq = torch.unique(labels)
    uniq = uniq[uniq > 0]
    remapped = torch.zeros_like(labels)
    for i, u in enumerate(uniq, start=1):
        remapped[labels == u] = i
    return remapped, int(uniq.numel())


@torch.no_grad()
def largest_component_centroid(mask: torch.Tensor) -> tuple[float, float]:
    """Centroid of the largest connected component; image center if empty."""
    H, W = mask.shape[-2:]
    lbl, n = label_components(mask > 0.5)
    if n == 0:
        return H / 2.0, W / 2.0
    sizes = torch.zeros(n + 1, device=mask.device, dtype=torch.long)
    sizes.scatter_add_(0, lbl.view(-1), torch.ones(lbl.numel(), device=mask.device, dtype=torch.long))
    sizes[0] = 0
    k = int(sizes.argmax().item())
    ys, xs = torch.where(lbl == k)
    return float(ys.float().mean().item()), float(xs.float().mean().item())


@torch.no_grad()
def sobel_magnitude(img: torch.Tensor) -> torch.Tensor:
    """Normalized Sobel edge magnitude for a 2D image tensor."""
    x = img.float().view(1, 1, *img.shape[-2:])
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device, dtype=torch.float32
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device, dtype=torch.float32
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    edge = torch.sqrt(gx * gx + gy * gy).view(*img.shape[-2:])
    e_min, e_max = edge.min(), edge.max()
    if float(e_max - e_min) > 0:
        edge = (edge - e_min) / (e_max - e_min)
    return edge


@torch.no_grad()
def gaussian_blur2d(img: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Separable Gaussian blur (odd kernel ~ 6*sigma+1)."""
    radius = max(1, int(3.0 * sigma + 0.5))
    k = 2 * radius + 1
    coords = torch.arange(k, device=img.device, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    x = img.float().view(1, 1, *img.shape[-2:])
    x = F.conv2d(x, kernel_1d.view(1, 1, 1, k), padding=(0, radius))
    x = F.conv2d(x, kernel_1d.view(1, 1, k, 1), padding=(radius, 0))
    return x.view(*img.shape[-2:])


@torch.no_grad()
def dice(a: torch.Tensor, b: torch.Tensor, smooth: float = 1e-5) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float((2.0 * (a * b).sum() + smooth) / (a.sum() + b.sum() + smooth))


@torch.no_grad()
def hd95(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    dist_b: torch.Tensor | None = None,
) -> float:
    a = mask_a.bool()
    b = mask_b.bool()
    if not bool(a.any()) or not bool(b.any()):
        return min(30.0, float(mask_a.shape[0]))
    dist_a = distance_transform_edt(~a)
    if dist_b is None:
        dist_b = distance_transform_edt(~b)
    d_ab = dist_b[a]
    d_ba = dist_a[b]
    vals = torch.cat([d_ab, d_ba])
    return float(torch.quantile(vals, 0.95).item())
