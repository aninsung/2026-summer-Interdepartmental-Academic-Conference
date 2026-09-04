"""Supervised dual-head refiner: candidate + local fix."""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class DualHeadRefiner(nn.Module):
    def __init__(self, in_ch: int = 4):
        # default 4 = 2 MRI (t1ce+flair) + rough + prob
        super().__init__()
        self.in_ch = in_ch
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(True),
        )
        self.cand = nn.Conv2d(32, 1, 1)
        self.fix = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        f = self.enc(x)
        return self.cand(f), self.fix(f)


def _stack_sl_input(image, rough, prob) -> np.ndarray:
    """Build (C_img+2, H, W) float32 array."""
    img = np.asarray(image, dtype=np.float32)
    r = np.asarray(rough, dtype=np.float32)
    p = np.asarray(prob, dtype=np.float32)
    if img.ndim == 2:
        img = img[None]
    if r.ndim == 3:
        r = r[0]
    if p.ndim == 3:
        p = p[0]
    return np.concatenate([img, r[None], p[None]], axis=0).astype(np.float32)


@torch.no_grad()
def apply_sl_refiner(
    net: DualHeadRefiner,
    image_2d: np.ndarray,
    rough: np.ndarray,
    prob: np.ndarray,
    device: torch.device,
    cand_thr: float = 0.4,
    morph_small: bool = False,
) -> np.ndarray:
    """image: (H,W) or (C,H,W). rough/prob: (H,W). Returns refined binary float mask."""
    from scipy.ndimage import binary_closing, binary_opening

    x_np = _stack_sl_input(image_2d, rough, prob)
    r = x_np[-2]
    x = torch.from_numpy(x_np[None]).float().to(device)
    expected = getattr(net, "in_ch", None)
    if expected is not None and x.shape[1] != expected:
        raise ValueError(
            f"SL refiner in_ch={expected} but input has {x.shape[1]} channels "
            f"(image channels + rough + prob)."
        )
    net.eval()
    lc, lf = net(x)
    cand = (torch.sigmoid(lc) > cand_thr).float()
    rb = x[:, -2:-1]
    blended = torch.where(cand > 0.5, torch.sigmoid(lf), rb)
    pred = (blended > 0.5).float()[0, 0].cpu().numpy()
    if float(pred.sum()) < 5:
        pred = (r > 0.5).astype(np.float32)
    if morph_small:
        m = pred > 0.5
        m = binary_closing(binary_opening(m, iterations=1), iterations=1)
        if m.sum() >= 3:
            pred = m.astype(np.float32)
    return pred.astype(np.float32)


def build_sl_refiner(in_ch: int = 4) -> DualHeadRefiner:
    return DualHeadRefiner(in_ch=in_ch)
