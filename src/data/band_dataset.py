"""경계 띠 보정 학습 샘플.

Stage 2 확률맵은 contour_dataset.build_stage2_entries 로 만든 것을 그대로 쓴다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.band import action_target, edit_band, signed_boundary_distance


class BandSliceDataset(Dataset):
    def __init__(self, entries, prob_lo: float = 0.35, prob_hi: float = 0.65,
                 radius: int = 2, augment: bool = False):
        self.entries = entries
        self.augment = augment
        self.bands = []
        self.phis = []
        self.targets = []
        band_pixels = 0
        edits = 0
        for entry in entries:
            band = edit_band(entry.probability.astype(np.float32), entry.stage2,
                             prob_lo, prob_hi, radius)
            target = action_target(entry.stage2, entry.gt)
            self.bands.append(band.astype(np.uint8))
            self.phis.append(signed_boundary_distance(entry.gt))
            self.targets.append(target)
            band_pixels += int(band.sum())
            edits += int(((target != 1) & band).sum())
        self.band_pixels = band_pixels
        self.edit_pixels = edits

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        image = entry.model_input()
        stage2 = entry.stage2.astype(np.float32)
        gt = entry.gt.astype(np.float32)
        band = self.bands[idx].astype(np.float32)
        phi = self.phis[idx]
        target = self.targets[idx].copy()

        if self.augment:
            if np.random.rand() < 0.5:
                image = image[:, :, ::-1]
                stage2 = stage2[:, ::-1]
                gt = gt[:, ::-1]
                band = band[:, ::-1]
                phi = phi[:, ::-1]
                target = target[:, ::-1]
            if np.random.rand() < 0.5:
                image = image[:, ::-1, :]
                stage2 = stage2[::-1, :]
                gt = gt[::-1, :]
                band = band[::-1, :]
                phi = phi[::-1, :]
                target = target[::-1, :]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "stage2": torch.from_numpy(np.ascontiguousarray(stage2[None])),
            "gt": torch.from_numpy(np.ascontiguousarray(gt[None])),
            "band": torch.from_numpy(np.ascontiguousarray(band[None])),
            "phi": torch.from_numpy(np.ascontiguousarray(phi[None])),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
        }
