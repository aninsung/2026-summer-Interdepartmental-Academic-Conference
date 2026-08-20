"""Small 학습용 <min_area 슬라이스 반복 샘플링."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class FragmentOversample(Dataset):
    def __init__(self, base: Dataset, min_area: float = 50.0, repeats: int = 4):
        self.base = base
        ids = []
        samples = getattr(base, "_samples", None)
        n = len(base)
        for i in range(n):
            if samples is not None:
                area = float(np.sum(samples[i][1]))
            else:
                area = float(base[i]["gt_mask"].sum().item())
            extra = repeats if 0 < area < min_area else 1
            ids.extend([i] * extra)
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.base[self.ids[idx]]
