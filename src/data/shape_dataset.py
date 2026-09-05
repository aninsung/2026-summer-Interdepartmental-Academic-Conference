import torch
from torch.utils.data import Dataset
import numpy as np


def size_class_from_area(area: float) -> int:
    if area < 300:
        return 0
    if area < 700:
        return 1
    return 2


def soft_label_from_area(area: float, margin: float = 20.0) -> np.ndarray:
    """Hard one-hot, with linear blend near 300 / 700 boundaries."""
    y = np.zeros(3, dtype=np.float32)
    c = size_class_from_area(area)
    y[c] = 1.0
    # near 300: blend Small↔Medium
    if 300.0 - margin <= area < 300.0:
        t = (area - (300.0 - margin)) / margin  # 0 at far small → 1 at 300
        y[:] = 0
        y[0] = 1.0 - t
        y[1] = t
    elif 300.0 <= area < 300.0 + margin:
        t = (area - 300.0) / margin
        y[:] = 0
        y[0] = 1.0 - t
        y[1] = t
    # near 700: blend Medium↔Large
    elif 700.0 - margin <= area < 700.0:
        t = (area - (700.0 - margin)) / margin
        y[:] = 0
        y[1] = 1.0 - t
        y[2] = t
    elif 700.0 <= area < 700.0 + margin:
        t = (area - 700.0) / margin
        y[:] = 0
        y[1] = 1.0 - t
        y[2] = t
    return y


def _rotate_deg(img: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate (C,H,W) by angle_deg with bilinear sampling and zero padding."""
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
    theta = img.new_tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)
    return torch.nn.functional.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(0)


def augment_mri(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """img: (C,H,W) float tensor in roughly [0,1]. H/V flip, ±15° rotate, brightness/contrast."""
    x = img
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-2])
    # ±15° continuous rotation (keeps size cues better than 90° jumps)
    if rng.random() < 0.7:
        angle = float(rng.uniform(-15.0, 15.0))
        x = _rotate_deg(x, angle)
    # brightness / contrast
    if rng.random() < 0.8:
        bright = float(rng.uniform(-0.1, 0.1))
        contrast = float(rng.uniform(0.85, 1.15))
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = (x - mean) * contrast + mean + bright
        x = x.clamp(0.0, 1.0)
    return x


class ShapeDataset(Dataset):
    """
    BraTS2020Dataset을 래핑하여 크기 클래스(Small, Medium, Large) 레이블을 제공하는 데이터셋.
    """
    def __init__(
        self,
        brats_dataset,
        augment: bool = False,
        soft_boundary: bool = False,
        boundary_margin: float = 20.0,
        seed: int = 42,
    ):
        self.brats_dataset = brats_dataset
        self.augment = bool(augment)
        self.soft_boundary = bool(soft_boundary)
        self.boundary_margin = float(boundary_margin)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.brats_dataset)

    def __getitem__(self, idx):
        sample = self.brats_dataset[idx]
        img = sample["image"]
        gt = sample["gt_mask"]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        img = img.float()
        area = float(torch.sum(gt).item()) if torch.is_tensor(gt) else float(np.sum(gt))
        class_id = size_class_from_area(area)

        if self.augment:
            img = augment_mri(img, self.rng)

        class_t = torch.tensor(class_id, dtype=torch.long)
        if self.soft_boundary:
            soft = torch.from_numpy(soft_label_from_area(area, self.boundary_margin))
            return img, class_t, soft
        return img, class_t
