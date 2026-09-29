import torch
from torch.utils.data import Dataset


class ShapeDataset(Dataset):
    """크기 클래스 레이블. 학습에서는 경계 면적 슬라이스를 뺀다.

    정답 경계는 그대로다. Small < 300, Medium < 700, 그 외 Large.
    margin 안(기본 80px)은 이웃 클래스와 MRI가 거의 같아서 학습 손실에서 제외한다.
    """

    SMALL_MAX = 300.0
    MEDIUM_MAX = 700.0

    def __init__(self, brats_dataset, margin=80.0, drop_boundary=False, augment=False, use_25d=True):
        self.brats_dataset = brats_dataset
        self.margin = float(margin)
        self.augment = augment
        self.use_25d = use_25d
        self.indices = []
        for i in range(len(brats_dataset)):
            area = float(brats_dataset[i]["gt_mask"].sum())
            if drop_boundary and not self.is_clear(area):
                continue
            self.indices.append(i)

    def is_clear(self, area: float) -> bool:
        area = float(area)
        margin = self.margin
        if area < self.SMALL_MAX:
            return area < self.SMALL_MAX - margin
        if area < self.MEDIUM_MAX:
            return area >= self.SMALL_MAX + margin and area < self.MEDIUM_MAX - margin
        return area >= self.MEDIUM_MAX + margin

    @classmethod
    def class_id(cls, area: float) -> int:
        if area < cls.SMALL_MAX:
            return 0
        if area < cls.MEDIUM_MAX:
            return 1
        return 2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.brats_dataset[self.indices[idx]]
        img = sample["image_25d"] if self.use_25d else sample["image"]
        gt = sample["gt_mask"]
        area = float(gt.sum())
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[-1])
            img = img * (0.9 + 0.2 * torch.rand(1).item())
        class_t = torch.tensor(self.class_id(area), dtype=torch.long)
        clear = torch.tensor(self.is_clear(area), dtype=torch.bool)
        return img, class_t, clear
