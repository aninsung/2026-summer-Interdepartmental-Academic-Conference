import torch
from torch.utils.data import Dataset
import numpy as np

class ShapeDataset(Dataset):
    """
    BraTS2020Dataset을 래핑하여 크기 클래스(Small, Medium, Large) 레이블을 제공하는 데이터셋.
    """
    def __init__(self, brats_dataset):
        self.brats_dataset = brats_dataset
        
    def __len__(self):
        return len(self.brats_dataset)
        
    def __getitem__(self, idx):
        sample = self.brats_dataset[idx]
        img = sample['image']
        gt = sample['gt_mask']
        
        area = torch.sum(gt).item()
        
        # 클래스 매핑
        if area < 300:
            class_id = 0 # Small
        elif area < 700:
            class_id = 1 # Medium
        else:
            class_id = 2 # Large
            
        # img는 이미 (1, 128, 128) 텐서임
        class_t = torch.tensor(class_id, dtype=torch.long)
        
        return img, class_t
