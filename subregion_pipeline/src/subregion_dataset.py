"""
subregion_pipeline/src/subregion_dataset.py
--------------------------------------------
BraTS2021 2D 슬라이스 서브리전(Subregion) 데이터셋 로더.
BraTS GT `seg` 라벨 정의:
  - 0: 배경 (Background)
  - 1: NCR/NET (Necrotic / Non-Enhancing Tumor Core)
  - 2: ED (Peritumoral Edema)
  - 4: ET (Enhancing Tumor)

3대 서브리전 마스크 변환:
  - WT (Whole Tumor) : seg > 0 (1, 2, 4 포함)
  - TC (Tumor Core)  : (seg == 1) | (seg == 4)
  - ET (Enhancing Tumor) : seg == 4

슬라이스 주도 서브리전 (Subregion Label):
  - Label 0 (ET-heavy) : ET 면적 > 20px (조영 증식 종양 우세)
  - Label 1 (TC-heavy) : TC 면적 > 40px 및 ET <= 20px (종양 핵심 우세)
  - Label 2 (WT-heavy) : 기타 부종 우세 및 대형 전체 종양
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


def _normalize_volume(vol: np.ndarray) -> np.ndarray:
    brain = vol > 0
    if brain.sum() == 0:
        return np.zeros_like(vol, dtype=np.float32)
    mean = vol[brain].mean()
    std  = vol[brain].std() + 1e-8
    normed = (vol - mean) / std
    p1, p99 = np.percentile(normed[brain], [1, 99])
    normed = np.clip(normed, p1, p99)
    normed = (normed - p1) / (p99 - p1 + 1e-8)
    return normed.astype(np.float32)


def _load_volume(path: str) -> np.ndarray:
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32)


def _find_patient_dirs(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        if (Path("src/data/archive")).exists():
            root_path = Path("src/data/archive")
        else:
            return []

    if (root_path / "BraTS2021_Training_Data").is_dir():
        root_path = root_path / "BraTS2021_Training_Data"

    dirs = sorted([
        p for p in root_path.iterdir()
        if p.is_dir() and ("BraTS20" in p.name or "BraTS2021" in p.name) and p.name != "BraTS2021_Training_Data"
    ])
    return dirs


def _find_modality_file(pdir: Path, pid: str, modality: str) -> Optional[Path]:
    for ext in [".nii.gz", ".nii"]:
        candidate = pdir / f"{pid}_{modality}{ext}"
        if candidate.exists():
            return candidate
    return None


def _find_seg_file(pdir: Path, pid: str) -> Optional[Path]:
    for ext in [".nii.gz", ".nii"]:
        candidate = pdir / f"{pid}_seg{ext}"
        if candidate.exists():
            return candidate
    return None


class SubregionBraTSDataset(Dataset):
    """
    BraTS 2D 서브리전(Subregion: ET, TC, WT) 데이터셋
    """
    def __init__(
        self,
        root_dir: str = "src/data/archive",
        modality: str = "t1ce+flair",
        target_size: int = 128,
        max_patients: Optional[int] = None,
    ):
        super().__init__()
        self.target_size = target_size
        self.modality = modality

        pdirs = _find_patient_dirs(root_dir)
        if max_patients is not None and max_patients > 0:
            pdirs = pdirs[:max_patients]

        self.samples = []

        print(f"[Subregion Dataset] Loading {len(pdirs)} patients from {root_dir}...")
        for pdir in pdirs:
            pid = pdir.name
            seg_p = _find_seg_file(pdir, pid)
            t1ce_p = _find_modality_file(pdir, pid, "t1ce")
            flair_p = _find_modality_file(pdir, pid, "flair")

            if not (seg_p and t1ce_p and flair_p):
                continue

            seg_vol = _load_volume(str(seg_p))
            t1ce_vol = _normalize_volume(_load_volume(str(t1ce_p)))
            flair_vol = _normalize_volume(_load_volume(str(flair_p)))

            D = seg_vol.shape[2]
            for z in range(D):
                sl_seg = seg_vol[:, :, z]
                wt_area = (sl_seg > 0).sum()
                if wt_area < 25:
                    continue

                sl_t1ce = t1ce_vol[:, :, z]
                sl_flair = flair_vol[:, :, z]

                if sl_seg.shape[0] != target_size or sl_seg.shape[1] != target_size:
                    sl_seg_t = torch.from_numpy(sl_seg).unsqueeze(0).unsqueeze(0).float()
                    sl_t1ce_t = torch.from_numpy(sl_t1ce).unsqueeze(0).unsqueeze(0).float()
                    sl_flair_t = torch.from_numpy(sl_flair).unsqueeze(0).unsqueeze(0).float()

                    sl_seg = torch.nn.functional.interpolate(sl_seg_t, size=(target_size, target_size), mode='nearest').squeeze().numpy()
                    sl_t1ce = torch.nn.functional.interpolate(sl_t1ce_t, size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze().numpy()
                    sl_flair = torch.nn.functional.interpolate(sl_flair_t, size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze().numpy()

                img_2ch = np.stack([sl_t1ce, sl_flair], axis=0).astype(np.float32)

                wt_mask = (sl_seg > 0).astype(np.float32)
                tc_mask = ((sl_seg == 1) | (sl_seg == 4)).astype(np.float32)
                et_mask = (sl_seg == 4).astype(np.float32)

                et_area = et_mask.sum()
                tc_area = tc_mask.sum()

                if et_area >= 20:
                    sub_label = 0  # ET-heavy
                elif tc_area >= 40:
                    sub_label = 1  # TC-heavy
                else:
                    sub_label = 2  # WT-heavy

                self.samples.append({
                    "image": img_2ch,
                    "gt_wt": np.expand_dims(wt_mask, 0),
                    "gt_tc": np.expand_dims(tc_mask, 0),
                    "gt_et": np.expand_dims(et_mask, 0),
                    "subregion_label": sub_label,
                    "patient_id": pid,
                    "slice_idx": z,
                })

        print(f"[Subregion Dataset] Total valid slices loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "image": torch.from_numpy(s["image"]),
            "gt_wt": torch.from_numpy(s["gt_wt"]),
            "gt_tc": torch.from_numpy(s["gt_tc"]),
            "gt_et": torch.from_numpy(s["gt_et"]),
            "subregion_label": torch.tensor(s["subregion_label"], dtype=torch.long),
        }

    def get_numpy_arrays(self):
        imgs = np.stack([s["image"] for s in self.samples], axis=0)
        gt_wts = np.stack([s["gt_wt"][:, 0] if s["gt_wt"].ndim == 3 else s["gt_wt"] for s in self.samples], axis=0)
        gt_tcs = np.stack([s["gt_tc"][:, 0] if s["gt_tc"].ndim == 3 else s["gt_tc"] for s in self.samples], axis=0)
        gt_ets = np.stack([s["gt_et"][:, 0] if s["gt_et"].ndim == 3 else s["gt_et"] for s in self.samples], axis=0)
        labels = np.array([s["subregion_label"] for s in self.samples], dtype=np.int64)
        return imgs, gt_wts, gt_tcs, gt_ets, labels
