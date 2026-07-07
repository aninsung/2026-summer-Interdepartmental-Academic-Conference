"""
실제 BraTS2020 데이터셋 로더 (NIfTI .nii 기반)
-----------------------------------------------
BraTS2020 폴더 구조:
  <root>/
    BraTS20_Training_001/
      BraTS20_Training_001_flair.nii
      BraTS20_Training_001_t1.nii
      BraTS20_Training_001_t1ce.nii
      BraTS20_Training_001_t2.nii
      BraTS20_Training_001_seg.nii   ← GT 레이블
    BraTS20_Training_002/ ...

GT 레이블(seg) 값:
  0 = 배경
  1 = Necrotic / Non-Enhancing Tumor Core (NCR/NET)
  2 = Peritumoral Edema (ED)
  4 = GD-Enhancing Tumor (ET)
  → 이진화: 0 이외 = 종양 (Whole Tumor)

출력 슬라이스:
  - image   : (1, H, W)  — t1ce 단일 채널 (가장 종양 대비 선명)
  - gt_mask : (1, H, W)  — 이진 Whole Tumor 마스크
  - rough_mask : (1, H, W) — make_noisy_mask()로 시뮬레이션한 U-Net 초기 예측
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def _normalize_volume(vol: np.ndarray) -> np.ndarray:
    """뇌 마스크 내에서 z-score 정규화 후 0~1 클리핑."""
    brain = vol > 0
    if brain.sum() == 0:
        return np.zeros_like(vol, dtype=np.float32)
    mean = vol[brain].mean()
    std  = vol[brain].std() + 1e-8
    normed = (vol - mean) / std
    # 99 퍼센타일 기준 클리핑 후 0~1 스케일
    p1, p99 = np.percentile(normed[brain], [1, 99])
    normed = np.clip(normed, p1, p99)
    normed = (normed - p1) / (p99 - p1 + 1e-8)
    return normed.astype(np.float32)


def _load_volume(path: str) -> np.ndarray:
    """NIfTI 파일을 (H, W, D) float32 배열로 반환."""
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32)


def _find_patient_dirs(root: str) -> List[Path]:
    """루트 디렉토리 아래 BraTS20_* 폴더 목록을 정렬하여 반환."""
    root_path = Path(root)
    dirs = sorted([p for p in root_path.iterdir() if p.is_dir() and "BraTS20" in p.name])
    return dirs


def _select_slices(
    seg_vol: np.ndarray,
    min_tumor_ratio: float = 0.002,
) -> List[int]:
    """
    종양 픽셀 비율이 min_tumor_ratio 이상인 유효 슬라이스 인덱스 반환.
    (D 차원 = 마지막 축)
    """
    D = seg_vol.shape[2]
    H, W = seg_vol.shape[0], seg_vol.shape[1]
    valid = []
    for z in range(D):
        sl = seg_vol[:, :, z]
        tumor_ratio = (sl > 0).sum() / (H * W)
        if tumor_ratio >= min_tumor_ratio:
            valid.append(z)
    return valid


def make_noisy_mask(
    gt_mask: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    erosion_prob: float = 0.5,
    max_morph_px: int = 5,
) -> np.ndarray:
    """
    GT 마스크에 형태학적 노이즈를 추가하여 U-Net의 '초기 울퉁불퉁한 마스크'를 시뮬레이션합니다.
    """
    from scipy.ndimage import binary_erosion, binary_dilation

    if rng is None:
        rng = np.random.default_rng()

    mask = gt_mask.copy().astype(bool)
    n_ops = rng.integers(2, 6)

    for _ in range(n_ops):
        radius = rng.integers(1, max_morph_px + 1)
        struct = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
        if rng.random() < erosion_prob:
            mask = binary_erosion(mask, structure=struct)
        else:
            mask = binary_dilation(mask, structure=struct)

    # 경계에 픽셀 단위 소금-후추 노이즈
    boundary = np.zeros_like(gt_mask, dtype=bool)
    dilated = np.array(binary_dilation(mask, np.ones((3, 3))))
    eroded = np.array(binary_erosion(mask, np.ones((3, 3))))
    boundary = dilated ^ eroded
    flip_idx = np.where(boundary)
    flip_mask = rng.random(len(flip_idx[0])) < 0.25
    noisy = mask.astype(np.float32)
    noisy[flip_idx[0][flip_mask], flip_idx[1][flip_mask]] = 1.0 - noisy[flip_idx[0][flip_mask], flip_idx[1][flip_mask]]

    return noisy.astype(np.float32)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class BraTS2020Dataset(Dataset):
    """
    실제 BraTS2020 2D 슬라이스 데이터셋.

    Parameters
    ----------
    root_dir : str
        BraTS20_Training_* 폴더들이 있는 최상위 경로.
        예) "src/data/archive (1)/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    modality : str
        사용할 MRI 모달리티 ('t1ce', 't1', 't2', 'flair'). 기본 t1ce.
    target_size : int
        슬라이스를 리사이즈할 해상도 (H=W=target_size). 0이면 원본 크기 유지.
    max_patients : int or None
        로드할 최대 환자 수. None이면 전체.
    min_tumor_ratio : float
        유효 슬라이스로 인정할 최소 종양 픽셀 비율.
    noise_seed : int
        rough_mask 생성용 랜덤 시드.
    simulate_rough : bool
        True면 make_noisy_mask()로 rough_mask를 합성.
        False면 gt_mask를 그대로 rough_mask로 사용 (디버그용).
    """

    def __init__(
        self,
        root_dir: str,
        modality: str = "t1ce",
        target_size: int = 128,
        max_patients: Optional[int] = None,
        min_tumor_ratio: float = 0.002,
        noise_seed: int = 42,
        simulate_rough: bool = True,
    ):
        self.root_dir      = root_dir
        self.modality      = modality
        self.target_size   = target_size
        self.min_tumor_ratio = min_tumor_ratio
        self.simulate_rough  = simulate_rough
        self.rng = np.random.default_rng(noise_seed)

        self._samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._build(max_patients)

    # ── 내부 빌더 ──────────────────────────────────────────
    def _build(self, max_patients: Optional[int]) -> None:
        patient_dirs = _find_patient_dirs(self.root_dir)
        if max_patients is not None:
            patient_dirs = patient_dirs[:max_patients]

        try:
            from tqdm import tqdm
            _iter = tqdm(patient_dirs, desc="[BraTS2020] Loading patients", unit="pt", ncols=80, ascii=True)
        except ImportError:
            _iter = patient_dirs
            print(f"[BraTS2020Dataset] {len(patient_dirs)}명 환자 로딩 중...")

        import sys
        sys.stdout.reconfigure(errors='replace') if hasattr(sys.stdout, 'reconfigure') else None
        total_slices = 0

        for pdir in _iter:
            pid = pdir.name
            mod_path = pdir / f"{pid}_{self.modality}.nii"
            seg_path = pdir / f"{pid}_seg.nii"

            if not mod_path.exists() or not seg_path.exists():
                if hasattr(_iter, 'write'):
                    _iter.write(f"  [SKIP] 파일 없음: {pdir.name}")
                else:
                    print(f"  [SKIP] 파일 없음: {pdir.name}")
                continue

            # 볼륨 로드
            mod_vol = _normalize_volume(_load_volume(str(mod_path)))  # (H,W,D)
            seg_vol = _load_volume(str(seg_path))                     # (H,W,D) 레이블

            # 유효 슬라이스 선택
            valid_zs = _select_slices(seg_vol, self.min_tumor_ratio)
            for z in valid_zs:
                img_sl  = mod_vol[:, :, z]                           # (H,W)
                gt_sl   = (seg_vol[:, :, z] > 0).astype(np.float32)  # 이진화

                # 리사이즈
                if self.target_size > 0:
                    img_sl = self._resize(img_sl)
                    gt_sl  = self._resize(gt_sl, is_mask=True)

                # rough_mask 생성
                if self.simulate_rough:
                    rough_sl = make_noisy_mask(gt_sl, self.rng)
                else:
                    rough_sl = gt_sl.copy()

                self._samples.append((img_sl, gt_sl, rough_sl))

            total_slices += len(valid_zs)
            if hasattr(_iter, 'set_postfix'):
                _iter.set_postfix({"slices": total_slices, "this_pt": len(valid_zs)})

        print(f"[BraTS2020Dataset] 완료: 총 {total_slices}개 유효 슬라이스 로드.")

    def _resize(self, arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """간단한 바이선형/최근접 이웃 리사이즈 (PIL 없이 skimage)."""
        from skimage.transform import resize as sk_resize
        order = 0 if is_mask else 1  # 마스크는 최근접, 이미지는 바이선형
        resized = sk_resize(
            arr,
            (self.target_size, self.target_size),
            order=order,
            mode="constant",
            anti_aliasing=(not is_mask),
            preserve_range=True,
        )
        return resized.astype(np.float32)

    # ── Dataset API ────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        img, gt, rough = self._samples[idx]
        return {
            "image":      torch.from_numpy(img).unsqueeze(0),    # (1,H,W)
            "gt_mask":    torch.from_numpy(gt).unsqueeze(0),     # (1,H,W)
            "rough_mask": torch.from_numpy(rough).unsqueeze(0),  # (1,H,W)
        }

    def get_numpy_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RL 환경 초기화용 NumPy 배열 반환.
        Returns: images (N,H,W), gt_masks (N,H,W), rough_masks (N,H,W)
        """
        imgs   = np.stack([s[0] for s in self._samples], axis=0)
        gts    = np.stack([s[1] for s in self._samples], axis=0)
        roughs = np.stack([s[2] for s in self._samples], axis=0)
        return imgs, gts, roughs


# ──────────────────────────────────────────────
# BraTS2020 경로 헬퍼
# ──────────────────────────────────────────────

# 기본 데이터 경로 (configs/ppo_brats.yaml 또는 직접 오버라이드)
_DATA_DIR = Path(__file__).parent / "archive (1)"
DEFAULT_TRAIN_ROOT = str(
    _DATA_DIR / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
)
DEFAULT_VAL_ROOT = str(
    _DATA_DIR / "BraTS2020_ValidationData" / "MICCAI_BraTS2020_ValidationData"
)


def build_brats_datasets(
    train_root: str = DEFAULT_TRAIN_ROOT,
    val_root: str   = DEFAULT_VAL_ROOT,
    modality: str   = "t1ce",
    target_size: int = 128,
    max_train_patients: Optional[int] = None,
    max_val_patients:   Optional[int] = None,
) -> Tuple["BraTS2020Dataset", "BraTS2020Dataset"]:
    """
    학습/검증용 BraTS2020 데이터셋을 한 번에 생성합니다.

    Usage
    -----
    >>> from src.data.brats2020_dataset import build_brats_datasets
    >>> train_ds, val_ds = build_brats_datasets()
    """
    train_ds = BraTS2020Dataset(
        root_dir=train_root,
        modality=modality,
        target_size=target_size,
        max_patients=max_train_patients,
        simulate_rough=True,
    )
    val_ds = BraTS2020Dataset(
        root_dir=val_root,
        modality=modality,
        target_size=target_size,
        max_patients=max_val_patients,
        simulate_rough=True,
    )
    return train_ds, val_ds
