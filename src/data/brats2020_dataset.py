"""
BraTS 데이터셋 로더 (BraTS2020 / BraTS2021 공용, NIfTI .nii / .nii.gz 지원)
--------------------------------------------------------------------------
BraTS2021 폴더 구조:
  <root>/
    BraTS2021_00000/
      BraTS2021_00000_flair.nii.gz
      BraTS2021_00000_t1.nii.gz
      BraTS2021_00000_t1ce.nii.gz
      BraTS2021_00000_t2.nii.gz
      BraTS2021_00000_seg.nii.gz   ← GT 레이블
    BraTS2021_00002/ ...

BraTS2020 폴더 구조:
  <root>/
    BraTS20_Training_001/
      BraTS20_Training_001_flair.nii
      ...

GT 레이블(seg) 값:
  0 = 배경
  1 = Necrotic / Non-Enhancing Tumor Core (NCR/NET)
  2 = Peritumoral Edema (ED)
  4 = GD-Enhancing Tumor (ET)
  → 이진화: 0 이외 = 종양 (Whole Tumor)

출력 슬라이스:
  - image       : (C, H, W)     — 중심 슬라이스 MRI (모달리티 수 C)
  - image_25d   : (3C, H, W)    — z-1 / z / z+1 스택 (Small 2.5D용)
  - gt_mask     : (1, H, W)     — 이진 Whole Tumor 마스크
  - gt_regions  : (2, H, W)     — ED(label 2), TC(NCR∪ET)
  - rough_mask  : (1, H, W)     — make_noisy_mask()로 시뮬레이션한 초기 예측
"""

import os
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
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
    # 99 퍼센타일 기준 클리핑 후 0~1 스케일 (속도를 위해 샘플링)
    vals = normed[brain]
    sub_vals = vals[::8] if len(vals) > 100000 else vals
    p1, p99 = np.percentile(sub_vals, [1, 99])
    normed = np.clip(normed, p1, p99)
    normed = (normed - p1) / (p99 - p1 + 1e-8)
    return normed.astype(np.float32)


def _resize_volume_torch(vol: np.ndarray, target_size: int, is_mask: bool = False) -> np.ndarray:
    """PyTorch interpolate를 활용한 초고속 3D 볼륨 배치 리사이즈."""
    if target_size <= 0:
        return vol.astype(np.float32)
    H, W, D = vol.shape
    if H == target_size and W == target_size:
        return vol.astype(np.float32)
    
    # (H, W, D) -> (D, 1, H, W)
    t = torch.from_numpy(vol).permute(2, 0, 1).unsqueeze(1)
    mode = 'nearest' if is_mask else 'bilinear'
    align_corners = None if is_mask else False
    t_resized = torch.nn.functional.interpolate(
        t, size=(target_size, target_size), mode=mode, align_corners=align_corners
    )
    # (D, 1, target_size, target_size) -> (target_size, target_size, D)
    res = t_resized.squeeze(1).permute(1, 2, 0).numpy()
    return res.astype(np.float32)


def _process_single_patient_job(args: Tuple) -> Tuple[str, Optional[Tuple[list, list, list]], Optional[str]]:
    (
        pdir,
        modality_list,
        target_size,
        slice_selection,
        min_tumor_ratio,
        simulate_rough,
        noise_seed,
    ) = args

    pid = pdir.name
    rng = np.random.default_rng(noise_seed)

    mod_paths = [_find_modality_file(pdir, pid, m) for m in modality_list]
    seg_path = _find_seg_file(pdir, pid)

    if any(p is None for p in mod_paths) or seg_path is None:
        return pid, None, f"[SKIP] 파일 없음: {pid}"

    try:
        raw_mod_vols = [_load_volume(str(p)) for p in mod_paths]
        seg_vol = _load_volume(str(seg_path))

        mod_vols = [_normalize_volume(v) for v in raw_mod_vols]

        if target_size > 0:
            mod_vols = [_resize_volume_torch(v, target_size, is_mask=False) for v in mod_vols]
            seg_vol = _resize_volume_torch(seg_vol, target_size, is_mask=True)

        has_et = bool((seg_vol == 4.0).any())

        valid_zs = (
            range(mod_vols[0].shape[2])
            if slice_selection == "all"
            else _select_slices(seg_vol, min_tumor_ratio)
        )

        samples = []
        sample_pids = []
        sample_zs = []
        depth = int(mod_vols[0].shape[2])

        for z in valid_zs:
            z_prev = _clip_z(z - 1, depth)
            z_curr = _clip_z(z, depth)
            z_next = _clip_z(z + 1, depth)

            img_sl = np.stack([m[:, :, z_curr] for m in mod_vols], axis=0).astype(np.float32)
            prev_sl = np.stack([m[:, :, z_prev] for m in mod_vols], axis=0).astype(np.float32)
            next_sl = np.stack([m[:, :, z_next] for m in mod_vols], axis=0).astype(np.float32)

            img_25d = np.concatenate([prev_sl, img_sl, next_sl], axis=0)
            img_sl_out = img_sl[0] if img_sl.shape[0] == 1 else img_sl

            gt_sl = (seg_vol[:, :, z_curr] > 0).astype(np.float32)
            seg_sl = np.rint(seg_vol[:, :, z_curr]).astype(np.float32)

            if simulate_rough:
                rough_sl = make_noisy_mask(gt_sl, rng)
            else:
                rough_sl = gt_sl.copy()

            samples.append((img_sl_out, gt_sl, rough_sl, has_et, img_25d, seg_sl))
            sample_pids.append(pid)
            sample_zs.append(int(z))

        return pid, (samples, sample_pids, sample_zs), None

    except Exception as e:
        return pid, None, f"[ERROR] {pid}: {e}"



def _load_volume(path: str) -> np.ndarray:
    """NIfTI 파일(.nii / .nii.gz)을 (H, W, D) float32 배열로 반환."""
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32)


def _find_patient_dirs(root: str) -> List[Path]:
    """
    루트 디렉토리 아래 BraTS20* 또는 BraTS2021* 폴더 목록을 정렬하여 반환.
    BraTS2020: BraTS20_Training_XXX
    BraTS2021: BraTS2021_XXXXX
    """
    root_path = Path(root)
    if not root_path.exists():
        if root_path.name == "BraTS2021_Training_Data" and root_path.parent.exists():
            root_path = root_path.parent
        elif (Path("src/data/archive")).exists():
            root_path = Path("src/data/archive")
        elif (Path(__file__).parent / "archive").exists():
            root_path = Path(__file__).parent / "archive"
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
    """
    환자 폴더에서 지정 모달리티 파일을 찾아 반환.
    .nii.gz → .nii 순서로 탐색.
    """
    for ext in [".nii.gz", ".nii"]:
        candidate = pdir / f"{pid}_{modality}{ext}"
        if candidate.exists():
            return candidate
    return None


def _find_seg_file(pdir: Path, pid: str) -> Optional[Path]:
    """환자 폴더에서 seg 파일을 찾아 반환."""
    for ext in [".nii.gz", ".nii"]:
        candidate = pdir / f"{pid}_seg{ext}"
        if candidate.exists():
            return candidate
    return None


def _clip_z(z: int, depth: int) -> int:
    return int(max(0, min(depth - 1, z)))


def regions_from_seg(seg: np.ndarray) -> np.ndarray:
    """BraTS 레이블 → (2, H, W) = ED(label 2), TC(NCR 1 ∪ ET 4)."""
    ed = (seg == 2).astype(np.float32)
    tc = np.isin(seg, (1, 4)).astype(np.float32)
    return np.stack([ed, tc], axis=0)


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
    BraTS2020 / BraTS2021 2D 슬라이스 데이터셋.
    .nii 와 .nii.gz 형식을 모두 자동 감지합니다.

    Parameters
    ----------
    root_dir : str
        BraTS20_Training_* 또는 BraTS2021_* 폴더들이 있는 최상위 경로.
        예) "src/data/archive/BraTS2021_Training_Data"
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
        patient_ids: Optional[List[str]] = None,
        slice_selection: str = "tumor",
        num_workers: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        if slice_selection not in {"tumor", "all"}:
            raise ValueError("slice_selection must be tumor or all")
        self.slice_selection = slice_selection
        if num_workers is None:
            num_workers = os.environ.get("BRATS_NUM_WORKERS", "1")
        self.num_workers = max(1, int(num_workers))
        self.cache_dir = cache_dir or os.environ.get("BRATS_CACHE_DIR")
        self.root_dir        = root_dir
        self.modality        = modality
        self.target_size     = target_size
        self.min_tumor_ratio = min_tumor_ratio
        self.simulate_rough  = simulate_rough
        self.noise_seed      = noise_seed
        self.patient_ids     = set(patient_ids) if patient_ids is not None else None
        self.rng = np.random.default_rng(noise_seed)

        self._samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._sample_pids: List[str] = []
        self._sample_zs: List[int] = []
        if self._load_cache():
            return
        self._build(max_patients)
        self._save_cache()

    def _cache_path(self) -> Optional[Path]:
        if not self.cache_dir:
            return None
        ids = sorted(self.patient_ids) if self.patient_ids is not None else []
        key = repr((str(self.root_dir), self.modality, self.target_size,
                    self.slice_selection, self.simulate_rough, self.min_tumor_ratio,
                    self.noise_seed, self.patient_ids is not None, ids)).encode()
        digest = hashlib.sha1(key).hexdigest()[:16]
        return Path(self.cache_dir) / f"brats_{digest}.pkl"

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if path is None or not path.is_file():
            return False
        try:
            with path.open("rb") as f:
                samples, pids, zs = pickle.load(f)
            self._samples = samples
            self._sample_pids = pids
            self._sample_zs = zs
            print(f"[BraTS Cache] loaded {len(samples)} slices: {path}")
            return True
        except Exception as exc:
            print(f"[BraTS Cache] ignoring invalid cache {path}: {exc}")
            return False

    def _save_cache(self) -> None:
        path = self._cache_path()
        if path is None or not self._samples:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump((self._samples, self._sample_pids, self._sample_zs), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        print(f"[BraTS Cache] saved {len(self._samples)} slices: {path}")

    # ── 내부 빌더 ──────────────────────────────────────────
    def _build(self, max_patients: Optional[int]) -> None:
        patient_dirs = _find_patient_dirs(self.root_dir)
        if not patient_dirs:
            print(f"[BraTS2020Dataset] 경고: '{self.root_dir}' 에서 환자 폴더를 찾을 수 없습니다.")
            return

        if self.patient_ids is not None:
            patient_dirs = [p for p in patient_dirs if p.name in self.patient_ids]
        elif max_patients is not None:
            patient_dirs = patient_dirs[:max_patients]

        # 모달리티 파싱 (단일 't1ce' 또는 복합 't1ce+flair', 't1ce+t2' 등 지원)
        if isinstance(self.modality, str):
            self.modality_list = [m.strip() for m in self.modality.replace(',', '+').split('+')]
        else:
            self.modality_list = list(self.modality)

        jobs = []
        for pdir in patient_dirs:
            patient_seed = int(hashlib.md5(f"{self.noise_seed}_{pdir.name}".encode()).hexdigest()[:8], 16) % (2**31)
            jobs.append((
                pdir,
                self.modality_list,
                self.target_size,
                self.slice_selection,
                self.min_tumor_ratio,
                self.simulate_rough,
                patient_seed,
            ))

        total_slices = 0
        skipped = 0
        max_w = min(self.num_workers, len(jobs))
        print(f"[BraTS Dataset] {len(jobs)}명 환자 병렬 로딩 중 (workers={max_w})...")

        if max_w > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing
            with ProcessPoolExecutor(max_workers=max_w, mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = {executor.submit(_process_single_patient_job, job): job[0].name for job in jobs}
                completed = []
                for future in as_completed(futures):
                    completed.append(future.result())
                for pid, res, err_msg in sorted(completed, key=lambda row: row[0]):
                    if err_msg or res is None:
                        skipped += 1
                        print(f"  {err_msg}")
                    else:
                        p_samples, p_pids, p_zs = res
                        self._samples.extend(p_samples)
                        self._sample_pids.extend(p_pids)
                        self._sample_zs.extend(p_zs)
                        total_slices += len(p_samples)
        else:
            for job in jobs:
                pid, res, err_msg = _process_single_patient_job(job)
                if err_msg or res is None:
                    skipped += 1
                    print(f"  {err_msg}")
                else:
                    p_samples, p_pids, p_zs = res
                    self._samples.extend(p_samples)
                    self._sample_pids.extend(p_pids)
                    self._sample_zs.extend(p_zs)
                    total_slices += len(p_samples)

        print(f"[BraTS Dataset] 완료: 총 {total_slices}개 유효 슬라이스 로드. (건너뜀: {skipped}명)")

    def _modal_slice(self, mod_vols: List[np.ndarray], z: int) -> np.ndarray:
        """모달리티 볼륨에서 z 슬라이스를 (C, H, W)로 반환. 범위 밖은 가장자리로 클램프."""
        z = _clip_z(z, int(mod_vols[0].shape[2]))
        channels = []
        for m_vol in mod_vols:
            sl = m_vol[:, :, z]
            if self.target_size > 0:
                sl = self._resize(sl)
            channels.append(sl)
        return np.stack(channels, axis=0).astype(np.float32)

    def _resize(self, arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """간단한 바이선형/최근접 이웃 리사이즈 (skimage)."""
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
        sample = self._samples[idx]
        img, gt, rough, has_et = sample[0], sample[1], sample[2], sample[3]
        img_25d = sample[4] if len(sample) > 4 else None
        seg_sl = sample[5] if len(sample) > 5 else None
        if img.ndim == 2:
            img_tensor = torch.from_numpy(img).unsqueeze(0)
        else:
            img_tensor = torch.from_numpy(img)

        if img_25d is None:
            img_25d_t = torch.cat([img_tensor, img_tensor, img_tensor], dim=0)
        else:
            img_25d_t = torch.from_numpy(img_25d)

        if seg_sl is not None:
            gt_regions = torch.from_numpy(regions_from_seg(seg_sl))
        else:
            wt = torch.from_numpy(gt).unsqueeze(0)
            gt_regions = torch.cat([wt, wt], dim=0)

        return {
            "image":      img_tensor,
            "image_25d":  img_25d_t,
            "gt_mask":    torch.from_numpy(gt).unsqueeze(0),
            "gt_regions": gt_regions,
            "rough_mask": torch.from_numpy(rough).unsqueeze(0),
            "has_et":     has_et,
        }

    def get_numpy_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RL 환경 초기화용 NumPy 배열 반환.
        Returns: images (N,C,H,W) or (N,H,W), gt_masks (N,H,W), rough_masks (N,H,W)
        """
        imgs   = np.stack([s[0] for s in self._samples], axis=0)
        gts    = np.stack([s[1] for s in self._samples], axis=0)
        roughs = np.stack([s[2] for s in self._samples], axis=0)
        return imgs, gts, roughs

    def get_numpy_25d_arrays(self) -> np.ndarray:
        """(N, 3C, H, W) prev/center/next. 없으면 center를 세 번 복제."""
        stacks = []
        for s in self._samples:
            if len(s) > 4 and s[4] is not None:
                stacks.append(s[4])
                continue
            img = s[0]
            if img.ndim == 2:
                stacks.append(np.stack([img, img, img], axis=0))
            else:
                stacks.append(np.concatenate([img, img, img], axis=0))
        return np.stack(stacks, axis=0)

    def get_et_presence_array(self) -> np.ndarray:
        """
        각 슬라이스별로 해당 환자 볼륨의 ET(레이블 4) 존재 여부를 bool 배열로 반환.
        Returns: has_ets (N,)
        """
        return np.array([s[3] for s in self._samples], dtype=bool)


# ──────────────────────────────────────────────
# BraTS 경로 헬퍼
# ──────────────────────────────────────────────

# BraTS2021 기본 데이터 경로
_DATA_DIR = Path(__file__).parent / "archive"
DEFAULT_TRAIN_ROOT = str(_DATA_DIR)
DEFAULT_VAL_ROOT   = str(_DATA_DIR)  # BraTS2021은 train/val 분리가 없으므로 동일 경로 사용


def build_brats_datasets(
    train_root: str = DEFAULT_TRAIN_ROOT,
    val_root:   str = DEFAULT_VAL_ROOT,
    modality:   str = "t1ce",
    target_size: int = 128,
    max_train_patients: Optional[int] = None,
    max_val_patients:   Optional[int] = None,
) -> Tuple["BraTS2020Dataset", "BraTS2020Dataset"]:
    """
    학습/검증용 BraTS 데이터셋을 한 번에 생성합니다.
    BraTS2021은 별도 val 폴더가 없으므로 train 데이터를 분할하여 사용합니다.

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
