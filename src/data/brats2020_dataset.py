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


def _apply_clahe_2d(img_2d: np.ndarray, clip_limit: float = 0.02) -> np.ndarray:
    """skimage 기반 2D CLAHE 적용 (float32 [0, 1] 입력)."""
    from skimage.exposure import equalize_adapthist
    if img_2d.max() <= 1e-6:
        return img_2d
    return equalize_adapthist(np.clip(img_2d, 0, 1), clip_limit=clip_limit).astype(np.float32)


def _apply_bilateral_2d(img_2d: np.ndarray, sigma_color: float = 0.05, sigma_spatial: float = 2.0) -> np.ndarray:
    """skimage 기반 Bilateral Filter (Anisotropic Diffusion 근사) 적용."""
    from skimage.restoration import denoise_bilateral
    if img_2d.max() <= 1e-6:
        return img_2d
    return denoise_bilateral(
        np.clip(img_2d, 0, 1), 
        sigma_color=sigma_color, 
        sigma_spatial=sigma_spatial, 
        channel_axis=None
    ).astype(np.float32)


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


def _gaussian_blur_batch(x: "torch.Tensor", sigma: float = 2.0, truncate: float = 4.0) -> "torch.Tensor":
    """Separable Gaussian blur for (N, H, W) float tensors (scipy-like truncate)."""
    import torch
    import torch.nn.functional as F

    radius = max(1, int(truncate * sigma + 0.5))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    n = x.shape[0]
    x4 = x.unsqueeze(1)
    x4 = F.conv2d(F.pad(x4, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
    x4 = F.conv2d(F.pad(x4, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))
    return x4.view(n, *x.shape[-2:])


def make_noisy_masks_batch(
    gt_masks: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    erosion_prob: float = 0.5,
    max_morph_px: int = 5,
    sigma: float = 2.0,
    device: Optional[str] = None,
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched GPU (or CPU) morph noise + Gaussian prob maps.

    Returns (noisy_roughs, soft_probs) each float32 with shape (N, H, W).
    Semantics match make_noisy_mask + scipy.ndimage.gaussian_filter(sigma=2).
    """
    import torch
    import torch.nn.functional as F

    if rng is None:
        rng = np.random.default_rng()
    if gt_masks.ndim != 3:
        raise ValueError(f"gt_masks must be (N,H,W), got {gt_masks.shape}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    n_total = int(gt_masks.shape[0])
    rough_out = np.empty(gt_masks.shape, dtype=np.float32)
    prob_out = np.empty(gt_masks.shape, dtype=np.float32)
    if n_total == 0:
        return rough_out, prob_out

    max_morph_px = int(max_morph_px)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        n = end - start
        masks = torch.from_numpy(gt_masks[start:end] > 0.5).to(dev, non_blocking=True)

        n_ops = rng.integers(2, 6, size=n)
        max_ops = int(n_ops.max()) if n else 0
        for step in range(max_ops):
            active = n_ops > step
            if not np.any(active):
                continue
            radii = rng.integers(1, max_morph_px + 1, size=n)
            do_erode = rng.random(n) < erosion_prob
            active_t = torch.as_tensor(active, device=dev)
            for r in range(1, max_morph_px + 1):
                k = 2 * r + 1
                pad = r
                dil_sel = active_t & torch.as_tensor((radii == r) & (~do_erode), device=dev)
                ero_sel = active_t & torch.as_tensor((radii == r) & do_erode, device=dev)
                if bool(dil_sel.any()):
                    m = masks[dil_sel].float().unsqueeze(1)
                    m = F.max_pool2d(m, kernel_size=k, stride=1, padding=pad) > 0.5
                    masks[dil_sel] = m.squeeze(1)
                if bool(ero_sel.any()):
                    m = (~masks[ero_sel]).float().unsqueeze(1)
                    m = F.max_pool2d(m, kernel_size=k, stride=1, padding=pad) > 0.5
                    masks[ero_sel] = ~m.squeeze(1)

        dil = F.max_pool2d(masks.float().unsqueeze(1), 3, 1, 1).squeeze(1) > 0.5
        ero = ~((F.max_pool2d((~masks).float().unsqueeze(1), 3, 1, 1).squeeze(1) > 0.5))
        boundary = dil ^ ero
        flip = boundary & (torch.rand(masks.shape, device=dev) < 0.25)
        noisy = masks.float()
        noisy = torch.where(flip, 1.0 - noisy, noisy)
        probs = _gaussian_blur_batch(noisy, sigma=sigma)

        rough_out[start:end] = noisy.detach().cpu().numpy().astype(np.float32, copy=False)
        prob_out[start:end] = probs.detach().cpu().numpy().astype(np.float32, copy=False)

    return rough_out, prob_out


def _resize_slice(arr: np.ndarray, target_size: int, is_mask: bool = False) -> np.ndarray:
    from skimage.transform import resize as sk_resize

    if target_size <= 0:
        return arr.astype(np.float32)
    order = 0 if is_mask else 1
    resized = sk_resize(
        arr,
        (target_size, target_size),
        order=order,
        mode="constant",
        anti_aliasing=(not is_mask),
        preserve_range=True,
    )
    return resized.astype(np.float32)


def _modal_slice_from_vols(
    mod_vols: List[np.ndarray], z: int, target_size: int, modality_list: List[str]
) -> np.ndarray:
    z = _clip_z(z, int(mod_vols[0].shape[2]))
    channels = []
    for i, m_vol in enumerate(mod_vols):
        sl = m_vol[:, :, z]
        
        # 적용: 단계 2. 기존 2채널 전처리 품질 극대화
        mod_name = modality_list[i].lower()
        if mod_name == "t1ce":
            sl = _apply_clahe_2d(sl)
        elif mod_name == "flair":
            sl = _apply_bilateral_2d(sl)
            
        if target_size > 0:
            sl = _resize_slice(sl, target_size, is_mask=False)
        channels.append(sl)
    return np.stack(channels, axis=0).astype(np.float32)


def _cache_dir_for(
    root_dir: str,
    modality_list: List[str],
    target_size: int,
    min_tumor_ratio: float,
    simulate_rough: bool,
    noise_seed: int,
) -> Path:
    key = (
        f"{'+'.join(modality_list)}_s{target_size}_"
        f"r{min_tumor_ratio:g}_rough{int(simulate_rough)}_seed{noise_seed}"
    )
    base = Path(root_dir)
    if base.name == "BraTS2021_Training_Data":
        base = base.parent
    return base / ".brats_cache" / key


def _load_one_patient(
    pdir_str: str,
    modality_list: List[str],
    target_size: int,
    min_tumor_ratio: float,
    simulate_rough: bool,
    noise_seed: int,
    cache_dir: Optional[str] = None,
) -> Tuple[str, list, int, Optional[str]]:
    """(pid, samples, n_slices, error). samples keep BraTS2020Dataset tuple layout."""
    pdir = Path(pdir_str)
    pid = pdir.name

    if cache_dir:
        cpath = Path(cache_dir) / f"{pid}.npz"
        if cpath.is_file():
            try:
                data = np.load(cpath, allow_pickle=False)
                n = int(data["n"])
                samples = [
                    (
                        data[f"img_{i}"],
                        data[f"gt_{i}"],
                        data[f"rough_{i}"],
                        bool(data[f"has_et_{i}"]),
                        data[f"img25d_{i}"],
                        data[f"seg_{i}"],
                    )
                    for i in range(n)
                ]
                return pid, samples, n, None
            except Exception:
                pass

    mod_paths = [_find_modality_file(pdir, pid, m) for m in modality_list]
    seg_path = _find_seg_file(pdir, pid)
    if any(p is None for p in mod_paths) or seg_path is None:
        return pid, [], 0, f"파일 없음: {pid}"

    try:
        mod_vols = [_normalize_volume(_load_volume(str(p))) for p in mod_paths]
        seg_vol = _load_volume(str(seg_path))
        has_et = bool((seg_vol == 4.0).any())
        valid_zs = _select_slices(seg_vol, min_tumor_ratio)
        rng = np.random.default_rng(noise_seed + (abs(hash(pid)) % 1_000_000_007))

        samples = []
        for z in valid_zs:
            img_sl = _modal_slice_from_vols(mod_vols, z, target_size, modality_list)
            img_25d = np.concatenate(
                [
                    _modal_slice_from_vols(mod_vols, z - 1, target_size, modality_list),
                    img_sl,
                    _modal_slice_from_vols(mod_vols, z + 1, target_size, modality_list),
                ],
                axis=0,
            )
            if img_sl.shape[0] == 1:
                img_sl = img_sl[0]

            gt_sl = (seg_vol[:, :, z] > 0).astype(np.float32)
            seg_sl = seg_vol[:, :, z].astype(np.float32)
            if target_size > 0:
                gt_sl = _resize_slice(gt_sl, target_size, is_mask=True)
                seg_sl = np.rint(_resize_slice(seg_sl, target_size, is_mask=True)).astype(np.float32)

            rough_sl = make_noisy_mask(gt_sl, rng) if simulate_rough else gt_sl.copy()
            samples.append((img_sl, gt_sl, rough_sl, has_et, img_25d, seg_sl))

        if cache_dir and samples:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            payload = {"n": np.asarray(len(samples), dtype=np.int32)}
            for i, (img, gt, rough, het, img25, seg) in enumerate(samples):
                payload[f"img_{i}"] = img
                payload[f"gt_{i}"] = gt
                payload[f"rough_{i}"] = rough
                payload[f"has_et_{i}"] = np.asarray(het, dtype=np.bool_)
                payload[f"img25d_{i}"] = img25
                payload[f"seg_{i}"] = seg
            # numpy savez appends .npz if missing — keep final suffix as .npz
            out = Path(cache_dir) / f"{pid}.npz"
            tmp = Path(cache_dir) / f"{pid}.tmp.npz"
            np.savez_compressed(tmp, **payload)
            tmp.replace(out)

        return pid, samples, len(samples), None
    except Exception as e:
        return pid, [], 0, f"{pid}: {e}"


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class BraTS2020Dataset(Dataset):
    """
    BraTS2020 / BraTS2021 2D 슬라이스 데이터셋.
    .nii 와 .nii.gz 형식을 모두 자동 감지합니다.

    Parallel load via BRATS_NUM_WORKERS (default min(32, CPU)).
    Slice cache under <archive>/.brats_cache/ (disable with BRATS_NO_CACHE=1).
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
        num_workers: Optional[int] = None,
        use_cache: bool = True,
    ):
        self.root_dir        = root_dir
        self.modality        = modality
        self.target_size     = target_size
        self.min_tumor_ratio = min_tumor_ratio
        self.simulate_rough  = simulate_rough
        self.patient_ids     = set(patient_ids) if patient_ids is not None else None
        self.noise_seed      = noise_seed
        self.rng = np.random.default_rng(noise_seed)
        if num_workers is None:
            num_workers = int(os.environ.get("BRATS_NUM_WORKERS", "0")) or min(
                32, (os.cpu_count() or 4)
            )
        self.num_workers = max(1, int(num_workers))
        self.use_cache = bool(use_cache) and os.environ.get("BRATS_NO_CACHE", "") != "1"

        self._samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._sample_pids: List[str] = []
        self._build(max_patients)

    def _build(self, max_patients: Optional[int]) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        patient_dirs = _find_patient_dirs(self.root_dir)
        if not patient_dirs:
            print(f"[BraTS2020Dataset] 경고: '{self.root_dir}' 에서 환자 폴더를 찾을 수 없습니다.")
            return

        if self.patient_ids is not None:
            patient_dirs = [p for p in patient_dirs if p.name in self.patient_ids]
        elif max_patients is not None:
            patient_dirs = patient_dirs[:max_patients]

        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")

        if isinstance(self.modality, str):
            self.modality_list = [m.strip() for m in self.modality.replace(",", "+").split("+")]
        else:
            self.modality_list = list(self.modality)

        cache_path = None
        if self.use_cache:
            cache_path = _cache_dir_for(
                self.root_dir,
                self.modality_list,
                self.target_size,
                self.min_tumor_ratio,
                self.simulate_rough,
                self.noise_seed,
            )
            cache_path.mkdir(parents=True, exist_ok=True)
            print(
                f"[BraTS Dataset] workers={self.num_workers}, cache={cache_path} "
                f"({len(patient_dirs)} patients)"
            )
        else:
            print(f"[BraTS Dataset] workers={self.num_workers}, cache=OFF")

        total_slices = 0
        skipped = 0
        results_by_pid = {}

        def _job(pdir: Path):
            return _load_one_patient(
                str(pdir),
                self.modality_list,
                self.target_size,
                self.min_tumor_ratio,
                self.simulate_rough,
                self.noise_seed,
                str(cache_path) if cache_path is not None else None,
            )

        try:
            from src.utils.progress import want_tqdm
            from tqdm import tqdm
            pbar = (
                tqdm(
                    total=len(patient_dirs),
                    desc="[BraTS] Loading patients",
                    unit="pt",
                    ncols=80,
                    ascii=True,
                )
                if want_tqdm()
                else None
            )
        except ImportError:
            pbar = None

        n_patients = len(patient_dirs)
        log_every = max(1, n_patients // 10)
        done_pts = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futs = {pool.submit(_job, p): p.name for p in patient_dirs}
            for fut in as_completed(futs):
                pid, samples, n, err = fut.result()
                if err:
                    skipped += 1
                    if pbar is not None and hasattr(pbar, "write"):
                        pbar.write(f"  [SKIP] {err}")
                    else:
                        print(f"  [SKIP] {err}")
                else:
                    results_by_pid[pid] = samples
                    total_slices += n
                done_pts += 1
                if pbar is not None:
                    pbar.set_postfix({"slices": total_slices})
                    pbar.update(1)
                elif done_pts == 1 or done_pts % log_every == 0 or done_pts >= n_patients:
                    print(
                        f"[BraTS] Loading patients: {done_pts}/{n_patients} "
                        f"(slices={total_slices})",
                        flush=True,
                    )
        if pbar is not None:
            pbar.close()

        for pdir in patient_dirs:
            samples = results_by_pid.get(pdir.name)
            if not samples:
                continue
            for s in samples:
                self._samples.append(s)
                self._sample_pids.append(pdir.name)

        print(f"[BraTS Dataset] 완료: 총 {total_slices}개 유효 슬라이스 로드. (건너뜀: {skipped}명)")

    def _modal_slice(self, mod_vols: List[np.ndarray], z: int) -> np.ndarray:
        return _modal_slice_from_vols(mod_vols, z, self.target_size, self.modality_list)

    def _resize(self, arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
        return _resize_slice(arr, self.target_size, is_mask=is_mask)

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
