"""환자 단위 train/val 분할."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.brats2020_dataset import BraTS2020Dataset, _find_patient_dirs

log = logging.getLogger(__name__)

DEFAULT_SPLIT_PATH = "checkpoints/patient_split.json"


def list_patient_ids(root_dir: str) -> List[str]:
    return [p.name for p in _find_patient_dirs(root_dir)]


def _select_patient_pool(
    root_dir: str,
    max_patients: Optional[int],
    seed: int,
) -> List[str]:
    """전체 환자 ID 중 seed 기반 무작위 샘플(max_patients명)을 선택."""
    ids = list_patient_ids(root_dir)
    if max_patients is not None and len(ids) > int(max_patients):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(ids), size=int(max_patients), replace=False)
        ids = sorted(ids[i] for i in pick)
    return ids


def _split_cache_valid(
    split: Dict,
    root_dir: str,
    max_patients: Optional[int],
    val_ratio: float,
    seed: int,
) -> bool:
    return (
        split.get("root_dir") == root_dir
        and split.get("max_patients") == max_patients
        and split.get("val_ratio") == val_ratio
        and split.get("seed") == seed
    )


def load_or_create_patient_split(
    root_dir: str,
    max_patients: Optional[int] = 210,
    split_path: str = DEFAULT_SPLIT_PATH,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict:
    path = Path(split_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            split = json.load(f)
        if _split_cache_valid(split, root_dir, max_patients, val_ratio, seed):
            log.info(
                "환자 분할 로드: %s (train=%d, val=%d)",
                path,
                len(split.get("train", [])),
                len(split.get("val", [])),
            )
            return split
        log.warning(
            "환자 분할 설정 변경 감지 (max_patients/root/seed/val_ratio) → %s 재생성",
            path,
        )

    ids = _select_patient_pool(root_dir, max_patients, seed)
    if not ids:
        raise ValueError(f"환자 폴더를 찾을 수 없습니다: {root_dir}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_val = max(1, int(round(len(ids) * val_ratio)))
    if len(ids) > 1:
        n_val = min(n_val, len(ids) - 1)
    val_idx = set(int(i) for i in perm[:n_val])
    train = [ids[i] for i in range(len(ids)) if i not in val_idx]
    val = [ids[i] for i in range(len(ids)) if i in val_idx]
    split = {
        "seed": seed,
        "val_ratio": val_ratio,
        "max_patients": max_patients,
        "root_dir": root_dir,
        "train": train,
        "val": val,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    log.info(
        "환자 분할 생성: %s (train=%d, val=%d, seed=%d)",
        path,
        len(train),
        len(val),
        seed,
    )
    return split


def filter_dataset_by_size(ds: BraTS2020Dataset, refinement_mode: str) -> None:
    ref_m = refinement_mode.lower()
    old_len = len(ds._samples)
    pids = getattr(ds, "_sample_pids", [None] * old_len)
    keep_s, keep_p = [], []
    for sample, pid in zip(ds._samples, pids):
        area = float(np.sum(sample[1]))
        ok = (
            (ref_m == "small" and 0 < area < 300)
            or (ref_m == "medium" and 300 <= area < 700)
            or (ref_m == "large" and area >= 700)
        )
        if ok:
            keep_s.append(sample)
            keep_p.append(pid)
    ds._samples = keep_s
    ds._sample_pids = keep_p
    log.info("[%s] 크기 필터: %d → %d 슬라이스", ref_m.upper(), old_len, len(ds._samples))


def clone_filtered_by_size(ds: BraTS2020Dataset, refinement_mode: str) -> BraTS2020Dataset:
    """Shallow-clone dataset then apply size filter (does not mutate the source)."""
    out = copy.copy(ds)
    out._samples = list(ds._samples)
    pids = getattr(ds, "_sample_pids", None)
    out._sample_pids = list(pids) if pids is not None else [None] * len(out._samples)
    filter_dataset_by_size(out, refinement_mode)
    return out


def split_dataset_by_patients(
    ds: BraTS2020Dataset,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> Tuple[BraTS2020Dataset, BraTS2020Dataset]:
    train_set, val_set = set(train_ids), set(val_ids)
    pids = getattr(ds, "_sample_pids", None)
    if not pids or len(pids) != len(ds._samples):
        raise ValueError("데이터셋에 환자 ID가 없어 환자 단위 분할을 할 수 없습니다.")

    tr_s, tr_p, va_s, va_p = [], [], [], []
    for sample, pid in zip(ds._samples, pids):
        if pid in train_set:
            tr_s.append(sample)
            tr_p.append(pid)
        elif pid in val_set:
            va_s.append(sample)
            va_p.append(pid)

    train_ds = copy.copy(ds)
    val_ds = copy.copy(ds)
    train_ds._samples, train_ds._sample_pids = tr_s, tr_p
    val_ds._samples, val_ds._sample_pids = va_s, va_p
    return train_ds, val_ds


def prepare_train_val_datasets(
    full_ds: BraTS2020Dataset,
    train_root: str,
    max_patients: Optional[int],
    patient_split: Optional[str] = None,
    refinement_mode: Optional[str] = None,
) -> Tuple[BraTS2020Dataset, BraTS2020Dataset]:
    split_path = patient_split or DEFAULT_SPLIT_PATH
    split = load_or_create_patient_split(train_root, max_patients, split_path)
    train_ds, val_ds = split_dataset_by_patients(full_ds, split["train"], split["val"])
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"분할 결과가 비었습니다 (train={len(train_ds)}, val={len(val_ds)}). "
            f"데이터셋에 실린 환자와 {split_path} 의 환자 목록이 어긋났을 가능성이 큽니다. "
            "BraTS2020Dataset 을 patient_ids= 로 로드하거나 split 파일을 삭제해 재생성하세요."
        )
    if refinement_mode:
        filter_dataset_by_size(train_ds, refinement_mode)
        filter_dataset_by_size(val_ds, refinement_mode)
    log.info(
        "환자 단위 분할: train %d명/%d장, val %d명/%d장",
        len(split["train"]),
        len(train_ds),
        len(split["val"]),
        len(val_ds),
    )
    return train_ds, val_ds


def load_split_brats_datasets(
    train_root: str,
    modality: str,
    target_size: int,
    max_patients: Optional[int],
    patient_split: Optional[str] = None,
    refinement_mode: Optional[str] = None,
    simulate_rough: bool = True,
) -> Tuple[BraTS2020Dataset, BraTS2020Dataset]:
    # 분할을 먼저 확정하고 그 환자만 로드한다.
    # max_patients 를 데이터셋에 그대로 넘기면 "정렬 순 앞 N명"이 실리는데,
    # 분할은 전체에서 무작위 N명을 뽑으므로 둘이 어긋나 슬라이스가 0이 된다.
    split_path = patient_split or DEFAULT_SPLIT_PATH
    split = load_or_create_patient_split(train_root, max_patients, split_path)
    pool = list(split["train"]) + list(split["val"])

    full_ds = BraTS2020Dataset(
        root_dir=train_root,
        modality=modality,
        target_size=target_size,
        max_patients=None,
        patient_ids=pool,
        simulate_rough=simulate_rough,
    )
    return prepare_train_val_datasets(
        full_ds, train_root, max_patients, split_path, refinement_mode
    )
