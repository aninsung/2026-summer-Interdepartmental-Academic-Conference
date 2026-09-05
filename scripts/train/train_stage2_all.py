"""
Stage 2: train CaraNet / UNet++ / SegResNet with a single BraTS load.

Loads train/val once (no size filter), then clones+filters per expert class.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_TRAIN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _TRAIN)

from src.data.patient_split import clone_filtered_by_size, load_split_brats_datasets
from src.utils.seed import set_seed

from train_caranet import train_caranet
from train_segresnet import train_segresnet
from train_unetplusplus import train_unetplusplus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: all experts, one BraTS load")
    parser.add_argument("--use_real_data", action="store_true", default=True)
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--max_train_patients", type=int, default=None)
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--no_multi_region", action="store_true", help="SegResNet: disable ED/TC heads")
    args = parser.parse_args()
    set_seed(args.seed, args.deterministic)
    augment = not args.no_augment

    log.info("Stage2 ALL: BraTS train/val 1회 로드 (크기 필터는 expert별로 복제 후 적용)")
    train_full, val_full = load_split_brats_datasets(
        train_root=args.train_root,
        modality=args.modality,
        target_size=args.target_size,
        max_patients=args.max_train_patients,
        patient_split=args.patient_split,
        refinement_mode=None,
        simulate_rough=False,
    )
    log.info("공유 풀: train=%d  val=%d", len(train_full), len(val_full))

    common = dict(
        use_real_data=True,
        train_root=args.train_root,
        modality=args.modality,
        target_size=args.target_size,
        max_train_patients=args.max_train_patients,
        patient_split=args.patient_split,
        seed=args.seed,
        deterministic=args.deterministic,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        augment=augment,
    )

    # Small — CaraNet
    tr = clone_filtered_by_size(train_full, "small")
    va = clone_filtered_by_size(val_full, "small")
    log.info("=== Stage2 Small (CaraNet) train=%d val=%d ===", len(tr), len(va))
    train_caranet(
        **common,
        refinement_mode="small",
        save_path="checkpoints/caranet_best.pt",
        train_ds=tr,
        val_ds=va,
    )

    # Medium — UNet++
    tr = clone_filtered_by_size(train_full, "medium")
    va = clone_filtered_by_size(val_full, "medium")
    log.info("=== Stage2 Medium (UNet++) train=%d val=%d ===", len(tr), len(va))
    train_unetplusplus(
        **common,
        refinement_mode="medium",
        save_path="checkpoints/unetplusplus_best.pt",
        train_ds=tr,
        val_ds=va,
    )

    # Large — SegResNet
    tr = clone_filtered_by_size(train_full, "large")
    va = clone_filtered_by_size(val_full, "large")
    log.info("=== Stage2 Large (SegResNet) train=%d val=%d ===", len(tr), len(va))
    train_segresnet(
        **common,
        refinement_mode="large",
        save_path="checkpoints/segresnet_best.pt",
        multi_region=not args.no_multi_region,
        loss_type="bce_dice",
        train_ds=tr,
        val_ds=va,
    )
    log.info("Stage2 ALL 완료 (CaraNet / UNet++ / SegResNet)")


if __name__ == "__main__":
    main()
