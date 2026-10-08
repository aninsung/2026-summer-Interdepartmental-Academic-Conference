"""경계 띠 픽셀 보정 지도학습.

Medium 슬라이스만 쓴다. 띠 밖은 Stage 2를 유지하고, 띠 안은 끄기/유지/켜기를
GT에 맞춘다. 손실은 전체 마스크 Dice와 띠 안의 경계 거리다.
검증에서 저장하는 점수는 FLAIR 가드를 적용한 슬라이스 DSC/HD95다.

사용 예:
  BRATS_NUM_WORKERS=16 python -u scripts/train/train_band_refine.py \
      --classes medium --epochs 20 --save_path checkpoints/band_refine_medium.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.band_dataset import BandSliceDataset
from src.data.contour_dataset import build_stage2_entries
from src.models.band_refine import build_band_refine, predict_pair, reconstruct_mask
from src.utils.evaluation_records import measured_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_INDEX = {"small": 0, "medium": 1, "large": 2}


def band_loss(logits, stage2, gt, band, phi, target, boundary_weight: float, ce_weight: float):
    """띠 CE + 전체 Dice + 띠 경계거리. CE는 유지 픽셀이 대부분이므로 균등 평균만 쓴다."""
    pred = reconstruct_mask(logits, stage2, band)
    editable = band[:, 0] > 0.5
    n_edit = editable.sum().clamp(min=1)
    per_pixel = F.cross_entropy(logits, target, reduction="none")
    ce = (per_pixel * editable).sum() / n_edit

    p = pred.flatten(1)
    g = gt.flatten(1)
    inter = (p * g).sum(dim=1)
    dice = 1.0 - ((2.0 * inter + 1e-5) / (p.sum(dim=1) + g.sum(dim=1) + 1e-5)).mean()

    boundary = (pred * phi * band).sum() / n_edit
    loss = ce_weight * ce + dice + boundary_weight * boundary
    return loss, ce, dice, boundary


@torch.no_grad()
def evaluate_slices(model, entries, device, prob_lo, prob_hi, radius, flair_index) -> dict:
    """가드 적용/미적용을 Stage 2와 같은 슬라이스에서 비교한다."""
    model.eval()
    stage2_dsc, raw_dsc, guard_dsc = [], [], []
    stage2_hd, raw_hd, guard_hd = [], [], []
    for entry in entries:
        base = measured_metrics(entry.stage2, entry.gt)
        raw, guarded = predict_pair(model, entry, device, prob_lo, prob_hi, radius, flair_index)
        raw_m = measured_metrics(raw, entry.gt)
        guard_m = measured_metrics(guarded, entry.gt)
        stage2_dsc.append(base["dsc"])
        raw_dsc.append(raw_m["dsc"])
        guard_dsc.append(guard_m["dsc"])
        if base["hd95_surface_px"] is not None:
            stage2_hd.append(base["hd95_surface_px"])
        if raw_m["hd95_surface_px"] is not None:
            raw_hd.append(raw_m["hd95_surface_px"])
        if guard_m["hd95_surface_px"] is not None:
            guard_hd.append(guard_m["hd95_surface_px"])

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "stage2_dsc": mean(stage2_dsc),
        "raw_dsc": mean(raw_dsc),
        "guard_dsc": mean(guard_dsc),
        "stage2_hd95": mean(stage2_hd),
        "raw_hd95": mean(raw_hd),
        "guard_hd95": mean(guard_hd),
        "n": len(stage2_dsc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="경계 띠 픽셀 보정 학습")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--max_train_patients", type=int, default=400)
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--flair_index", type=int, default=1)
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--classes", type=str, default="medium")
    parser.add_argument("--prob_lo", type=float, default=0.35)
    parser.add_argument("--prob_hi", type=float, default=0.65)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--boundary_weight", type=float, default=0.3)
    parser.add_argument("--ce_weight", type=float, default=0.15)
    parser.add_argument("--dsc_slack", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--stage2_thresholds", type=str, default="0.80,0.80,0.50")
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", type=str, default="", help="이어서 지도학습할 가중치")
    parser.add_argument("--save_path", type=str, default="checkpoints/band_refine_medium.pt")
    args = parser.parse_args()

    from src.utils.seed import set_seed
    set_seed(args.seed, deterministic=False)

    classes = tuple(CLASS_INDEX[c.strip()] for c in args.classes.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | 클래스: %s | 띠 확률 %.2f–%.2f, 반경 %d",
             device, args.classes, args.prob_lo, args.prob_hi, args.radius)

    from src.data.patient_split import load_split_brats_datasets
    train_ds, val_ds = load_split_brats_datasets(
        train_root=args.train_root,
        modality=args.modality,
        target_size=args.target_size,
        max_patients=args.max_train_patients,
        patient_split=args.patient_split,
        refinement_mode=None,
        simulate_rough=False,
    )
    log.info("원본 슬라이스: train %d, val %d", len(train_ds), len(val_ds))

    from src.models.dynamic_router import AdaptivePipeline
    images, _, _ = train_ds.get_numpy_arrays()
    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch, strict_checkpoints=True)
    pipeline.eval()

    thresholds = [float(t) for t in args.stage2_thresholds.split(",")]
    if len(thresholds) != 3:
        raise SystemExit("--stage2_thresholds 는 소형,중형,대형 3개 값이어야 합니다.")
    log.info("Stage 2 임계값 %s", thresholds)
    log.info("Stage 2 확률맵 계산 (train)")
    train_entries = build_stage2_entries(
        train_ds, pipeline, device, classes, thresholds=thresholds, use_tta=not args.no_tta,
    )
    log.info("Stage 2 확률맵 계산 (val)")
    val_entries = build_stage2_entries(
        val_ds, pipeline, device, classes, thresholds=thresholds, use_tta=not args.no_tta,
    )
    del pipeline
    torch.cuda.empty_cache()
    if not train_entries or not val_entries:
        raise SystemExit("해당 클래스 슬라이스가 없습니다.")

    train_set = BandSliceDataset(train_entries, args.prob_lo, args.prob_hi, args.radius,
                                 augment=not args.no_augment)
    log.info("학습 띠 픽셀 %d, 그중 GT와 다른 픽셀 %d (%.2f%%)",
             train_set.band_pixels, train_set.edit_pixels,
             100.0 * train_set.edit_pixels / max(train_set.band_pixels, 1))
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=len(train_set) > args.batch_size,
    )

    sample_in = train_set[0]["image"].shape[0]
    model = build_band_refine(in_channels=sample_in, width=args.width).to(device)
    if args.pretrained:
        pack = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(pack["model"])
        log.info("지도학습 초기 가중치 로드: %s", args.pretrained)
    log.info("모델 파라미터: %.2fM | 입력 채널 %d",
             sum(p.numel() for p in model.parameters()) / 1e6, sample_in)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    before = evaluate_slices(model, val_entries, device, args.prob_lo, args.prob_hi,
                             args.radius, args.flair_index)
    log.info("학습 전 Stage2 DSC %.4f HD95 %.3f | raw %.4f/%.3f | guard %.4f/%.3f",
             before["stage2_dsc"], before["stage2_hd95"], before["raw_dsc"], before["raw_hd95"],
             before["guard_dsc"], before["guard_hd95"])

    best_hd = before["stage2_hd95"]
    best_dsc = before["guard_dsc"]
    best_state = None
    best_raw_hd = float("inf")
    best_raw_pack = None
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    config = {
        "in_channels": sample_in,
        "width": args.width,
        "classes": args.classes,
        "prob_lo": args.prob_lo,
        "prob_hi": args.prob_hi,
        "radius": args.radius,
        "flair_index": args.flair_index,
        "boundary_weight": args.boundary_weight,
        "ce_weight": args.ce_weight,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_ce = total_dice = total_bd = 0.0
        n_batch = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            stage2 = batch["stage2"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            band = batch["band"].to(device, non_blocking=True)
            phi = batch["phi"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss, ce, dice, boundary = band_loss(
                logits, stage2, gt, band, phi, target, args.boundary_weight, args.ce_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            total_ce += float(ce.detach())
            total_dice += float(dice.detach())
            total_bd += float(boundary.detach())
            n_batch += 1
        scheduler.step()
        n_batch = max(1, n_batch)

        stats = evaluate_slices(model, val_entries, device, args.prob_lo, args.prob_hi,
                                args.radius, args.flair_index)
        log.info(
            "Epoch [%02d/%d] loss=%.4f (ce %.3f, dice %.3f, bd %.3f) | "
            "guard DSC %.4f→%.4f HD95 %.3f→%.3f | raw DSC %.4f HD95 %.3f",
            epoch, args.epochs, total / n_batch, total_ce / n_batch, total_dice / n_batch,
            total_bd / n_batch, before["stage2_dsc"], stats["guard_dsc"],
            before["stage2_hd95"], stats["guard_hd95"], stats["raw_dsc"], stats["raw_hd95"],
        )

        keeps_dsc = stats["guard_dsc"] + args.dsc_slack >= before["stage2_dsc"]
        if keeps_dsc and stats["guard_hd95"] < best_hd:
            best_hd = stats["guard_hd95"]
            best_dsc = stats["guard_dsc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "model": best_state,
                "config": config,
                "val_guard_dsc": best_dsc,
                "val_guard_hd95": best_hd,
                "val_stage2_dsc": before["stage2_dsc"],
                "val_stage2_hd95": before["stage2_hd95"],
                "val_raw_dsc": stats["raw_dsc"],
                "val_raw_hd95": stats["raw_hd95"],
            }, args.save_path)
            log.info("  ✔ 저장 (guard DSC %.4f, HD95 %.3f)", best_dsc, best_hd)
        raw_keeps = stats["raw_dsc"] + args.dsc_slack >= before["stage2_dsc"]
        if raw_keeps and stats["raw_hd95"] < best_raw_hd:
            best_raw_hd = stats["raw_hd95"]
            best_raw_pack = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "val_guard_dsc": stats["guard_dsc"],
                "val_guard_hd95": stats["guard_hd95"],
                "val_raw_dsc": stats["raw_dsc"],
                "val_raw_hd95": stats["raw_hd95"],
            }

    if best_state is None:
        pack = best_raw_pack or {
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "val_guard_dsc": stats["guard_dsc"],
            "val_guard_hd95": stats["guard_hd95"],
            "val_raw_dsc": stats["raw_dsc"],
            "val_raw_hd95": stats["raw_hd95"],
        }
        torch.save({
            "model": pack["model"],
            "config": config,
            "val_guard_dsc": pack["val_guard_dsc"],
            "val_guard_hd95": pack["val_guard_hd95"],
            "val_stage2_dsc": before["stage2_dsc"],
            "val_stage2_hd95": before["stage2_hd95"],
            "val_raw_dsc": pack["val_raw_dsc"],
            "val_raw_hd95": pack["val_raw_hd95"],
            "saved_without_gate": True,
        }, args.save_path)
        log.info("가드 지표가 Stage 2 HD95를 넘지 못해 raw 최적을 저장했다. HD95 %.3f",
                 pack["val_raw_hd95"])
    else:
        log.info("학습 완료. 저장 모델 guard DSC %.4f / HD95 %.3f (Stage2 %.4f / %.3f)",
                 best_dsc, best_hd, before["stage2_dsc"], before["stage2_hd95"])


if __name__ == "__main__":
    main()
