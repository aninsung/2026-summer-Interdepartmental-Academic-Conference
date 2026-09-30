"""지도학습 윤곽 진화 학습 (MARL-MambaContour 타당성 검증).

Stage 2 Expert 마스크의 연결요소마다 닫힌 윤곽을 초기화하고, 각 점을 GT 경계의
대응점으로 이동시키도록 지도학습한다. RL은 쓰지 않는다. 논문 ablation에서 지도학습
베이스라인이 MARL 대비 mDice 4.02%p 낮은 정도였으므로, 이 베이스라인이 Stage 2를
못 넘기면 MARL로 올려도 넘기기 어렵다고 판단할 수 있다.

사용 예:
  BRATS_NUM_WORKERS=16 python scripts/train/train_contour_evolve.py \
      --max_train_patients 400 --classes medium,large --epochs 30
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

from src.data.contour_dataset import (
    ContourComponentDataset,
    build_contour_samples,
    build_stage2_entries,
)
from src.utils.contour import match_gt_component, rasterize
from src.utils.evaluation_records import measured_metrics
from src.utils.metrics import dice

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_INDEX = {"small": 0, "medium": 1, "large": 2}


def smoothness_penalty(points: torch.Tensor) -> torch.Tensor:
    """닫힌 윤곽의 2차 차분. 논문의 협력 정규화항(윤곽 매끄러움)에 대응한다."""
    second = torch.roll(points, 1, dims=1) - 2.0 * points + torch.roll(points, -1, dims=1)
    return second.norm(dim=-1).mean()


def point_to_polyline(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    """닫힌 폴리라인의 선분까지 최단 거리. points (B,M,2), polyline (B,N,2) → (B,M)."""
    start = polyline
    end = torch.roll(polyline, shifts=-1, dims=1)
    segment = end - start
    length2 = segment.pow(2).sum(dim=-1).clamp(min=1e-6)
    rel = points[:, :, None, :] - start[:, None, :, :]
    t = (rel * segment[:, None, :, :]).sum(dim=-1) / length2[:, None, :]
    closest = start[:, None, :, :] + t.clamp(0.0, 1.0).unsqueeze(-1) * segment[:, None, :, :]
    return (points[:, :, None, :] - closest).norm(dim=-1).min(dim=-1).values


def sample_edges(polyline: torch.Tensor, per_edge: int = 4) -> torch.Tensor:
    """선분 위의 점을 뽑아 꼭짓점만 줄어드는 손실을 막는다. (B,N,2) → (B,N*per_edge,2)."""
    start = polyline
    end = torch.roll(polyline, shifts=-1, dims=1)
    t = torch.linspace(0.0, 1.0, per_edge + 1, device=polyline.device)[:-1]
    points = start[:, None, :, :] + t.view(1, -1, 1, 1) * (end - start)[:, None, :, :]
    return points.reshape(polyline.shape[0], -1, 2)


def tail_mean(dist: torch.Tensor, frac: float = 0.05) -> torch.Tensor:
    """HD95에 맞춰 거리의 상위 frac 평균만 쓴다. dist는 (B, M)."""
    k = max(1, int(round(frac * dist.shape[1])))
    return dist.topk(k, dim=1).values.mean()


def contour_loss(stages: list[torch.Tensor], target: torch.Tensor, boundary: torch.Tensor,
                 smooth_weight: float, surface_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """점 대응 + 선분 샘플의 양방향 상위 거리(HD95 proxy) + 매끄러움."""
    reg = torch.zeros((), device=target.device)
    cover = torch.zeros((), device=target.device)
    boundary_samples = sample_edges(boundary)
    for pts in stages:
        reg = reg + F.smooth_l1_loss(pts, target, beta=2.0)
        gt_to_pred = point_to_polyline(boundary_samples, pts)
        pred_to_gt = point_to_polyline(sample_edges(pts), boundary)
        cover = cover + 0.5 * (tail_mean(gt_to_pred) + tail_mean(pred_to_gt))
    reg = reg / len(stages)
    cover = cover / len(stages)
    smooth = smoothness_penalty(stages[-1])
    return reg + surface_weight * cover + smooth_weight * smooth, reg, cover


@torch.no_grad()
def evaluate_component_dsc(model, loader, device, entries, samples) -> dict:
    """컴포넌트 단위로 초기 윤곽 대비 예측 윤곽의 DSC를 비교한다."""
    model.eval()
    init_dsc, pred_dsc, target_dsc, l1 = [], [], [], []
    init_hd, pred_hd = [], []
    offset = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        init = batch["init_points"].to(device, non_blocking=True)
        target = batch["target_points"].to(device, non_blocking=True)
        pred = model(image, init)[-1]
        l1.append(float((pred - target).norm(dim=-1).mean()))

        init_np = init.cpu().numpy()
        pred_np = pred.cpu().numpy()
        tgt_np = target.cpu().numpy()
        for b in range(len(init_np)):
            sample = samples[offset + b]
            gt_comp = entries[sample.slice_index].gt
            shape = gt_comp.shape
            # 컴포넌트 단위 비교이므로 대응 GT 연결요소만 놓고 본다.
            comp_mask = rasterize([init_np[b]], shape)
            gt_local = match_gt_component(comp_mask, gt_comp)
            if gt_local is None:
                continue
            pred_mask = rasterize([pred_np[b]], shape)
            init_m = measured_metrics(comp_mask, gt_local)
            pred_m = measured_metrics(pred_mask, gt_local)
            init_dsc.append(init_m["dsc"])
            pred_dsc.append(pred_m["dsc"])
            target_dsc.append(dice(rasterize([tgt_np[b]], shape), gt_local))
            if init_m["hd95_surface_px"] is not None:
                init_hd.append(init_m["hd95_surface_px"])
            if pred_m["hd95_surface_px"] is not None:
                pred_hd.append(pred_m["hd95_surface_px"])
        offset += len(init_np)

    return {
        "init_dsc": float(np.mean(init_dsc)) if init_dsc else float("nan"),
        "pred_dsc": float(np.mean(pred_dsc)) if pred_dsc else float("nan"),
        "target_dsc": float(np.mean(target_dsc)) if target_dsc else float("nan"),
        "point_l1": float(np.mean(l1)) if l1 else float("nan"),
        "init_hd95": float(np.mean(init_hd)) if init_hd else float("nan"),
        "pred_hd95": float(np.mean(pred_hd)) if pred_hd else float("nan"),
        "n": len(init_dsc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="지도학습 윤곽 진화 학습")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--max_train_patients", type=int, default=400)
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--classes", type=str, default="medium,large")
    parser.add_argument("--n_points", type=int, default=128)
    parser.add_argument("--target_mode", type=str, default="nearest",
                        choices=["nearest", "arclength"])
    parser.add_argument("--n_iters", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=10)
    parser.add_argument("--max_shift", type=float, default=8.0)
    parser.add_argument("--smooth_weight", type=float, default=0.15)
    parser.add_argument("--surface_weight", type=float, default=0.15,
                        help="선분 양방향 상위 5% 거리(HD95 proxy) 가중치")
    parser.add_argument("--dsc_slack", type=float, default=0.005,
                        help="HD95가 줄면 이 범위까지 DSC 하락은 허용하고 저장")
    parser.add_argument("--pretrained", type=str, default="",
                        help="이어서 학습할 체크포인트")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_tta", action="store_true", help="Stage 2 확률맵 TTA 비활성화")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="checkpoints/contour_evolve.pt")
    args = parser.parse_args()

    from src.utils.seed import set_seed
    set_seed(args.seed, deterministic=False)

    classes = tuple(CLASS_INDEX[c.strip()] for c in args.classes.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | 클래스: %s", device, args.classes)

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

    log.info("Stage 2 확률맵 계산 (train)")
    train_entries = build_stage2_entries(train_ds, pipeline, device, classes,
                                        use_tta=not args.no_tta)
    log.info("Stage 2 확률맵 계산 (val)")
    val_entries = build_stage2_entries(val_ds, pipeline, device, classes,
                                       use_tta=not args.no_tta)
    del pipeline
    torch.cuda.empty_cache()

    train_samples = build_contour_samples(train_entries, args.n_points, args.min_area,
                                          args.target_mode)
    val_samples = build_contour_samples(val_entries, args.n_points, args.min_area,
                                        args.target_mode)
    if not train_samples or not val_samples:
        raise SystemExit("윤곽 샘플이 없습니다. 클래스 필터나 Expert 체크포인트를 확인하세요.")

    train_set = ContourComponentDataset(train_entries, train_samples,
                                        augment=not args.no_augment)
    val_set = ContourComponentDataset(val_entries, val_samples, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    from src.models.contour_evolve import build_contour_evolve
    sample_in = train_set[0]["image"].shape[0]
    model = build_contour_evolve(in_channels=sample_in, n_iters=args.n_iters,
                                 max_shift=args.max_shift).to(device)
    if args.pretrained:
        prev = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(prev["model"])
        log.info("초기 가중치: %s", args.pretrained)
    log.info("모델 파라미터: %.2fM | 입력 채널 %d", sum(p.numel() for p in model.parameters()) / 1e6, sample_in)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    baseline = evaluate_component_dsc(model, val_loader, device, val_entries, val_samples)
    log.info("학습 전: DSC %.4f, HD95 %.3f (목표윤곽 DSC 상한 %.4f)",
             baseline["init_dsc"], baseline["init_hd95"], baseline["target_dsc"])

    # 시작 모델보다 HD95가 더 작고, DSC는 초기 윤곽에서 slack 안에 있을 때만 저장.
    best_hd = baseline["pred_hd95"] if args.pretrained else float("inf")
    best_dsc = baseline["pred_dsc"]
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, total_reg, total_cover = 0.0, 0.0, 0.0
        n_batch = max(1, len(train_loader))
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            init = batch["init_points"].to(device, non_blocking=True)
            target = batch["target_points"].to(device, non_blocking=True)
            boundary = batch["gt_boundary"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            stages = model(image, init)
            loss, reg, cover = contour_loss(stages, target, boundary,
                                            args.smooth_weight, args.surface_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            total_reg += float(reg.detach())
            total_cover += float(cover.detach())
        scheduler.step()

        stats = evaluate_component_dsc(model, val_loader, device, val_entries, val_samples)
        log.info("Epoch [%02d/%d] loss=%.4f (dist %.4f, cover %.3f) | DSC %.4f→%.4f "
                 "HD95 %.3f→%.3f",
                 epoch, args.epochs, total / n_batch, total_reg / n_batch, total_cover / n_batch,
                 stats["init_dsc"], stats["pred_dsc"], stats["init_hd95"], stats["pred_hd95"])

        keeps_dsc = stats["pred_dsc"] + args.dsc_slack >= baseline["init_dsc"]
        if keeps_dsc and stats["pred_hd95"] < best_hd:
            best_hd = stats["pred_hd95"]
            best_dsc = stats["pred_dsc"]
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "in_channels": sample_in, "n_iters": args.n_iters,
                    "n_points": args.n_points, "min_area": args.min_area,
                    "max_shift": args.max_shift, "classes": args.classes,
                    "target_mode": args.target_mode,
                    "surface_weight": args.surface_weight,
                },
                "val_component_dsc": best_dsc,
                "val_component_hd95": best_hd,
            }, args.save_path)
            log.info("  ✔ 저장 (DSC %.4f, HD95 %.3f)", best_dsc, best_hd)

    log.info("학습 완료. 저장 모델 DSC %.4f / HD95 %.3f (초기 DSC %.4f, HD95 %.3f)",
             best_dsc, best_hd, baseline["init_dsc"], baseline["init_hd95"])


if __name__ == "__main__":
    main()
