"""BraTS 2021 상위 입상 방법(KAIST 1위 / NVAUTO 2위) 베이스라인 학습 스크립트.

본 프로젝트의 적응형 파이프라인과 공정하게 비교하기 위해 데이터, 환자 분할,
모달리티, 해상도, 지표를 모두 동일하게 맞춘다. 즉 checkpoints/patient_split.json
을 그대로 재사용하므로 두 베이스라인과 파이프라인은 같은 168명으로 학습하고
같은 42명으로 검증한다.

사용 예시
    python baselines/train_baseline.py --method kaist
    python baselines/train_baseline.py --method nvauto --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.common import check_split_compatibility
from baselines.losses import (
    BarlowTwinsLoss,
    BceBatchDiceLoss,
    DeepSupervisionLoss,
    soft_dice_loss,
)
from baselines.models import build_kaist_nnunet, build_nvauto_segresnet
from src.utils.metrics import dice as np_dice
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

METHODS = ("kaist", "nvauto")


# ────────────────────────────── 데이터 ──────────────────────────────


def make_collate(augment: bool):
    """기존 Expert 학습 스크립트와 같은 방식의 증강을 쓴다(면적 보존 변환만)."""

    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        gts = torch.stack([b["gt_mask"] for b in batch])

        if augment:
            for i in range(images.size(0)):
                if torch.rand(1).item() > 0.5:
                    images[i] = torch.flip(images[i], dims=[-1])
                    gts[i] = torch.flip(gts[i], dims=[-1])
                if torch.rand(1).item() > 0.5:
                    images[i] = torch.flip(images[i], dims=[-2])
                    gts[i] = torch.flip(gts[i], dims=[-2])
                if torch.rand(1).item() > 0.75:
                    k = torch.randint(1, 4, (1,)).item()
                    images[i] = torch.rot90(images[i], k, dims=[-2, -1])
                    gts[i] = torch.rot90(gts[i], k, dims=[-2, -1])

        return {"image": images, "gt_mask": gts}

    return collate


def perturb_view(
    img: torch.Tensor, gt: torch.Tensor, max_shift: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVAUTO 의 뷰별 교란: 강도 변환 + 작은 평행이동.

    논문은 공유 미러 플립 이후 사본마다 강도/공간 변환을 독립적으로 적용한다.
    평행이동은 프로젝션 전 16배 average pooling 이 흡수할 수 있는 범위로 제한한다.
    """
    # 강도 변환: 밝기, 대비, 가우시안 노이즈
    b = img.shape[0]
    scale = 1.0 + 0.1 * (torch.rand(b, 1, 1, 1, device=img.device) * 2 - 1)
    shift = 0.1 * (torch.rand(b, 1, 1, 1, device=img.device) * 2 - 1)
    out = img * scale + shift
    out = out + 0.01 * torch.randn_like(out)

    # 공간 변환: 정수 픽셀 평행이동(마스크에도 같이 적용해 정답 정합을 유지)
    dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    if dy or dx:
        out = torch.roll(out, shifts=(dy, dx), dims=(-2, -1))
        gt = torch.roll(gt, shifts=(dy, dx), dims=(-2, -1))
    return out, gt


# ────────────────────────────── 학습 ──────────────────────────────


def build_model(method: str, in_channels: int, input_size: int, device: torch.device):
    if method == "kaist":
        model = build_kaist_nnunet(
            in_channels=in_channels, out_channels=1, input_size=input_size
        )
    elif method == "nvauto":
        model = build_nvauto_segresnet(in_channels=in_channels, out_channels=1)
    else:
        raise ValueError(f"알 수 없는 method: {method} (가능한 값: {METHODS})")
    return model.to(device)


def build_optimizer(method: str, model: torch.nn.Module, lr: float, epochs: int):
    """각 논문이 명시한 최적화 설정을 쓴다."""
    if method == "kaist":
        # nnU-Net: SGD + nesterov(momentum 0.99) + poly 스케줄
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.99, nesterov=True, weight_decay=3e-5
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda e: (1.0 - e / epochs) ** 0.9
        )
    else:
        # NVAUTO: AdamW, lr 1e-4, weight decay 1e-5, cosine 스케줄
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return opt, sched


def train(
    method: str = "kaist",
    train_root: str = "src/data/archive",
    modality: str = "t1ce+flair",
    target_size: int = 128,
    max_train_patients: int | None = 210,
    patient_split: str = "checkpoints/patient_split.json",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float | None = None,
    bt_weight: float = 0.01,
    save_path: str | None = None,
    augment: bool = True,
    seed: int = 42,
    deterministic: bool = True,
) -> Dict[str, float]:
    if method not in METHODS:
        raise ValueError(f"알 수 없는 method: {method} (가능한 값: {METHODS})")

    set_seed(seed, deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if lr is None:
        lr = 1e-2 if method == "kaist" else 1e-4
    if save_path is None:
        save_path = f"baselines/checkpoints/{method}_best.pt"

    log.info(f"=== 베이스라인 학습: {method} ===")
    log.info(f"Device: {device} | seed={seed} deterministic={deterministic}")

    # 파이프라인과 동일한 환자 분할을 재사용한다.
    # 설정이 어긋나면 분할 파일이 덮어써지므로 그 전에 멈춘다.
    check_split_compatibility(train_root, max_train_patients, patient_split, seed=42)

    from src.data.patient_split import load_split_brats_datasets

    train_ds, val_ds = load_split_brats_datasets(
        train_root=train_root,
        modality=modality,
        target_size=target_size,
        max_patients=max_train_patients,
        patient_split=patient_split,
        refinement_mode=None,  # 크기 구간을 나누지 않는다: 단일 모델이 전 범위를 담당
        simulate_rough=False,
    )
    log.info(f"학습 슬라이스 {len(train_ds)} | 검증 슬라이스 {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True, collate_fn=make_collate(augment),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=True, collate_fn=make_collate(False),
    )

    in_channels = train_ds[0]["image"].shape[0]
    model = build_model(method, in_channels, target_size, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"파라미터 수: {n_params:,}")

    optimizer, scheduler = build_optimizer(method, model, lr, epochs)
    seg_criterion = DeepSupervisionLoss(BceBatchDiceLoss(bce_weight=0.5, batch_dice=True))
    bt_criterion = BarlowTwinsLoss(lambda_coeff=0.005)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    best_val_dsc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running, running_bt = 0.0, 0.0

        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            gt = batch["gt_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if method == "nvauto":
                    # 같은 슬라이스를 독립적으로 두 번 교란해 두 뷰를 만든다.
                    img1, gt1 = perturb_view(img, gt)
                    img2, gt2 = perturb_view(img, gt)
                    logits1, z1 = model(img1, return_embedding=True)
                    logits2, z2 = model(img2, return_embedding=True)
                    seg = 0.5 * (
                        soft_dice_loss(logits1, gt1, batch_dice=True)
                        + soft_dice_loss(logits2, gt2, batch_dice=True)
                    )
                    bt = bt_criterion(z1.float(), z2.float())
                    loss = seg + bt_weight * bt
                    running_bt += float(bt.detach())
                else:
                    outputs = model(img)
                    loss = seg_criterion(outputs, gt)
                    seg = loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(seg.detach())

        epoch_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_loss = running / max(1, len(train_loader))

        # ── 검증: 파이프라인과 같은 임계값 0.5 이진화 후 DSC ──
        model.eval()
        dsc_sum, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device, non_blocking=True)
                gt = batch["gt_mask"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(img)
                prob = torch.sigmoid(logits.float())
                pred = (prob > 0.5).float().cpu().numpy()
                gt_np = gt.cpu().numpy()
                for i in range(pred.shape[0]):
                    dsc_sum += np_dice(pred[i, 0], gt_np[i, 0])
                    n += 1
        val_dsc = dsc_sum / max(1, n)

        extra = f" bt={running_bt / max(1, len(train_loader)):.2f}" if method == "nvauto" else ""
        log.info(
            f"Epoch [{epoch:02d}/{epochs}] loss={train_loss:.4f}{extra} "
            f"lr={epoch_lr:.2e} val_DSC={val_dsc:.4f}"
        )

        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            torch.save(
                {
                    "method": method,
                    "state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "target_size": target_size,
                    "val_dsc": best_val_dsc,
                },
                save_path,
            )
            log.info(f"  ✔ 저장 (val_DSC={best_val_dsc:.4f}) → {save_path}")

    log.info(f"=== {method} 학습 완료. Best val DSC: {best_val_dsc:.4f} ===")
    return {"method": method, "best_val_dsc": best_val_dsc, "n_params": n_params}


def main():
    p = argparse.ArgumentParser(description="BraTS 2021 상위 입상 방법 베이스라인 학습")
    p.add_argument("--method", type=str, default="kaist", choices=METHODS,
                   help="kaist: 1위(Extending nnU-Net) / nvauto: 2위(SegResNet + 중복 감소)")
    p.add_argument("--train_root", type=str, default="src/data/archive")
    p.add_argument("--modality", type=str, default="t1ce+flair")
    p.add_argument("--target_size", type=int, default=128)
    p.add_argument("--max_train_patients", type=int, default=210)
    p.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="기본값: kaist 1e-2(SGD), nvauto 1e-4(AdamW)")
    p.add_argument("--bt_weight", type=float, default=0.01,
                   help="Barlow Twins 항 가중치. 원 논문이 Dice 와의 균형비를 명시하지 않아 조정 대상이다.")
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_deterministic", action="store_true")
    args = p.parse_args()

    train(
        method=args.method,
        train_root=args.train_root,
        modality=args.modality,
        target_size=args.target_size,
        max_train_patients=args.max_train_patients,
        patient_split=args.patient_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        bt_weight=args.bt_weight,
        save_path=args.save_path,
        augment=not args.no_augment,
        seed=args.seed,
        deterministic=not args.no_deterministic,
    )


if __name__ == "__main__":
    main()
