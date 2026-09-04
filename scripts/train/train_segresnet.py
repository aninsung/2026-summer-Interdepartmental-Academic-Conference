"""
Step 1 (대안): SegResNet 학습 스크립트
경량 2D U-Net 대신 MONAI SegResNet을 사용하여 초기 마스크를 생성합니다.

사용 예시:
  # 실제 BraTS2021 데이터 (기본값)
  python train_segresnet.py

  # 환자 수 제한 + 커스텀 저장 경로
  python train_segresnet.py --max_train_patients 100 --save_path checkpoints/segresnet_best.pt

  # init_filters 조정 (표현력↑)
  python train_segresnet.py --init_filters 32 --epochs 30

비교 실행 (U-Net vs SegResNet):
  python train_unet.py      --save_path checkpoints/unet_best.pt
  python train_segresnet.py --save_path checkpoints/segresnet_best.pt
  # 이후 evaluate.py 에서 --unet_path 를 각각 지정하여 성능 비교
"""

import os
import sys
import argparse
import logging

import torch
from torch.utils.data import DataLoader

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.segresnet import (
    build_segresnet,
    DiceLoss,
    BCEDiceLoss,
    BoundaryLoss,
    MultiChannelBCEDiceLoss,
    compute_dice,
    region_logits_to_wt,
)
from src.utils.dataloader import loader_kwargs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class _SegResCollate:
    """Picklable collate for SegResNet (optional multi-region + augment)."""

    def __init__(self, do_augment: bool = True, multi_region: bool = True):
        self.do_augment = do_augment
        self.multi_region = multi_region

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        gt_masks = torch.stack([b["gt_mask"] for b in batch])
        rough_masks = torch.stack([b["rough_mask"] for b in batch])
        gt_regions = (
            torch.stack([b["gt_regions"] for b in batch]) if self.multi_region else None
        )

        if self.do_augment:
            for i in range(images.size(0)):
                if torch.rand(1).item() > 0.5:
                    images[i] = torch.flip(images[i], dims=[-1])
                    gt_masks[i] = torch.flip(gt_masks[i], dims=[-1])
                    if gt_regions is not None:
                        gt_regions[i] = torch.flip(gt_regions[i], dims=[-1])
                if torch.rand(1).item() > 0.5:
                    images[i] = torch.flip(images[i], dims=[-2])
                    gt_masks[i] = torch.flip(gt_masks[i], dims=[-2])
                    if gt_regions is not None:
                        gt_regions[i] = torch.flip(gt_regions[i], dims=[-2])
                if torch.rand(1).item() > 0.75:
                    k = torch.randint(1, 4, (1,)).item()
                    images[i] = torch.rot90(images[i], k, dims=[-2, -1])
                    gt_masks[i] = torch.rot90(gt_masks[i], k, dims=[-2, -1])
                    if gt_regions is not None:
                        gt_regions[i] = torch.rot90(gt_regions[i], k, dims=[-2, -1])

        out = {"image": images, "gt_mask": gt_masks, "rough_mask": rough_masks}
        if gt_regions is not None:
            out["gt_regions"] = gt_regions
        return out


def train_segresnet(
    # 데이터 설정
    use_real_data: bool = True,
    train_root: str = "src/data/archive",
    val_root: str = "",  # BraTS2021은 별도 val 폴더 없음 → train 80/20 분할
    modality: str = "t1ce+flair",
    target_size: int = 128,
    max_train_patients: int = None,
    max_val_patients: int = None,
    # SegResNet 구조 설정
    init_filters: int = 16,
    dropout_prob: float = 0.2,
    # 학습 설정
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    save_path: str = "checkpoints/segresnet_best.pt",
    device: str = "auto",
    loss_type: str = "bce_dice", # "bce_dice", "dice", "boundary"
    augment: bool = True,
    refinement_mode: str = None, # Added for True Expert filtering
    pretrained_path: str = "",   # Added for Fine-tuning
    patient_split: str = None,
    seed: int = 42,
    deterministic: bool = False,
    multi_region: bool = True,
) -> None:
    from src.utils.seed import set_seed
    set_seed(seed, deterministic)

    if device == "auto":
        from src.utils.device import get_torch_device
        device = get_torch_device()
    else:
        device = torch.device(device)
    log.info(f"Device: {device}")

    # Fine-tuning 설정 조율
    if refinement_mode and pretrained_path:
        if epochs == 20:
            epochs = 8
        if lr == 3e-4:
            lr = 5e-5
        log.info(f"🔧 [Fine-Tuning Mode] 자동 설정 변경: epochs={epochs}, lr={lr:.1e}")

    # ── 데이터 ──────────────────────────────────────────────
    if use_real_data:
        log.info("실제 BraTS2021 데이터 사용 (환자 단위 train/val 분할)")
        from src.data.patient_split import load_split_brats_datasets
        train_ds, val_ds = load_split_brats_datasets(
            train_root=train_root,
            modality=modality,
            target_size=target_size,
            max_patients=max_train_patients,
            patient_split=patient_split,
            refinement_mode=refinement_mode,
            simulate_rough=False,
        )
    else:
        raise ValueError(
            "합성 데이터 생성기가 삭제되어 더 이상 합성 데이터를 사용할 수 없습니다. "
            "--use_real_data 옵션을 사용해 주세요."
        )

    log.info(f"학습 슬라이스: {len(train_ds)}  |  검증 슬라이스: {len(val_ds)}")
    if multi_region:
        log.info("Large 영역 헤드: ED + TC(NCR∪ET) → 추론 WT = ED ∪ TC")

    dl_kw = loader_kwargs()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_SegResCollate(do_augment=augment, multi_region=multi_region),
        **dl_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_SegResCollate(do_augment=False, multi_region=multi_region),
        **dl_kw,
    )
    log.info(f"DataLoader: num_workers={dl_kw['num_workers']}, batch_size={batch_size}")

    # ── 모델 ────────────────────────────────────────────────
    sample_item = train_ds[0]
    sample_img = sample_item["image"]
    in_ch = sample_img.shape[0] if sample_img.ndim == 3 else 1
    out_ch = 2 if multi_region else 1
    model = build_segresnet(
        in_channels=in_ch,
        out_channels=out_ch,
        init_filters=init_filters,
        dropout_prob=dropout_prob,
    ).to(device)

    init_path = pretrained_path if pretrained_path and os.path.exists(pretrained_path) else (
        save_path if os.path.exists(save_path) else ""
    )
    if init_path:
        from src.utils.weight_adapt import load_adapted_state_dict
        log.info(f"기존 가중치에서 초기화: {init_path}")
        load_adapted_state_dict(model, init_path, device, logger=log)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"SegResNet 파라미터 수: {n_params:,}  (init_filters={init_filters})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if multi_region:
        criterion = MultiChannelBCEDiceLoss(bce_weight=0.5).to(device)
        log.info("손실 함수: MultiChannelBCEDiceLoss (ED + TC)")
    elif loss_type == "boundary":
        criterion = BoundaryLoss().to(device)
        log.info("손실 함수: BoundaryLoss")
    elif loss_type == "bce_dice":
        criterion = BCEDiceLoss(bce_weight=0.5).to(device)
        log.info("손실 함수: BCEDiceLoss (BCE 0.5 + Dice 0.5)")
    else:
        criterion = DiceLoss().to(device)
        log.info("손실 함수: DiceLoss")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── AMP (자동 혼합 정밀도) ──────────────────────────────
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log.info(f"AMP(FP16 혼합 정밀도): {'✅ 활성화' if use_amp else '❌ 비활성화 (CPU)'}")

    best_val_dsc = 0.0
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
    )

    try:
        from tqdm import tqdm as _tqdm

        USE_TQDM = True
    except ImportError:
        USE_TQDM = False

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0

        if USE_TQDM:
            pbar = _tqdm(
                train_loader,
                desc=f"Epoch [{epoch:02d}/{epochs}] Train",
                unit="batch",
                ncols=90,
                ascii=True,
                leave=False,
            )
        else:
            pbar = train_loader

        for step, batch in enumerate(pbar, 1):
            img = batch["image"].to(device, non_blocking=True)
            if multi_region:
                gt = batch["gt_regions"].to(device, non_blocking=True)
            else:
                gt = batch["gt_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(img)
                loss = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if USE_TQDM:
                pbar.set_postfix({"loss": f"{train_loss / step:.4f}"})

        if USE_TQDM:
            pbar.close()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_dsc = 0.0
        with torch.no_grad():
            val_iter = (
                _tqdm(
                    val_loader,
                    desc=f"Epoch [{epoch:02d}/{epochs}] Val  ",
                    unit="batch",
                    ncols=90,
                    ascii=True,
                    leave=False,
                )
                if USE_TQDM
                else val_loader
            )
            for batch in val_iter:
                img = batch["image"].to(device, non_blocking=True)
                gt_wt = batch["gt_mask"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = torch.sigmoid(model(img))
                pred_wt = region_logits_to_wt(pred, from_logits=False)
                pred_bin = (pred_wt > 0.5).float()
                val_dsc += compute_dice(pred_bin, gt_wt)
            if USE_TQDM:
                val_iter.close()
        val_dsc /= len(val_loader)

        log.info(f"Epoch [{epoch:02d}/{epochs}] loss={train_loss:.4f}  val_DSC={val_dsc:.4f}")

        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            torch.save(model.state_dict(), save_path)
            log.info(f"  ✔ Best model saved (val_DSC={best_val_dsc:.4f})")

    log.info(f"\n=== SegResNet 학습 완료. Best val DSC: {best_val_dsc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1 (대안): Train SegResNet (MONAI) on BraTS2021"
    )
    # 데이터 관련
    parser.add_argument(
        "--use_real_data",
        action="store_true",
        default=True,
        help="실제 BraTS NIfTI 데이터 사용 (기본값)",
    )
    parser.add_argument(
        "--train_root",
        type=str,
        default="src/data/archive",
    )
    parser.add_argument(
        "--val_root",
        type=str,
        default="",
        help="BraTS2021은 별도 val 폴더 없음. 비워두면 train 80/20 분할.",
    )
    parser.add_argument(
        "--modality", type=str, default="t1ce+flair",
    )
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument(
        "--max_train_patients", type=int, default=None, help="학습 환자 수 제한 (None=전체)"
    )
    parser.add_argument(
        "--max_val_patients", type=int, default=None, help="검증 환자 수 제한 (None=전체)"
    )
    parser.add_argument(
        "--refinement_mode", type=str, default=None, help="True Expert 학습을 위한 타겟 크기 클래스"
    )
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--seed", type=int, default=42, help="전역 시드")
    parser.add_argument("--deterministic", action="store_true", help="cuDNN 결정적 모드 (느려짐)")
    parser.add_argument(
        "--pretrained_path", type=str, default="", help="파인튜닝할 사전 학습 가중치 경로"
    )
    # SegResNet 구조
    parser.add_argument(
        "--init_filters",
        type=int,
        default=16,
        help="SegResNet 첫 블록 채널 수 (기본 16; 32/64로 늘리면 표현력↑)",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.2,
        help="드롭아웃 비율 (0=비활성화)",
    )
    # 학습 관련
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, default="checkpoints/segresnet_best.pt", help="최우수 모델 저장 경로")
    parser.add_argument("--loss", type=str, default="bce_dice", choices=["bce_dice", "dice", "boundary"], help="사용할 손실 함수")
    
    # ── 기타 옵션 ──
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--use_bce_dice",
        action="store_true",
        default=True,
        help="BCE+Dice 복합 손실 함수 사용 (기본값: True)",
    )
    parser.add_argument(
        "--no_bce_dice",
        action="store_true",
        default=False,
        help="DiceLoss만 사용 (BCE+Dice 비활성화)",
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        default=False,
        help="Data Augmentation 비활성화",
    )
    parser.add_argument(
        "--no_multi_region",
        action="store_true",
        help="ED/TC 분리 헤드 비활성화 (단일 WT 채널)",
    )
    args = parser.parse_args()
    # --no_bce_dice 플래그 처리
    if args.no_bce_dice:
        args.use_bce_dice = False
    # --no_augment → augment=False
    args.augment = not args.no_augment
    args.multi_region = not args.no_multi_region
    # argparse 전용 키 제거
    d = vars(args)
    d["loss_type"] = d.pop("loss", "bce_dice")
    d.pop("no_bce_dice", None)
    d.pop("no_augment", None)
    d.pop("use_bce_dice", None)
    d.pop("no_multi_region", None)
    train_segresnet(**d)
