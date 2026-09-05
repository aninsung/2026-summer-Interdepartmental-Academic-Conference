"""
Step 1: CaraNet 학습 스크립트
CaraNet을 기반으로 초기 마스크 생성기를 학습합니다.

사용 예시:
  # 실제 BraTS2021 데이터 (기본값)
  python train_caranet.py

  # 환자 수 제한 + 커스텀 저장 경로
  python train_caranet.py --max_train_patients 100 --save_path checkpoints/caranet_best.pt
"""

import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.caranet import build_caranet
from src.models.segresnet import BCEDiceLoss, DiceLoss, compute_dice
from src.utils.dataloader import loader_kwargs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class _CaraNetCollate:
    """Picklable collate for Windows DataLoader workers."""

    def __init__(self, do_augment: bool, do_zoom: bool, zoom_prob: float = 0.5, zoom_patch: int = 64):
        self.do_augment = do_augment
        self.do_zoom = do_zoom
        self.zoom_prob = zoom_prob
        self.zoom_patch = zoom_patch

    def __call__(self, batch):
        def _img(item):
            t = item.get("image_25d")
            return t if t is not None else item["image"]

        images = torch.stack([_img(b) for b in batch])
        gt_masks = torch.stack([b["gt_mask"] for b in batch])
        rough_masks = torch.stack([b["rough_mask"] for b in batch])

        if self.do_augment:
            for i in range(images.size(0)):
                if torch.rand(1).item() > 0.5:
                    images[i] = torch.flip(images[i], dims=[-1])
                    gt_masks[i] = torch.flip(gt_masks[i], dims=[-1])
                if torch.rand(1).item() > 0.5:
                    images[i] = torch.flip(images[i], dims=[-2])
                    gt_masks[i] = torch.flip(gt_masks[i], dims=[-2])
                if torch.rand(1).item() > 0.75:
                    k = torch.randint(1, 4, (1,)).item()
                    images[i] = torch.rot90(images[i], k, dims=[-2, -1])
                    gt_masks[i] = torch.rot90(gt_masks[i], k, dims=[-2, -1])

        if self.do_zoom:
            from src.utils.zoom_crop import zoom_tensors
            out_size = int(images.shape[-1])
            for i in range(images.size(0)):
                if torch.rand(1).item() < self.zoom_prob:
                    images[i], gt_masks[i] = zoom_tensors(
                        images[i], gt_masks[i], patch_size=self.zoom_patch, out_size=out_size
                    )

        return {"image": images, "gt_mask": gt_masks, "rough_mask": rough_masks}

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        tp = (inputs * targets).sum()
        fp = (inputs * (1.0 - targets)).sum()
        fn = ((1.0 - inputs) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky) ** self.gamma


def train_caranet(
    # 데이터 설정
    use_real_data: bool = True,
    train_root: str = "src/data/archive",
    val_root: str = "",  # BraTS2021은 별도 val 폴더 없음 → train 80/20 분할
    modality: str = "t1ce+flair",
    target_size: int = 128,
    max_train_patients: int = None,
    max_val_patients: int = None,
    # 학습 설정
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    save_path: str = "checkpoints/caranet_best.pt",
    device: str = "auto",
    use_bce_dice: bool = True,
    augment: bool = True,
    refinement_mode: str = None, # Added for True Expert filtering
    pretrained_path: str = "",   # Added for Fine-tuning
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    patient_split: str = None,
    seed: int = 42,
    deterministic: bool = False,
    zoom_crop: bool = True,
    zoom_patch: int = 64,
    zoom_prob: float = 0.5,
    focal_tversky: bool = False,
    fragment_repeats: int = 4,
    fragment_area: float = 50.0,
    train_ds=None,
    val_ds=None,
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
    if train_ds is not None and val_ds is not None:
        log.info("사전 로드된 train/val 데이터셋 사용 (재로드 생략)")
    elif use_real_data:
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

    if refinement_mode == "small" and fragment_repeats > 1:
        from src.data.fragment_oversample import FragmentOversample
        n0 = len(train_ds)
        train_ds = FragmentOversample(train_ds, min_area=fragment_area, repeats=fragment_repeats)
        log.info(
            f"파편 오버샘플: {n0} → {len(train_ds)} "
            f"(GT < {fragment_area:.0f}px ×{fragment_repeats})"
        )

    use_zoom = bool(zoom_crop and (refinement_mode == "small"))
    if use_zoom:
        log.info(f"Small zoom-crop: patch={zoom_patch}, train_prob={zoom_prob}")

    dl_kw = loader_kwargs()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_CaraNetCollate(do_augment=augment, do_zoom=use_zoom,
                                   zoom_prob=zoom_prob, zoom_patch=zoom_patch),
        **dl_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_CaraNetCollate(do_augment=False, do_zoom=False),
        **dl_kw,
    )
    log.info(f"DataLoader: num_workers={dl_kw['num_workers']}, batch_size={batch_size}")

    # ── 모델 ────────────────────────────────────────────────
    sample_item = train_ds[0]
    sample_img = sample_item["image_25d"] if "image_25d" in sample_item else sample_item["image"]
    in_ch = sample_img.shape[0] if sample_img.ndim == 3 else 1
    log.info(f"CaraNet in_channels={in_ch} (2.5D stacked)" if in_ch > 2 else f"CaraNet in_channels={in_ch}")
    model = build_caranet(
        in_channels=in_ch,
        out_channels=1,
    ).to(device)

    init_path = pretrained_path if pretrained_path and os.path.exists(pretrained_path) else (
        save_path if os.path.exists(save_path) else ""
    )
    if init_path:
        from src.utils.weight_adapt import load_adapted_state_dict
        log.info(f"기존 가중치에서 초기화: {init_path}")
        load_adapted_state_dict(model, init_path, device, logger=log)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if focal_tversky:
        criterion = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=2.0)
        log.info(f"손실 함수: FocalTverskyLoss (alpha={tversky_alpha}, beta={tversky_beta}, gamma=2.0)")
    elif use_bce_dice:
        criterion = BCEDiceLoss(bce_weight=0.5)
        log.info("손실 함수: BCEDiceLoss (BCE 0.5 + Dice 0.5)")
    else:
        criterion = DiceLoss()
        log.info("손실 함수: DiceLoss")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── AMP (자동 혼합 정밀도) ──────────────────────────────
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log.info(f"AMP(FP16 혼합 정밀도): {'✅ 활성화' if use_amp else '❌ 비활성화 (CPU)'}")

    best_val_dsc = 0.0
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    try:
        from tqdm import tqdm as _tqdm
        from src.utils.progress import want_tqdm
        USE_TQDM = want_tqdm()
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
            gt  = batch["gt_mask"].to(device, non_blocking=True)
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
                gt  = batch["gt_mask"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = torch.sigmoid(model(img))
                pred_bin = (pred > 0.5).float()
                val_dsc += compute_dice(pred_bin, gt)
            if USE_TQDM:
                val_iter.close()
        val_dsc /= len(val_loader)

        log.info(f"Epoch [{epoch:02d}/{epochs}] loss={train_loss:.4f}  val_DSC={val_dsc:.4f}")

        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            torch.save(model.state_dict(), save_path)
            log.info(f"  ✔ Best model saved (val_DSC={best_val_dsc:.4f})")

    log.info(f"\n=== CaraNet 학습 완료. Best val DSC: {best_val_dsc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Train CaraNet (MONAI style Custom)")
    # 데이터 관련
    parser.add_argument("--use_real_data", action="store_true", default=True, help="실제 데이터 사용")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--val_root", type=str, default="", help="비워두면 train 80/20 분할")
    parser.add_argument("--modality", type=str, default="t1ce+flair", )
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--max_train_patients", type=int, default=None, help="학습 환자 수 제한")
    parser.add_argument("--max_val_patients", type=int, default=None, help="검증 환자 수 제한")
    parser.add_argument("--refinement_mode", type=str, default=None, help="True Expert 학습을 위한 타겟 크기 클래스")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--seed", type=int, default=42, help="전역 시드")
    parser.add_argument("--deterministic", action="store_true", help="cuDNN 결정적 모드 (느려짐)")
    parser.add_argument("--pretrained_path", type=str, default="", help="파인튜닝할 사전 학습 가중치 경로")
    parser.add_argument("--tversky_alpha", type=float, default=0.3, help="FocalTverskyLoss alpha")
    parser.add_argument("--tversky_beta", type=float, default=0.7, help="FocalTverskyLoss beta")
    parser.add_argument("--focal_tversky", action="store_true", help="Small에도 FocalTversky 사용 (기본은 BCEDice)")
    parser.add_argument("--zoom_crop", action="store_true", default=True, help="Small 학습 시 GT 중심 zoom-crop 혼합")
    parser.add_argument("--no_zoom_crop", action="store_true", help="zoom-crop 비활성화")
    parser.add_argument("--zoom_patch", type=int, default=64, help="zoom-crop 패치 한 변")
    parser.add_argument("--zoom_prob", type=float, default=0.5, help="학습 배치에서 zoom-crop 적용 확률")
    parser.add_argument("--fragment_repeats", type=int, default=4, help="Small GT <fragment_area 슬라이스 반복 횟수")
    parser.add_argument("--fragment_area", type=float, default=50.0, help="파편으로 볼 GT 면적 상한")
    # 학습 관련
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, default="checkpoints/caranet_best.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_bce_dice", action="store_true", default=True, help="BCE+Dice 사용")
    parser.add_argument("--no_bce_dice", action="store_true", default=False, help="DiceLoss만 사용")
    parser.add_argument("--no_augment", action="store_true", default=False, help="Data Augmentation 비활성화")
    
    args = parser.parse_args()
    if args.no_bce_dice:
        args.use_bce_dice = False
    args.augment = not args.no_augment
    if args.no_zoom_crop:
        args.zoom_crop = False
    d = vars(args)
    d.pop("no_bce_dice", None)
    d.pop("no_augment", None)
    d.pop("no_zoom_crop", None)
    
    train_caranet(**d)
