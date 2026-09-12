"""
Step 1: UNet 3+ (UNet3+) 학습 스크립트
개량된 UNet3+ 모델을 기반으로 초기 마스크 생성기를 학습합니다.

사용 예시:
  # 실제 BraTS2021 데이터 (기본값)
  python train_unet3plus.py
"""

import os
import sys
import argparse
import logging

import torch
from torch.utils.data import DataLoader

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.unet3plus import build_unet3plus
from src.models.segresnet import BCEDiceLoss, DiceLoss, compute_dice

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_unet3plus(
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
    save_path: str = "checkpoints/unet3plus_best.pt",
    device: str = "auto",
    use_bce_dice: bool = True,
    augment: bool = True,
    use_dsv: bool = False,
    refinement_mode: str = None,
    pretrained_path: str = None,
) -> None:
    if device == "auto":
        from src.utils.device import get_torch_device
        device = get_torch_device()
    else:
        device = torch.device(device)
    log.info(f"Device: {device}")

    # ── 데이터 ──────────────────────────────────────────────
    if use_real_data:
        log.info("실제 BraTS2021 데이터 사용")
        from src.data.brats2020_dataset import BraTS2020Dataset
        import numpy as np
        full_ds = BraTS2020Dataset(
            root_dir=train_root,
            modality=modality,
            target_size=target_size,
            max_patients=max_train_patients,
            simulate_rough=False,
        )
        
        # 크기별 맞춤형 전문가 학습을 위한 데이터셋 필터링 (True Expert 적용)
        if refinement_mode:
            ref_m = refinement_mode.lower()
            filtered = []
            for sample in full_ds._samples:
                # sample = (img_sl, gt_sl, rough_sl, has_et)
                gt = sample[1]
                area = np.sum(gt)
                if ref_m == "small" and 0 < area < 300:
                    # 64x64 Zoom-in Patch 추출 (Small Expert 전용)
                    img_sl, gt_sl, rough_sl, has_et = sample
                    y_indices, x_indices = np.where(gt_sl > 0)
                    cy = int(np.mean(y_indices)) if len(y_indices) > 0 else gt_sl.shape[0] // 2
                    cx = int(np.mean(x_indices)) if len(x_indices) > 0 else gt_sl.shape[1] // 2
                    
                    # 64x64 Crop
                    patch_size = 64
                    half = patch_size // 2
                    H, W = gt_sl.shape
                    y1 = max(0, min(H - patch_size, cy - half))
                    x1 = max(0, min(W - patch_size, cx - half))
                    y2 = y1 + patch_size
                    x2 = x1 + patch_size
                    
                    crop_img = img_sl[:, y1:y2, x1:x2] if img_sl.ndim == 3 else img_sl[y1:y2, x1:x2]
                    crop_gt = gt_sl[y1:y2, x1:x2]
                    crop_rough = rough_sl[y1:y2, x1:x2]
                    
                    filtered.append((crop_img, crop_gt, crop_rough, has_et))
                elif ref_m == "medium" and 300 <= area < 700:
                    filtered.append(sample)
                elif ref_m == "large" and area >= 700:
                    filtered.append(sample)
            
            old_len = len(full_ds._samples)
            full_ds._samples = filtered
            log.info(f"[{refinement_mode.upper()} Expert] 필터링 완료: {len(full_ds._samples)}개 슬라이스 사용 (기존 {old_len} 중)")

        from src.data.patient_split import prepare_train_val_datasets
        train_ds, val_ds = prepare_train_val_datasets(
            full_ds, train_root, max_train_patients, patient_split=None, refinement_mode=None
        )
    else:
        raise ValueError(
            "합성 데이터 생성기가 삭제되어 더 이상 합성 데이터를 사용할 수 없습니다. "
            "--use_real_data 옵션을 사용해 주세요."
        )

    log.info(f"학습 슬라이스: {len(train_ds)}  |  검증 슬라이스: {len(val_ds)}")

    # ── Data Augmentation ──────────────────────────────────
    def augment_batch(batch):
        images = torch.stack([b["image"] for b in batch])
        gt_masks = torch.stack([b["gt_mask"] for b in batch])
        rough_masks = torch.stack([b["rough_mask"] for b in batch])

        if augment:
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

        return {"image": images, "gt_mask": gt_masks, "rough_mask": rough_masks}

    n_workers = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
        collate_fn=augment_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
        collate_fn=augment_batch,
    )
    log.info(f"DataLoader: num_workers={n_workers}, batch_size={batch_size}, prefetch_factor=4")

    # ── 모델 ────────────────────────────────────────────────
    sample_item = train_ds[0]
    sample_img = sample_item["image"]
    in_ch = sample_img.shape[0] if sample_img.ndim == 3 else 1
    model = build_unet3plus(
        in_channels=in_ch,
        out_channels=1,
        DSV=use_dsv
    ).to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        log.info(f"사전 학습된 가중치 로드 중: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        model_state = model.state_dict()
        for k, v in list(state_dict.items()):
            if k in model_state and model_state[k].shape != v.shape:
                log.warning(f"Shape mismatch for {k}: checkpoint {v.shape} vs model {model_state[k].shape}. Adapting weights...")
                if v.ndim == 4 and v.shape[1] == 1 and model_state[k].shape[1] > 1:
                    state_dict[k] = v.repeat(1, model_state[k].shape[1], 1, 1) / model_state[k].shape[1]
                else:
                    del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_bce_dice:
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
    if use_dsv:
        log.info("Deep Supervision(DSV) 활성화됨: 모든 스케일 예측에 손실 함수 적용")

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
                if use_dsv:
                    preds = model(img) # list of tensors
                    loss = 0.0
                    for p in preds:
                        loss += criterion(p, gt)
                    loss = loss / len(preds)
                else:
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
                    if use_dsv:
                        preds = model(img)
                        # 평가 시에는 가장 고해상도 예측 맵인 dec0의 출력(0번째 원소)을 사용합니다.
                        pred = torch.sigmoid(preds[0])
                    else:
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

    log.info(f"\n=== UNet 3+ 학습 완료. Best val DSC: {best_val_dsc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Train UNet 3+ (UNet3Plus)")
    # 데이터 관련
    parser.add_argument("--use_real_data", action="store_true", default=True, help="실제 데이터 사용")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--val_root", type=str, default="", help="비워두면 train 80/20 분할")
    parser.add_argument("--modality", type=str, default="t1ce+flair", )
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--max_train_patients", type=int, default=None, help="학습 환자 수 제한")
    parser.add_argument("--max_val_patients", type=int, default=None, help="검증 환자 수 제한")
    # 학습 관련
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, default="checkpoints/unet3plus_best.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_bce_dice", action="store_true", default=True, help="BCE+Dice 사용")
    parser.add_argument("--no_bce_dice", action="store_true", default=False, help="DiceLoss만 사용")
    parser.add_argument("--no_augment", action="store_true", default=False, help="Data Augmentation 비활성화")
    parser.add_argument("--use_dsv", action="store_true", default=False, help="Deep Supervision(DSV) 활성화")
    parser.add_argument("--refinement_mode", type=str, default=None, help="크기별 특화 모드")
    parser.add_argument("--pretrained_path", type=str, default=None, help="사전 학습 가중치 파일 경로")
    
    args = parser.parse_args()
    if args.no_bce_dice:
        args.use_bce_dice = False
    args.augment = not args.no_augment
    d = vars(args)
    d.pop("no_bce_dice", None)
    d.pop("no_augment", None)
    
    train_unet3plus(**d)
