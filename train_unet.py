"""
Step 1: U-Net 학습 스크립트
합성 데이터 또는 실제 BraTS2020 데이터로 초기 마스크 생성기를 학습합니다.

사용 예시:
  # 합성 데이터 (기본값)
  python train_unet.py

  # 실제 BraTS2020 데이터
  python train_unet.py --use_real_data --max_train_patients 50
"""

import os
import sys
import argparse
import logging

import torch
from torch.utils.data import DataLoader, random_split

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import build_unet, DiceLoss, compute_dice

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_unet(
    # 데이터 설정
    use_real_data: bool = True,
    train_root: str = r"src\data\archive\BraTS2021_Training_Data",
    val_root:   str = "",  # BraTS2021은 별도 val 폴더 없음 → train 80/20 분할
    modality:   str = "t1ce",
    target_size: int = 128,
    max_train_patients: int = None,
    max_val_patients:   int = None,
    # 합성 데이터 설정 (use_real_data=False 시)
    num_samples: int = 400,
    # 학습 설정
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 3e-4,
    save_path: str = "checkpoints/unet_best.pt",
    device: str = "auto",
) -> None:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    log.info(f"Device: {device}")

    # ── 데이터 ──────────────────────────────────────────────
    if use_real_data:
        log.info("실제 BraTS2021 데이터 사용")
        from src.data.brats2020_dataset import BraTS2020Dataset
        full_ds = BraTS2020Dataset(
            root_dir=train_root,
            modality=modality,
            target_size=target_size,
            max_patients=max_train_patients,
            simulate_rough=True,
        )
        # BraTS2021은 별도 val 폴더가 없음 → train 데이터 80/20 분할
        if val_root and val_root != train_root and val_root != "":
            try:
                val_ds = BraTS2020Dataset(
                    root_dir=val_root,
                    modality=modality,
                    target_size=target_size,
                    max_patients=max_val_patients,
                    simulate_rough=True,
                )
                if len(val_ds) == 0:
                    raise ValueError("Validation 데이터가 비어 있습니다.")
                train_ds = full_ds
            except Exception as e:
                log.warning(f"Validation 데이터 로드 실패 ({e}). Train 20%를 Val로 분할합니다.")
                n_val = max(1, int(len(full_ds) * 0.2))
                n_train = len(full_ds) - n_val
                train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        else:
            log.info("BraTS2021: 별도 val 폴더 없음 → Train 80% / Val 20% 자동 분할")
            n_val = max(1, int(len(full_ds) * 0.2))
            n_train = len(full_ds) - n_val
            train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    else:
        raise ValueError("합성 데이터 생성기(synthetic_brats.py)가 삭제되어 더 이상 합성 데이터를 사용할 수 없습니다. --use_real_data 옵션을 사용해 주세요.")

    log.info(f"학습 슬라이스: {len(train_ds)}  |  검증 슬라이스: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── 모델 ────────────────────────────────────────────────
    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_dsc = 0.0
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

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
            img = batch["image"].to(device)
            gt  = batch["gt_mask"].to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
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
            val_iter = _tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{epochs}] Val  ", unit="batch", ncols=90, ascii=True, leave=False) if USE_TQDM else val_loader
            for batch in val_iter:
                img = batch["image"].to(device)
                gt  = batch["gt_mask"].to(device)
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

    log.info(f"\n=== U-Net 학습 완료. Best val DSC: {best_val_dsc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Train U-Net (합성 or 실제 BraTS2020)")
    # 데이터 관련
    parser.add_argument("--use_real_data", action="store_true", default=True, help="실제 BraTS NIfTI 데이터 사용 (기본값)")
    parser.add_argument("--train_root", type=str,
                        default=r"src\data\archive\BraTS2021_Training_Data")
    parser.add_argument("--val_root",   type=str, default="",
                        help="BraTS2021은 별도 val 폴더 없음. 비워두면 train 80/20 분할.")
    parser.add_argument("--modality",   type=str, default="t1ce", choices=["t1ce","t1","t2","flair"])
    parser.add_argument("--target_size",type=int, default=128)
    parser.add_argument("--max_train_patients", type=int, default=None, help="학습 환자 수 제한 (None=전체)")
    parser.add_argument("--max_val_patients",   type=int, default=None, help="검증 환자 수 제한 (None=전체)")
    # 합성 데이터
    parser.add_argument("--num_samples", type=int, default=400, help="합성 데이터 샘플 수 (use_real_data=False 시)")
    # 학습 관련
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--save_path",  type=str,   default="checkpoints/unet_best.pt")
    args = parser.parse_args()
    train_unet(**vars(args))
