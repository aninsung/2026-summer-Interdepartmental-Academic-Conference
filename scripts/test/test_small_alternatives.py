"""
test_small_alternatives.py
--------------------------------------------------------------------------
Small Expert (소형 뇌종양 마스크) 대안 모델 및 후처리 비교 실험 스크립트.

실험 배경:
  3-Stage Adaptive Pipeline에서 Small 종양(<300px) 영역은 분할 난이도가 매우 높음.
  본 스크립트는 Small 종양 분할 성능 극대화를 위해 3가지 대안을 비교 평가합니다:
    - Alternative A: UNet 3+ (Zoom-in Patches 기반 학습 및 평가)
    - Alternative B: Attention U-Net + Recall Optimization (Tversky Loss, beta=0.8)
    - Alternative C: Connected Component Filtering (최소 픽셀 크기 10, 15, 20px 후처리)

사용 예시:
  # 전체 파이프라인 대안 비교 실행
  python test_small_alternatives.py

  # 환자 수 제한 및 빠른 검증 실행
  python test_small_alternatives.py --max_patients 50 --epochs 4 --quick_run
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.ndimage import label, center_of_mass

# 프로젝트 루트를 Python 패스에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.attention_unet import build_attention_unet
from src.models.unet3plus import build_unet3plus
from src.models.segresnet import BCEDiceLoss, compute_dice
from src.data.brats2020_dataset import BraTS2020Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Loss Functions & Utility Functions
# ──────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky Loss (Recall Optimization Mode)
    alpha=0.2, beta=0.8 설정으로 False Negative(미검출)에 높은 패널티 부여하여 Recall 향상.
    """
    def __init__(self, alpha: float = 0.2, beta: float = 0.8, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


def calc_dsc(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-5) -> float:
    """Numpy 기반 Dice Similarity Coefficient 계산."""
    p = (pred_mask > 0.5).astype(float)
    g = (gt_mask > 0.5).astype(float)
    intersection = (p * g).sum()
    total = p.sum() + g.sum()
    return float((2.0 * intersection + smooth) / (total + smooth))


def apply_connected_component_filter(mask: np.ndarray, min_size: int = 10) -> np.ndarray:
    """
    Connected Component Filtering (Alternative C)
    예측 마스크에서 min_size 픽셀 미만의 고립 노이즈 컴포넌트 제거.
    """
    binary_mask = (mask > 0.5).astype(np.uint8)
    labeled, num_features = label(binary_mask)
    if num_features == 0:
        return binary_mask.astype(np.float32)

    filtered = np.zeros_like(binary_mask, dtype=np.float32)
    for i in range(1, num_features + 1):
        component = (labeled == i)
        if component.sum() >= min_size:
            filtered[component] = 1.0
    return filtered


# ──────────────────────────────────────────────
# 2. Datasets & Zoom-in Helper
# ──────────────────────────────────────────────

class SimpleSliceDataset(Dataset):
    """(image, gt_mask, rough_mask) 슬라이스 튜플 커스텀 데이터셋."""
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, gt, rough = self.samples[idx]
        return (
            torch.from_numpy(img).unsqueeze(0).float(),
            torch.from_numpy(gt).unsqueeze(0).float(),
            torch.from_numpy(rough).unsqueeze(0).float(),
        )


def crop_zoom_patch(img: np.ndarray, gt: np.ndarray, rough: np.ndarray, patch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GT 종양 중심점을 기준으로 patch_size x patch_size 영역 크롭 (Alternative A Zoom-in).
    """
    H, W = gt.shape
    if gt.sum() > 0:
        cy, cx = center_of_mass(gt > 0)
        cy, cx = int(cy), int(cx)
    else:
        cy, cx = H // 2, W // 2

    half = patch_size // 2
    y1 = max(0, min(H - patch_size, cy - half))
    x1 = max(0, min(W - patch_size, cx - half))
    y2 = y1 + patch_size
    x2 = x1 + patch_size

    crop_img = img[:, y1:y2, x1:x2] if img.ndim == 3 else img[y1:y2, x1:x2]
    crop_gt = gt[y1:y2, x1:x2]
    crop_rough = rough[y1:y2, x1:x2]
    return crop_img, crop_gt, crop_rough


class SyntheticSmallDataset(Dataset):
    """합성 데이터셋 (BraTS 데이터셋 미존재시 폴백)."""
    def __init__(self, num_samples: int = 200, img_size: int = 128, is_zoom: bool = False):
        self.num_samples = num_samples
        self.img_size = 64 if is_zoom else img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        np.random.seed(idx)
        img = np.random.randn(1, self.img_size, self.img_size).astype(np.float32)
        gt = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)

        # 소형 종양 패치 (10 ~ 250 px)
        cy, cx = np.random.randint(16, self.img_size - 16, size=2)
        r = np.random.randint(3, 8)
        y, x = np.ogrid[:self.img_size, :self.img_size]
        mask = (y - cy)**2 + (x - cx)**2 <= r**2
        gt[0, mask] = 1.0

        # Rough mask: noisy version
        rough = gt.copy()
        noise = np.random.rand(*gt.shape) < 0.05
        rough = np.logical_xor(rough, noise).astype(np.float32)

        return torch.from_numpy(img), torch.from_numpy(gt), torch.from_numpy(rough)


# ──────────────────────────────────────────────
# 3. Alternative Runners
# ──────────────────────────────────────────────

def run_alternative_b(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 3e-4,
    tversky_beta: float = 0.8,
) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
    """
    Alternative B: Attention U-Net + Recall Optimization (Tversky Loss with beta=0.8)
    """
    log.info(f"--- Training Alternative B: Attention U-Net + Recall Optimization (beta={tversky_beta}) ---")
    model = build_attention_unet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = TverskyLoss(alpha=1.0 - tversky_beta, beta=tversky_beta)

    best_val_dsc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, gts, _ in train_loader:
            images, gts = images.to(device), gts.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, gts)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_dscs = []
        with torch.no_grad():
            for images, gts, _ in val_loader:
                images, gts = images.to(device), gts.to(device)
                logits = model(images)
                preds = (torch.sigmoid(logits) > 0.5).float()
                for p, g in zip(preds, gts):
                    val_dscs.append(calc_dsc(p.cpu().numpy(), g.cpu().numpy()))

        epoch_val_dsc = float(np.mean(val_dscs)) if val_dscs else 0.0
        best_val_dsc = max(best_val_dsc, epoch_val_dsc)
        log.info(f"  Epoch [{epoch}/{epochs}] Loss: {train_loss:.4f} | Val DSC: {epoch_val_dsc:.4f}")

    # 최종 검증 예측 및 GT 수집
    val_preds, val_gts = [], []
    model.eval()
    with torch.no_grad():
        for images, gts, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            for p, g in zip(probs, gts.numpy()):
                val_preds.append(p[0])
                val_gts.append(g[0])

    return best_val_dsc, val_preds, val_gts


def run_alternative_a(
    train_zoom_loader: DataLoader,
    val_zoom_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 3e-4,
) -> float:
    """
    Alternative A: UNet 3+ on Zoom-in Patches
    """
    log.info("--- Training Alternative A: UNet 3+ on Zoom-in Patches ---")
    model = build_unet3plus(in_channels=1, out_channels=1, channels=(32, 64, 128, 256)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss()

    best_val_dsc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, gts, _ in train_zoom_loader:
            images, gts = images.to(device), gts.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, gts)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_zoom_loader.dataset)

        # Validation
        model.eval()
        val_dscs = []
        with torch.no_grad():
            for images, gts, _ in val_zoom_loader:
                images, gts = images.to(device), gts.to(device)
                logits = model(images)
                preds = (torch.sigmoid(logits) > 0.5).float()
                for p, g in zip(preds, gts):
                    val_dscs.append(calc_dsc(p.cpu().numpy(), g.cpu().numpy()))

        epoch_val_dsc = float(np.mean(val_dscs)) if val_dscs else 0.0
        best_val_dsc = max(best_val_dsc, epoch_val_dsc)
        log.info(f"  Epoch [{epoch}/{epochs}] Loss: {train_loss:.4f} | Val DSC: {epoch_val_dsc:.4f}")

    return best_val_dsc


def run_alternative_c(
    val_preds: List[np.ndarray],
    val_gts: List[np.ndarray],
    min_sizes: List[int] = [10, 15, 20],
) -> dict:
    """
    Alternative C: Connected Component Filtering
    """
    log.info("--- Evaluating Alternative C: Connected Component Filtering ---")
    results = {}
    for min_size in min_sizes:
        dscs = []
        for p, g in zip(val_preds, val_gts):
            filtered_p = apply_connected_component_filter(p, min_size=min_size)
            dscs.append(calc_dsc(filtered_p, g))
        mean_dsc = float(np.mean(dscs)) if dscs else 0.0
        results[min_size] = mean_dsc
        log.info(f"  CC Filter (min={min_size}px) Val DSC: {mean_dsc:.4f}")
    return results


# ──────────────────────────────────────────────
# 4. Main Evaluation Workflow
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Small Expert Alternatives Comparison")
    parser.add_argument("--train_root", type=str, default="src/data/archive", help="Path to BraTS data root")
    parser.add_argument("--modality", type=str, default="t1ce", help="MRI modality")
    parser.add_argument("--max_patients", type=int, default=None, help="Max patients to load")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs per alternative model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--patch_size", type=int, default=64, help="Zoom-in patch size for Alt A")
    parser.add_argument("--quick_run", action="store_true", help="Quick run for rapid validation")
    args = parser.parse_args()

    if args.quick_run and args.max_patients is None:
        args.max_patients = 20

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    log.info(f"Running Small Expert alternatives testing on {device_str}...")
    log.info("Loading full dataset...")

    # Load dataset or synthetic fallback
    small_samples = []
    zoom_samples = []

    use_synthetic = False
    try:
        full_ds = BraTS2020Dataset(
            root_dir=args.train_root,
            modality=args.modality,
            target_size=128,
            max_patients=args.max_patients,
            simulate_rough=True,
        )
        if len(full_ds) == 0:
            raise ValueError("BraTS2020Dataset length is 0.")

        log.info("Filtering and cropping small class slices...")
        for sample in full_ds._samples:
            img, gt, rough, _ = sample
            tumor_area = (gt > 0).sum()
            if 0 < tumor_area < 300:
                small_samples.append((img, gt, rough))
                c_img, c_gt, c_rough = crop_zoom_patch(img, gt, rough, patch_size=args.patch_size)
                zoom_samples.append((c_img, c_gt, c_rough))

        log.info(f"Total Small slices: {len(small_samples)}")
        if len(small_samples) == 0:
            log.warning("No small slices found in BraTS dataset. Falling back to synthetic dataset.")
            use_synthetic = True
    except Exception as e:
        log.warning(f"BraTS Dataset load exception ({e}). Falling back to synthetic dataset.")
        use_synthetic = True

    if use_synthetic:
        num_syn = 100 if args.quick_run else 300
        syn_train = SyntheticSmallDataset(num_samples=num_syn, img_size=128, is_zoom=False)
        syn_val = SyntheticSmallDataset(num_samples=num_syn // 2, img_size=128, is_zoom=False)
        syn_zoom_train = SyntheticSmallDataset(num_samples=num_syn, img_size=args.patch_size, is_zoom=True)
        syn_zoom_val = SyntheticSmallDataset(num_samples=num_syn // 2, img_size=args.patch_size, is_zoom=True)

        train_loader = DataLoader(syn_train, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(syn_val, batch_size=args.batch_size, shuffle=False)
        train_zoom_loader = DataLoader(syn_zoom_train, batch_size=args.batch_size, shuffle=True)
        val_zoom_loader = DataLoader(syn_zoom_val, batch_size=args.batch_size, shuffle=False)
    else:
        if args.quick_run and len(small_samples) > 200:
            small_samples = small_samples[:200]
            zoom_samples = zoom_samples[:200]

        ds_full = SimpleSliceDataset(small_samples)
        ds_zoom = SimpleSliceDataset(zoom_samples)

        val_size = max(1, int(len(ds_full) * 0.2))
        train_size = len(ds_full) - val_size

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(ds_full, [train_size, val_size], generator=generator)
        train_zoom_ds, val_zoom_ds = random_split(ds_zoom, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        train_zoom_loader = DataLoader(train_zoom_ds, batch_size=args.batch_size, shuffle=True)
        val_zoom_loader = DataLoader(val_zoom_ds, batch_size=args.batch_size, shuffle=False)

    epochs = 2 if args.quick_run else args.epochs

    # Run Alternative B (Attention U-Net + Recall Optimization)
    alt_b_dsc, val_preds, val_gts = run_alternative_b(
        train_loader, val_loader, device, epochs=epochs, lr=args.lr, tversky_beta=0.8
    )

    # Run Alternative A (UNet 3+ on Zoom-in Patches)
    alt_a_dsc = run_alternative_a(
        train_zoom_loader, val_zoom_loader, device, epochs=epochs, lr=args.lr
    )

    # Run Alternative C (Connected Component Filtering on Alt B predictions)
    alt_c_results = run_alternative_c(val_preds, val_gts, min_sizes=[10, 15, 20])

    # Print Formatted Summary
    print("\n=======================================================")
    print("           Small Expert Alternatives Comparison")
    print("=======================================================")
    print(f" - Alternative B: Attention U-Net (Tversky beta=0.8) : {alt_b_dsc:.4f} ({alt_b_dsc*100:.2f}%)")
    print(f" - Alternative A: UNet 3+ (Zoom-in)                  : {alt_a_dsc:.4f} ({alt_a_dsc*100:.2f}%)")
    for min_size in [10, 15, 20]:
        cc_dsc = alt_c_results.get(min_size, 0.0)
        print(f" - Alternative C: Connected Component Filter (min={min_size}px): {cc_dsc:.4f} ({cc_dsc*100:.2f}%)")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
