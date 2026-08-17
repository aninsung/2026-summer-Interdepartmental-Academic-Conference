"""
subregion_pipeline/scripts/train_subregion_experts.py
------------------------------------------------------
Stage 2: Subregion Expert 백본 모델 학습 스크립트
  - Subregion 0 (ET): Attention U-Net
  - Subregion 1 (TC): UNet++
  - Subregion 2 (WT): SegResNet
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from subregion_pipeline.src.subregion_dataset import SubregionBraTSDataset
from src.models.attention_unet import build_attention_unet
from src.models.unetplusplus import build_unetplusplus
from src.models.segresnet import build_segresnet, BCEDiceLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_expert(
    subregion_target: str = "et", # "et", "tc", "wt"
    train_root: str = "src/data/archive",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 3e-4,
    max_train_patients: int = 210,
    save_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device)
    if save_path is None:
        save_path = f"subregion_pipeline/checkpoints/expert_{subregion_target}_best.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    log.info(f"Loading Dataset for Subregion Expert [{subregion_target.upper()}] Training...")
    ds = SubregionBraTSDataset(root_dir=train_root, target_size=128, max_patients=max_train_patients)

    target_map = {"et": 0, "tc": 1, "wt": 2}
    target_idx = target_map[subregion_target]

    # Filter dataset for specific subregion mask presence
    if subregion_target == "et":
        filtered_indices = [i for i, s in enumerate(ds.samples) if s["gt_et"].sum() >= 20]
    elif subregion_target == "tc":
        filtered_indices = [i for i, s in enumerate(ds.samples) if s["gt_tc"].sum() >= 20]
    else:
        filtered_indices = [i for i, s in enumerate(ds.samples) if s["gt_wt"].sum() >= 20]

    if len(filtered_indices) == 0:
        filtered_indices = list(range(len(ds)))  # Fallback

    sub_ds = Subset(ds, filtered_indices)
    n_val = max(1, int(len(sub_ds) * 0.2))
    n_train = len(sub_ds) - n_val
    train_ds, val_ds = random_split(sub_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    log.info(f"[{subregion_target.upper()} Expert] Training samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Model Selection
    if subregion_target == "et":
        model = build_attention_unet(in_channels=2, out_channels=1).to(device)
        gt_key = "gt_et"
    elif subregion_target == "tc":
        model = build_unetplusplus(in_channels=2, out_channels=1).to(device)
        gt_key = "gt_tc"
    else:
        model = build_segresnet(in_channels=2, out_channels=1).to(device)
        gt_key = "gt_wt"

    criterion = BCEDiceLoss(smooth=1e-5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_dsc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            gts  = batch[gt_key].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, gts)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_dsc_list = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                gts  = batch[gt_key].to(device)

                probs = torch.sigmoid(model(imgs))
                preds = (probs > 0.5).float()
                
                intersection = (preds * gts).sum(dim=(-2, -1))
                union = preds.sum(dim=(-2, -1)) + gts.sum(dim=(-2, -1))
                dsc = (2.0 * intersection + 1e-5) / (union + 1e-5)
                val_dsc_list.extend(dsc.cpu().numpy().tolist())

        val_dsc = float(np.mean(val_dsc_list))
        log.info(f"Epoch [{epoch:02d}/{epochs:02d}] loss={train_loss:.4f} val_DSC={val_dsc:.4f}")

        if val_dsc > best_dsc:
            best_dsc = val_dsc
            torch.save(model.state_dict(), save_path)
            log.info(f"  ✔ Best model saved (val_DSC={val_dsc:.4f})")

    log.info(f"=== [{subregion_target.upper()} Expert] Training Complete. Best val DSC: {best_dsc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subregion", type=str, default="et", choices=["et", "tc", "wt"])
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_train_patients", type=int, default=210)
    args = parser.parse_args()

    import numpy as np
    train_expert(
        subregion_target=args.subregion,
        train_root=args.train_root,
        batch_size=args.batch_size,
        max_train_patients=args.max_train_patients
    )
