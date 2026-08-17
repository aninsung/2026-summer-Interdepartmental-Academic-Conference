"""
subregion_pipeline/scripts/train_subregion_classifier.py
---------------------------------------------------------
Stage 1: Subregion Classifier (ET / TC / WT 분류기) 학습 스크립트
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from subregion_pipeline.src.subregion_dataset import SubregionBraTSDataset
from subregion_pipeline.src.subregion_classifier import SubregionClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_classifier(
    train_root: str = "src/data/archive",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_train_patients: int = 210,
    save_path: str = "subregion_pipeline/checkpoints/subregion_classifier.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device(device)
    log.info(f"Loading Subregion Dataset for Classifier Training (Max Patients: {max_train_patients})...")

    ds = SubregionBraTSDataset(root_dir=train_root, target_size=128, max_patients=max_train_patients)
    if len(ds) == 0:
        raise ValueError("데이터셋이 비어 있습니다.")

    n_val = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SubregionClassifier(in_channels=2, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["subregion_label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()

        train_loss /= len(train_ds)
        train_acc = train_correct / len(train_ds)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["subregion_label"].to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)

        log.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            log.info(f"  -> Best model saved! (Val Acc: {val_acc:.4f})")

    log.info(f"Subregion Classifier Training Complete! Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_train_patients", type=int, default=210)
    args = parser.parse_args()
    train_classifier(train_root=args.train_root, batch_size=args.batch_size, max_train_patients=args.max_train_patients)
