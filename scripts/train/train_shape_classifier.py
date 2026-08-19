import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.shape_dataset import ShapeDataset
from src.models.shape_classifier import build_shape_classifier
from src.utils.device import get_torch_device
import time

def main():
    parser = argparse.ArgumentParser(description="Shape Classifier Training")
    parser.add_argument("--train_root", type=str, default="src/data/archive", help="데이터셋 경로")
    parser.add_argument("--max_train_patients", type=int, default=None, help="학습 환자 수 제한 (None이면 전체)")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=15, help="에폭 수")
    parser.add_argument("--modality", type=str, default="t1ce+flair", help="MRI 모달리티 ('t1ce', 't1ce+flair' 등)")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW 가중치 감쇠")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="ImageNet 사전학습 없이 무작위 초기화 (ablation용)")
    parser.add_argument("--seed", type=int, default=42, help="전역 시드")
    parser.add_argument("--deterministic", action="store_true", help="cuDNN 결정적 모드 (느려짐)")
    args = parser.parse_args()

    from src.utils.seed import set_seed
    set_seed(args.seed, args.deterministic)
    print(f"Seed: {args.seed} (deterministic={args.deterministic})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print(f"Loading BraTS Dataset from {args.train_root} (Modality: {args.modality})...")
    from src.data.patient_split import load_split_brats_datasets
    train_brats, val_brats = load_split_brats_datasets(
        train_root=args.train_root,
        modality=args.modality,
        target_size=128,
        max_patients=args.max_train_patients,
        patient_split=args.patient_split,
        refinement_mode=None,
        simulate_rough=False,
    )
    train_set = ShapeDataset(train_brats)
    val_set = ShapeDataset(val_brats)
    train_size = len(train_set)
    val_size = len(val_set)
    print(f"Total valid slices: {train_size + val_size} (train={train_size}, val={val_size})")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 3. 모델 초기화
    sample_img, _ = train_set[0]
    in_channels = sample_img.shape[0] if sample_img.ndim == 3 else 1
    pretrained = not args.no_pretrained
    print(f"Building Shape Classifier (Input Channels: {in_channels}, Pretrained: {pretrained})...")
    model = build_shape_classifier(
        in_channels=in_channels, num_classes=3, pretrained=pretrained
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. 훈련 루프
    num_epochs = args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            
        epoch_train_loss = train_loss / train_size
        epoch_train_acc = train_correct.double() / train_size
        
        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = val_correct.double() / val_size
        
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.1f}s] lr={cur_lr:.2e} "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "checkpoints/shape_classifier_best.pt")
            print(f"  -> Best model saved! (Val Acc: {best_val_acc:.4f})")
            
    print(f"Training Complete! Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
