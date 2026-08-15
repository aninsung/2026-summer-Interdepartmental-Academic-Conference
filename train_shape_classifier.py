import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.shape_dataset import ShapeDataset
from src.models.shape_classifier import build_shape_classifier
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. 원본 데이터셋 로드 (최대 100명의 환자 = 약 6,000장 이상의 슬라이스)
    #    평가를 빨리 확인하기 위해 100명으로 제한하여 훈련합니다.
    print("Loading BraTS Dataset...")
    brats_dataset = BraTS2020Dataset(
        root_dir="src/data/archive/BraTS2021_Training_Data",
        modality="t1ce",
        target_size=128,
        max_patients=100, 
        simulate_rough=False # GT 기반으로 넓이를 재므로 상관없음
    )
    
    # 2. ShapeDataset 래핑
    shape_dataset = ShapeDataset(brats_dataset)
    total_size = len(shape_dataset)
    print(f"Total valid slices: {total_size}")
    
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_set, val_set = random_split(shape_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)
    
    # 3. 모델 초기화
    model = build_shape_classifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 4. 훈련 루프
    num_epochs = 15
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
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.1f}s] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "checkpoints/shape_classifier_best.pt")
            print(f"  -> Best model saved! (Val Acc: {best_val_acc:.4f})")
            
    print(f"Training Complete! Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
