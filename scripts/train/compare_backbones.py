import json
import time
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, ".")

from scripts.train.train_boundary_3d import load_case, build_backbone, measure, predict, sample_patch
from monai.losses import DiceCELoss

def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = ["unet", "attention_unet", "unet++", "unet3plus", "segresnet", "caranet3d"]
    
    # 1. Load small subset of real data
    print("Loading 40 training cases and 10 validation cases...")
    data_root = Path("src/data")
    all_patients = sorted([p.name for p in data_root.glob("BraTS2021_*") if p.is_dir()])
    train_ids = all_patients[:40]
    val_ids = all_patients[40:50]
    
    config = {
        "env": {"tolerance_mm": 5.0},
        "resize": 96,
        "backbone_patch_size": 96,
        "backbone_batch_size": 1,
        "backbone_lr": 0.001
    }
    
    train_cases = []
    for pid in train_ids:
        c = load_case(data_root, pid, resize=config["resize"])
        train_cases.append(c)
        
    val_cases = []
    for pid in val_ids:
        c = load_case(data_root, pid, resize=config["resize"])
        val_cases.append(c)
        
    rng = np.random.default_rng(42)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    
    results = []
    
    print("\nStarting Benchmark...")
    print(f"{'Model':<15} | {'Params':<10} | {'VRAM (GB)':<10} | {'Step Time':<10} | {'End DSC':<10}")
    print("-" * 65)
    
    for name in models:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        config["backbone"] = name
        
        try:
            model = build_backbone(config, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["backbone_lr"])
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            
            n_params = get_model_params(model)
            
            # Train for 500 steps
            model.train()
            start_time = time.perf_counter()
            for step in range(500):
                samples = [sample_patch(train_cases[int(rng.integers(len(train_cases)))], config["backbone_patch_size"], rng)
                           for _ in range(config["backbone_batch_size"])]
                images = torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
                targets = torch.from_numpy(np.stack([s[1] for s in samples]))[:, None].to(device)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    out = model(images)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    loss = criterion(out, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            elapsed = time.perf_counter() - start_time
            time_per_step = elapsed / 500
            
            max_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Evaluate on val
            scores = []
            for c in val_cases:
                out = predict(model, c["image"], device, config["backbone_patch_size"])
                if isinstance(out, (list, tuple)):
                    out = out[0]
                pred = out.argmax(0)
                m = measure(c, pred, config)
                scores.append(m["mean"]["dice"])
            val_dsc = float(np.mean(scores))
            
            print(f"{name:<15} | {n_params:<10,} | {max_vram:<10.2f} | {time_per_step:<10.3f} | {val_dsc:<10.4f}")
            results.append((name, n_params, max_vram, time_per_step, val_dsc))
            
        except Exception as e:
            print(f"{name:<15} | ERROR: {str(e)[:40]}")
            
    # Write report
    with open("runs/compare_report.md", "w") as f:
        f.write("# 3D Backbone Comparison Report (500 steps, 40 patients)\n\n")
        f.write("| Model | Parameters | VRAM (GB) | Time/Step (s) | Val DSC (500 steps) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in results:
            f.write(f"| {r[0]} | {r[1]:,} | {r[2]:.2f} | {r[3]:.3f} | {r[4]:.4f} |\n")
            
    print("\nSaved report to runs/compare_report.md")

if __name__ == "__main__":
    main()
