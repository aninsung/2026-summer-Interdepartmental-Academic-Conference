import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.shape_dataset import ShapeDataset
from src.models.shape_classifier import SUPPORTED_BACKBONES, build_shape_classifier
from src.utils.dataloader import loader_kwargs


@torch.no_grad()
def _eval_detailed(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    cm = torch.zeros(3, 3, dtype=torch.long)
    for batch in loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
    recalls = []
    for c in range(3):
        denom = cm[c].sum().item()
        recalls.append((cm[c, c].item() / denom) if denom > 0 else 0.0)
    macro = float(np.mean(recalls))
    return {
        "acc": correct / max(total, 1),
        "macro_recall": macro,
        "recall_small": recalls[0],
        "recall_medium": recalls[1],
        "recall_large": recalls[2],
        "cm": cm.tolist(),
        "n": total,
    }


def _class_weights_from_loader(dataset, device):
    counts = np.zeros(3, dtype=np.float64)
    for i in range(len(dataset)):
        item = dataset[i]
        y = int(item[1].item())
        counts[y] += 1
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32, device=device), counts


def _batch_loss(logits, labels, soft, class_weight, use_soft, ordinal_weight):
    if use_soft and soft is not None:
        log_p = F.log_softmax(logits, dim=1)
        ce_vec = -(soft * log_p).sum(dim=1)
        if class_weight is not None:
            ce_vec = ce_vec * class_weight[labels]
        ce = ce_vec.mean()
    else:
        ce = F.cross_entropy(logits, labels, weight=class_weight)

    loss = ce
    if ordinal_weight > 0:
        t_med = (labels >= 1).float()
        t_large = (labels >= 2).float()
        s = logits
        p_med = torch.sigmoid(s[:, 1] + s[:, 2] - s[:, 0])
        p_large = torch.sigmoid(s[:, 2] - 0.5 * (s[:, 0] + s[:, 1]))
        ord_loss = F.binary_cross_entropy(p_med, t_med) + F.binary_cross_entropy(p_large, t_large)
        loss = loss + ordinal_weight * ord_loss
    return loss


def main():
    parser = argparse.ArgumentParser(description="Shape Classifier Training")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--max_train_patients", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=list(SUPPORTED_BACKBONES))
    parser.add_argument("--save_path", type=str, default="checkpoints/shape_classifier_best.pt")
    parser.add_argument("--metrics_out", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--aug", action="store_true", default=True, help="P1: MRI aug (flip/±15°/bright-contrast)")
    parser.add_argument("--no_aug", action="store_true", help="Disable MRI augmentation")
    parser.add_argument("--class_weight", action="store_true", default=True, help="P1: inverse-freq CE weights")
    parser.add_argument("--no_class_weight", action="store_true", help="Disable CE class weights")
    parser.add_argument(
        "--select_metric",
        type=str,
        default="macro_recall",
        choices=["acc", "macro_recall", "joint"],
        help="Best ckpt: acc | macro_recall | joint(=0.5*acc+0.5*macroR)",
    )
    parser.add_argument("--patience", type=int, default=8, help="early stop patience (0=off)")
    parser.add_argument("--boundary_soft", action="store_true", help="P2: soft labels near 300/700")
    parser.add_argument("--boundary_margin", type=float, default=20.0)
    parser.add_argument("--ordinal_weight", type=float, default=0.0, help="P2: ordinal aux loss weight")
    parser.add_argument(
        "--recipe",
        type=str,
        default="p1",
        choices=["", "baseline", "p1", "p2"],
        help="preset: baseline | p1 (default) | p2 (overrides related flags)",
    )
    args = parser.parse_args()

    if args.no_aug:
        args.aug = False
    if args.no_class_weight:
        args.class_weight = False

    if args.recipe == "baseline":
        args.aug = False
        args.class_weight = False
        args.boundary_soft = False
        args.ordinal_weight = 0.0
        args.select_metric = "acc"
        args.lr = 1e-3
        args.epochs = 15
        args.patience = 0
    elif args.recipe == "p1":
        args.aug = True
        args.class_weight = True
        args.boundary_soft = False
        args.ordinal_weight = 0.0
        args.select_metric = "macro_recall"
        args.lr = 3e-4
        args.epochs = 30
        args.patience = 8
    elif args.recipe == "p2":
        args.aug = True
        args.class_weight = True
        args.boundary_soft = True
        args.ordinal_weight = 0.3
        args.select_metric = "macro_recall"
        args.lr = 3e-4
        args.epochs = 30
        args.patience = 8

    from src.utils.seed import set_seed
    set_seed(args.seed, args.deterministic)
    print(f"Seed: {args.seed} (deterministic={args.deterministic})")
    print(
        f"Recipe flags: aug={args.aug} class_weight={args.class_weight} "
        f"boundary_soft={args.boundary_soft} ordinal={args.ordinal_weight} "
        f"select={args.select_metric} lr={args.lr} epochs={args.epochs} patience={args.patience}"
    )

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
    train_set = ShapeDataset(
        train_brats,
        augment=args.aug,
        soft_boundary=args.boundary_soft,
        boundary_margin=args.boundary_margin,
        seed=args.seed,
    )
    val_set = ShapeDataset(val_brats, augment=False, soft_boundary=False, seed=args.seed + 1)
    train_size = len(train_set)
    val_size = len(val_set)
    print(f"Total valid slices: {train_size + val_size} (train={train_size}, val={val_size})")

    dl_kw = loader_kwargs()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **dl_kw)
    print(f"DataLoader: num_workers={dl_kw['num_workers']}, batch_size={args.batch_size}")

    sample = train_set[0]
    sample_img = sample[0]
    in_channels = sample_img.shape[0] if sample_img.ndim == 3 else 1
    pretrained = not args.no_pretrained
    print(
        f"Building Shape Classifier backbone={args.backbone} "
        f"(in_ch={in_channels}, pretrained={pretrained})..."
    )
    model = build_shape_classifier(
        in_channels=in_channels,
        num_classes=3,
        pretrained=pretrained,
        backbone=args.backbone,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {n_params:.2f}M")

    class_weight = None
    if args.class_weight:
        class_weight, counts = _class_weights_from_loader(
            ShapeDataset(train_brats, augment=False, soft_boundary=False, seed=args.seed),
            device,
        )
        print(f"Class counts={counts.tolist()} weights={class_weight.detach().cpu().tolist()}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_epochs = args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_score = -1.0
    best_metrics = None
    bad_epochs = 0
    os.makedirs(os.path.dirname(args.save_path) or "checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            soft = batch[2].to(device) if (args.boundary_soft and len(batch) > 2) else None
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = _batch_loss(
                outputs, labels, soft, class_weight,
                use_soft=args.boundary_soft, ordinal_weight=args.ordinal_weight,
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_train_loss = train_loss / train_size
        epoch_train_acc = train_correct / train_size
        val = _eval_detailed(model, val_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if args.select_metric == "acc":
            score = val["acc"]
        elif args.select_metric == "joint":
            score = 0.5 * val["acc"] + 0.5 * val["macro_recall"]
        else:
            score = val["macro_recall"]
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{num_epochs} [{elapsed:.1f}s] lr={cur_lr:.2e} "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Acc: {val['acc']:.4f} macroR={val['macro_recall']:.4f} "
            f"R(S/M/L)={val['recall_small']:.3f}/{val['recall_medium']:.3f}/{val['recall_large']:.3f}"
        )

        # Prefer higher select score; tie-break on Val Acc then Large recall
        better = score > best_score + 1e-8
        if (not better) and best_metrics is not None and abs(score - best_score) <= 1e-8:
            better = (
                val["acc"] > best_metrics["best_val_acc"] + 1e-8
                or (
                    abs(val["acc"] - best_metrics["best_val_acc"]) <= 1e-8
                    and val["recall_large"] > best_metrics["recall_large"] + 1e-8
                )
            )
        if better:
            best_score = score
            bad_epochs = 0
            best_metrics = {
                "backbone": args.backbone,
                "recipe": args.recipe or "custom",
                "best_epoch": epoch + 1,
                "best_val_acc": val["acc"],
                "best_macro_recall": val["macro_recall"],
                "select_metric": args.select_metric,
                "best_score": best_score,
                "train_acc_at_best": epoch_train_acc,
                "recall_small": val["recall_small"],
                "recall_medium": val["recall_medium"],
                "recall_large": val["recall_large"],
                "cm": val["cm"],
                "params_m": n_params,
                "in_channels": in_channels,
                "seed": args.seed,
                "epochs": num_epochs,
                "aug": args.aug,
                "class_weight": args.class_weight,
                "boundary_soft": args.boundary_soft,
                "ordinal_weight": args.ordinal_weight,
                "lr": args.lr,
            }
            torch.save(model.state_dict(), args.save_path)
            print(
                f"  -> Best saved {args.save_path} "
                f"(select={args.select_metric}={best_score:.4f}, acc={val['acc']:.4f})"
            )
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stop at epoch {epoch+1} (patience={args.patience})")
                break

    print(
        f"Training Complete! best {args.select_metric}={best_score:.4f} "
        f"acc={best_metrics['best_val_acc']:.4f}" if best_metrics else "Training Complete!"
    )
    if args.metrics_out and best_metrics is not None:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)
        print(f"Metrics saved: {args.metrics_out}")


if __name__ == "__main__":
    main()
