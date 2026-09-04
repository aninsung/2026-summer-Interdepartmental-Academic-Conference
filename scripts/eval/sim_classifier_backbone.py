"""
Stage 1 크기 분류기 백본 비교 시뮬 (메인 코드 미변경).

ResNet18 vs EfficientNet-B0 vs ConvNeXt-Tiny
- ImageNet pretrained, CE, 3-class (Small/Medium/Large)
- 동일 patient split / modality / epochs

결과: results/sim_classifier_backbone.md
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.data.patient_split import load_split_brats_datasets
from src.data.shape_dataset import ShapeDataset
from src.utils.dataloader import loader_kwargs
from src.utils.seed import set_seed


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int, pretrained_weight: torch.Tensor | None):
    if in_channels == conv.in_channels:
        return conv
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False,
    )
    if pretrained_weight is not None:
        # RGB → 합산 후 채널 균등 분배
        merged = pretrained_weight.sum(dim=1, keepdim=True) / in_channels
        new.weight.data.copy_(merged.repeat(1, in_channels, 1, 1))
    return new


def build_backbone(name: str, in_channels: int = 2, num_classes: int = 3, pretrained: bool = True):
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)
        old = m.conv1.weight.data.clone() if weights is not None else None
        m.conv1 = _adapt_first_conv(m.conv1, in_channels, old)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        old_conv = m.features[0][0]
        old_w = old_conv.weight.data.clone() if weights is not None else None
        m.features[0][0] = _adapt_first_conv(old_conv, in_channels, old_w)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=weights)
        old_conv = m.features[0][0]
        old_w = old_conv.weight.data.clone() if weights is not None else None
        m.features[0][0] = _adapt_first_conv(old_conv, in_channels, old_w)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    # per-class
    cm = torch.zeros(3, 3, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1
    acc = correct / max(total, 1)
    # macro recall
    recalls = []
    for c in range(3):
        denom = cm[c].sum().item()
        recalls.append((cm[c, c].item() / denom) if denom > 0 else 0.0)
    return {
        "acc": acc,
        "loss": loss_sum / max(total, 1),
        "recall_small": recalls[0],
        "recall_medium": recalls[1],
        "recall_large": recalls[2],
        "n": total,
    }


def train_one(name: str, train_loader, val_loader, in_ch: int, device, epochs: int, lr: float, seed: int):
    print(f"\n=== {name} | epochs={epochs} ===", flush=True)
    set_seed(seed, deterministic=False)
    model = build_backbone(name, in_channels=in_ch, pretrained=True).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best = {"acc": -1.0}
    t0 = time.time()
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_correct = 0
        tr_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.numel()
        sched.step()
        tr_acc = tr_correct / max(tr_total, 1)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "train_acc": tr_acc, **{f"val_{k}": v for k, v in val.items() if k != "n"}})
        print(
            f"  ep {ep:02d}/{epochs}  train={tr_acc:.4f}  val={val['acc']:.4f}  "
            f"R(S/M/L)={val['recall_small']:.3f}/{val['recall_medium']:.3f}/{val['recall_large']:.3f}",
            flush=True,
        )
        if val["acc"] > best["acc"]:
            best = {**val, "epoch": ep, "train_acc": tr_acc}

    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": name,
        "best_val_acc": best["acc"],
        "best_epoch": best["epoch"],
        "best_train_acc": best.get("train_acc", float("nan")),
        "recall_small": best["recall_small"],
        "recall_medium": best["recall_medium"],
        "recall_large": best["recall_large"],
        "params_m": n_params,
        "train_sec": elapsed,
        "history": history,
    }


def main():
    epochs = 8
    batch_size = 64
    lr = 1e-3
    seed = 42
    set_seed(seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    train_brats, val_brats = load_split_brats_datasets(
        train_root="src/data/archive",
        modality="t1ce+flair",
        target_size=128,
        max_patients=210,
        patient_split="checkpoints/patient_split.json",
        refinement_mode=None,
        simulate_rough=False,
    )
    train_set = ShapeDataset(train_brats)
    val_set = ShapeDataset(val_brats)
    sample, _ = train_set[0]
    in_ch = sample.shape[0] if sample.ndim == 3 else 1
    print(f"train={len(train_set)} val={len(val_set)} in_ch={in_ch}", flush=True)

    # Windows + large in-memory dataset: workers crash with Errno 22 / pickle truncate
    dl_kw = loader_kwargs(num_workers=0, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **dl_kw)

    names = ["resnet18", "efficientnet_b0", "convnext_tiny"]
    rows = []
    for name in names:
        try:
            rows.append(
                train_one(name, train_loader, val_loader, in_ch, device, epochs, lr, seed)
            )
        except Exception as e:
            import traceback

            print(f"FAIL {name}: {e}", flush=True)
            traceback.print_exc()
            rows.append(
                {
                    "name": name,
                    "best_val_acc": float("nan"),
                    "best_epoch": -1,
                    "best_train_acc": float("nan"),
                    "recall_small": float("nan"),
                    "recall_medium": float("nan"),
                    "recall_large": float("nan"),
                    "params_m": float("nan"),
                    "train_sec": 0.0,
                    "note": str(e),
                }
            )

    out = ROOT / "results" / "sim_classifier_backbone.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 1 classifier backbone simulation",
        "",
        f"- Data: BraTS t1ce+flair, patient split 210, train={len(train_set)}, val={len(val_set)}",
        f"- Setup: ImageNet pretrained, CE, 3-class, AdamW lr={lr}, cosine, epochs={epochs}, seed={seed}",
        "- Main pipeline / checkpoints **not modified**",
        "",
        "| Backbone | Best Val Acc | Best Ep | Train Acc@best | Rec S/M/L | Params(M) | Time(s) |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        note = r.get("note", "")
        def f(x):
            return "—" if x != x else f"{x:.4f}"

        rec = f"{f(r['recall_small'])}/{f(r['recall_medium'])}/{f(r['recall_large'])}"
        pm = r["params_m"]
        pm_s = "—" if pm != pm else f"{pm:.1f}"
        lines.append(
            f"| {r['name']} | {f(r['best_val_acc'])} | {r['best_epoch']} | "
            f"{f(r['best_train_acc'])} | {rec} | {pm_s} | {r['train_sec']:.0f} |"
            + (f" <!-- {note} -->" if note else "")
        )
    # ranking
    valid = [r for r in rows if r["best_val_acc"] == r["best_val_acc"]]
    if valid:
        best = max(valid, key=lambda r: r["best_val_acc"])
        lines += [
            "",
            f"**Best in this sim:** `{best['name']}` Val Acc **{best['best_val_acc']:.4f}**",
            "",
            "## Note",
            "- Short epoch budget; ranking may shift with full 15 epochs.",
            "- Documented ResNet18 baseline was Val Acc 0.8512 (longer run).",
            "",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\nsaved", out, flush=True)
    for r in rows:
        acc = r["best_val_acc"]
        acc_s = "nan" if acc != acc else f"{acc:.4f}"
        pm = r["params_m"]
        pm_s = "nan" if pm != pm else f"{pm:.1f}"
        print(
            f"{r['name']}: val={acc_s} @ep{r['best_epoch']}  "
            f"params={pm_s}M  time={r['train_sec']:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
