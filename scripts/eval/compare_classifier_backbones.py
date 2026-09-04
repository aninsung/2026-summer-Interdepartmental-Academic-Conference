"""Train ResNet18 vs EfficientNet-B0 Stage1 classifiers and write a comparison table."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(backbone: str) -> dict:
    ckpt = ROOT / "checkpoints" / f"shape_classifier_{backbone}.pt"
    metrics = ROOT / "results" / f"classifier_{backbone}_metrics.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/train/train_shape_classifier.py"),
        "--backbone", backbone,
        "--batch_size", "64",
        "--epochs", "15",
        "--seed", "42",
        "--deterministic",
        "--max_train_patients", "210",
        "--patient_split", "checkpoints/patient_split.json",
        "--modality", "t1ce+flair",
        "--save_path", str(ckpt),
        "--metrics_out", str(metrics),
    ]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return json.loads(metrics.read_text(encoding="utf-8"))


def main():
    rows = []
    for name in ("resnet18", "efficientnet_b0"):
        rows.append(run(name))

    out = ROOT / "results" / "classifier_backbone_compare.md"
    lines = [
        "# Stage1 classifier: ResNet18 vs EfficientNet-B0",
        "",
        "- Data: BraTS t1ce+flair, patient split 210 (train/val from `patient_split.json`)",
        "- Setup: ImageNet pretrained, CE, AdamW 1e-3, cosine, **15 epochs**, seed=42, deterministic",
        "",
        "| Backbone | Best Val Acc | Best Ep | Train Acc@best | Rec S/M/L | Params(M) | Checkpoint |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for r in rows:
        rec = (
            f"{r['recall_small']:.3f}/{r['recall_medium']:.3f}/{r['recall_large']:.3f}"
        )
        ckpt = f"`checkpoints/shape_classifier_{r['backbone']}.pt`"
        lines.append(
            f"| {r['backbone']} | {r['best_val_acc']:.4f} | {r['best_epoch']} | "
            f"{r['train_acc_at_best']:.4f} | {rec} | {r['params_m']:.1f} | {ckpt} |"
        )

    winner = max(rows, key=lambda x: x["best_val_acc"])
    lines += [
        "",
        f"**Winner:** `{winner['backbone']}` Val Acc **{winner['best_val_acc']:.4f}**",
        "",
        "To use the winner in the pipeline:",
        "```",
        f"copy checkpoints\\shape_classifier_{winner['backbone']}.pt checkpoints\\shape_classifier_best.pt",
        "```",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\nsaved", out, flush=True)
    print(f"WINNER: {winner['backbone']} {winner['best_val_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
