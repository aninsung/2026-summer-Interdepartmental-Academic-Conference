"""Compare Stage1 classifier: baseline vs P1 vs P2."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(recipe: str) -> dict:
    ckpt = ROOT / "checkpoints" / f"shape_classifier_{recipe}.pt"
    metrics = ROOT / "results" / f"classifier_{recipe}_metrics.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/train/train_shape_classifier.py"),
        "--recipe", recipe,
        "--backbone", "resnet18",
        "--batch_size", "64",
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
    rows = [run(r) for r in ("baseline", "p1", "p2")]
    out = ROOT / "results" / "classifier_p1_p2_compare.md"
    lines = [
        "# Stage1 classifier: baseline vs P1 vs P2",
        "",
        "- **baseline**: ResNet18, no aug, CE, lr=1e-3, 15 ep, select=acc",
        "- **p1**: aug + class weight + lr=3e-4 + 30 ep + early stop + select=macro_recall",
        "- **p2**: p1 + boundary soft labels (±20px) + ordinal aux (0.3)",
        "",
        "| Recipe | Val Acc | MacroR | Rec S/M/L | Best Ep | select | Checkpoint |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    for r in rows:
        rec = f"{r['recall_small']:.3f}/{r['recall_medium']:.3f}/{r['recall_large']:.3f}"
        lines.append(
            f"| {r.get('recipe', '?')} | {r['best_val_acc']:.4f} | {r['best_macro_recall']:.4f} | "
            f"{rec} | {r['best_epoch']} | {r['select_metric']} | "
            f"`checkpoints/shape_classifier_{r.get('recipe','?')}.pt` |"
        )
    by_acc = max(rows, key=lambda x: x["best_val_acc"])
    by_macro = max(rows, key=lambda x: x["best_macro_recall"])
    lines += [
        "",
        f"**Best Val Acc:** `{by_acc.get('recipe')}` = **{by_acc['best_val_acc']:.4f}**",
        f"**Best Macro Recall:** `{by_macro.get('recipe')}` = **{by_macro['best_macro_recall']:.4f}**",
        "",
        "Pipeline에 쓰려면 승자 체크포인트를 `shape_classifier_best.pt`로 복사.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\nsaved", out, flush=True)
    for r in rows:
        print(
            f"{r.get('recipe')}: acc={r['best_val_acc']:.4f} macroR={r['best_macro_recall']:.4f} "
            f"R={r['recall_small']:.3f}/{r['recall_medium']:.3f}/{r['recall_large']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
