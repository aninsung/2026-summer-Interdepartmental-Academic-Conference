# Stage1 classifier: baseline vs P1 vs P2

- **baseline**: ResNet18, no aug, CE, lr=1e-3, 15 ep, select=acc
- **p1**: H/V flip + ±15° + bright/contrast, inverse-freq CE, lr=3e-4, 30 ep, early stop, select=macro_recall
- **p2**: p1 + boundary soft labels (±20px) + ordinal aux (0.3)

| Recipe | Val Acc | MacroR | Rec S/M/L | Best Ep | select | Checkpoint |
|---|---:|---:|---|---:|---|---|
| baseline | 0.8488 | 0.8181 | 0.886/0.874/0.694 | 15 | acc | `checkpoints/shape_classifier_baseline.pt` |
| p1 | 0.8603 | 0.8745 | 0.902/0.812/0.910 | 2 | macro_recall | `checkpoints/shape_classifier_p1.pt` |
| p2 | 0.8763 | 0.8845 | 0.921/0.835/0.897 | 23 | macro_recall | `checkpoints/shape_classifier_p2.pt` |

**Best Val Acc:** `p2` = **0.8763**
**Best Macro Recall:** `p2` = **0.8845**

Pipeline에 쓰려면 승자 체크포인트를 `shape_classifier_best.pt`로 복사.
