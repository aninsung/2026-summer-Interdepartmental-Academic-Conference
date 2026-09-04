# Stage1 classifier: ResNet18 vs EfficientNet-B0

- Data: BraTS t1ce+flair, patient split 210 (train/val from `patient_split.json`)
- Setup: ImageNet pretrained, CE, AdamW 1e-3, cosine, **15 epochs**, seed=42, deterministic

| Backbone | Best Val Acc | Best Ep | Train Acc@best | Rec S/M/L | Params(M) | Checkpoint |
|---|---:|---:|---:|---|---:|---|
| resnet18 | 0.8644 | 6 | 0.9667 | 0.911/0.872/0.739 | 11.2 | `checkpoints/shape_classifier_resnet18.pt` |
| efficientnet_b0 | 0.8583 | 15 | 0.9987 | 0.884/0.847/0.835 | 4.0 | `checkpoints/shape_classifier_efficientnet_b0.pt` |

**Winner:** `resnet18` Val Acc **0.8644**

To use the winner in the pipeline:
```
copy checkpoints\shape_classifier_resnet18.pt checkpoints\shape_classifier_best.pt
```
