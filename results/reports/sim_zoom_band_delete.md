# Zoom-band delete 시뮬레이션 (강화: FP-가중 + 더 긴 학습)

합성 과분할 rough + 외곽 FP. Stage2 체크포인트 미사용.

- Train 400 / Eval 80
- Full SL: 25 ep (풀슬라이스), Zoom SL: 40 ep (패치 12/slice, FP-가중)
- Morph shrink: 전역 erosion 1px
- Zoom SL delete: 외곽 2× 확대 → rem_band에서만 OFF

| Size | Method | DSC ↑ | HD95 ↓ | ΔDSC vs Rough | FP removed | TP lost |
|---|---|---:|---:|---:|---:|---:|
| medium | Rough | 0.9009 | 1.23 | +0.0000 | +0.0 | +0.0 |
| medium | Morph shrink | 0.9654 | 0.54 | +0.0645 | +98.4 | +0.0 |
| medium | Full SL shrink | 0.9983 | 0.00 | +0.0974 | +147.8 | +1.7 |
| medium | **Zoom SL delete** | 1.0000 | 0.00 | +0.0991 | +148.2 | +0.0 |
| large | Rough | 0.9442 | 0.54 | +0.0000 | +0.0 | +0.0 |
| large | Morph shrink | 0.9806 | 0.00 | +0.0364 | +176.0 | +0.0 |
| large | Full SL shrink | 0.9991 | 0.00 | +0.0549 | +266.9 | +3.8 |
| large | **Zoom SL delete** | 1.0000 | 0.00 | +0.0558 | +267.1 | +0.0 |

총 소요: 60.8s

> 합성 실험. BraTS 절대수치와 비교하지 말고 상대 효과만 볼 것.
