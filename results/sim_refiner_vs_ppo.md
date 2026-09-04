# Refiner vs PPO 짧은 시뮬레이션

메인 파이프라인·체크포인트는 변경하지 않음. 합성 blob + Expert 노이즈 rough.

- Rough: 초기 마스크
- Morphology: opening→closing
- PPO: `MaskRefinementEnv` 4k steps (CPU, 짧은 학습)
- Refiner: Tiny U-Net, 입력 `[MRI, rough, prob]`, BCE+Dice 10 epoch

| Size | Method | DSC ↑ | HD95 ↓ | ΔDSC vs Rough | Train sec |
|---|---|---:|---:|---:|---:|
| small | Rough | 0.4652 | 71.58 | +0.0000 | 0.0 |
| small | Morphology | 0.8042 | 1.26 | +0.3390 | 0.0 |
| small | PPO | 0.8046 | 1.22 | +0.3394 | 22.3 |
| small | **Refiner** | 0.4468 | 7.66 | -0.0185 | 1.1 |
| medium | Rough | 0.8347 | 39.01 | +0.0000 | 0.0 |
| medium | Morphology | 0.9264 | 1.09 | +0.0918 | 0.0 |
| medium | PPO | 0.9258 | 1.09 | +0.0911 | 42.0 |
| medium | **Refiner** | 0.9411 | 0.49 | +0.1065 | 0.7 |
| large | Rough | 0.9317 | 1.08 | +0.0000 | 0.0 |
| large | Morphology | 0.9596 | 0.22 | +0.0279 | 0.0 |
| large | PPO | 0.9589 | 0.22 | +0.0273 | 46.9 |
| large | **Refiner** | 0.9740 | 0.00 | +0.0424 | 0.6 |

## Overall (3 size mean)

| Method | DSC | HD95 |
|---|---:|---:|
| Rough | 0.7439 | 37.22 |
| Morphology | 0.8967 | 0.86 |
| PPO | 0.8964 | 0.84 |
| **Refiner** | **0.7873** | **2.72** |

총 소요: 123.6s

> 합성 데이터·짧은 학습이므로 절대 수치는 BraTS 논문 수치와 직접 비교하지 말 것. 같은 조건에서 **상대 순위**만 본다.
