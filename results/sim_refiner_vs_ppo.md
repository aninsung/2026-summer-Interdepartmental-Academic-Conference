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
| small | PPO | 0.8025 | 3.75 | +0.3373 | 114.7 |
| small | **Refiner** | 0.4448 | 7.67 | -0.0204 | 0.8 |
| medium | Rough | 0.8347 | 39.01 | +0.0000 | 0.0 |
| medium | Morphology | 0.9264 | 1.09 | +0.0918 | 0.0 |
| medium | PPO | 0.9254 | 1.09 | +0.0907 | 129.2 |
| medium | **Refiner** | 0.9412 | 0.49 | +0.1065 | 0.5 |
| large | Rough | 0.9317 | 1.08 | +0.0000 | 0.0 |
| large | Morphology | 0.9596 | 0.22 | +0.0279 | 0.0 |
| large | PPO | 0.9587 | 0.22 | +0.0270 | 131.1 |
| large | **Refiner** | 0.9740 | 0.00 | +0.0423 | 0.5 |

## Overall (3 size mean)

| Method | DSC | HD95 |
|---|---:|---:|
| Rough | 0.7439 | 37.22 |
| Morphology | 0.8967 | 0.86 |
| PPO | 0.8955 | 1.69 |
| **Refiner** | **0.7867** | **2.72** |

총 소요: 416.3s

> 합성 데이터·짧은 학습이므로 절대 수치는 BraTS 논문 수치와 직접 비교하지 말 것. 같은 조건에서 **상대 순위**만 본다.
