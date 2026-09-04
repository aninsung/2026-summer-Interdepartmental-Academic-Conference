# Small Zoom+Residual Refiner 시뮬레이션

메인 파이프라인 미변경. 합성 Small blob (`area<300`), micro(`<50`) 오버샘플 학습.

- Global: 128×128 Tiny-UNet `[MRI,rough,prob]→mask`
- Zoom+Residual: 64×64 COM crop, `logit = f(x) + α·logit(rough)`, boundary Dice, empty-crop fallback
- PPO: MaskRefinementEnv small, 4k steps

eval n=48, micro n=3

| Method | DSC ↑ | HD95 ↓ | ΔDSC | Micro DSC | Train sec |
|---|---:|---:|---:|---:|---:|
| Rough | 0.5298 | 68.65 | +0.0000 | 0.3419 | 0.0 |
| Morphology | 0.8101 | 1.26 | +0.2803 | 0.7108 | 0.0 |
| Global Refiner | 0.7828 | 1.75 | +0.2530 | 0.7283 | 1.1 |
| **Zoom+Residual** | 0.5759 | 69.41 | +0.0460 | 0.3875 | 0.7 |
| PPO | 0.8084 | 1.23 | +0.2786 | 0.7213 | 27.4 |

총 소요: 34.4s

> 합성·짧은 학습. BraTS 절대수치와 비교하지 말고 Small 설계 상대 효과만 볼 것.
