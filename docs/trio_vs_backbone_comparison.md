# TRIO vs 단일 백본+PPO 비교

평가 조건: BraTS 2021, 환자 210명 풀.  
상세 요약: [`docs/trio_vs_backbone_summary.md`](trio_vs_backbone_summary.md)

- **TRIO**: 크기별 Expert 라우팅 + 크기별 PPO (메인, t1ce+flair, val 42명, n=2,434, 2026-08-20)
- **U-Net / UNet++ / SegResNet**: 단일 백본 + 단일 PPO (`ppo/`, t1ce, hold-out 20%, n=2,449)

## Overall

> TRIO RL **0.9031†** = GT 상한. 논문 비교는 Rough **0.8948**.

| Method | Rough DSC | RL DSC | RL HD95 ↓ | n |
|---|---:|---:|---:|---:|
| **TRIO** | **0.8948** | 0.9031† | 1.61† | 2,434 |
| U-Net | 0.7157 | 0.7198 | 5.17 | 2,449 |
| UNet++ | 0.7631 | 0.7661 | 4.53 | 2,449 |
| SegResNet | 0.7542 | 0.7638 | 4.89 | 2,449 |

## Size-stratified (TRIO = Stage 2; 단일 백본 = RL 후)

### DSC ↑

| Size | TRIO Stage 2 | U-Net | UNet++ | SegResNet |
|---|---:|---:|---:|---:|
| Small | **0.8286** | 0.5499 | 0.6649 | 0.6599 |
| Medium | **0.9264** | 0.8288 | 0.8273 | 0.8281 |
| Large | **0.9546** | 0.9034 | 0.8857 | 0.8827 |

### HD95 ↓ (px)

| Size | TRIO Stage 2 | U-Net | UNet++ | SegResNet |
|---|---:|---:|---:|---:|
| Small | **3.16** | 9.18 | 5.93 | 6.72 |
| Medium | **1.08** | 2.23 | 3.49 | 3.52 |
| Large | **0.37** | 1.88 | 3.39 | 3.47 |

## 정성

- 3×4 RL only (클래스 평균 DSC 근접): `results/rl_refined_3x4_comparison.png`
- Rough = Stage 2 / Expert 초기 분할
- TRIO RL† = 과거 단조 게이트 상한 (논문 메인 금지)
