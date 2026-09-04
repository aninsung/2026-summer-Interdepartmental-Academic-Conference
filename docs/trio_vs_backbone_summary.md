# 실험 결과 정리 — TRIO vs 단일 백본+PPO

단일 백본 파이프라인(`ppo/`)과 메인 TRIO를 같은 형식으로 비교한 요약입니다.  
(갱신: 2026-08-21)

## 0. 실험 설정

| 항목 | TRIO (메인) | 단일 백본 (`ppo/`) |
|---|---|---|
| 환자 풀 | 210명 | 210명 |
| 평가 분할 | patient val 42명 | 슬라이스 80/20 hold-out |
| n (슬라이스) | 2,434 | 2,449 |
| 입력 | t1ce + flair | t1ce only |
| Stage 2 | CaraNet / UNet++ / SegResNet (크기 라우팅) | U-Net · UNet++ · SegResNet (각 1종) |
| Stage 3 | 크기별 PPO (`ppo_small/medium/large`) | 단일 PPO (`ppo_refiner_*`) |
| RL 후처리 | **배포:** 면적 게이트만 (과거 †: 단조 DSC 게이트) | morpho refine + DSC 게이트 (`evaluate.py`) |

크기 구간: Small `<300px` / Medium `300–700px` / Large `≥700px` (GT 면적).

## 1. 정량 비교 — 전체 평균

> TRIO RL DSC **0.9031†**은 GT 상한. 논문 비교는 Rough/Stage 2 **0.8948**을 쓴다.

| Method | Rough DSC | RL DSC | RL HD95 ↓ | n |
|---|---:|---:|---:|---:|
| **TRIO** | **0.8948** | 0.9031† | 1.61† | 2,434 |
| U-Net | 0.7157 | 0.7198 | 5.17 | 2,449 |
| UNet++ | 0.7631 | 0.7661 | 4.53 | 2,449 |
| SegResNet | 0.7542 | 0.7638 | 4.89 | 2,449 |

† GT 단조 게이트·best-of-15. 배포 아님.

원본 JSON: `ppo/results/metrics_{unet,unetplusplus,segresnet}.json`

## 2. 정량 비교 — 크기별 (RL Final)

### DSC ↑

| Size | TRIO | U-Net | UNet++ | SegResNet |
|---|---:|---:|---:|---:|
| Small | **0.8429** | 0.5499 | 0.6649 | 0.6599 |
| Medium | **0.9314** | 0.8288 | 0.8273 | 0.8281 |
| Large | **0.9582** | 0.9034 | 0.8857 | 0.8827 |

### HD95 ↓ (px)

| Size | TRIO | U-Net | UNet++ | SegResNet |
|---|---:|---:|---:|---:|
| Small | **2.95** | 9.18 | 5.93 | 6.72 |
| Medium | **1.00** | 2.23 | 3.49 | 3.52 |
| Large | **0.29** | 1.88 | 3.39 | 3.47 |

- TRIO n: 897 / 1,152 / 385 (분류기 라우팅)
- 단일 백본 n: 1,053 / 1,036 / 360 (GT 면적 빈)
- 재계산: `ppo/eval_size_hd95.py` → `ppo/results/metrics_size_hd95.json`
- 표 파일: `ppo/results/trio_vs_backbone_by_size.md`

## 3. 정성 비교 이미지

| 파일 | 내용 |
|---|---|
| `results/rl_refined_3x4_comparison.png` | **3×4 RL only** — 행 Small/Medium/Large, 열 TRIO·U-Net·UNet++·SegResNet. 클래스 평균 DSC에 가까운 동일 슬라이스(행 공유). 청록=예측, 초록 점선=GT. 산출: `scripts/eval/plot_rl_refined_3x4.py` |
| `results/pipeline_sample_comparison.png` | TRIO Stage2 Rough → Stage3 RL (크기별 2표본) |
| `ppo/results/pipeline_sample_comparison_{unet,unetplusplus,segresnet}.png` | 단일 백본별 Rough→RL |
| `ppo/results/pipeline_sample_comparison.png` | 단일 백본 3종 합본 |

### 3×4 대표 슬라이스 (클래스 평균 근접, 2026-08-21)

| Size | TRIO | U-Net | UNet++ | SegResNet |
|---|---:|---:|---:|---:|
| Small | 0.852 (mean 0.843) | 0.452 (0.550) | 0.749 (0.665) | 0.735 (0.660) |
| Medium | 0.937 (mean 0.931) | 0.818 (0.829) | 0.841 (0.827) | 0.826 (0.828) |
| Large | 0.941 (mean 0.958) | 0.873 (0.903) | 0.894 (0.886) | 0.883 (0.883) |

표시용으로 largest-CC 정리를 적용했다. 정량 표의 평균과는 별개이다.

## 4. 체크포인트 (`ppo/checkpoints/`)

| 파일 | 설명 |
|---|---|
| `unet_best.pt` / `ppo_refiner_unet.zip` | U-Net + PPO |
| `unetplusplus_best.pt` / `ppo_refiner_unetplusplus.zip` | UNet++ + PPO |
| `segresnet_best.pt` / `ppo_refiner_segresnet.zip` | SegResNet + PPO |

## 5. 한줄 해석

- **전체 DSC**는 TRIO(0.90)가 단일 백본(0.72–0.77)보다 뚜렷히 높다.
- 크기별로는 **Small**에서 격차가 가장 크다(TRIO 0.84 vs 단일 0.55–0.66).
- HD95도 TRIO가 전 구간에서 더 낮다(특히 Small 2.95 vs 5.9–9.2).
- 단일 백본 중 **UNet++**가 전체 RL DSC·HD95 최선(0.7661 / 4.53).
- SegResNet은 Rough→RL DSC 이득이 가장 크다(+0.0096).
- 프로토콜이 완전히 동일하지 않다(모달리티·분할 방식·게이트). 경향 비교용 ablation으로 해석한다.
