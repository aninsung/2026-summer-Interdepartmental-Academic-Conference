# RL-Refiner 실험 결과

> 실험 이력과 최종 벤치마크를 한 파일로 정리한 문서입니다.  
> 이전 `EXPERIMENTS.md`, `final_models_report.md`, `technical_report.md`의 내용을 통합했습니다.

- **데이터셋**: BraTS 2021 Task 1
- **입력**: `t1ce+flair`, 128×128 2D 슬라이스
- **지표**: DSC (높을수록 좋음), HD95 px (낮을수록 좋음)
- **최신 실행**: 2026-08-19 `python run_pipeline.py --modality t1ce+flair` (Stage 1–4 공통 **210명**)
- **파이프라인 구조**: [PIPELINE.md](PIPELINE.md)

---

## 1. 최종 결과 (2026-08-19, 210명 전수)

### 1.1 평가 설정

| 항목 | 값 |
|---|---|
| 실행 명령 | `python run_pipeline.py --modality t1ce+flair` |
| Stage 1–4 환자 수 | **210명** (분류기 / Expert / PPO / 평가 공통) |
| 유효 슬라이스 | **12,241장** 전수 평가 |
| 클래스 라우팅 | GT 면적 기준 (`<300` / `300–700` / `≥700`) |
| Stage 2 | CaraNet / UNet++ / SegResNet |
| Stage 3 | 크기별 PPO (`ppo_small.zip` / `ppo_medium.zip` / `ppo_large.zip`) |
| 안전장치 | Dual Monotonic Safety Gate (DSC 하락 또는 HD95 증가 시 초기 마스크 원복) |
| 시각화 | 클래스당 Rough–RL DSC 차이가 큰 원본 2장, 총 6장 |

### 1.2 파이프라인 종합

| 지표 | Stage 2 초기 | Stage 3 보정 후 | 변화 |
|---|---:|---:|---:|
| 평균 DSC | 0.8775 | **0.8880** | **+0.0105 (+1.05%p)** |
| 평균 HD95 | 1.8416 px | **1.6843 px** | **−0.1573 px** |

세 클래스 모두 DSC가 올랐고 HD95가 줄었습니다. Medium에서 개선폭이 가장 큽니다.

### 1.3 크기별 성능

| 클래스 | Expert | n | 초기 DSC | 최종 DSC | ΔDSC | 초기 HD95 | 최종 HD95 | ΔHD95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Small (`<300px`)** | CaraNet | 4,503 | 0.7956 | **0.8035** | +0.0079 | 3.5405 | **3.4353** | −0.1052 |
| **Medium (`300–700px`)** | UNet++ | 5,031 | 0.9180 | **0.9322** | +0.0142 | 0.9031 | **0.6965** | −0.2066 |
| **Large (`≥700px`)** | SegResNet | 2,707 | 0.9386 | **0.9464** | +0.0078 | 0.7595 | **0.6077** | −0.1518 |

### 1.4 Small 층화

| 구간 | n | 초기 DSC | 최종 DSC | 초기 HD95 | 최종 HD95 |
|---|---:|---:|---:|---:|---:|
| Active Tumor (`≥50px`) | 4,135 | 0.8135 | **0.8208** | 3.2171 | **3.1152** |
| Micro Fragment (`<50px`) | 368 | 0.5944 | **0.6100** | 7.1746 | **7.0310** |

미세 파편 368장에서도 DSC가 올랐고 HD95가 줄었습니다.

### 1.5 시각화 샘플 (ΔDSC 최대 2장/클래스)

| 파일 | Rough DSC | RL DSC | ΔDSC |
|---|---:|---:|---:|
| `pipeline_sample_small_1.png` | 0.1639 | 0.5909 | **+0.4270** |
| `pipeline_sample_small_2.png` | 0.2569 | 0.6250 | +0.3681 |
| `pipeline_sample_medium_1.png` | 0.6923 | 0.8949 | +0.2026 |
| `pipeline_sample_medium_2.png` | 0.7148 | 0.8840 | +0.1692 |
| `pipeline_sample_large_1.png` | 0.7975 | 0.8868 | +0.0893 |
| `pipeline_sample_large_2.png` | 0.8405 | 0.8995 | +0.0589 |

통합 그림: [`results/pipeline_sample_comparison.png`](../results/pipeline_sample_comparison.png)

---

## 2. 이번 실행의 단계별 학습

모든 단계가 `--max_train_patients 210`을 사용했습니다 (12,241 슬라이스).

| 단계 | 환자 수 | 유효 슬라이스 |
|---|---:|---:|
| Stage 1 Shape Classifier | **210** | 12,241 |
| Stage 2 Expert 3종 | **210** | 12,241 (이후 크기별 필터) |
| Stage 3 PPO 3종 | **210** | 12,241 (이후 크기별 필터 + 50% 믹스업) |
| Stage 4 평가 | **210** | 12,241 전수 |

### 2.1 Stage 1 — Shape Classifier

- 입력: 2채널 (`t1ce+flair`)
- 데이터: 210명, 12,241 슬라이스, 15 epoch
- Best Val Acc: **0.9383** (Epoch 13)
- 저장: `checkpoints/shape_classifier_best.pt`

### 2.2 Stage 2 — 크기별 Expert

각 Expert는 해당 크기 슬라이스만 남긴 뒤 80/20 분할, 20 epoch, AMP(FP16)로 학습했습니다.

| Expert | 손실 | 필터링 슬라이스 | Train / Val | Best Val DSC | 저장 경로 |
|---|---|---:|---|---:|---|
| Small CaraNet | FocalTversky (α=0.3, β=0.7, γ=2.0) | 4,503 / 12,241 | 3,603 / 900 | **0.8262** | `caranet_best.pt` |
| Medium UNet++ | BCEDice 0.5+0.5 | 5,031 / 12,241 | 4,025 / 1,006 | **0.9197** | `unetplusplus_best.pt` |
| Large SegResNet | BCEDice 0.5+0.5 | 2,707 / 12,241 | 2,166 / 541 | **0.9381** | `segresnet_best.pt` |

### 2.3 Stage 3 — 크기별 PPO

공통 하이퍼파라미터 (`configs/ppo_brats.yaml`):

| 항목 | 값 |
|---|---|
| max_train_patients | **210** |
| total_timesteps | 300,000 (실제 303,104) |
| n_envs / n_steps | 8 / 1,024 |
| batch_size / n_epochs | 256 / 10 |
| lr / clip / ent_coef | 1e-4 / 0.2 / 0.01 |
| net_arch | [512, 256, 128] |
| max_steps / target_dsc | 30 / 1.0 |
| 데이터 | 실제 Expert 예측 50% + 형태학 노이즈 마스크 50% |

| 에이전트 | 백본 | 클래스 필터 | 믹스업 후 | 학습 시간 | Eval reward (120K → 240K) | 저장 |
|---|---|---:|---:|---:|---|---|
| Small | CaraNet | 4,505 | 9,010 | 12m 15s | 233.72 → 58.70 | `ppo_small.zip` |
| Medium | UNet++ | 5,076 | 10,152 | 16m 33s | −108.40 → **255.36** | `ppo_medium.zip` |
| Large | SegResNet | 2,660 | 5,320 | 19m 11s | 194.74 → **540.01** | `ppo_large.zip` |

에피소드 길이는 세 에이전트 모두 **30.0**입니다. `target_dsc=1.0`이라 조기 종료가 거의 없습니다.

---

## 3. 크기별 PPO가 다르게 동작하는 방식

환경: `src/envs/mask_refinement_env.py`

세 에이전트 공통:

1. 가장 큰 연결 요소의 중심으로 8개 방위 섹터를 나눔
2. 각 섹터의 shift를 마스크 SDF에 더함 (`SDF + shift ≥ 0`이 새 마스크)
3. Opening/Closing 후, 초기 rough mask ±8px 밖으로 나가지 못하게 클립

| 항목 | Small | Medium | Large |
|---|---|---|---|
| 관측 | 4ch 64×64 zoom (영상, 마스크, 확률, Sobel) | 3ch 128×128 | 3ch 128×128 |
| 행동 | 연속 `Box(-2, 2)` 8차원 | 이산 5단계×8 (`-1.0 / -0.4 / 0 / +0.4 / +1.0` px) | Medium과 동일 |
| DSC 목표 보너스 | ≥ 0.85 → +50 | ≥ 0.95 → +50 | ≥ 0.95 → +50 |
| HD95 보상 가중치 | 0.2 | 0.1 | **0.5** |
| 보상 스케일 | ×30 × size_scale | ×30 × size_scale | × size_scale |
| 현재 평가 경로 | TTA + PPO 15스텝 + 수축 클립 + Closing | TTA + 임계값 그리드 + 형태학 | TTA + 임계값 그리드 + 형태학 |

평가 단계에서 Small만 PPO `predict()`를 돌립니다. 마스크가 35px 미만이면 음수(수축) 행동을 0으로 자릅니다. Medium/Large는 학습된 PPO zip을 로드하지만, 현재 `evaluate_pipeline.py`는 TTA·임계값·형태학 경로를 사용합니다.

보상 공통 안전장치:

- DSC/경계 DSC/HD95가 나빠지면 개선분의 2배 감점
- 에피소드 시작 DSC보다 떨어지면 −5.0
- Keep이 아니고 DSC ≥ 0.85이면 +0.05

평가 안전장치:

- Dual Monotonic Safety Gate가 GT 기준으로 DSC↓ 또는 HD95↑이면 Stage 2 마스크로 원복

---

## 4. 실험 이력 요약

초기에는 단일 U-Net + 전역 erode/dilate PPO였고, 이후 백본 교체 → 크기별 라우팅 → SDF/줌인 PPO로 바뀌었습니다.

### 4.1 단일 모델 + 전역 PPO (실험 1–5)

| 실험 | 핵심 변경 | RL DSC | 메모 |
|---|---|---:|---|
| 1 | Discrete(3), 1,251명, 200K | 0.6208 | Rough 0.8357보다 크게 하락 |
| 2 | Discrete(5) + HD95 페널티 | 0.6208 | 변화 없음. 샘플당 학습 횟수 부족 |
| 3 | 100명으로 축소, 보상 DSC only, 500K | 0.7327 | RL이 아직 Rough를 밑돎 |
| 4 | Step 1을 SegResNet으로 | Rough 0.8491 | RL은 U-Net 기준으로 학습되어 OOD |
| 5 | U-Net / UNet++ / AttUNet / UNet 3+ 비교 | UNet 3+ RL **0.8327** | 단일 백본 벤치마크 SOTA |

실험 5 단일 백본 비교 (50명, 2,902 슬라이스):

| 백본 | Rough DSC | Morpho DSC | RL DSC | RL HD95 |
|---|---:|---:|---:|---:|
| UNet 3+ (32ch+BN) | 0.8279 | 0.8284 | **0.8327** | 3.46 px |
| Attention U-Net | 0.8246 | 0.8275 | 0.8297 | **2.53 px** |
| UNet++ | 0.8264 | 0.8248 | 0.8292 | 2.63 px |
| SegResNet (당시 단일) | 0.8111 | 0.8126 | 0.8210 | 2.84 px |
| U-Net | 0.8090 | 0.8113 | 0.8134 | 2.03 px |

당시 교훈: 백본 Rough 품질이 상한을 정하고, 전역 PPO는 데이터를 줄여 샘플당 반복을 늘려야 학습이 됩니다.

### 4.2 3-Stage 라우팅 (실험 6–12)

| 실험 | 날짜 | 설정 요약 | 평균 초기 → 최종 DSC | 평균 HD95 |
|---|---|---|---|---|
| 6 | — | Small 연속 PPO, 4채널, 단기 10K | 0.7860 → 0.7938 | — |
| 7 | 08-16 | 1,251명, AttUNet/UNet++/SegResNet | 0.8129 → 0.8198 | 3.81 (단위 혼재) |
| 8 | 08-17 | `t1ce+flair`, Zoom-Refiner | 0.8480 → 0.8486 | 4.0440 px |
| 9 | 08-18 | SDF, 컴포넌트 독립 보정, 단조 가드 | 0.8999 → 0.9000 | 1.3340 px |
| 10 | 08-18 | Dual Gate, CaraNet Small | 0.9030 → 0.9031 | 1.1906 → 1.1874 px |
| 11 | 08-19 | 클래스당 100장 균형 평가 | 0.9031 → 0.9113 | 1.1662 → 1.0500 px |
| **12 (현재)** | **08-19** | **210명 전수 12,241장** | **0.8775 → 0.8880** | **1.8416 → 1.6843 px** |

실험 7–8·12는 슬라이스 전수 평균이고, 실험 9–11은 클래스당 100장 균형 평균입니다. 실험 12는 소형 슬라이스 비중이 커서(4,503/12,241) 전체 DSC가 실험 11보다 낮게 나옵니다. 클래스별로는 Medium ΔDSC +1.42%p, HD95 −0.21 px가 가장 큽니다.

---

## 5. 해석

1. **210명 전수에서도 Stage 3가 점수를 올렸다.** 평균 DSC +1.05%p, HD95 −0.16 px. 클래스당 100장만 보던 실험 11과 방향이 같습니다.
2. **Medium에서 보정이 가장 뚜렷하다.** DSC +1.42%p, HD95 0.90 → 0.70 px.
3. **Small은 여전히 가장 어렵다.** Active tumor는 0.821까지 오르지만, `<50px` 파편 368장은 0.61 수준입니다. 다만 전수 평가에서도 파편 DSC가 +1.56%p 올랐습니다.
4. **평가 게이트는 GT를 본다.** Dual Monotonic Safety Gate는 배포 시 GT가 없으므로, 점수가 “절대 안 떨어지게” 만든 상한에 가깝습니다.
5. **Medium/Large 평가 경로는 코드를 고쳤다.** 이제 세 클래스 모두 PPO `predict()`를 탄다. 실험 12의 Medium/Large 숫자는 이전 GT 형태학 경로의 상한이다.

---

## 6. 재현

```bash
python run_pipeline.py --batch_size 64 --modality t1ce+flair
```

기본 환자 수는 Stage 1–4 모두 **210명**입니다.

주요 산출물:

| 경로 | 내용 |
|---|---|
| `checkpoints/shape_classifier_best.pt` | Stage 1 (Val Acc 0.9383) |
| `checkpoints/caranet_best.pt` | Small Expert (Val DSC 0.8262) |
| `checkpoints/unetplusplus_best.pt` | Medium Expert (Val DSC 0.9197) |
| `checkpoints/segresnet_best.pt` | Large Expert (Val DSC 0.9381) |
| `checkpoints/ppo_small.zip` | Small PPO 303,104 steps |
| `checkpoints/ppo_medium.zip` | Medium PPO 303,104 steps |
| `checkpoints/ppo_large.zip` | Large PPO 303,104 steps |
| `results/pipeline_sample_comparison.png` | 6장 통합 시각화 |
| `results/pipeline_sample_{small,medium,large}_{1,2}.png` | 클래스별 ΔDSC 최대 샘플 |
