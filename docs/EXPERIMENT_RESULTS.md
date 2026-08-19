# RL-Refiner 실험 결과

> 실험 이력과 최종 벤치마크를 한 파일로 정리한 문서입니다.
> 이전 `EXPERIMENTS.md`, `final_models_report.md`, `technical_report.md`의 내용을 통합했습니다.

- **데이터셋**: BraTS 2021 Task 1
- **입력**: `t1ce+flair`, 128×128 2D 슬라이스
- **지표**: DSC (↑), HD95 px (↓), Precision, Recall — 구현은 `src/utils/metrics.py`
- **최신 실행**: 2026-08-19 `python run_pipeline.py batch_size=64` (seed 42, 결정적 모드 ON)
- **파이프라인 구조**: [PIPELINE.md](PIPELINE.md) / **베이스라인**: [baselines/README.md](../baselines/README.md)

---

## 1. 최종 결과 (2026-08-19, val hold-out)

### 1.1 평가 설정

| 항목 | 값 |
|---|---|
| 실행 명령 | `python run_pipeline.py batch_size=64` |
| 환자 풀 | 210명 → **train 168명 / val 42명** (`checkpoints/patient_split.json`, seed 42) |
| 학습 슬라이스 | 9,868 |
| **평가 슬라이스** | **2,373** (val 환자 전수) |
| 클래스 라우팅 | **Stage 1 분류기 예측** (Oracle 아님) |
| Stage 2 이진화 임계값 | **0.80 / 0.80 / 0.60** (Small / Medium / Large) |
| Stage 3 | 세 클래스 **모두** PPO `predict()` 15스텝 |
| 안전장치 | GT-free 면적 게이트(0.2×–4×) + **Monotonic DSC Gate**(DSC만 판정) |
| 시드 / 결정적 모드 | 42 / ON |

이전 문서의 **Dual Monotonic Safety Gate**(DSC 하락 **또는 HD95 증가** 시 원복)는 현재 코드에 없습니다. 지금은 DSC만 판정합니다.

### 1.2 파이프라인 종합

| 지표 | Stage 2 초기 | Stage 3 보정 후 | 변화 |
|---|---:|---:|---:|
| 평균 DSC | 0.8672 | **0.8752** | **+0.0080 (+0.80%p)** |
| 평균 HD95 | 2.3365 px | **2.2897 px** | **−0.0468 px** |

세 클래스 모두 DSC가 올랐고 HD95가 줄었습니다.

### 1.3 크기별 성능

분류기 라우팅 결과 분포: Small 922 / Medium 935 / Large 516 (GT 기준은 942 / 847 / 584).

| 클래스 | Expert | n | 초기 DSC | 최종 DSC | ΔDSC | 초기 HD95 | 최종 HD95 | ΔHD95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Small (`<300px`)** | CaraNet | 922 | 0.7753 | **0.7892** | +0.0139 | 4.4593 | **4.4459** | −0.0134 |
| **Medium (`300–700px`)** | UNet++ | 935 | 0.9223 | **0.9270** | +0.0047 | 1.0764 | **1.0170** | −0.0594 |
| **Large (`≥700px`)** | SegResNet | 516 | 0.9316 | **0.9347** | +0.0031 | 0.8271 | **0.7432** | −0.0839 |

### 1.4 Precision / Recall

`P < R`이면 과분할(경계가 GT 밖으로 번짐), `P > R`이면 과소분할(종양을 놓침)입니다.

| 클래스 | 초기 P | 최종 P | 초기 R | 최종 R | 해석 |
|---|---:|---:|---:|---:|---|
| Small | 0.7797 | **0.8044** | 0.8232 | 0.8199 | 과분할. PPO가 Precision을 +2.5%p 올려 균형에 접근 |
| Medium | 0.9385 | 0.9394 | 0.9176 | **0.9249** | 약한 과소분할. PPO가 Recall을 올림 |
| Large | 0.9017 | 0.9052 | 0.9730 | 0.9751 | **강한 과분할** (격차 0.07). 임계값을 더 올릴 여지 |

임계값을 0.50에서 0.80/0.80/0.60으로 올린 뒤에도 Small과 Large는 여전히 과분할입니다.

### 1.5 Small 층화

| 구간 | n | 초기 DSC | 최종 DSC | 초기 HD95 | 최종 HD95 |
|---|---:|---:|---:|---:|---:|
| Active Tumor (`≥50px`) | 851 | 0.7928 | **0.8039** | 4.1623 | **4.1440** |
| Micro Fragment (`<50px`) | 71 | 0.5644 | **0.6128** | 8.0184 | 8.0642 |

미세 파편은 DSC가 +4.8%p 올랐지만 **HD95는 늘었습니다**. 게이트가 DSC만 보므로 DSC를 올리면서 경계 최악값을 악화시키는 보정이 통과할 수 있습니다.

### 1.6 Monotonic DSC Gate 발동률

| 항목 | 값 |
|---|---:|
| 평가 슬라이스 | 2,373 |
| 게이트 발동 (최종 < 초기 → Stage 2 유지) | **858 (36.2%)** |

PPO는 슬라이스 3분의 1 이상에서 DSC를 떨어뜨렸고, 보고된 최종 점수는 그 손실을 **GT로 걸러낸** 값입니다. 게이트는 GT를 요구하므로 배포 시에는 쓸 수 없으며, 최종 DSC는 상한으로 읽어야 합니다.

### 1.7 시각화 샘플 (ΔDSC 최대 2장/클래스)

| 파일 | Rough DSC | RL DSC | ΔDSC |
|---|---:|---:|---:|
| `pipeline_sample_small_1.png` | 0.4941 | 0.7222 | **+0.2281** |
| `pipeline_sample_small_2.png` | 0.6825 | 0.8944 | +0.2118 |
| `pipeline_sample_medium_1.png` | 0.7792 | 0.8453 | +0.0661 |
| `pipeline_sample_medium_2.png` | 0.8675 | 0.9310 | +0.0635 |
| `pipeline_sample_large_1.png` | 0.7960 | 0.8578 | +0.0618 |
| `pipeline_sample_large_2.png` | 0.8152 | 0.8612 | +0.0460 |

통합 그림: [`results/pipeline_sample_comparison.png`](../results/pipeline_sample_comparison.png)

개선폭이 가장 큰 표본이므로 평균을 대표하지 않습니다.

---

## 2. 베이스라인 비교 (BraTS 2021 상위 입상 방법)

같은 42명 val 환자, 같은 2,373 슬라이스, 같은 지표 구현으로 측정했습니다. 구현은 `baselines/`에 있습니다.

### 2.1 전체 비교

| 방법 | DSC | HD95 (px) | Precision | Recall |
|---|---:|---:|---:|---:|
| 파이프라인 Stage 2 (게이트 없음) | 0.8672 | 2.3365 | — | — |
| 파이프라인 Stage 3 (Monotonic DSC Gate, **GT 사용**) | 0.8752 | 2.2897 | — | — |
| Extending nnU-Net (KAIST, 최종 테스트 1위) | 0.8923 | 1.9591 | 0.9215 | 0.8893 |
| SegResNet + 중복 감소 (NVAUTO, 최종 2위) | **0.8971** | **1.9022** | 0.9077 | 0.9029 |

**파이프라인은 GT 게이트를 켜고도 두 베이스라인보다 낮습니다.** GT를 쓰지 않는 Stage 2 기준으로는 격차가 −0.025에서 −0.030까지 벌어집니다.

### 2.2 크기별 비교

베이스라인은 GT 크기 구간(942 / 847 / 584)으로, 파이프라인은 분류기 라우팅 결과(922 / 935 / 516)로 집계했습니다.

| 클래스 | 파이프라인 초기 → 최종 | KAIST | NVAUTO |
|---|---|---:|---:|
| Small | 0.7753 → 0.7892 | 0.8278 | **0.8377** |
| Medium | 0.9223 → **0.9270** | 0.9165 | 0.9200 |
| Large | 0.9316 → 0.9347 | **0.9614** | 0.9594 |

HD95 (px):

| 클래스 | KAIST | NVAUTO |
|---|---:|---:|
| Small | 3.8017 | **3.3599** |
| Medium | **1.0619** | 1.2052 |
| Large | **0.2881** | 0.5618 |

Medium만 파이프라인이 앞섭니다. **Large 격차(−0.027)가 가장 크고**, HD95는 0.74 vs 0.29로 2배 이상 벌어집니다. 큰 종양에서는 단일 강한 백본이 유리하고, 그 위에 분류기 오분류(Val Acc 0.85)가 얹힙니다.

### 2.3 임계값 스윕

베이스라인만 0.5로 두고 파이프라인은 조정하면 불공정하므로 양쪽 모두 스윕했습니다.

| 방법 | 최적 임계값 | 최적 DSC | 0.5에서의 DSC | 이득 |
|---|---:|---:|---:|---:|
| KAIST | 0.30 | 0.8932 | 0.8923 | +0.0009 |
| NVAUTO | 0.40 | 0.8972 | 0.8971 | +0.0001 |

두 베이스라인은 임계값에 거의 둔감합니다(전 구간 변동 0.001 내외). 반대로 파이프라인 Expert는 임계값을 0.5에서 0.8로 올릴 때 이득이 컸는데, 이는 베이스라인의 확률 보정이 더 잘 돼 있다는 뜻으로 읽는 편이 맞습니다.

### 2.4 각색에 대한 주의

두 베이스라인은 원 논문의 재현이 아니라 **2D 각색**입니다. 3D 패치·4모달리티·5-fold 앙상블·BraTS 후처리를 모두 제외했으므로 대회 리더보드 점수(예: NVAUTO의 WT DSC 0.9265)와 직접 비교할 수 없습니다. 상세 차이표는 [baselines/README.md](../baselines/README.md)에 있습니다.

### 2.5 산출 그림

| 파일 | 내용 |
|---|---|
| `baselines/results/baseline_by_size.png` | 크기 구간별 DSC / HD95 막대그래프 |
| `baselines/results/baseline_threshold_sweep.png` | 임계값별 DSC / Precision / Recall 곡선 |
| `baselines/results/baseline_samples_median.png` | 중앙값 DSC 부근 정성 표본 |
| `baselines/results/baseline_samples_worst.png` | 최악 사례 정성 표본 |

정성 표본은 체리피킹을 피하려고 최고 사례가 아니라 **중앙값 부근**과 **최악**을 골랐습니다.

---

## 3. 이번 실행의 단계별 학습

| 단계 | 환자 | 학습 슬라이스 | 검증 슬라이스 |
|---|---:|---:|---:|
| Stage 1 Shape Classifier | 168 / 42 | 9,868 | 2,373 |
| Stage 2 Expert 3종 | 168 / 42 | 크기별 필터 (아래) | 크기별 필터 |
| Stage 3 PPO 3종 | 168 / 42 | 크기별 필터 + 50% 믹스업 | 크기별 필터 (hold-out) |
| Stage 4 평가 | 42 | — | 2,373 전수 |

GT 크기 구간 분포:

| 역할 | 슬라이스 | Small | Medium | Large |
|---|---:|---:|---:|---:|
| train | 9,868 | 3,561 | 4,184 | 2,123 |
| val | 2,373 | 942 | 847 | 584 |

### 3.1 Stage 1 — Shape Classifier

| 항목 | 값 |
|---|---|
| 모델 | ResNet18, `conv1` 2채널, `fc` 3클래스 |
| 초기화 | **ImageNet 사전학습** (기본). `--no_pretrained`로 해제 |
| 옵티마이저 | **AdamW** (lr 1e-3, weight decay 1e-4) + **CosineAnnealingLR** |
| 에폭 / 배치 | 15 / 64 |
| 분할 | **환자 단위** |
| Best Val Acc | **0.8512** |
| 저장 | `checkpoints/shape_classifier_best.pt` |

사전학습 ablation:

| 초기화 | Best Val Acc | 체크포인트 |
|---|---:|---|
| ImageNet 사전학습 | **0.8512** | `shape_classifier_best.pt` |
| 무작위 (scratch) | 0.8424 | `shape_classifier_scratch_backup.pt` |

이득이 +0.9%p에 그쳤고 Train Acc는 0.96–0.99까지 올라 과적합이 남습니다. 병목은 특징 품질이 아니라 **면적 300 / 700 경계 부근의 레이블 모호성**입니다. 몇 픽셀 차이로 클래스가 갈리므로 분류 문제로는 줄일 수 없는 잡음이 있습니다.

> **과거 수치 정정:** 이전 문서의 Val Acc **0.9383**은 **슬라이스 단위 80/20 분할**로 측정한 값입니다. 같은 환자의 인접 슬라이스가 train과 val에 함께 들어가 누출이 있었으므로 현재 환자 단위 수치와 비교할 수 없습니다.

### 3.2 Stage 2 — 크기별 Expert

공통: 20 epoch, batch 64, Adam lr 3e-4, CosineAnnealingLR, AMP(FP16), 환자 단위 분할 + GT 면적 필터.

| Expert | 손실 | train / val 슬라이스 | 저장 경로 |
|---|---|---:|---|
| Small CaraNet | FocalTversky (α=0.3, β=0.7, γ=2.0) | 3,561 / 942 | `caranet_best.pt` |
| Medium UNet++ | BCEDice 0.5+0.5 | 4,184 / 847 | `unetplusplus_best.pt` |
| Large SegResNet | BCEDice 0.5+0.5 | 2,123 / 584 | `segresnet_best.pt` |

Small에 `β > α`인 FocalTversky를 쓰는 이유는 미검출(FN)에 더 큰 벌점을 주기 위함입니다.

Expert 단독 성능은 이진화 임계값에 좌우되므로, val에서의 실측치는 §1.3의 **초기 DSC**(Small 0.7753 / Medium 0.9223 / Large 0.9316)를 봅니다. 이 값은 임계값 0.80/0.80/0.60과 TTA를 적용한 뒤의 수치입니다.

> **과거 수치 정정:** 이전 문서의 Expert Val DSC 0.8262 / 0.9197 / 0.9381은 크기별로 필터한 뒤 **슬라이스 단위 80/20**으로 나눈 학습 중 최고값입니다. 환자 단위 hold-out 수치와 직접 비교할 수 없습니다.

### 3.3 Stage 3 — 크기별 PPO

공통 하이퍼파라미터 (`configs/ppo_brats.yaml`):

| 항목 | 값 |
|---|---|
| max_train_patients | **210** |
| total_timesteps | 300,000 (실제 303,104) |
| n_envs / n_steps | 8 / 1,024 |
| batch_size / n_epochs | 256 / 10 |
| lr / clip / ent_coef | 1e-4 / 0.2 / 0.01 |
| gamma / GAE λ | 0.99 / 0.95 |
| net_arch | [512, 256, 128] |
| max_steps / target_dsc | 30 / 1.0 |
| step_penalty | 0.001 |
| 학습 데이터 | 실제 Expert 예측 50% + 형태학 노이즈 마스크 50% |
| 평가 환경 | **val 환자 hold-out**, 믹스업 없음 |

조기 종료 장치:

| 장치 | 설정 | 이유 |
|---|---|---|
| `stop_dsc_target` | **0 (비활성화)** | Medium/Large rough DSC가 이미 기본 임계값 0.92를 넘어 첫 평가(2,048 스텝)에서 즉시 중단됐다 |
| `stop_plateau` | **false (비활성화)** | 세 에이전트 모두 완주시킨다 |
| Milestone Snapshot | 25 / 50 / 75% | `checkpoints/snapshots/{small,medium,large}/`에 **클래스별 하위 폴더** 저장 |

과거에는 스냅샷이 클래스와 무관한 한 폴더에 저장돼 마지막 에이전트가 앞선 것을 덮어썼습니다. 지금은 `refinement_mode`별 하위 폴더로 분리됩니다.

| 에이전트 | 백본 | 클래스 필터 (train) | 믹스업 후 | val hold-out | 저장 |
|---|---|---:|---:|---:|---|
| Small | CaraNet | 3,561 | 7,122 | 942 | `ppo_small.zip` |
| Medium | UNet++ | 4,184 | 8,368 | 847 | `ppo_medium.zip` |
| Large | SegResNet | 2,123 | 4,246 | 584 | `ppo_large.zip` |

Large 에이전트 상세: 25m45s, eval 보상 120K **375.20** → 240K **387.60**, 에피소드 길이 30.0 고정. `target_dsc=1.0`이라 조기 종료가 거의 없습니다.

최종 모델은 eval 보상이 가장 좋았던 `checkpoints/best_{mode}/best_model.zip`을 복사한 것입니다.

---

## 4. 크기별 PPO가 다르게 동작하는 방식

환경: `src/envs/mask_refinement_env.py`

세 에이전트 공통:

1. 가장 큰 연결 요소의 중심으로 8개 방위 섹터를 나눔
2. 각 섹터의 shift를 마스크 SDF에 더함 (`SDF + shift ≥ 0`이 새 마스크)
3. Closing/Opening 후, 초기 rough mask ±8px 밖으로 나가지 못하게 클립

| 항목 | Small | Medium | Large |
|---|---|---|---|
| 관측 | 4ch 64×64 zoom (영상, 마스크, 확률, Sobel) | 3ch 128×128 | 3ch 128×128 |
| 행동 | 연속 `Box(-2, 2)` 8차원 | 이산 5단계×8 (`-1.0 / -0.4 / 0 / +0.4 / +1.0` px) | Medium과 동일 |
| DSC 목표 보너스 | ≥ 0.85 → +50 | ≥ 0.95 → +50 | ≥ 0.95 → +50 |
| HD95 보상 가중치 | 0.2 | 0.1 | **0.5** |
| 보상 스케일 | ×30 × size_scale | ×30 × size_scale | × size_scale |

관측은 영상·마스크·확률·에지로만 구성되며 **GT를 포함하지 않습니다.** GT는 보상 계산에만 쓰입니다.

보상 공통 안전장치:

- DSC / 경계 DSC(GT ±3px 밴드) / HD95가 나빠지면 개선분의 2배 감점
- 에피소드 시작 DSC보다 떨어지면 −5.0
- Keep이면서 DSC ≥ 0.85이면 +0.05
- DSC 목표 보너스는 조건을 만족하는 **매 스텝** 부여

평가 단계에서는 **세 클래스 모두** PPO `predict()`를 15스텝 돌립니다. Small은 마스크가 35px 미만이면 음수(수축) 행동을 0으로 자릅니다.

> **과거 서술 정정:** 이전 문서에는 "평가에서 Small만 PPO를 돌리고 Medium/Large는 TTA·임계값·형태학 경로를 쓴다"고 적혀 있었습니다. 현재 코드는 세 클래스 모두 PPO를 사용합니다.

---

## 5. 실험 이력 요약

초기에는 단일 U-Net + 전역 erode/dilate PPO였고, 이후 백본 교체 → 크기별 라우팅 → SDF/줌인 PPO → 환자 단위 분할·임계값 조정으로 바뀌었습니다.

### 5.1 단일 모델 + 전역 PPO (실험 1–5)

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

### 5.2 3-Stage 라우팅 (실험 6–13)

| 실험 | 날짜 | 설정 요약 | 평균 초기 → 최종 DSC | 평균 HD95 |
|---|---|---|---|---|
| 6 | — | Small 연속 PPO, 4채널, 단기 10K | 0.7860 → 0.7938 | — |
| 7 | 08-16 | 1,251명, AttUNet/UNet++/SegResNet | 0.8129 → 0.8198 | 3.81 (단위 혼재) |
| 8 | 08-17 | `t1ce+flair`, Zoom-Refiner | 0.8480 → 0.8486 | 4.0440 px |
| 9 | 08-18 | SDF, 컴포넌트 독립 보정, 단조 가드 | 0.8999 → 0.9000 | 1.3340 px |
| 10 | 08-18 | Dual Gate, CaraNet Small | 0.9030 → 0.9031 | 1.1906 → 1.1874 px |
| 11 | 08-19 | 클래스당 100장 균형 평가 | 0.9031 → 0.9113 | 1.1662 → 1.0500 px |
| 12 | 08-19 | 210명 **전수** 12,241장, Oracle 라우팅, Dual Gate | 0.8775 → 0.8880 | 1.8416 → 1.6843 px |
| **13 (현재)** | **08-19** | **val 42명 hold-out 2,373장, 분류기 라우팅, 임계값 0.8/0.8/0.6, 시드 고정** | **0.8672 → 0.8752** | **2.3365 → 2.2897 px** |

**실험 12와 13은 직접 비교할 수 없습니다.** 프로토콜이 세 군데에서 바뀌었습니다.

| 항목 | 실험 12 | 실험 13 |
|---|---|---|
| 평가 집합 | 210명 **전수** (학습 데이터 포함) | **val 42명만** (hold-out) |
| 라우팅 | **Oracle** (GT 면적) | **분류기 예측** |
| 게이트 | Dual (DSC + HD95) + Medium/Large GT 형태학 | Monotonic **DSC만** |

실험 13의 점수가 낮은 것은 성능 퇴행이 아니라 **누출과 Oracle 이점을 제거한 결과**입니다. 실험 7–8·12는 슬라이스 전수 평균, 실험 9–11은 클래스당 100장 균형 평균입니다.

### 5.3 실험 13에서 고친 것

| 항목 | 이전 | 현재 |
|---|---|---|
| 분할 | 슬라이스 단위 80/20 (같은 환자가 양쪽에) | **환자 단위** 168 / 42 |
| 데이터 로딩 | `max_patients`로 "정렬 순 앞 N명" 로드 → 무작위 표본과 불일치 | 분할을 먼저 확정한 뒤 해당 환자 ID만 로드 |
| 라우팅 | Oracle (GT 면적) | 분류기 예측 |
| 이진화 임계값 | 고정 0.50 | 클래스별 0.80 / 0.80 / 0.60 |
| 미세 파편 처리 | 0.15–0.45 고정 후보 + 점수 휴리스틱 (임계값 0.80과 단절) | `stage2_threshold`에서 0.05씩 내려가는 **사다리** |
| 게이트 | Dual (DSC + HD95) | Monotonic DSC만 |
| Medium/Large 평가 | TTA + 임계값 그리드 + GT 형태학 | PPO `predict()` 15스텝 |
| 지표 | DSC, HD95 | + **Precision, Recall** |
| PPO 스냅샷 | 한 폴더 공유 → 덮어씀 | `snapshots/{mode}/` 분리 |
| PPO 조기 종료 | `stop_dsc_target 0.92`로 즉시 중단 | 비활성화(0), plateau도 false |
| 재현성 | 시드 미고정 | seed 42 + 결정적 모드 기본 ON |
| 분류기 | scratch ResNet18 + Adam | ImageNet 사전학습 + AdamW + Cosine |

---

## 6. 해석

1. **Stage 3는 여전히 점수를 올리지만 폭이 작아졌다.** 평균 DSC +0.80%p, HD95 −0.05 px. 실험 12의 +1.05%p보다 작은데, Oracle 라우팅과 GT 형태학 이점이 빠졌기 때문입니다.
2. **게이트가 없으면 PPO의 순효과는 불확실하다.** 2,373장 중 **858장(36.2%)** 에서 PPO가 DSC를 떨어뜨렸고, 그 손실은 GT 게이트로 걸러졌습니다. 배포 환경에서는 이 게이트를 쓸 수 없습니다.
3. **베이스라인이 더 강하다.** 같은 val 환자에서 KAIST 0.8923, NVAUTO 0.8971로, 파이프라인의 게이트 적용 후 0.8752보다 높습니다. Medium 구간만 파이프라인이 앞섭니다.
4. **Large가 가장 약한 고리다.** DSC 0.9347 vs 베이스라인 0.9594–0.9614, HD95 0.74 vs 0.29–0.56. Precision 0.905 / Recall 0.975로 과분할이 심해 임계값을 더 올릴 여지가 있습니다.
5. **분류기가 라우팅 상한을 제한한다.** Val Acc 0.8512이므로 슬라이스 약 15%가 엉뚱한 Expert로 갑니다. 면적 300 / 700 경계의 레이블 모호성이 원인이라 분류 정확도만 올리는 방식으로는 한계가 있습니다.
6. **임계값 조정 이득이 PPO 이득에 필적한다.** 0.50 → 0.80/0.80/0.60 변경만으로 DSC가 PPO 보정분과 비슷한 폭으로 올랐습니다. 반대로 두 베이스라인은 임계값에 거의 둔감합니다(변동 0.001 내외).
7. **Small은 여전히 가장 어렵다.** Active tumor는 0.804까지 오르지만 `<50px` 파편 71장은 0.61 수준이고, 이 구간에서는 HD95가 오히려 늘었습니다.

---

## 7. 재현

```bash
python run_pipeline.py batch_size=64
```

기본 환자 풀은 210명(train 168 / val 42)이고, seed 42 · 결정적 모드가 기본입니다.

```bash
# 평가만
python scripts/eval/evaluate_pipeline.py --split_role val

# Stage 2 단독 (베이스라인과 비교용)
python scripts/eval/evaluate_pipeline.py --split_role val --skip_ppo

# 임계값 스윕
python scripts/eval/sweep_threshold.py --split_role val --plot results/threshold_sweep.png

# 베이스라인
python baselines/run_comparison.py --epochs 20 --batch_size 32 --sweep
```

주요 산출물:

| 경로 | 내용 |
|---|---|
| `checkpoints/patient_split.json` | 환자 분할 (train 168 / val 42, seed 42) |
| `checkpoints/shape_classifier_best.pt` | Stage 1 (Val Acc 0.8512) |
| `checkpoints/shape_classifier_scratch_backup.pt` | Stage 1 사전학습 없음 ablation (0.8424) |
| `checkpoints/caranet_best.pt` | Small Expert |
| `checkpoints/unetplusplus_best.pt` | Medium Expert |
| `checkpoints/segresnet_best.pt` | Large Expert |
| `checkpoints/ppo_{small,medium,large}.zip` | PPO 303,104 steps |
| `checkpoints/snapshots/{mode}/` | 25 / 50 / 75% 마일스톤 |
| `results/pipeline_sample_comparison.png` | 6장 통합 시각화 |
| `results/pipeline_sample_{small,medium,large}_{1,2}.png` | 클래스별 ΔDSC 최대 샘플 |
| `results/threshold_sweep.png` | 임계값 스윕 곡선 |
| `baselines/results/comparison.md` | 베이스라인 비교 표 |
| `baselines/results/{kaist,nvauto}_metrics.json` | 구간별 지표 + 스윕 |
| `baselines/results/baseline_*.png` | 베이스라인 정량·정성 그림 |
