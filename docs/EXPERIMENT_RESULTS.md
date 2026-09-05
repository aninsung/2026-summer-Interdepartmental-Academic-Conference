# RL-Refiner 실험 결과

> **논문 보고 프로토콜 (중요).** 메인 숫자는 **Stage 2 DSC 0.8948 / HD95 1.7336 px**이다.
> 아래 표의 Stage 3 Final **0.9031**은 GT 단조 게이트 + best-of-15 **상한**이며 배포·논문 본표에 쓰지 않는다.
> 배포 평가는 `python scripts/eval/evaluate_pipeline.py --split_role val` (기본 deploy) 또는 `--deploy_mode`.
> 옛 상한 재현만 `--gt_upper_bound`.

> 실험 이력과 최종 벤치마크를 한 파일로 정리한 문서입니다.
> 이전 `EXPERIMENTS.md`, `final_models_report.md`, `technical_report.md`의 내용을 통합했습니다.

- **데이터셋**: BraTS 2021 Task 1
- **입력**: `t1ce+flair`, 128×128 2D 슬라이스
- **지표**: DSC (↑), HD95 px (↓), Precision, Recall — 구현은 `src/utils/metrics.py`
- **최신 실행**: 2026-08-20 `python run_pipeline.py batch_size=64` (seed 42, 결정적 모드 ON) — 당시 Stage 3는 GT 상한 프로토콜
- **비교 그림 표기**: 파이프라인 = **TRIO** (논문 비교는 Stage 2 또는 배포형 Stage 3)
- **파이프라인 구조**: [PIPELINE.md](PIPELINE.md) / **베이스라인**: [baselines/README.md](../baselines/README.md)

---

## 1. 최종 결과 (2026-08-20, val hold-out)

### 1.1 평가 설정

| 항목 | 값 |
|---|---|
| 실행 명령 | `python run_pipeline.py batch_size=64` |
| 환자 풀 | 210명 → **train 168명 / val 42명** (`checkpoints/patient_split.json`, seed 42) |
| 학습 슬라이스 | 9,868 |
| **평가 슬라이스** | **2,434** (val 환자 전수, 분류기 라우팅) |
| 클래스 라우팅 | **Stage 1 분류기 예측** (Oracle 아님) |
| Stage 2 이진화 임계값 | **0.80 / 0.80 / 0.50** (Small / Medium / Large) |
| CC filter | **0 / 15 / 25** (Small 끔) |
| Small Expert | CaraNet **2.5D** + `<50px` ×4 오버샘플 + zoom-crop, BCEDice |
| Large Expert | 전 구간 학습 + **ED/TC** 2채널 → WT 합집합 |
| Stage 3 | 세 클래스 **모두** PPO `predict()` 15스텝. 초안 마스크는 평가와 같은 임계값으로 이진화 |
| 안전장치 | **GT-free 면적**(0.85×–1.2×, 배포) · Large Stage3 skip · **Monotonic DSC Gate**(GT, 상한 전용) |
| Confidence skip | 기본 **OFF** (`--confidence_threshold` 미지정) |
| 시드 / 결정적 모드 | 42 / ON |

이전 문서의 **Dual Monotonic Safety Gate**(DSC 하락 **또는 HD95 증가** 시 원복)는 현재 코드에 없습니다. 확률–에지 정합 Fallback도 없습니다. 지금은 면적 게이트와 DSC 단조 게이트만 기본으로 켭니다. 상세는 [PIPELINE.md §8.5](PIPELINE.md)를 봅니다.

### 1.2 파이프라인 종합

| 지표 | Stage 2 초기 | Stage 3 보정 후 | 변화 |
|---|---:|---:|---:|
| 평균 DSC | 0.8948 | **0.9031** | **+0.0083 (+0.83%p)** |
| 평균 HD95 | 1.7336 px | **1.6060 px** | **−0.1276 px** |

세 클래스 모두 DSC가 올랐고 HD95가 줄었습니다.

### 1.3 크기별 성능

분류기 라우팅 결과 분포: Small 897 / Medium 1,152 / Large 385.

| 클래스 | Expert | n | 초기 DSC | 최종 DSC | ΔDSC | 초기 HD95 | 최종 HD95 | ΔHD95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Small (`<300px`)** | CaraNet 2.5D | 897 | 0.8286 | **0.8429** | +0.0143 | 3.1618 | **2.9482** | −0.2136 |
| **Medium (`300–700px`)** | UNet++ | 1,152 | 0.9264 | **0.9314** | +0.0050 | 1.0790 | **1.0000** | −0.0790 |
| **Large (`≥700px`)** | SegResNet ED/TC | 385 | 0.9546 | **0.9582** | +0.0036 | 0.3650 | **0.2919** | −0.0731 |

### 1.4 Precision / Recall

`P < R`이면 과분할(경계가 GT 밖으로 번짐), `P > R`이면 과소분할(종양을 놓침)입니다.

| 클래스 | 초기 P | 최종 P | 초기 R | 최종 R | 해석 |
|---|---:|---:|---:|---:|---|
| Small | 0.9005 | **0.9195** | 0.7958 | 0.8008 | **과소분할**. PPO가 Precision을 더 올림 |
| Medium | 0.9529 | 0.9545 | 0.9089 | **0.9165** | 약한 과소분할. PPO가 Recall을 올림 |
| Large | 0.9596 | 0.9613 | 0.9515 | **0.9568** | 거의 균형. 임계값 0.50이 Recall을 회복 |

Large 임계값을 0.70에서 0.50으로 내린 뒤 P/R이 맞춰졌습니다.

### 1.5 Small 층화

| 구간 | n | 초기 DSC | 최종 DSC | 초기 HD95 | 최종 HD95 |
|---|---:|---:|---:|---:|---:|
| Active Tumor (`≥50px`) | 828 | 0.8427 | **0.8529** | 2.8934 | **2.7456** |
| Micro Fragment (`<50px`) | 69 | 0.6595 | **0.7239** | 6.3828 | **5.3794** |

미세 파편 DSC +6.44%p, HD95도 줄었습니다.

### 1.6 안전 가드와 Monotonic DSC 발동률

기본 평가 순서: Stage3(SL, Large skip) → 면적 게이트(0.85×–1.2×). 단조 DSC 게이트는 `--gt_upper_bound` 상한만. Confidence skip은 끄고 돌렸습니다.

| 항목 | 값 |
|---|---:|
| 평가 슬라이스 | 2,434 |
| 단조 게이트 발동 (최종 DSC &lt; 초기 → Stage 2 유지) | **858 (35.3%)** |

PPO는 슬라이스 3분의 1 이상에서 DSC를 떨어뜨렸고, 보고된 최종 점수는 그 손실을 **GT로 걸러낸** 값입니다. “클래스별 점수 하락 0%”는 이 게이트 이후의 이야기이며, 게이트 없는 Stage 2가 배포에 가까운 숫자입니다.

### 1.7 시각화 샘플 (클래스 평균 Final DSC에 가깝고 Final > Initial)

| 파일 | Rough DSC | RL DSC | ΔDSC |
|---|---:|---:|---:|
| `pipeline_sample_small_1.png` | 0.8322 | 0.8429 | +0.0106 |
| `pipeline_sample_small_2.png` | 0.8127 | 0.8432 | +0.0305 |
| `pipeline_sample_medium_1.png` | 0.9003 | 0.9315 | +0.0313 |
| `pipeline_sample_medium_2.png` | 0.9209 | 0.9316 | +0.0107 |
| `pipeline_sample_large_1.png` | 0.9535 | 0.9582 | +0.0047 |
| `pipeline_sample_large_2.png` | 0.9520 | 0.9583 | +0.0063 |

통합 그림: [`results/pipeline_sample_comparison.png`](../results/pipeline_sample_comparison.png). 원본 MRI는 `*_original.png`.

---

## 2. 베이스라인 비교 (BraTS 2021 상위 입상 방법)

같은 42명 val 환자, 같은 지표 구현으로 측정했습니다. 베이스라인은 2,373장, 이번 파이프라인은 2,434장이라 슬라이스 수가 완전히 같지는 않습니다. 구현은 `baselines/`에 있습니다.

### 2.1 전체 비교

| 방법 | DSC | HD95 (px) | Precision | Recall |
|---|---:|---:|---:|---:|
| TRIO Stage 2 (게이트 없음) | 0.8948 | 1.7336 | — | — |
| **TRIO** Stage 3 (Monotonic DSC Gate, **GT 사용**) | **0.9031** | **1.6060** | — | — |
| Extending nnU-Net (KAIST, 최종 테스트 1위) | 0.8923 | 1.9591 | 0.9215 | 0.8893 |
| SegResNet + 중복 감소 (NVAUTO, 최종 2위) | 0.8971 | 1.9022 | 0.9077 | 0.9029 |

**Stage 2(0.8948)는 KAIST(0.8923)를 넘고 NVAUTO(0.8971)에 근접합니다.** Stage 3(TRIO)는 더 높지만 GT 게이트 상한입니다.

### 2.2 크기별 비교

베이스라인은 GT 크기 구간(942 / 847 / 584)으로, 파이프라인은 분류기 라우팅 결과(897 / 1,152 / 385)로 집계했습니다.

| 클래스 | 파이프라인 초기 → 최종 | KAIST | NVAUTO |
|---|---|---:|---:|
| Small | 0.8286 → **0.8429** | 0.8278 | 0.8377 |
| Medium | 0.9264 → **0.9314** | 0.9165 | 0.9200 |
| Large | 0.9546 → **0.9582** | **0.9614** | 0.9594 |

HD95 (px):

| 클래스 | KAIST | NVAUTO |
|---|---:|---:|
| Small | 3.8017 | **3.3599** |
| Medium | **1.0619** | 1.2052 |
| Large | **0.2881** | 0.5618 |

Medium은 계속 앞섭니다. Small 최종 0.8429는 NVAUTO 0.8377을 넘습니다. **Large 격차는 −0.027에서 약 −0.003으로 줄었고**, HD95 0.29는 KAIST 0.29와 같습니다.

### 2.3 임계값 스윕

베이스라인만 0.5로 두고 파이프라인은 조정하면 불공정하므로 양쪽 모두 스윕했습니다.

| 방법 | 최적 임계값 | 최적 DSC | 0.5에서의 DSC | 이득 |
|---|---:|---:|---:|---:|
| KAIST | 0.30 | 0.8932 | 0.8923 | +0.0009 |
| NVAUTO | 0.40 | 0.8972 | 0.8971 | +0.0001 |

두 베이스라인은 임계값에 거의 둔감합니다(전 구간 변동 0.001 내외). Small·Medium Expert는 0.5에서 0.8로 올릴 때 이득이 컸고, Large는 ED/TC 이후 과소분할이라 0.50을 유지합니다.

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

### 2.6 같은 슬라이스 정성 비교 (TRIO / KAIST / NVAUTO)

2026-08-20에 세 방법을 **같은 val 슬라이스**에서 그리는 3행×6열 그리드를 추가했습니다.

| 항목 | 값 |
|---|---|
| 산출 파일 | `results/method_comparison_3x6.png` |
| 스크립트 | `scripts/eval/plot_three_method_grid.py` |
| 레이아웃 | 행: **TRIO**, KAIST, NVAUTO · 열: Small 1, Small 2, Medium 1, Medium 2, Large 1, Large 2 |
| 제목 | `Representative slices near class-mean DSC  ·  dashed green = GT` |
| TRIO 행 | 파이프라인 Stage 3 + Monotonic DSC 게이트 (임계값 0.80 / 0.80 / 0.50, CC 0 / 15 / 25) |
| 선택 기준 | 클래스 평균 DSC에 가깝고, 세 방법 DSC를 같은 장에서 비교 |
| 고정 인덱스 | **113, 2286, 1121, 2295, 95, 1480** |

그림에 찍힌 슬라이스 DSC:

| | Small 1 | Small 2 | Medium 1 | Medium 2 | Large 1 | Large 2 |
|---|---:|---:|---:|---:|---:|---:|
| TRIO | 0.840 | 0.849 | 0.930 | 0.931 | 0.959 | 0.957 |
| KAIST | 0.825 | 0.835 | 0.915 | 0.923 | 0.962 | 0.960 |
| NVAUTO | 0.839 | 0.850 | 0.914 | 0.923 | 0.956 | 0.958 |

**정량 표와 그림의 베이스라인 가중치는 다릅니다.** 08-19 비교 표(DSC 0.8923 / 0.8971)는 `baselines/results/{kaist,nvauto}_metrics.json`(2,373장)입니다. 비교 그리드를 그릴 때 체크포인트가 없어서 같은 분할·20 epoch·seed 42로 **다시 학습**했고, 학습 중 val DSC는 KAIST **0.9021**, NVAUTO **0.8999**였습니다. 그림의 KAIST/NVAUTO 윤곽은 이 재학습 가중치입니다.

```bash
python scripts/eval/plot_three_method_grid.py --indices 113,2286,1121,2295,95,1480 --out results/method_comparison_3x6.png
```

베이스라인 가중치가 없으면 먼저:

```bash
python baselines/train_baseline.py --method kaist  --epochs 20 --batch_size 32 --seed 42 --no_deterministic
python baselines/train_baseline.py --method nvauto --epochs 20 --batch_size 32 --seed 42 --no_deterministic
```

KAIST axial attention 역전파가 비결정적이라 재학습은 `--no_deterministic`을 썼습니다.

---

## 3. 이번 실행의 단계별 학습

| 단계 | 환자 | 학습 슬라이스 | 검증 슬라이스 |
|---|---:|---:|---:|
| Stage 1 Shape Classifier | 168 / 42 | 9,868 | 2,373 |
| Stage 2 Expert 3종 | 168 / 42 | 크기별 필터 (아래) | 크기별 필터 |
| Stage 3 PPO 3종 | 168 / 42 | 크기별 필터 + 50% 믹스업 | 크기별 필터 (hold-out) |
| Stage 4 평가 | 42 | — | 2,434 전수 |

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

공통: 20 epoch, batch 64, Adam lr 3e-4, CosineAnnealingLR, AMP(FP16), 환자 단위 분할.

| Expert | 손실 | 학습 범위 | 저장 경로 |
|---|---|---|---|
| Small CaraNet | BCEDice 0.5+0.5, 2.5D, `<50px` ×4 | Small만 | `caranet_best.pt` |
| Medium UNet++ | BCEDice 0.5+0.5 | Medium만 | `unetplusplus_best.pt` |
| Large SegResNet | MultiChannelBCEDice (ED+TC) | **전 구간** | `segresnet_best.pt` |

Expert 단독 성능은 이진화 임계값에 좌우되므로, val에서의 실측치는 §1.3의 **초기 DSC**(Small 0.8286 / Medium 0.9264 / Large 0.9546)를 봅니다. 이 값은 임계값 0.80/0.80/0.50과 TTA를 적용한 뒤의 수치입니다.

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

Large 에이전트 상세: 27m56s, eval 보상 120K **606.76** → 240K **639.43**, 에피소드 길이 30.0 고정. `target_dsc=1.0`이라 조기 종료가 거의 없습니다.

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

### 5.2 3-Stage 라우팅 (실험 6–14)

| 실험 | 날짜 | 설정 요약 | 평균 초기 → 최종 DSC | 평균 HD95 |
|---|---|---|---|---|
| 6 | — | Small 연속 PPO, 4채널, 단기 10K | 0.7860 → 0.7938 | — |
| 7 | 08-16 | 1,251명, AttUNet/UNet++/SegResNet | 0.8129 → 0.8198 | 3.81 (단위 혼재) |
| 8 | 08-17 | `t1ce+flair`, Zoom-Refiner | 0.8480 → 0.8486 | 4.0440 px |
| 9 | 08-18 | SDF, 컴포넌트 독립 보정, 단조 가드 | 0.8999 → 0.9000 | 1.3340 px |
| 10 | 08-18 | Dual Gate, CaraNet Small | 0.9030 → 0.9031 | 1.1906 → 1.1874 px |
| 11 | 08-19 | 클래스당 100장 균형 평가 | 0.9031 → 0.9113 | 1.1662 → 1.0500 px |
| 12 | 08-19 | 210명 **전수** 12,241장, Oracle 라우팅, Dual Gate | 0.8775 → 0.8880 | 1.8416 → 1.6843 px |
| 13 | 08-19 | val 42명 hold-out 2,373장, 분류기 라우팅, 임계값 0.8/0.8/0.6 | 0.8672 → 0.8752 | 2.3365 → 2.2897 px |
| **14 (현재 정량)** | **08-20** | **2.5D CaraNet, Large ED/TC, 임계값 0.8/0.8/0.5, val 2,434장** | **0.8948 → 0.9031** | **1.7336 → 1.6060 px** |
| 15 | 08-20 | TRIO/KAIST/NVAUTO **같은 슬라이스** 3×6 그리드. 행 이름 TRIO, 제목 영어. 베이스라인 재학습(학습 중 val DSC 0.9021 / 0.8999) | 그림만 | — |

**실험 12와 13은 직접 비교할 수 없습니다.** 프로토콜이 세 군데에서 바뀌었습니다.

| 항목 | 실험 12 | 실험 13 |
|---|---|---|
| 평가 집합 | 210명 **전수** (학습 데이터 포함) | **val 42명만** (hold-out) |
| 라우팅 | **Oracle** (GT 면적) | **분류기 예측** |
| 게이트 | Dual (DSC + HD95) + Medium/Large GT 형태학 | Monotonic **DSC만** |

실험 13의 점수가 낮은 것은 성능 퇴행이 아니라 **누출과 Oracle 이점을 제거한 결과**입니다. 실험 7–8·12는 슬라이스 전수 평균, 실험 9–11은 클래스당 100장 균형 평균입니다.

실험 13과 14는 같은 val 42명·분류기 라우팅이지만 슬라이스 수(2,373 vs 2,434)와 Expert 구조가 달라 1:1은 아닙니다. 14에서 Stage 2가 0.8672 → 0.8948로 오른 것은 2.5D CaraNet, Large ED/TC, Large 임계값 0.50이 겹친 결과입니다.

### 5.3 실험 13–14에서 고친 것

| 항목 | 이전 | 실험 13 | 실험 14 (현재) |
|---|---|---|---|
| 분할 | 슬라이스 단위 80/20 (같은 환자가 양쪽에) | **환자 단위** 168 / 42 | 동일 |
| 데이터 로딩 | `max_patients`로 "정렬 순 앞 N명" 로드 | 분할을 먼저 확정한 뒤 해당 환자 ID만 로드 | 동일 |
| 라우팅 | Oracle (GT 면적) | 분류기 예측 | 동일 |
| 이진화 임계값 | 고정 0.50 | 0.80 / 0.80 / 0.60 | **0.80 / 0.80 / 0.50** |
| CC filter | — | 15 / 15 / 25 | **0 / 15 / 25** (Small 끔) |
| Small Expert | FocalTversky, 2D | CaraNet 2D | **2.5D + `<50px` ×4 + zoom-crop, BCEDice** |
| Large Expert | Large만 학습, WT 1채널 | 동일 | **전 구간 학습 + ED/TC 2채널** |
| 미세 파편 처리 | 고정 후보 + 휴리스틱 | 임계값 사다리 | 동일 |
| 게이트 | Dual (DSC + HD95) | Monotonic DSC만 | 동일 |
| Medium/Large 평가 | TTA + GT 형태학 | PPO `predict()` 15스텝 | 동일 |
| 지표 | DSC, HD95 | + Precision, Recall | 동일 |
| PPO 스냅샷 | 한 폴더 공유 → 덮어씀 | `snapshots/{mode}/` 분리 | 동일 |
| PPO 조기 종료 | `stop_dsc_target 0.92`로 즉시 중단 | 비활성화 | 동일 |
| 재현성 | 시드 미고정 | seed 42 + 결정적 모드 기본 ON | 동일 |
| 분류기 | scratch ResNet18 + Adam | ImageNet 사전학습 + AdamW + Cosine | 동일 |
| 시각화 표본 | ΔDSC 최대 | — | **클래스 평균 Final DSC + Final > Initial** |
| 방법 비교 그리드 | 없음 | — | **TRIO vs KAIST vs NVAUTO, 같은 슬라이스 3×6, 제목 영어** |

---

## 6. 해석

1. **Stage 3 이득은 +0.83%p DSC, HD95 −0.13 px.** 게이트가 858장(35.3%)을 Stage 2로 되돌리므로 최종 DSC는 상한입니다.
2. **Stage 2가 베이스라인과 경쟁한다.** 0.8948은 KAIST 0.8923을 넘고 NVAUTO 0.8971에 근접합니다. 공정 비교는 게이트 없는 Stage 2입니다.
3. **Large 격차가 거의 닫혔다.** 최종 0.9582 vs KAIST 0.9614 / NVAUTO 0.9594. ED/TC 헤드와 임계값 0.50이 과소분할(이전 P 0.97 / R 0.93)을 균형으로 돌렸습니다.
4. **Small은 2.5D 이후 0.83까지 올랐지만 과소분할이 남습니다** (P 0.92 / R 0.80).
5. **분류기가 라우팅 상한을 제한한다.** Val Acc 0.8512이므로 슬라이스 약 15%가 엉뚱한 Expert로 갑니다.
6. **시각화 표본은 클래스 평균 Final DSC에 가깝고 실제로 개선된 슬라이스**입니다. 예전처럼 ΔDSC 최대 이상치를 고르지 않습니다.
7. **비교 그림의 방법 이름은 TRIO**입니다. Stage 3 + 단조 DSC 게이트와 같습니다.
8. **Medium UNet++ 학습 스크립트의 val 로더는 train 증강을 그대로 씁니다** (`train_unetplusplus.py`의 `collate_fn=augment_batch`). 학습 중 val DSC만 흔들릴 수 있고, Stage 4 평가 로더는 증강이 없습니다. Medium을 다시 학습하지 않는 한 보고된 0.9264 → 0.9314는 그대로입니다.

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

# TRIO / KAIST / NVAUTO 같은 슬라이스 그리드
python scripts/eval/plot_three_method_grid.py --indices 113,2286,1121,2295,95,1480 --out results/method_comparison_3x6.png
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
| `results/pipeline_sample_{small,medium,large}_{1,2}.png` | 클래스 평균에 가까운 개선 샘플 |
| `results/pipeline_sample_*_original.png` | 해당 슬라이스 원본 MRI |
| `results/method_comparison_3x6.png` | TRIO / KAIST / NVAUTO 같은 슬라이스 3×6 그리드 |
| `results/threshold_sweep.png` | 임계값 스윕 곡선 |
| `baselines/results/comparison.md` | 베이스라인 비교 표 |
| `baselines/results/{kaist,nvauto}_metrics.json` | 구간별 지표 + 스윕 |
| `baselines/results/baseline_*.png` | 베이스라인 정량·정성 그림 |
