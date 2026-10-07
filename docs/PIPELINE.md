# RL-Refiner 파이프라인

분류 → 크기별 Expert → **경계 띠 PPO 하나** → 평가. 코드 기준 설명이다.

논문은 [paper_draft_ko.md](paper_draft_ko.md)다. 수치는 BraTS 2021 **1251명** 중 개발 400명(train 280 / val 60 / 방법 선택 test 60)을 뺀 **851명** 환자 평균이다. 고정 임계값 Stage 2 대비 DSC 0.8337→**0.8604** (짝 차이 +0.0267, 95% CI 0.0256–0.0278). 검증 60명으로 임계값을 0.45/0.80/0.20에 다시 고른 Stage 2(0.8411)와 비교해도 **+0.0193** (0.0184–0.0202)이 남는다. HD95 평균 4.888→4.609 px는 빈 마스크 제외가 달라 짝비교가 아니다. 이 0.8337→0.8604는 **2026-09-30** 잠금 가중치다. 상세는 [EXPERIMENT_RESULTS.md §0.11–0.13](EXPERIMENT_RESULTS.md). 2026-10-07에 같은 분할로 다시 학습한 시드 42·7·123의 PPO DSC는 0.8597, 0.8620, 0.8609(평균 0.8609, 범위 0.0023, 0.003보다 좁다)이고 Stage 2는 0.8339, 0.8370, 0.8352, 대형 전 슬라이스는 0.8503, 0.8540, 0.8500이다. 빈 슬라이스를 포함한 131,905장에서 시드 42의 거짓 양성은 Stage 2 0.567, PPO 0.521, 대형 0.930이다. 양쪽 마스크가 있는 공통 슬라이스의 HD95 환자 평균은 Stage 2 6.072 px(11.536 mm), PPO 5.649 px(10.734 mm)이고, z를 쌓은 3D Dice는 0.779, 0.813, 0.799이다. mm는 픽셀×1.9이다. 시드 7은 거짓 양성 0.532, 0.497, 0.821과 3D Dice 0.786, 0.811, 0.796이다. 시드 123은 0.601, 0.571, 0.829와 0.771, 0.809, 0.759이다. 시드 7·123의 PPO 공통 HD95는 10.351 mm, 10.731 mm이다. 추론 ablation은 5스텝 0.8597, 1스텝 0.8460, 지도학습 1스텝 0.8383, 대형+띠 0.8676, 정답 라우팅 0.8698, 이전 풀 148명을 뺀 703명 0.8579, 반경 0은 0.8427, 스텝 3·가드 해제·띠 폭은 0.8575–0.8601이다. 단일 U-Net을 같은 280명으로 20에폭 학습하고 임계값 0.5로 평가하면 DSC 0.8156(95% CI 0.8067–0.8243), HD95 6.939 px이다. 전역 1픽셀 수축·유지·팽창 바닐라 PPO 15스텝은 DSC 0.8133(0.8046–0.8220), HD95 6.955 px, 짝 차이 −0.0023(95% CI −0.0029–−0.0017)이다. 정답 크기별 DSC는 소형 0.7215→0.7197, 중형 0.8758→0.8736(757명), 대형 0.8973→0.8939(408명)이다. 가중치는 `checkpoints/seeds/{42,7,123}/`와 `checkpoints/single_backbone/unet.pt`, `ppo_unet_vanilla.zip`이다. 파일은 `results/protocol_gaps/seed_{42,7,123}.json`, `results/vanilla_unet_851.json`, `results/ablation_review/summary.json`이다. 로그는 [§0.15](EXPERIMENT_RESULTS.md)다. 2026-08-20 기록은 [history_2026-08-20.md](history_2026-08-20.md)다.

`ppo_v5`는 폐기했다. 8방위 SDF 에이전트(`ppo_v2`, `ppo_v4`)와 2026-08-20 TRIO(단조 DSC 게이트)는 이전 구현이다. 수치는 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) §0.1–0.4와 §1에만 두고, 이 문서의 본문과 섞지 않는다. 이전 품질 gate 절차는 [QUALITY_GATE.md](QUALITY_GATE.md) 아래쪽에 남아 있다.

구조 그림: [`results/pipeline_overview.jpg`](../results/pipeline_overview.jpg), [`results/fig2_ppo_internal_route.jpg`](../results/fig2_ppo_internal_route.jpg).  
Stage 2와 경계 띠 PPO: [`results/band_ppo_delta_matched/delta_matched_comparison.png`](../results/band_ppo_delta_matched/delta_matched_comparison.png).  
단일 백본: [`results/single_backbone_ppo_grid.png`](../results/single_backbone_ppo_grid.png).

---

## 1. 한눈에 보는 구조

입력은 BraTS 2021 환자의 **T1ce + FLAIR** 2채널, 128×128 슬라이스다. 출력은 Whole Tumor 이진 마스크다. Small Expert만 인접 슬라이스를 붙인 **2.5D**(6채널)를 쓴다.

| 단계 | 하는 일 | 산출물 |
|:---:|---|---|
| 1 | 종양 면적으로 Small / Medium / Large 분류 | `checkpoints/shape_classifier_best.pt` |
| 2 | 클래스에 맞는 Expert가 확률 맵을 만들고 이진화 | `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt` |
| 3 | 정책 하나가 경계 띠 픽셀을 끄기 / 유지 / 켜기 | `checkpoints/band_ppo.pt` |
| 4 | DSC / HD95 / Precision / Recall | `results/band_ppo_locked.json` |

```mermaid
flowchart LR
    IN["T1ce + FLAIR"] --> CLS["Stage 1 크기 분류"]
    CLS --> EXP["Stage 2 크기별 Expert"]
    EXP --> BAND["Stage 3 경계 띠 PPO"]
    BAND --> OUT["최종 WT 마스크"]
```

구현 위치:

- 진입점: `run_pipeline.py` (`--refinement_profile band_ppo`는 학습을 건너뛰고 평가만)
- 라우터: `src/models/dynamic_router.py`의 `AdaptivePipeline`
- 띠와 보상: `src/utils/band.py`
- 정책: `src/models/band_refine.py`의 `BandActorCritic`
- 학습: `scripts/train/train_band_ppo.py`
- 논문 표: `scripts/eval/evaluate_band_ppo_locked.py`
- 지표: `src/utils/metrics.py`

---

## 2. 데이터와 분할

로더: `src/data/brats2020_dataset.py`. 경로 `src/data/archive`.

| 항목 | 값 |
|---|---|
| 모달리티 | `t1ce+flair` |
| 해상도 | 128×128 |
| 레이블 | WT: seg ≠ 0. Large 학습용 영역 헤드: ED(2), TC(NCR 1 ∪ ET 4) |
| 정규화 | 뇌 마스크 안 z-score → 1–99 퍼센타일 클리핑 → 0–1 |
| 종양 슬라이스 | 종양 픽셀 비율 ≥ 0.002 |

분할 파일 `checkpoints/patient_split.json` (seed 42)은 **개발 400명**이다.

| 역할 | 환자 | 용도 |
|---|---:|---|
| train | 280 | Expert·분류기·띠 PPO 학습 |
| val | 60 | 학습 중 모델 선택 |
| test | 60 | 방법 선택. 논문 성능이 아님 |
| 미사용 | 851 | 확정 평가. 학습과 방법 선택에 넣지 않음 |

같은 환자의 인접 슬라이스가 train과 val에 동시에 들어가지 않는다. `max_patients`를 로더에 직접 넘기면 정렬 순 앞 N명이 실려 이 분할과 어긋난다. `src/data/patient_split.py`가 분할을 먼저 확정한 뒤 그 ID만 넘긴다.

1251명을 train 875 / val 188 / test 188로 나눈 기록은 다른 프로토콜이다. 그 수치는 §0.1의 `ppo_v4` 기록이지 현재 분할이 아니다.

---

## 3. 크기 클래스

면적 정의는 `src/utils/metrics.py`의 `gt_size_class`와 `src/data/shape_dataset.py`가 공유한다.

| 클래스 | 면적 | Expert |
|---|---|---|
| 0 Small | `0 < area < 300` | CaraNet 2.5D |
| 1 Medium | `300 ≤ area < 700` | UNet++ |
| 2 Large | `area ≥ 700` | SegResNet (ED+TC → WT) |

학습 중 Expert와 방법 선택 test는 **GT 면적**으로 클래스를 고른다. 851명 확정 평가는 **분류기 예측**으로 Expert를 고른다. 그래서 그 평가 슬라이스의 일부는, 학습 때 그 크기에 쓰이던 Expert가 아닌 마스크를 보정한다.

---

## 4. 실행 (`run_pipeline.py`)

```bash
python run_pipeline.py --refinement_profile band_ppo \
  --max_train_patients 1251 --split_role all --slice_selection tumor \
  --no_deterministic --output_dir results/band_ppo_pipeline_1251 --no_plots
```

`band_ppo`는 `checkpoints/band_ppo.pt`가 있어야 한다. 분류기·Expert·에이전트 학습은 건너뛰고 Stage 4만 돈다. 평가 모드는 `ppo_raw`다. 이 명령의 DSC 0.8503→0.8758은 **개발 400명을 포함한 슬라이스 평균**이라 논문 본문 성능이 아니다.

논문 표:

```bash
python scripts/eval/evaluate_band_ppo_locked.py
```

결과는 `results/band_ppo_locked.json`. 개발 400명을 빼고 환자 평균으로 집계한다.

시드 기본값 42. `band_ppo` 학습·851명 평가·위 1251명 평가는 결정적 모드를 끄고 돌렸다(`--no_deterministic`). `src/utils/seed.py`의 `set_seed`가 `random` / `numpy` / `torch` 시드를 고정한다.

다른 `--refinement_profile`(`legacy`, `ppo_v2`, `ppo_v3`, `ppo_v4`)은 크기별 SDF 에이전트 경로다. 현재 Stage 3가 아니다.

---

## 5. Stage 1 — Shape Classifier

| 항목 | 내용 |
|---|---|
| 스크립트 | `scripts/train/train_shape_classifier.py` |
| 모델 | ResNet18. `conv1`은 2.5D 채널, `fc`는 3클래스 (`src/models/shape_classifier.py`) |
| 레이블 | GT 면적 → 0/1/2. 경계 ±80px(`margin=80`)는 학습에서 제외 |
| 손실 | CrossEntropy, label smoothing 0.05 |
| 이번 개발 분할 | val acc **0.8151**. 851명 슬라이스에서 정답 크기와 맞는 비율 **0.801** |

`conv1`을 입력 채널 수에 맞출 때 ImageNet RGB 커널을 채널 축으로 합한 뒤 나눈다.

---

## 6. Stage 2 — 크기별 Expert

| 항목 | 값 |
|---|---|
| 에폭 / 배치 | 20 / 64 |
| 옵티마이저 | Adam, lr `3e-4`, CosineAnnealingLR |
| 정밀도 | AMP |
| 이진화 | Small / Medium / Large **0.80 / 0.80 / 0.50** |
| 연결요소 최소 픽셀 | **0 / 15 / 25** |
| TTA | 원본 + 좌우 flip + 상하 flip 확률 평균 |

| Expert | 파일 | 입력 / 출력 | 손실 | 학습 범위 |
|---|---|---|---|---|
| Small CaraNet | `src/models/caranet.py` | 2.5D 6채널 / WT 1채널 | BCEDice. GT `<50px` 오버샘플, zoom-crop | Small만 |
| Medium UNet++ | `src/models/unetplusplus.py` | 중심 2채널 / WT 1채널 | BCEDice | Medium만 |
| Large SegResNet | `src/models/segresnet.py` | 중심 2채널 / ED+TC 2채널 → WT=`1-(1-p_ed)(1-p_tc)` | MultiChannelBCEDice | 전 구간, 추론 때 Large만 |

이번 400명 분할의 Small CaraNet val DSC는 **0.8159**다. 체크포인트 채널이 바뀌면 `src/utils/weight_adapt.py`가 첫 conv를 맞춘다.

순전파: 분류기는 중심 슬라이스, Small Expert는 2.5D, Medium/Large는 중심 2채널. Large 2채널은 `region_logits_to_wt`로 합친다. Small은 1차 예측 뒤 zoom-crop 재추론을 한 번 더 한다.

---

## 7. Stage 3 — 경계 띠 PPO

스크립트: `scripts/train/train_band_ppo.py`. 정책은 **하나**다. Small / Medium / Large를 같이 보고, 시작 가중치는 Medium만 지도학습한 `checkpoints/band_refine_medium.pt`다.

마스크 전체를 다시 그리지 않는다. 띠 밖은 Stage 2 라벨을 복사한다.

### 7.1 편집 띠

`src/utils/band.py`의 `edit_band`.

- Stage 2 확률 0.35–0.65, 또는
- 현재 마스크 경계 ±2px

확률이 그 구간에 들어가면 경계에서 멀리 있어도 바뀔 수 있다. 수정 범위는 ±2px만이 아니다.

### 7.2 네트워크

`BandRefineNet`은 폭 32인 소형 U-Net이다. 입력 4채널은 T1ce, FLAIR, **현재** 마스크, Expert 확률이다. 정답은 관측에 없다.

| 블록 | 채널 | 해상도 |
|---|---:|---:|
| 입력 | 4 | 128² |
| enc1 | 32 | 128² |
| enc2 | 64 | 64² |
| enc3 | 128 | 32² |
| dec2 | 64 | 64² |
| dec1 | 32 | 128² |

dec1에서 두 갈래다.

- **행동 머리** 3채널: 끄기 / 유지 / 켜기. 추론은 이 머리의 argmax.
- **가치 머리** 1채널: 학습 전용. 픽셀 가치.

띠 밖 로짓은 끄기·켜기를 `-1e9`로 막아 유지가 된다. 인코더–디코더 skip은 enc1→dec1, enc2→dec2다.

### 7.3 다섯 스텝

`refine_batch`. 확률 맵은 고정하고 **마스크만** 다음 입력으로 넣는다. 스텝 수는 5.

FLAIR 가드는 **마지막 스텝만**이다. 그 슬라이스 띠의 평균과 표준편차로 z를 내고, 평균보다 어두운 픽셀은 켜지 못하고 밝은 픽셀은 끄지 못한다. 절대 밝기 임계값은 쓰지 않는다. 띠 픽셀이 8개 미만이거나 분산이 없으면 그 슬라이스는 Stage 2를 유지한다.

추론은 DSC가 내려도 초기 마스크로 되돌리지 않는다. GT 단조 게이트는 이 경로에 없다.

### 7.4 학습

1. 개발 train 280명의 Stage 2 확률·마스크를 TTA로 만든다 (`src/data/contour_dataset.py`).
2. 세 클래스를 한 배치에 섞는다.
3. 5스텝을 샘플링하고, 띠 픽셀만으로 GAE를 계산한다.
4. 보상은 뒤집은 픽셀만 받는다. GT와 맞으면 +1, 틀리면 −1. HD95가 줄면 `0.25 × (이전 − 이후)`를 더한다 (`hd_coef=0.25`).
5. PPO clip 0.1, γ 0.9, GAE λ 0.95, entropy 0.001, value coef 0.5.
6. 옵티마이저는 AdamW. 행동 lr `1e-5`, 가치 lr `1e-4`. 에폭 6, PPO epoch 2, 배치 16.

저장 모델은 검증 macro HD95가 Stage 2보다 낮고 DSC가 가장 높았던 **5 epoch**이다. 검증 macro DSC 0.8972 / HD95 3.558 (Stage 2 0.8741 / 3.838).

방법 선택 test 60명, GT 면적 라우팅, 3495장 (`results/band_ppo.json`):

| | Stage 2 DSC | PPO DSC | Stage 2 HD95 | PPO HD95 |
|---|---:|---:|---:|---:|
| Small | 0.7500 | **0.7846** | 6.640 | **6.393** |
| Medium | 0.9035 | **0.9215** | 3.592 | **2.885** |
| Large | 0.9223 | **0.9330** | 4.312 | **4.073** |
| 전체 | 0.8494 | **0.8721** | 4.905 | **4.474** |

세 클래스 모두 Stage 2보다 DSC·HD95가 나아서 이 정책을 채택했다. 이 표는 논문 확정 수치가 아니다.

---

## 8. Stage 4 — 확정 평가

스크립트: `scripts/eval/evaluate_band_ppo_locked.py`.

- 환자: 1251명에서 개발 400명을 뺀 **851명**, 종양 슬라이스 **50,010**장.
- 라우팅: 분류기. `--oracle_routing`이 아니다.
- Expert 이진화와 TTA, 임계값 0.80/0.80/0.50, CC 0/15/25 다음 `band_ppo.pt` 5스텝.
- DSC는 슬라이스 값의 **환자 평균**. 3D 볼륨 Dice가 아니다.
- HD95는 128×128 격자에서 윤곽 사이 95% 거리, 단위는 픽셀이다. 한쪽 마스크가 비면 그 슬라이스는 HD95 평균에서 뺀다.
- 크기 행은 환자군이 아니다. 그 정답 면적의 슬라이스가 있는 환자 수이며, 같은 환자가 두 행 이상에 들어간다.

| 정답 크기 구간 | 해당 슬라이스가 있는 환자 | Stage 2 DSC | PPO DSC | 환자 짝 차이 (95% CI) |
|---|---:|---:|---:|---|
| 전체 | 851 | 0.8337 | **0.8604** | +0.0267 (0.0256–0.0278) |
| 300px 미만 | 851 | 0.7579 | **0.7933** | +0.0354 (0.0338–0.0370) |
| 300–700px | 757 | 0.8700 | **0.8952** | +0.0252 (0.0239–0.0265) |
| 700px 이상 | 408 | 0.8981 | **0.9169** | +0.0188 (0.0173–0.0205) |

DSC 환자 짝 차이는 **+0.0267** (95% CI 0.0256–0.0278)이다. 검증 60명에서 크기별 임계값을 0.45/0.80/0.20으로 다시 고른 Stage 2는 0.8411이고, PPO와의 짝 차이는 **+0.0193** (0.0184–0.0202)이다. 재현은 `python scripts/eval/evaluate_stage2_retuned.py`, 파일은 `results/stage2_retuned.json`이다. HD95 평균은 전체 4.888→4.609 px로 `results/band_ppo_locked.json`에 있으나, 빈 마스크를 빼는 슬라이스가 Stage 2와 PPO에서 달라 본문 짝비교로 쓰지 않는다.  단일 백본 다섯 개와 섹터 PPO는 [§0.14](EXPERIMENT_RESULTS.md)다. 학습은 같은 train 280명이고, 보고 숫자는 미사용 48명·슬라이스 2,754장의 슬라이스 평균이다. Small DSC는 TRIO 0.812, UNet++ 0.778, Attention U-Net 0.761, SegResNet 0.676, UNet+++ 0.543, U-Net 0.537이다. 그림은 `results/single_backbone_ppo_grid.png`다. 고정 임계값 0.80/0.80/0.50과 연결요소 기준은 이전 210명 풀에서 정했고, 그 풀의 148명이 이 851명에 들어 있다. 띠 설계는 이번 방법 선택 60명에서 정했고, 가중치는 이번 개발 400명으로 다시 학습했다. Stage 2와 PPO를 맞춘 여섯 슬라이스는 `results/band_ppo_delta_matched/delta_matched_comparison.png`다. 재현은 `python scripts/eval/plot_delta_matched.py`다. 열 제목의 뒤는 분류기가 고른 전문가이고, 중형 첫째는 CaraNet으로 라우팅되었다. 칸의 HD95는 그 장만의 값이며, 중형 첫째는 2.00에서 3.38 px로 늘었다.

`run_pipeline.py`의 1251명 `ppo_raw` 평가는 같은 가중치를 쓰지만 집계가 다르다. 개발 환자가 포함된 슬라이스 평균이다.

### 2026-10-07 재학습

위 표의 0.8337→0.8604는 2026-09-30 잠금 가중치다. 2026-10-07 재학습은 환자 분할 파일을 그대로 두고 시드만 42, 7, 123으로 다시 학습한 다른 가중치다.

| 시드 | Stage 2 DSC | 경계 띠 PPO | 대형 전문가 전 슬라이스 |
|---:|---:|---:|---:|
| 42 | 0.8339 | 0.8597 | 0.8503 |
| 7 | 0.8370 | 0.8620 | 0.8540 |
| 123 | 0.8352 | 0.8609 | 0.8500 |

PPO DSC 범위는 0.0023이고 세 시드 평균은 0.8609이다. 시드 42 추론 ablation은 5스텝 0.8597, 1스텝 0.8460, 중형 지도학습 초기화 1스텝 0.8383, 대형+띠 0.8676, 정답 크기 라우팅 0.8698, 이전 210명 풀과 겹치는 148명을 뺀 703명 0.8579, 반경 0은 0.8427, 스텝 3·가드 해제·띠 폭 변경은 0.8575–0.8601이다. 파일은 `results/ablation_review/summary.json`이다.

빈 슬라이스를 포함한 131,905장에서 거짓 양성은 시드 42가 Stage 2 0.567, PPO 0.521, 대형 0.930이다. 시드 7은 0.532, 0.497, 0.821이고 시드 123은 0.601, 0.571, 0.829이다. Stage 2와 PPO가 함께 마스크를 가진 슬라이스의 HD95 환자 평균은 시드 42에서 Stage 2 6.072 px(11.536 mm), PPO 5.649 px(10.734 mm)이다. 시드 7·123의 PPO 공통 HD95는 10.351 mm, 10.731 mm이다. mm는 픽셀×1.9이다. z를 쌓은 3D Dice 환자 평균은 시드 42가 0.779, 0.813, 0.799, 시드 7이 0.786, 0.811, 0.796, 시드 123이 0.771, 0.809, 0.759이다. 파일은 `results/protocol_gaps/seed_{42,7,123}.json`이다.

단일 U-Net은 같은 280명으로 20에폭 학습하고 임계값 0.5로 851명, 종양 슬라이스 50,010장을 평가한다. 환자 평균 DSC는 0.8156(95% CI 0.8067–0.8243), HD95는 6.939 px이다. 그 마스크에 전역 1픽셀 수축·유지·팽창 바닐라 PPO를 15스텝 적용하면 DSC 0.8133(0.8046–0.8220), HD95 6.955 px, 환자 짝 차이 −0.0023(95% CI −0.0029–−0.0017)이다. 정답 크기별 DSC는 소형 0.7215→0.7197, 중형 0.8758→0.8736(757명), 대형 0.8973→0.8939(408명)이다. 파일은 `results/vanilla_unet_851.json`이다. 가중치는 `checkpoints/seeds/{42,7,123}/`, `checkpoints/single_backbone/unet.pt`, `ppo_unet_vanilla.zip`이다.

---

## 9. 학습과 추론

| 항목 | 학습 | 851명 평가 |
|---|---|---|
| 환자 | 개발 train 280 | 미사용 851 |
| Expert 선택 | GT 면적 | 분류기 |
| 관측 | 영상, 현재 마스크, 확률. GT 없음 | 같음 |
| GT | 보상만 | 지표만 |
| 스텝 | 5, 마지막만 FLAIR 가드 | 같음 |
| 행동 | 띠 안에서 샘플링 | argmax |
| 나빠지면 되돌림 | 없음 | 없음 |

---

## 10. 모듈 지도

```
run_pipeline.py                          band_ppo면 평가만
scripts/train/train_shape_classifier.py  Stage 1
scripts/train/train_caranet.py           Stage 2 Small
scripts/train/train_unetplusplus.py      Stage 2 Medium
scripts/train/train_segresnet.py         Stage 2 Large
scripts/train/train_band_refine.py       Medium 띠 지도학습 (PPO 초기 가중치)
scripts/train/train_band_ppo.py          Stage 3
scripts/eval/evaluate_band_ppo_locked.py 851명 논문 표
scripts/eval/evaluate_band_ppo.py        개발 test 60명
scripts/eval/plot_delta_matched.py       Stage 2와 PPO 여섯 슬라이스
scripts/eval/evaluate_pipeline.py        1251명 슬라이스 평균 (band_ppo 분기)

src/models/band_refine.py                U-Net 행동/가치 머리
src/utils/band.py                        띠, FLAIR 가드, 픽셀 보상
src/models/dynamic_router.py             분류 + Expert
src/data/patient_split.py                개발 400명 분할
src/utils/metrics.py                     DSC, HD95, 크기 클래스
```

---

## 11. 이전 Stage 3

2026-08-20과 `ppo_v2` / `ppo_v4`는 크기별 PPO 세 개(`ppo_small.zip`, `ppo_medium.zip`, `ppo_large.zip`)가 종양 중심 기준 8방위로 SDF를 평행 이동했다. 08-20 평가에는 GT 단조 DSC 게이트가 있어 그 DSC는 상한이다. `ppo_v4`의 1251명 val GT-free DSC는 0.8345로 Stage 2 0.8344와 같다. `ppo_v5`는 0.7614로 폐기했고 체크포인트를 지웠다.

그 경로의 품질 gate·energy 모델·평가 모드는 [QUALITY_GATE.md](QUALITY_GATE.md)에 있다. `results/archive_2026-08-20/method_comparison_3x6.png`와 `results/archive_2026-08-20/pipeline_sample_comparison.png`는 이 예전 Stage 3다. 현재 Stage 2와 경계 띠 PPO 그림은 `results/band_ppo_delta_matched/delta_matched_comparison.png`다.
