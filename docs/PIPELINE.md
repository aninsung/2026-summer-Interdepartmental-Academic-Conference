# RL-Refiner 파이프라인 상세

3단계 동적 라우팅(분류 → 크기별 Expert 분할 → 크기별 경계 보정)과 4번째 평가 단계를 **코드 기준**으로 정리한 문서입니다.

실험 수치와 이력은 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md), 대회 상위 입상 방법과의 비교는 [baselines/README.md](../baselines/README.md), 논문 초안은 [paper_draft_ko.md](paper_draft_ko.md)를 봅니다. 비교 그림·표에서 이 파이프라인을 **TRIO**로 표기합니다.

> **논문 주의.** Stage 3 Final 0.9031(단조 DSC 게이트 + best-of-15)은 GT 상한이다. 논문 메인·초록에는 Stage 2(0.8948) 또는 `--deploy_mode --stage3_mode sl` 재측정값만 쓴다. **코드 정본 배포 경로 = SL Refiner + 면적 게이트**이다.

기준 실행: `python run_pipeline.py batch_size=64` (seed 42). Stage 2 임계값 0.80 / 0.80 / 0.50, CC 0 / 15 / 25.

---

## 1. 한눈에 보는 구조

입력은 BraTS 2021 환자의 **T1ce + FLAIR** 2채널 2D 슬라이스(128×128)입니다. Small Expert만 인접 슬라이스를 붙인 **2.5D**(6채널)를 씁니다. 출력은 Whole Tumor 이진 마스크입니다.

| 단계 | 이름 | 하는 일 | 학습 산출물 |
|:---:|---|---|---|
| 1 | Shape Classifier | 종양 면적으로 Small / Medium / Large 분류 | `checkpoints/shape_classifier_best.pt` |
| 2 | Size Expert | 클래스에 맞는 분할 모델이 확률 맵 생성 | `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt` |
| 3 | SL Refiner (+ PPO teacher) | 교대 SL ↔ PPO 증류. **배포 추론은 SL** | `sl_refiner_*.pt`, `ppo_*.zip`(teacher) |
| 4 | Evaluation | DSC / HD95, **deploy: area gate only** | `results/pipeline_slice_metrics_deploy.npz` |

원스톱 진입점은 `run_pipeline.py`입니다. 기본 환자 풀은 Stage 1–4 공통 **210명**이고, 이 중 **train 168명 / val 42명**으로 나뉩니다.



<img width="1005" height="481" alt="image" src="https://github.com/user-attachments/assets/c12bc699-bed1-4d28-9074-62934d864656" />


구현 위치:

- 라우터: `src/models/dynamic_router.py`의 `AdaptivePipeline`
- RL 환경: `src/envs/mask_refinement_env.py`의 `MaskRefinementEnv`
- 지표: `src/utils/metrics.py` (`dice`, `hd95`, `precision`, `recall`, `apply_monotonic_dsc_gate`, `gt_size_class`)
- 평가: `scripts/eval/evaluate_pipeline.py`

---

## 2. 데이터

로더: `src/data/brats2020_dataset.py` (`BraTS2020Dataset`). BraTS 2020/2021 NIfTI를 모두 읽습니다.

| 항목 | 값 |
|---|---|
| 경로 | `src/data/archive` |
| 모달리티 | `t1ce+flair` (2채널). `t1ce`, `t1ce+t2` 등도 가능 |
| 해상도 | 128×128 |
| 레이블 | WT: seg의 0 이외. Large 학습용 영역 헤드: ED(label 2), TC(NCR 1 ∪ ET 4) |
| 정규화 | 뇌 마스크 안 z-score → 1–99 퍼센타일 클리핑 → 0–1 |
| 유효 슬라이스 | 종양 픽셀 비율 ≥ 0.002 |
| 기본 환자 풀 | 210명 → **12,241 슬라이스** |

### 2.1 환자 단위 분할

분할 파일은 `checkpoints/patient_split.json` (80/20, seed=42)이며 **Stage 1–4가 같은 파일을 공유**합니다. 슬라이스가 아니라 **환자**를 나누므로, 같은 환자의 인접 슬라이스가 train과 val에 동시에 들어가는 누출이 없습니다.

| 역할 | 환자 | 슬라이스 | Small `<300` | Medium `300–700` | Large `≥700` |
|---|---:|---:|---:|---:|---:|
| train (GT 구간) | 168 | 9,868 | 3,561 | 4,184 | 2,123 |
| val GT 구간 (이전 집계) | 42 | 2,373 | 942 | 847 | 584 |
| val 2026-08-20 평가 | 42 | **2,434** | 897* | 1,152* | 385* |

\* 2026-08-20 클래스 수는 **분류기 라우팅** 분포입니다.

`src/data/patient_split.py`의 `load_split_brats_datasets`가 **분할을 먼저 확정한 뒤 그 환자 ID만** 데이터셋에 넘깁니다. `max_patients`를 로더에 직접 넘기면 "정렬 순 앞 N명"이 실려 무작위 표본과 어긋나므로 그렇게 하지 않습니다.

기존 분할 파일이 현재 환자 풀과 호환되지 않으면 재생성하며, 호환되는 경우에는 절대 덮어쓰지 않습니다.

---

## 3. 크기 클래스

면적은 GT(또는 평가 시 예측 컴포넌트)의 픽셀 수입니다. 정의는 `src/utils/metrics.py`의 `gt_size_class`와 `src/data/shape_dataset.py`가 공유합니다.

| 클래스 | 면적 | Expert | Stage3 배포 |
|---|---|---|---|
| 0 Small | `0 < area < 300` | CaraNet | `sl_refiner_small.pt` |
| 1 Medium | `300 ≤ area < 700` | UNet++ | `sl_refiner_medium.pt` + shrink band + zoom delete |
| 2 Large | `area ≥ 700` | SegResNet | `sl_refiner_large.pt` + shrink band + zoom delete |

Stage 2 Expert 학습만 **GT 면적**으로 필터합니다. **Stage 3 학습·Stage 4 평가는 Stage1 분류기 예측**으로 Expert·Refiner를 고릅니다 (`class_filter=classifier`). Stage3 라우팅은 **슬라이스 분류기 클래스**이며(컴포넌트 면적 재분류 아님). Medium/Large는 외곽 FP 삭제를 우선한다.

---

## 4. 학습 파이프라인 (`run_pipeline.py`)

```bash
python run_pipeline.py batch_size=64
```

`key=value` 형태를 `--key value`로 자동 변환하므로 `--batch_size 64`와 같습니다.

공통 인자: `--train_root src/data/archive`, `--max_train_patients`(현재 기본·활성 split **1251**, 논문 표는 `patient_split_210.json`의 **210**/val 42), `--patient_split checkpoints/patient_split.json`, `--modality t1ce+flair`, `--seed 42`, `--deterministic`.

> **재현 주의.** 문서에 적힌 Stage2 DSC 0.8948 등은 **210명 풀** 기준이다. 기본 `patient_split.json`(1251)으로 돌리면 수치가 달라진다. 논문 재현은 `--max_train_patients 210 --patient_split checkpoints/patient_split_210.json`을 쓴다.

건너뛰기 플래그: `--skip_classifier`, `--skip_experts`, `--skip_agents`, `--skip_eval`.



PPO는 Stage 2 체크포인트가 있어야 `AdaptivePipeline`이 Expert를 로드할 수 있으므로, Expert 학습이 먼저입니다.

### 4.1 재현성

| 항목 | 값 |
|---|---|
| 시드 | `--seed 42` (Stage 1–4 공통) |
| 결정적 모드 | **기본 활성화**. 해제는 `--no_deterministic` |
| 구현 | `src/utils/seed.py`의 `set_seed` |

`set_seed`는 `random` / `numpy` / `torch` / CUDA 시드를 고정하고, 결정적 모드에서 `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=True)`를 설정합니다. `--deterministic` 플래그는 하위 호환용으로 남아 있으며 기본값과 동일합니다.

---

## 5. Stage 1 — Shape Classifier

| 항목 | 내용 |
|---|---|
| 스크립트 | `scripts/train/train_shape_classifier.py` |
| 모델 | ResNet18, `conv1`을 2채널, `fc`를 3클래스 (`src/models/shape_classifier.py`) |
| 초기화 | **ImageNet 사전학습 기본 사용**. `--no_pretrained`로 무작위 초기화 (ablation) |
| 레이블 | GT 면적 → 0/1/2 (`ShapeDataset`) |
| 손실 | CrossEntropy |
| 옵티마이저 | **AdamW**, `--lr 1e-3`, `--weight_decay 1e-4` |
| 스케줄러 | **CosineAnnealingLR** (`T_max=epochs`) |
| 에폭 / 배치 | 15 / 64 |
| 분할 | **환자 단위** (train 9,868 / val 2,373 슬라이스) |
| 저장 | 검증 정확도 최고 가중치 |
| 이번 실행 | Best Val Acc **0.8512** |

`conv1`을 2채널로 바꿀 때 ImageNet RGB 커널을 채널 축으로 합한 뒤 입력 채널 수로 나눠 균등 분배합니다. 입력 채널이 같은 값일 때 원래 응답 크기가 유지됩니다.

### 5.1 과적합에 대한 기록

이번 실행에서 Train Acc는 0.96–0.99까지 오르지만 Val Acc는 0.85 부근에서 멈춥니다. ImageNet 사전학습 없이 무작위 초기화한 ablation은 Best Val Acc **0.8424** (`checkpoints/shape_classifier_scratch_backup.pt`)로, 사전학습의 이득은 약 +0.9%p에 그쳤습니다.

원인은 특징 품질이 아니라 **레이블 자체의 모호성**입니다. 면적 300 / 700 경계 바로 옆 슬라이스는 몇 픽셀 차이로 클래스가 갈리므로, 분류 문제로 보면 줄일 수 없는 잡음이 남습니다. 라우팅 오류의 영향은 Stage 4 결과에서 확인할 수 있습니다(§8.5).

과거 문서에 있던 Val Acc 0.9383은 **슬라이스 단위 80/20 분할**로 측정한 값입니다. 같은 환자의 인접 슬라이스가 train과 val에 함께 들어가 누출이 있었으므로, 현재 환자 단위 분할 수치와 비교하면 안 됩니다.

---

## 6. Stage 2 — 크기별 Expert

Small·Medium은 `--refinement_mode`로 자기 크기(GT 면적)만 남깁니다. **Large SegResNet은 크기 필터 없이 전 구간**을 학습하고, 추론 때만 Large로 라우팅된 슬라이스에 쓰입니다.

| 항목 | 값 |
|---|---|
| 에폭 / 배치 | 20 / 64 |
| 옵티마이저 | Adam, `--lr 3e-4` |
| 스케줄러 | CosineAnnealingLR (`T_max=epochs`) |
| 정밀도 | AMP (FP16) |

| Expert | 파일 | 왜 이 모델인가 | 입력 / 출력 | 손실 | 학습 범위 |
|---|---|---|---|---|---|
| Small CaraNet | `src/models/caranet.py` | 작은 물체용 Context Axial Reverse Attention | 2.5D 6채널 (`z-1,z,z+1`) / WT 1채널 | **BCEDice** 0.5+0.5. GT `<50px` ×4 오버샘플, zoom-crop | Small만 |
| Medium UNet++ | `src/models/unetplusplus.py` | MONAI BasicUNetPlusPlus, features `(16,32,64,128,256,16)` | 중심 2채널 / WT 1채널 | BCEDice 0.5+0.5 | Medium만 |
| Large SegResNet | `src/models/segresnet.py` | 잔차 인코더, `init_filters=16` | 중심 2채널 / **ED+TC 2채널** → WT=`1-(1-p_ed)(1-p_tc)` | MultiChannelBCEDice | **전 구간** |

Small 학습 보조: `src/data/fragment_oversample.py`, `src/utils/zoom_crop.py`. 체크포인트 채널이 바뀌면 `src/utils/weight_adapt.py`가 첫  conv를 맞춰 이어서 학습합니다.

`AdaptivePipeline` 로드 순서:

1. Small: `caranet_best.pt` → 없으면 `attention_unet_best.pt` → 없으면 랜덤 CaraNet
2. Medium: `unetplusplus_best.pt` → 없으면 `unet3plus_best.pt` → 없으면 랜덤 UNet++
3. Large: `segresnet_best.pt` → 없으면 랜덤 SegResNet (ckpt의 in/out 채널을 읽어 구성)

순전파: 분류기는 **중심 2채널**, Small Expert는 2.5D, Medium/Large는 중심 2채널. 채널 수가 안 맞으면 `_match_in_channels`가 중심 슬라이스를 꺼내거나 복제합니다. 클래스별로 해당 Expert → sigmoid → WT 확률 `(B, 1, H, W)`. Large가 2채널이면 `region_logits_to_wt`로 합칩니다. Small은 1차 예측 뒤 zoom-crop 재추론을 한 번 더 합니다.

Expert 단독 성능은 이진화 임계값에 따라 달라지므로 §8에서 함께 다룹니다.

---

## 7. Stage 3 — SL Refiner (+ PPO teacher)

`run_pipeline.py`는 `scripts/train/train_alt_stage3.py`를 호출합니다.

1. Stage2 rough를 **분류기 라우팅**으로 생성 (`class_filter=classifier`, `oracle_expert=False`)
2. DualHead SL을 GT로 학습 → (옵션) PPO teacher 성공 샘플 증류 → 반복
3. 산출: `checkpoints/sl_refiner_{mode}.pt` (배포), `checkpoints/ppo_{mode}.zip` (teacher)

PPO는 8방위 SDF `MaskRefinementEnv`에서 teacher로만 쓰이며, **배포 추론은 SL**입니다. 구 `train_agent.py` 단독 PPO 경로는 ablation/legacy입니다.

### 7.1 데이터

train 환자만 사용. 필터·Expert는 **분류기 예측 클래스**와 동일(배포와 정합). mixup=False.

---

## 8. Stage 4 — 평가·추론

### 8.1 기본(배포) 프로토콜

```bash
python scripts/eval/evaluate_pipeline.py --split_role val --deploy_mode --stage3_mode sl
```

- Stage1 **분류기**로 Expert·SL 선택 (**슬라이스 클래스**; 컴포넌트 면적으로 Stage3를 다시 고르지 않음)
- Stage2 이진화 → Init DSC (non-TTA)
- Stage3 SL은 **같은 Stage2 컴포넌트 마스크**에서 시작 (TTA는 soft prob 채널만) → Final−Init에 TTA 이득 미포함
- Medium/Large: **boundary shrink band** (`--boundary_band_px 3`) + **외곽 zoom delete** (`--sl_zoom_patches 16`) → rem_band에서만 OFF
- Large Stage3 활성 (skip 기본 해제). 이전 Large skip은 `--stage3_skip_classes 2`
- 면적 게이트(**0.85×–1.2×**)만. 단조 DSC·best-of-15 **OFF** (GT 필요 → 배포 보장으로 쓰지 말 것)
- GT 상한 재현: `--gt_upper_bound` (논문 메인 금지)
- 배포 끄기: `--no-deploy_mode`

### 8.2 주요 플래그

| 플래그 | 기본 | 의미 |
|---|---|---|
| `--stage3_mode` | `sl` | `sl` / `ppo` / `skip` |
| `--stage3_skip_classes` | `` (없음) | 예: `2`면 Large 생략 |
| `--area_gate_lo/hi` | `0.85` / `1.2` | GT-free 면적 게이트 |
| `--boundary_band_px` | `3` | Medium/Large shrink band |
| `--boundary_band_classes` | `1,2` | band 적용 클래스 |
| `--sl_zoom_patches` | `16` | 외곽 2× zoom 삭제 패치 수 (0=끔) |
| `--deploy_mode` / `--no-deploy_mode` | on | last mask + area gate |
| `--gt_upper_bound` | off | GT best-of-N + monotonic (상한) |
| `--oracle_routing` | off | GT 면적 라우팅 상한 |
| `--skip_ppo` / `stage3_mode=skip` | — | Stage2 단독 |

---

## 9. 베이스라인 비교

BraTS 2021 상위 입상 방법 두 개를 같은 42명 val 환자·같은 지표 구현으로 측정했습니다. 코드는 `baselines/`에 분리돼 있습니다.

| 방법 | DSC | HD95 | Precision | Recall |
|---|---:|---:|---:|---:|
| TRIO Stage 2 (게이트 없음) | 0.8948 | 1.7336 | — | — |
| **TRIO** Stage 3 (Monotonic DSC Gate, GT 사용) | **0.9031** | **1.6060** | — | — |
| Extending nnU-Net (KAIST, 1위) | 0.8923 | 1.9591 | 0.9215 | 0.8893 |
| SegResNet + 중복 감소 (NVAUTO, 2위) | 0.8971 | 1.9022 | 0.9077 | 0.9029 |

두 베이스라인은 3D·4모달리티·앙상블을 제외한 **2D 각색 구현**이므로 대회 리더보드 점수와 직접 비교할 수 없습니다. 베이스라인은 2,373장, 이번 파이프라인은 2,434장이라 슬라이스 수가 완전히 같지는 않습니다.

Stage 2 DSC 0.8948은 KAIST 0.8923을 넘고 NVAUTO 0.8971에 근접합니다. Stage 3는 더 높지만 GT 게이트 상한입니다.

| 클래스 | 파이프라인 초기 → 최종 | KAIST | NVAUTO |
|---|---|---:|---:|
| Small | 0.8286 → **0.8429** | 0.8278 | 0.8377 |
| Medium | 0.9264 → **0.9314** | 0.9165 | 0.9200 |
| Large | 0.9546 → **0.9582** | **0.9614** | 0.9594 |

Large 격차는 이전 −0.027에서 약 −0.003으로 줄었습니다.

---

## 10. 학습 vs 추론 차이

| 항목 | Stage2 학습 | Stage3 학습 | Stage4 배포 평가 |
|---|---|---|---|
| 환자 | train | train | val |
| 클래스/Expert | **GT 면적** | **분류기 예측** | **분류기 예측** |
| Stage3 모델 | — | SL (+ PPO teacher) | **SL** |
| Init 마스크 | — | Stage2 rough | Stage2 non-TTA (SL 시작점과 동일) |
| TTA | 없음 | 없음 | soft prob만 (마스크 Init에 미사용) |
| 게이트 | — | — | area gate only |

---

## 11. 모듈 지도

```
run_pipeline.py                          Stage 1–4 순차 실행
scripts/train/train_shape_classifier.py  Stage 1
scripts/train/train_caranet.py           Stage 2 Small
scripts/train/train_unetplusplus.py      Stage 2 Medium
scripts/train/train_segresnet.py         Stage 2 Large
scripts/train/train_alt_stage3.py        Stage 3 SL ↔ PPO 교대 (정본)
scripts/train/train_agent.py             load_real_data + legacy PPO
scripts/eval/evaluate_pipeline.py        Stage 4 배포 평가
src/models/sl_refiner.py                 DualHead SL (배포)
src/envs/mask_refinement_env.py          PPO teacher 환경
```

```bash
# 기본 (분류기 라우팅 + SL + area gate)
python scripts/eval/evaluate_pipeline.py --split_role val --deploy_mode --stage3_mode sl

# Stage 2 단독
python scripts/eval/evaluate_pipeline.py --split_role val --stage3_mode skip

# GT 상한 (논문 메인 금지)
python scripts/eval/evaluate_pipeline.py --split_role val --gt_upper_bound
```

---

## 12. 재현


```bash
pip install -r requirements.txt
# BraTS 2021를 src/data/archive 에 둔 뒤
python run_pipeline.py batch_size=64
```

이미 체크포인트가 있으면 평가만:

```bash
python run_pipeline.py --skip_classifier --skip_experts --skip_agents
```

또는 직접:

```bash
# 기본 (분류기 라우팅 + PPO + 게이트)
python scripts/eval/evaluate_pipeline.py --split_role val

# Stage 2 단독 (PPO·게이트 없음, 베이스라인과 비교용)
python scripts/eval/evaluate_pipeline.py --split_role val --skip_ppo

# 상한 (GT 면적 라우팅)
python scripts/eval/evaluate_pipeline.py --split_role val --oracle_routing
```

베이스라인:

```bash
python baselines/run_comparison.py --epochs 20 --batch_size 32 --sweep
```

같은 슬라이스 비교 그리드:

```bash
python scripts/eval/plot_three_method_grid.py --indices 113,2286,1121,2295,95,1480 --out results/method_comparison_3x6.png
```
