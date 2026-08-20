# RL-Refiner (TRIO)

2026 컴공&인지 연합학술제 연구트랙

**강화학습 기반 뇌종양 MRI 분할 경계 보정 시스템**

딥러닝 모델이 만든 초기 분할(Rough Mask)의 경계 오차를, 종양 크기별 전용 PPO 에이전트가 순차적으로 보정합니다. 비교 그림·표에서는 이 3단계 파이프라인(분류 → 크기별 Expert → PPO + 단조 DSC 게이트)을 **TRIO**로 표기합니다.

---

## 한눈에 보는 최종 결과 (2026-08-20)

`t1ce+flair` 2채널, BraTS 2021 환자 210명 중 **val 42명(2,434 슬라이스) hold-out** 평가입니다. 라우팅은 Stage 1 분류기 예측을 쓰고(Oracle 아님), Stage 2 임계값은 **0.80 / 0.80 / 0.50**, CC는 **0 / 15 / 25**, Monotonic DSC 게이트를 적용했습니다.

| 단계 | 산출 | 수치 |
|---|---|---|
| Stage 1 Shape Classifier | Best Val Acc | **0.8512** (ImageNet 사전학습) |
| Stage 2 Expert 3종 | val 초기 DSC (Small/Medium/Large) | **0.8286** / **0.9264** / **0.9546** |
| Stage 3 PPO 3종 | 학습 스텝 | 303,104 each |

| 지표 | Stage 2 (초기 분할) | Stage 3 (PPO 보정) | 변화 |
|---|---:|---:|---:|
| **평균 DSC** | 0.8948 | **0.9031** | **+0.83%p** |
| **평균 HD95** | 1.7336 px | **1.6060 px** | **−0.128 px** |

| 크기 클래스 | Expert | n | 초기 DSC → 최종 DSC | 초기 HD95 → 최종 HD95 |
|---|---|---:|---:|---:|
| Small (`<300px`) | CaraNet 2.5D | 897 | 0.8286 → **0.8429** | 3.1618 → **2.9482 px** |
| Medium (`300–700px`) | UNet++ | 1152 | 0.9264 → **0.9314** | 1.0790 → **1.0000 px** |
| Large (`≥700px`) | SegResNet ED/TC | 385 | 0.9546 → **0.9582** | 0.3650 → **0.2919 px** |

### 대회 상위 입상 방법과의 비교

베이스라인은 같은 42명 val 환자에서 이전에 측정한 값입니다(당시 2,373 슬라이스). 이번 파이프라인 평가는 2,434장이라 슬라이스 수가 완전히 같지는 않습니다.

| 방법 | DSC | HD95 (px) | Precision | Recall |
|---|---:|---:|---:|---:|
| TRIO Stage 2 (게이트 없음) | 0.8948 | 1.7336 | — | — |
| **TRIO** Stage 3 (Monotonic DSC 게이트) | **0.9031** | **1.6060** | — | — |
| Extending nnU-Net (KAIST, BraTS21 1위) | 0.8923 | 1.9591 | 0.9215 | 0.8893 |
| SegResNet + 중복 감소 (NVAUTO, BraTS21 2위) | 0.8971 | 1.9022 | 0.9077 | 0.9029 |

**Stage 2(0.8948)는 KAIST(0.8923)를 넘고 NVAUTO(0.8971)에 근접합니다.** Stage 3(0.9031)은 두 베이스라인보다 높지만, 아래 게이트 주석대로 상한입니다. 크기별로는 Medium이 계속 앞서고, Large 최종 DSC 0.9582는 NVAUTO 0.9594·KAIST 0.9614에 근접합니다. 두 베이스라인은 3D·4모달리티·앙상블을 제외한 2D 각색 구현이므로 대회 리더보드 점수와 직접 비교할 수 없습니다 → [베이스라인 상세](baselines/README.md).

> **Monotonic DSC 게이트는 GT를 봅니다.** 보정 후 DSC가 초기보다 낮으면 Stage 2 마스크로 되돌리므로, 최종 DSC는 배포 성능이 아니라 **상한**입니다. 이번 실행에서 2,434장 중 858장(35.3%)이 되돌려졌습니다. GT 없이 측정되는 값은 Stage 2 초기 DSC입니다.

파이프라인 단계·입출력·학습/평가 차이는 [파이프라인 상세](docs/PIPELINE.md), 학습 로그와 실험 이력은 [실험 결과 보고서](docs/EXPERIMENT_RESULTS.md)에 있습니다.

---

## 프로젝트 배경 및 필요성

의료 영상 분할에서 U-Net과 같은 딥러닝 모델은 종양의 대략적인 위치는 잘 파악하지만, 종양의 미세한 경계에서는 작은 오류가 발생할 수 있습니다.

**정밀도의 중요성.** 감마나이프와 같은 정밀 방사선 치료에서는 1mm 수준의 오차도 치료 결과에 영향을 줄 수 있습니다.

**비용 문제.** 의료진이 모든 경계를 직접 수정하는 과정은 많은 시간과 비용이 필요합니다.

이 프로젝트는 그 후처리를 **크기별 전문가 분할 모델 + PPO 경계 보정기**로 자동화합니다.

평가 지표는 겹침 비율 **DSC**(높을수록 좋음), 경계 거리 **HD95**(낮을수록 좋음), 그리고 과분할/과소분할을 구분하는 **Precision / Recall**입니다.

---

## 파이프라인
<img width="1024" height="507" alt="image" src="https://github.com/user-attachments/assets/78c1aab7-dab0-4c6c-a0bd-c5d98093b11a" />



| 단계 | 역할 | 산출물 |
|:---:|---|---|
| **1** | 종양 면적으로 Small / Medium / Large 분류 | `shape_classifier_best.pt` |
| **2** | 클래스별 Expert가 확률 맵 생성 → 클래스별 임계값으로 이진화 | `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt` |
| **3** | 클래스별 PPO가 8방위 경계를 SDF로 미세 조정 | `ppo_small.zip`, `ppo_medium.zip`, `ppo_large.zip` |
| **4** | DSC / HD95 / Precision / Recall 평가 및 시각화 | `results/pipeline_sample_*.png` |

Stage 3의 크기별 동작 차이:

- **Small**: CaraNet이 `z-1, z, z+1` 2.5D 입력을 쓰고, 종양 중심 64×64 crop, 4채널(영상·마스크·확률·에지), 연속 행동 `[-2, 2]⁸`
- **Medium / Large**: 전체 128×128, 3채널(영상·마스크·확률), 이산 5단계×8섹터. Large Expert는 ED/TC 2채널을 WT로 합친다. Large PPO는 보상에서 HD95 가중치를 더 크게 둠

관측에는 GT가 들어가지 않습니다. GT는 학습 시 보상 계산과 평가 시 게이트·지표에만 쓰입니다.

단계별 데이터 흐름, SDF 보정, 임계값 조정, 평가 게이트, 학습 vs 추론 차이는 [파이프라인 상세](docs/PIPELINE.md)를 봅니다.

---

## 데이터 분할

환자 단위로 나눕니다. 같은 환자의 인접 슬라이스가 train과 val에 동시에 들어가는 누출이 없습니다.

| 역할 | 환자 | 슬라이스 | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|
| train | 168 | 9,868 | 3,561 | 4,184 | 2,123 |
| val (GT 구간, 이전 집계) | 42 | 2,373 | 942 | 847 | 584 |
| val (2026-08-20 평가) | 42 | **2,434** | 897* | 1152* | 385* |

\* 2026-08-20 숫자는 **분류기 라우팅** 분포입니다. GT 면적 기준 구간과 다릅니다.

분할 파일 `checkpoints/patient_split.json`(seed 42)을 Stage 1–4와 베이스라인이 모두 공유합니다.

---

## 시각화

### TRIO vs KAIST vs NVAUTO (같은 슬라이스)

세 방법을 **같은 val 슬라이스 6장**에서 비교합니다. 행은 TRIO / KAIST / NVAUTO, 열은 Small 1–2 · Medium 1–2 · Large 1–2입니다. 초록 점선은 GT입니다. 슬라이스는 클래스 평균 DSC에 가깝도록 골랐고, 인덱스는 `113, 2286, 1121, 2295, 95, 1480`으로 고정합니다.

![TRIO vs KAIST vs NVAUTO](results/method_comparison_3x6.png)

다시 그리는 명령:

```bash
python scripts/eval/plot_three_method_grid.py --indices 113,2286,1121,2295,95,1480 --out results/method_comparison_3x6.png
```

그림에 쓰인 베이스라인 가중치는 원본이 없어서 2026-08-20에 같은 분할로 다시 학습한 것입니다(학습 중 val DSC KAIST 0.9021 / NVAUTO 0.8999). 위 정량 표의 0.8923 / 0.8971은 2026-08-19 `baselines/results/{kaist,nvauto}_metrics.json` 값입니다.

### 파이프라인 Stage 2 → Stage 3

클래스마다 **Final DSC가 그 클래스 평균에 가깝고**, Final DSC > Initial DSC인 원본 2장씩(총 6장)을 고릅니다. 초록=GT, 빨강=Stage 2 Rough, 하늘색=Stage 3 RL. 원본 MRI는 `results/pipeline_sample_{class}_{n}_original.png`로 따로 저장합니다.

![Pipeline overview](results/pipeline_sample_comparison.png)

| Small | Medium | Large |
|---|---|---|
| ![small 1](results/pipeline_sample_small_1.png) | ![medium 1](results/pipeline_sample_medium_1.png) | ![large 1](results/pipeline_sample_large_1.png) |
| ![small 2](results/pipeline_sample_small_2.png) | ![medium 2](results/pipeline_sample_medium_2.png) | ![large 2](results/pipeline_sample_large_2.png) |

베이스라인 비교 그림은 `baselines/results/`에 있습니다.

| 크기별 성능 | 임계값 스윕 |
|---|---|
| ![by size](baselines/results/baseline_by_size.png) | ![sweep](baselines/results/baseline_threshold_sweep.png) |

---

## 기술 스택

| 분야 | 사용 기술 |
|------|----------|
| Deep Learning | PyTorch, MONAI |
| Reinforcement Learning | Gymnasium, Stable-Baselines3 (PPO) |
| Dataset | BraTS 2021 (`t1ce+flair`, 128×128 2D 슬라이스) |
| Language | Python |

---

## 프로젝트 구조

```
├── run_pipeline.py                 # Stage 1–4 원스톱 실행 (시드·결정적 모드 전파)
├── configs/ppo_brats.yaml          # PPO 하이퍼파라미터
├── scripts/train/                  # 분류기 · Expert · PPO 학습
├── scripts/eval/                   # 파이프라인 평가, 임계값 스윕, TRIO 비교 그리드
├── src/data/                       # BraTS 로더, 환자 단위 분할, 2.5D·영역 레이블
├── src/envs/                       # MaskRefinementEnv (크기별 PPO 환경)
├── src/models/                     # Classifier, Experts, AdaptivePipeline
├── src/utils/                      # 지표(DSC/HD95/P/R), 게이트, zoom-crop, 시드
├── baselines/                      # BraTS21 1·2위 방법 2D 각색 비교 실험
├── checkpoints/                    # 학습된 가중치 + patient_split.json
├── results/                        # 평가 시각화 (pipeline_sample_*, method_comparison_3x6.png)
├── docs/PIPELINE.md                # 파이프라인 상세
├── docs/EXPERIMENT_RESULTS.md      # 실험 결과 통합본
└── docs/paper_draft_ko.md          # 논문 초안
```

---

## Getting Started

### 1. 환경 설치

```bash
git clone https://github.com/aninsung/2026-summer-Interdepartmental-Academic-Conference.git
cd 2026-summer-Interdepartmental-Academic-Conference
pip install -r requirements.txt
```

BraTS 2021 데이터는 `src/data/archive`에 둡니다.

### 2. 실행

```bash
python run_pipeline.py batch_size=64
```

`key=value` 형태를 `--key value`로 자동 변환하므로 `--batch_size 64`와 같습니다.

**재현성이 기본값입니다.** `--seed 42`와 결정적 모드(cuDNN deterministic)가 켜진 상태로 Stage 1–4에 전파됩니다. 속도를 위해 끄려면 `--no_deterministic`을 씁니다.

주요 옵션:

```bash
# 특정 단계만 건너뛰기
python run_pipeline.py --skip_classifier --skip_experts --skip_agents

# 평가만 직접 실행
python scripts/eval/evaluate_pipeline.py --split_role val

# Stage 2 단독 성능 (PPO·게이트 없음)
python scripts/eval/evaluate_pipeline.py --split_role val --skip_ppo

# 이진화 임계값 스윕 (확률 맵 캐시 → 재학습 불필요)
python scripts/eval/sweep_threshold.py --split_role val --plot results/threshold_sweep.png

# 베이스라인 학습·평가·비교
python baselines/run_comparison.py --epochs 20 --batch_size 32 --sweep

# TRIO / KAIST / NVAUTO 같은 슬라이스 비교 그리드
python scripts/eval/plot_three_method_grid.py --indices 113,2286,1121,2295,95,1480 --out results/method_comparison_3x6.png
```
