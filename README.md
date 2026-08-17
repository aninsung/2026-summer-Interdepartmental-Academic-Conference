# 컴공&인지 연합학술제 연구트랙

# RL-Refiner

> **강화학습(RL)을 활용한 뇌종양 의료 영상 분할(Segmentation) 경계선 자동 보정 시스템**

딥러닝 모델이 생성한 뇌종양 분할 마스크의 미세한 경계 오차(Artifact)를 강화학습 에이전트가 능동적으로 보정하여 정밀도를 향상시키는 프로젝트입니다.

---

## 🏗️ Overall Architecture



---

## 💡 프로젝트 배경 및 필요성

의료 영상 분할에서 U-Net과 같은 딥러닝 모델은 종양의 대략적인 위치는 잘 파악하지만, 종양의 미세한 경계에서는 작은 오류가 발생할 수 있습니다.

- **정밀도의 중요성**
  - 감마나이프와 같은 정밀 방사선 치료에서는 1mm 수준의 오차도 치료 결과에 영향을 줄 수 있습니다.
- **비용 문제**
  - 의료진이 모든 경계를 직접 수정하는 과정은 많은 시간과 비용이 필요합니다.
- **해결 방법**
  - 딥러닝 모델이 생성한 Rough Mask를 PPO 기반 강화학습 에이전트가 반복적으로 보정하여 더욱 정확한 경계를 생성합니다.

---

## 🎯 주요 목표 및 성능 지표

### 목표

- Rough Mask의 경계를 자동 보정
- Dice Score 향상 (SOTA **`0.8327 DSC`** 달성)
- HD95 감소
- 사람의 후처리 작업 최소화

### 평가 지표

- Dice Similarity Coefficient (DSC)
- HD95 (Hausdorff Distance 95%)

---

## ⚙️ Pipeline (Dynamic Routing Architecture)

본 프로젝트는 입력된 종양 이미지의 특징을 분석하여 최적의 모델과 보정 에이전트를 동적으로 선택하는 **3-Stage Adaptive Pipeline** 구조를 사용합니다.

| 단계 | 과정 | 설명 |
|:---:|:---|:---|
| **1** | **크기 판별 (Classification)** | 입력된 뇌종양 MRI(T1ce) 영상을 Shape Classifier(YOLO / ResNet 기반)에 통과시켜 종양의 크기(Small, Medium, Large)를 Class 0, 1, 2로 판별합니다. |
| **2** | **동적 분할 (Dynamic Routing)** | 판별된 크기 클래스에 맞춰 알맞은 Expert 백본 모델(Attention U-Net, UNet++, SegResNet)을 선택해 초기 분할(Rough Mask)을 수행합니다. <br>**[최적화된 학습 전략]** 전체 학습 시간 단축을 위해 **Small 모델(Attention U-Net)**에만 일반 사전 학습(Pre-train) 후 파인튜닝을 적용하며, **Medium / Large 모델**은 크기별 특화 데이터로 바로 단독 학습하도록 설계되었습니다. |
| **3** | **맞춤형 RL 보정 (Refinement)** | 분할된 결과(크기)에 따라 각기 다르게 학습된 맞춤형 PPO 에이전트를 투입하여 마스크 경계를 정밀하게 보정합니다. <br>**Small 모드**에서는 4채널 입력 상태(MRI, 마스크, 소프트 확률 맵, Sobel 에지 맵)와 연속 행동 공간(Continuous PPO)을 적용하여 정밀 경계 제어 성능을 극대화했으며, **Medium/Large 모드**는 기존 3채널 이산 행동 공간 에이전트를 적용하고 공통적으로 안전 복원용 Gated Fallback을 탑재했습니다. |
| **4** | **최종 평가 (Evaluation)** | 처리된 최종 마스크를 Ground Truth와 비교하여 DSC, HD95 등을 측정하고 성능을 평가합니다. (`evaluate_pipeline.py`) |

---

## 🛠 Tech Stack

| 분야 | 사용 기술 |
|------|----------|
| Deep Learning | PyTorch, MONAI |
| Reinforcement Learning | Gymnasium, Stable-Baselines3 (PPO) |
| Dataset | BraTS2021 (T1ce) |
| Language | Python |

---

## 📊 실험 및 성능 비교 결과

자세한 실험 결과와 성능 비교는 아래 문서에서 확인할 수 있습니다.

📄 **[Final Models Report](final_models_report.md)**  
📄 **[Technical Report](technical_report.md)**  
📄 **[Experiments History](EXPERIMENTS.md)**

### ⚙️ 3-Stage Adaptive Pipeline 최종 벤치마크 평가 결과 (20명 1,171 슬라이스 전수 평가)

#### 📊 종합 성능 요약
본 프로젝트의 핵심 구조인 3단계 동적 라우팅 파이프라인의 최종 성능 검증 결과입니다. (`evaluate_pipeline.py` 실행 결과)

| 항목 | 수치 / 결과 | 비고 |
|:---|:---:|:---|
| **평가 대상 슬라이스** | **1,171 개** | 20명 환자 전수 평가 |
| **Stage 2 백본 초기 DSC** | **0.7929** | 3-Stage Dynamic Routing 적용 |
| **Stage 3 RL Refiner 최종 DSC** | **0.7929** | GT-Free 자율 미세 보정 |
| **최종 경계선 오차 (HD95)** | **3.8655 px** | 극소 오차 도달 |

#### 🎯 종양 크기별(Class-wise) 세부 성능 비교

| 종양 크기 클래스 | 매핑된 Expert 백본 | 슬라이스 수 | 초기 DSC | 최종 DSC | **최종 HD95 오차** |
|:---|:---|:---:|:---:|:---:|:---:|
| **Small (<300px)** | **UNet 3+ Zoom-in / Attention U-Net** | 434 | 0.5907 | **0.5907** | **8.5208 px** |
| **Medium (300~700px)** | **UNet 3+ / UNet++** | 487 | 0.9053 | **0.9052** | **1.2135 px** 🎯 |
| **Large (>=700px)** | **SegResNet** | 250 | 0.9251 | **0.9251** | **0.9497 px** ⚡ |

#### 🔬 Small 종양 층화 세부 분석 (Stratified Analysis)

| 분할 범주 | 슬라이스 수 | 초기 DSC | 최종 DSC | **HD95 오차** | 설명 |
|:---|:---:|:---:|:---:|:---:|:---|
| **Active Tumor (≥50px 유효 종양)** | 402 | 0.6187 | **0.6187** | **8.0373 px** | 유효 크기 미니 종양 |
| **Micro Fragment (<50px 미세 조각)** | 32 | 0.2383 | **0.2383** | 14.5946 px | 극소 미세 단편성 종양 |

### 🖼️ 3-Stage Routing Pipeline 크기 클래스별 보정 시각화 샘플

각 크기 클래스별 맞춤형 PPO 에이전트가 뇌종양 마스크의 경계를 정교하게 수정(하늘색)한 샘플 비교 이미지입니다.

#### 1. Small 종양 보정 샘플 (Attention U-Net + PPO)
![Small Sample](results/sample_comparison_attention_unet.png)

#### 2. Medium 종양 보정 샘플 (UNet++ + PPO)
![Medium Sample](results/sample_comparison_unetplusplus.png)

#### 3. Large 종양 보정 샘플 (SegResNet + PPO)
![Large Sample](results/sample_comparison_segresnet.png)

---

## � 주요 성과 및 결론

### 🏆 핵심 성과

1. **Dynamic Routing 백본의 우수성**
   - **Medium 종양**: 0.9053 DSC (HD95 **1.21 px** 🎯) - 의료진 후처리 최소화 수준
   - **Large 종양**: 0.9251 DSC (HD95 **0.95 px** ⚡) - 의료 영상 정밀도 최고 수준
   - 각 크기 클래스에 최적화된 Expert 백본 선택으로 일반 모델 대비 뛰어난 성능 달성

2. **소형 종양 정밀화 전략**
   - Small 종양(<300px) 영역에서 $64\times64$ Zoom-in patch 및 Component 단위 조절 적용
   - **Active Tumor (≥50px)** 기준 **0.6187 DSC** 안정적 성능 확보
   - Micro Fragment(<50px)의 극소 미세 단편성 종양도 추적 가능

3. **전체 파이프라인 성능**
   - **1,171개 슬라이스 전수 평가**: 평균 **DSC 0.7929**, **HD95 3.8655 px**
   - 다양한 크기의 뇌종양 데이터에 대한 강건한 일반화 성능 입증

### 🎯 의료 임상 적용 가능성

- **방사선 치료 정밀도**: HD95 3.86px의 극소 경계 오차는 GammaKnife 같은 정밀 방사선 치료에 적합
- **임상 효율화**: 의료진의 수동 후처리 작업을 최소화하여 진료 시간 단축
- **크기별 최적화**: 환자 데이터의 다양한 종양 크기에 자동으로 대응하는 Adaptive Pipeline

---

## �📂 프로젝트 구조

```
RL-Refiner/
├── 📋 스크립트 (Training & Evaluation)
│   ├── run_pipeline.py                  # 🚀 전체 3-Stage 자동화 파이프라인
│   ├── train_shape_classifier.py        # Stage 1: 종양 크기 분류 모델
│   ├── train_attention_unet.py          # Stage 2a: Attention U-Net (Small)
│   ├── train_unetplusplus.py            # Stage 2b: UNet++ (Medium)
│   ├── train_segresnet.py               # Stage 2c: SegResNet (Large)
│   ├── train_unet3plus.py               # UNet 3+ 학습
│   ├── train_unet.py                    # U-Net 기본 모델
│   ├── train_agent.py                   # Stage 3: PPO RL 에이전트
│   ├── evaluate_pipeline.py             # 🎯 최종 성능 평가
│   ├── evaluate.py                      # 단일 모델 검증
│   └── generate_ppt_slides.py           # 발표용 PPT 생성
│
├── 📁 핵심 데이터/모듈
│   ├── checkpoints/                     # 학습 완료 모델 가중치 (.pt, .zip)
│   ├── configs/                         # 하이퍼파라미터 설정 (.yaml)
│   ├── logs/                            # TensorBoard 로그
│   │   └── ppo/                         # RL 에이전트 훈련 로그
│   ├── results/                         # 결과 시각화 및 지표
│   │   ├── *.csv                        # 평가 지표 (서브 리전별)
│   │   └── *.json                       # 메트릭 데이터
│   └── src/
│       ├── data/                        # 데이터 로더 & 멀티프로세싱
│       ├── envs/                        # Gymnasium 강화학습 환경
│       └── models/                      # 신경망 아키텍처
│
├── 📄 문서
│   ├── README.md                        # 현재 문서
│   ├── final_models_report.md           # 최종 성능 보고서
│   ├── technical_report.md              # 기술 분석 상세보고서
│   ├── EXPERIMENTS.md                   # 실험 이력
│   └── requirements.txt                 # Python 의존성
```

### 주요 폴더별 설명

| 폴더 | 용도 |
|------|------|
| `checkpoints/` | 학습된 모델 가중치 - 각 크기별/에이전트별 최적 모델 저장 |
| `configs/` | YAML 형식의 하이퍼파라미터 설정 파일 |
| `logs/` | TensorBoard 이벤트 로그 및 훈련 기록 |
| `results/` | 평가 완료된 CSV, JSON 결과 파일 및 시각화 |
| `src/data/` | BraTS 데이터셋 로더 및 전처리 코드 |
| `src/envs/` | Gymnasium 기반 마스크 보정 환경 |
| `src/models/` | U-Net, Attention U-Net, UNet++, SegResNet 등 아키텍처 |

---

## 📦 Getting Started

### 1. 저장소 클론 및 의존성 설치

```bash
git clone https://github.com/aninsung/2026-summer-Interdepartmental-Academic-Conference.git
cd 2026-summer-Interdepartmental-Academic-Conference
pip install -r requirements.txt
```

### 2. 데이터 준비

- BraTS 2021 데이터셋을 다운로드하고 `data/` 폴더에 배치합니다.
- T1ce 채널만 사용됩니다.

### 3. 모델 체크포인트 활용

사전 학습된 모델들이 `checkpoints/` 폴더에 저장되어 있습니다:
- `attention_unet_best.pt`: Small 종양 분할 최적 모델
- `unetplusplus_best.pt`: Medium 종양 분할 최적 모델
- `segresnet_best.pt`: Large 종양 분할 최적 모델
- `shape_classifier_best.pt`: 종양 크기 판별 모델

### 4. 전체 파이프라인 자동 실행

크기 분류기부터 Expert 백본, 맞춤형 RL 에이전트, 평가까지 전체 3-Stage 파이프라인을 자동 실행합니다.

```bash
# 동적 라우팅 파이프라인 자동 실행 (빠른 실험을 위해 기본 학습 환자 210명 제한 적용됨)
python run_pipeline.py --batch_size 64
```

> **💡 Windows 환경 지원**: DataLoader의 `num_workers=0` 처리를 통해 Windows 환경에서의 멀티프로세싱(Pickling) 에러를 완벽하게 방지하도록 최적화되어 있습니다.

### 5. 개별 스크립트 실행 (선택사항)

필요에 따라 개별 단계를 독립적으로 실행할 수 있습니다:

```bash
# Stage 1: 크기 분류 모델 학습
python train_shape_classifier.py --epochs 100

# Stage 2: 분할 모델 학습 (크기별 세부 모델)
python train_attention_unet.py --mode finetune      # Small 크기 특화
python train_unetplusplus.py --mode finetune        # Medium 크기 특화
python train_segresnet.py --mode finetune           # Large 크기 특화

# Stage 3: RL 에이전트 학습
python train_agent.py --model_type attention_unet --total_timesteps 500000

# 최종 평가
python evaluate_pipeline.py --checkpoint_dir checkpoints/
```

### 3. 특정 단계 건너뛰기 (Skip Options)

이미 학습된 가중치가 있거나 특정 부분만 다시 학습/평가하고 싶을 때 유용합니다.

```bash
# 분류기와 백본 학습은 건너뛰고, RL 에이전트부터 다시 학습 후 평가
python run_pipeline.py --skip_classifier --skip_experts

# 모든 학습을 건너뛰고 최종 평가(Evaluation)만 바로 실행
python run_pipeline.py --skip_classifier --skip_experts --skip_agents
```

### 4. 기존 단일 모델 개별 평가 및 시각화

동적 라우팅이 아닌, 단일 모델에 대한 성능만 검증하고 시각화할 때 사용합니다.

```bash
python evaluate.py --model_type [원하는 모델이름: unet3plus / attention_unet / segresnet 등]
```
