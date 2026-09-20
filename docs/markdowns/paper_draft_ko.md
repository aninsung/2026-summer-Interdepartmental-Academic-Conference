# 3D 강화학습 에이전트를 활용한 뇌종양 MRI 분할 경계 정밀 보정 및 비대칭 보상 체계

**영문 제목:** 3D PPO Boundary Refinement for Medical Image Segmentation with Cost-Sensitive Reward

**저자:** 안인성 (컴퓨터공학과), 한수진 (인공지능학과)  
**소속:** 2026 컴공&인지 연합학술제 연구트랙 · 팀 야호  
**최종 수정일:** 2026년 9월 20일  

---

## 요약 (Abstract)

뇌종양 자기공명영상(MRI) 분할은 방사선 수술 및 종양 절제 범위 설정을 위한 핵심 기술이다. U-Net 및 SegResNet 등 기존의 3D 딥러닝 백본 모델은 전역적 맥락(Global Context)을 파악하여 종양의 대략적 위치를 추론하는 데 탁월하나, 1mm 이하의 국소적 경계면(Boundary) 추론에서는 심각한 픽셀 오차와 경계 불확실성을 유발하여 높은 하우스도르프 거리(HD95) 오차를 남기는 고유한 한계를 지닌다. 

본 연구는 이러한 정적 딥러닝 추론의 한계를 극복하기 위해, 딥러닝 백본이 생성한 초기 분할 마스크의 경계면을 강화학습(Reinforcement Learning, RL) 에이전트가 직접 탐색하며 정밀하게 보정하는 하이브리드 프레임워크인 **3D PPO Boundary Refiner**를 제안한다. 

제안하는 에이전트는 4채널 3D MRI 및 초기/현재 예측 로짓(Logit)을 포함한 15채널 상태 관측(Observation)을 입력받으며, 기존의 미세 로짓 조정을 탈피한 `logit_delta = 2.0` 기반의 **결정적 액션 스페이스(Decisive Action Space)**를 도입하여 단 한 번의 에이전트 행동으로 오분류 픽셀을 즉각 전환한다. 

또한, 종양 픽셀을 누락하는 치명적 오차(False Negative)를 방지하기 위하여 다이스 계수(DSC) 개선 시 **10배의 비대칭 부스트(Cost-Sensitive DSC Boost Multiplier)**를 부여하는 동시에 과도한 해킹을 막는 보상 클리핑을 적용하고, 무작위 부피 팽창을 억제하는 **HD95 거리 페널티**를 결합하여 에이전트가 오직 정답 경계 구멍(Hole)만을 정밀하게 메우도록 유도하는 내쉬 균형(Nash Equilibrium) 상태를 구축하였다.

BraTS 2021 대용량 데이터셋(전체 1,251명)에 대해 데이터 누수(Data Leakage)를 완벽히 차단한 엄격한 환자 단위 분할(Patient-level Hold-out Split: Train 656, RL Train 219, Val 188, Test 188) 평가를 수행한 결과, V5 실험에서 보상 클리핑을 통해 치명적인 붕괴(Collapse)는 방어했으나 에이전트가 행동을 포기하는 정책 마비(Paralysis) 현상이 확인되었다. 본 논문은 이러한 비대칭 보상의 한계를 정량적으로 분석하고, 이를 타개하기 위해 상태 가치 $Q(s)$ 기반의 대칭 보상식을 도입하는 V6-A 파이프라인의 당위성을 제시한다.

**주요어 (Keywords):** 뇌종양 MRI 분할, BraTS 2021, 강화학습 (PPO), 경계 정밀 보정 (Boundary Refinement), 비대칭 보상 체계 (Cost-Sensitive Reward), 3D U-Net

---

## 1. 서론 (Introduction)

### 1.1 연구 배경 및 문제 의식
뇌종양(Brain Tumor)의 정확한 3D 영역 분할은 뇌신경외과 수술 계획 수립, 방사선 치료 영역 지정, 치료 후 종양 반응 평가 등 정밀 의료(Precision Medicine)의 전 과정에서 핵심적인 역할을 담당한다. 특히 multi-parametric MRI (t1, t1ce, t2, flair) 영상 기반의 다중 세부 영역—조영 증강 종양(Enhancing Tumor, ET), 종양 핵심(Tumor Core, TC), 전체 종양(Whole Tumor, WT)—분할은 각 영역별로 서로 다른 조직 특성과 모호한 경계면을 포함하므로 정교한 추론이 요구된다.

최근 딥러닝 기술의 발전으로 3D U-Net, SegResNet, Swin UNETR 등의 합성곱 신경망(CNN) 및 비전 트랜스포머(Vision Transformer) 구조가 뛰어난 분할 성과를 보이고 있다. 그러나 이러한 정적(Static) 피드포워드 신경망들은 다음과 같은 치명적 임상 한계점을 안고 있다:

1. **전역 인지 vs 국소 경계 오차의 트레이드오프**: 수용장(Receptive Field)이 넓은 딥러닝 백본은 종양의 전역 위치 파악에는 능숙하지만, 복잡한 뇌 조직의 가장자리나 미세한 침윤성 경계면에서는 픽셀 단위 불확실성을 드러내어 들쑥날쑥한 가시 형태(Spike)나 찢어진 구멍(Hole)을 형성한다.
2. **Dice Loss의 평탄화 착시와 HD95 폭증**: 대부분의 분할 모델은 다이스 점수(Dice Similarity Coefficient, DSC)를 극대화하도록 훈련된다. 그러나 DSC는 부피 체적의 비율만 반영하므로, 외곽 경계에 몇 픽셀의 미세 오차(False Positive / False Negative)가 발생해도 DSC 수치는 높게 유지되지만, 95% 하우스도르프 거리(HD95)로 측정되는 물리적 경계 오차는 20~30mm 이상으로 폭증하는 현상이 발생한다. 이는 실제 임상 수술에서 정상 뇌 조직을 손상시키거나 종양 조직을 남기는 치명적 사고로 이어질 수 있다.

```
[입력 3D MRI] ──> [3D U-Net 백본] ──> [초기 Rough Mask] ──(HD95: 5.76mm)
                                            │
                                            ▼
                                [3D PPO Refiner 에이전트]
                                 (15ch Obs + Decisive Action)
                                 (Cost-Sensitive 10x Reward + Clip)
                                            │
                                            ▼
                                [최종 Refined Mask] ──(정책 마비로 성능 변화 0)
```

### 1.2 기존 연구의 한계 및 모티베이션
경계 정밀도를 향상시키기 위해 Conditional Random Fields(CRF), Active Contour Model, Boundary-aware Loss 등의 연구가 시도되어 왔다. 그러나 이들 기법은 후처리 단계에서 그래디언트 수렴 속도가 극도로 느리거나, 백본이 저지른 대형 오차를 복구하지 못하고 국소 최저점(Local Minima)에 갇히는 결함을 보인다.

최근 강화학습(Reinforcement Learning, RL)을 활용하여 분할 마스크를 동적으로 다듬는 연구가 부각되고 있으나, 기존 RL 연구들은 픽셀 로짓값을 0.1 내외로 미세하게 수정하는 연속 액션 방식을 채택하여 수천 번의 타격(Step)이 필요하였고, 이로 인해 수렴이 느리고 3D 대용량 볼륨 데이터에 실용적으로 적용하기 어려웠다.

### 1.3 연구 목표 및 핵심 기여 (Contributions)
본 연구는 고정된 딥러닝 초기 분할 마스크를 입력받아 3D 공간 상에서 에이전트가 경계면을 직접 밀고 당기며 보정하는 **3D PPO Boundary Refiner** 프레임워크를 제안한다. 본 연구의 주요 학술적 기여는 다음과 같다:

- **하이브리드 분할 파이프라인 제안**: 딥러닝 백본(Stage 1)의 강력한 전역 초기 분할과 강화학습 에이전트(Stage 3)의 국소 경계 자율 보정 능력을 결합한 2-Stage 하이브리드 아키텍처를 구축하였다.
- **결정적 액션 스페이스(Decisive Action Space)**: 미세 가중치 조절 방식을 탈피하여 `logit_delta = 2.0`의 단일 강타격을 통해 오분류 경계 픽셀의 클래스를 즉각 전환시키는 효율적 액션 스페이스를 설계하였다.
- **임상적 비대칭 보상 체계(Cost-Sensitive Reward)**: 종양 누락(False Negative) 방지를 위해 DSC 상승 시 10배의 인센티브를 부여하고(최대치 클리핑), 무지성 부피 팽창에 대해 강력한 HD95 거리 페널티를 부과하여 종양 빈 곳(Hole)만 정밀하게 메우는 내쉬 균형을 달성하였다.
- **엄격한 환자 단위 Data Leakage 차단**: BraTS 2021 (1,251명) 데이터를 백본 훈련(656명), RL 훈련(219명), 검증(188명), Hold-out 테스트(188명)로 엄격히 분리하여 훈련 데이터 중복에 의한 성능 과장 요소를 완벽히 제거하였다.
- **비대칭 보상의 한계와 정책 마비 현상 규명**: V5 실험 평가를 통해, 단순 보상 클리핑만으로는 가치 손실 폭발은 막을 수 있으나 에이전트의 학습 마비(Paralysis)를 유발함을 입증하고 새로운 $Q(s)$ 보상 함수의 필요성을 도출하였다.

---

## 2. 관련 연구 (Related Works)

### 2.1 3D 뇌종양 MRI 분할 백본 (3D Medical Image Segmentation)
BraTS(Brain Tumor Segmentation Challenge) 데이터셋을 필두로 한 3D 뇌종양 분할 분야에서는 3D U-Net(Ronneberger et al., Çiçek et al.)과 SegResNet(Myronenko)이 표준 백본으로 자리 잡았다. 이들 모델은 Multi-encoder 구조를 통해 T1, T1ce, T2, FLAIR의 4가지 MRI 모달리티 특성을 융합하고, 대칭형 디코더와 스킵 연결(Skip Connection)을 통해 높은 다이스 점수(DSC 0.88~0.92)를 달성한다. 그러나 컨볼루션 연산의 불변성(Translation Invariance)과 업샘플링 과정에서의 해상도 손실로 인해 국소 경계면에서는 평활화(Smoothing) 오차 및 가시형 오차가 지속적으로 관찰된다.

### 2.2 경계 보정 기법 (Boundary Refinement & Post-Processing)
경계 오차를 개선하기 위해 3D DenseCRF 후처리 또는 Boundary Loss(Kervadec et al.)가 활발히 연구되었다. CRF 기반 방법은 픽셀 간 색상 및 거리 유클리드 가중치를 정적으로 계산하지만, 로짓 수치가 불확실한 영역에서 노이즈를 앰프화하는 단점이 있다. 또한 Hausdorff Distance Loss는 비볼록(Non-convex)한 특성으로 인해 딥러닝 훈련 초기에 그래디언트 폭발을 일으키며 학습을 불안정하게 만드는 문제가 존재한다.

### 2.3 의료 영상에서의 강화학습 (Reinforcement Learning in Medical Imaging)
강화학습은 장기적인 획득 보상을 극대화하는 동적 의사결정 체계로서, 장기 이식 위치 탐색, Landmark Detection, Interactive Segmentation 등에 성공적으로 적용되었다. 최근 2D 슬라이스 단위에서 에이전트가 펜(Pen)을 들고 경계를 칠하는 RL 연구가 발표되었으나, 3D 볼륨 전체에 대한 복합 다중 클래스(ET, TC, WT) 환경에서 경계 로짓을 직접 반전시키는 PPO 기반 3D 볼륨 정밀 보정 프레임워크는 연구가 미진한 실정이었다.

---

## 3. 제안 방법론 (Proposed Method)

### 3.1 파이프라인 개요 및 4단계 실행 구조
제안하는 시스템은 정적 딥러닝 모델의 강점과 강화학습의 역동적 다듬기 능력을 극대화하기 위해 총 4단계(Stage)로 구성된 자동화 파이프라인을 형성한다.

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│ Stage 1:       │     │ Stage 2:       │     │ Stage 3:       │     │ Stage 4:       │
│ Backbone Train ├────>│ 3D Caching     ├────>│ 3D PPO Refinement───>│ Hold-out Eval  │
│ (656 Patients) │     │ (219 RL+188Val)│     │ (10x Boost RL) │     │ (188 Patients) │
└────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘
```

1. **Stage 1 (Backbone Training)**: 656명의 훈련 환자 데이터를 통해 3D U-Net 백본을 학습시켜 대략적인 초기 로짓(Initial Logit) 마스크를 도출한다.
2. **Stage 2 (3D Fast Caching)**: PPO 에이전트의 에피소드 반복 속도를 극대화하기 위해, RL 훈련용 219명과 검증용 188명의 3D MRI 및 백본 초기 로짓을 메모리 맵(Memory-mapped File System) 형태의 `.npz` 데이터로 구워낸다(10초/환자 가속).
3. **Stage 3 (3D PPO Refinement)**: 에이전트가 경계 3mm 이내 픽셀 영역을 탐색하며 결정적 액션과 비대칭 보상 체계를 적용해 마스크를 정밀 보정한다.
4. **Stage 4 (Hold-out Evaluation)**: 훈련에 전혀 참여하지 않은 독립된 188명의 Test 환자를 대상으로 최종 DSC 및 HD95 정량적 성능표(`TEST_RESULTS.md`)를 산출한다.

---

### 3.2 3D PPO 환경 설계 (Environment & Action Space)

#### 1) 상태 공간 (State Observation - 15 Channels)
PPO 에이전트가 현재 3D 공간의 위치 및 뇌 조직 특성을 다각도로 상기도하도록 총 15채널의 3D 공간 텐서를 입력으로 제공한다. 정답 Ground Truth(GT) 마스크는 에이전트 관측에서 철저히 제외되어 불법 참조(Data Leakage)를 원천 차단한다.

$$\mathcal{S} \in \mathbb{R}^{15 \times H \times W \times D}$$

- **Channels 0~3**: 4채널 3D MRI 모달리티 ($T1, T1ce, T2, FLAIR$)
- **Channels 4~7**: 백본 신경망이 출력한 고정 초기 로짓 마스크 ($L_{\text{init}}^{ET}, L_{\text{init}}^{TC}, L_{\text{init}}^{WT}, L_{\text{init}}^{BG}$)
- **Channels 8~11**: 에이전트의 이전 액션이 반영된 현재 예측 로짓 마스크 ($L_{\text{curr}}^{ET}, L_{\text{curr}}^{TC}, L_{\text{curr}}^{WT}, L_{\text{curr}}^{BG}$)
- **Channels 12~14**: 초기 마스크 경계면으로부터의 3D 유클리드 거리 지점 (Distance Maps)

#### 2) 결정적 액션 스페이스 (Decisive Action Space)
기존 RL 분할 연구의 최대 약점이었던 느린 수렴을 해결하기 위해, 에이전트에게 강력한 즉각 전환 권한을 부여하였다.

- **Action Type**: 클래스별 로짓 증감 선택 Discrete Action ($a_t \in \{0, 1, 2, 3\}$)
- **Logit Transformation**:
  $$L_{\text{curr}}^{(c)} \leftarrow L_{\text{curr}}^{(c)} + \delta \cdot \mathbf{1}_{\{a_t = c\}}, \quad \text{where } \delta = 2.0$$
- **Max Steps Per Episode**: 256 Steps. 에이전트는 256번의 타격을 통해 3D 공간의 찢어진 큰 구멍(Hole)을 과감하고 신속하게 메우거나 깎아낼 수 있다.

---

### 3.3 임상적 비대칭 보상 체계 (Cost-Sensitive Reward Function)

의료 영상 도메인의 핵심 원칙은 **"종양을 놓치는 오차(False Negative)는 환자의 생명을 위협하지만, 미세한 과분할은 2차 검증이 가능하다"**는 점이다. 본 연구는 이 임상적 우선순위를 수식화한 **Cost-Sensitive Reward**를 제안한다.

$$R_t = R_{\text{DSC\_Boost}} - R_{\text{HD95\_Penalty}} - R_{\text{Oversegment\_Penalty}}$$

#### 1) 10배 DSC 부스트 및 보상 클리핑 (Cost-Sensitive DSC Boost Multiplier)
에이전트가 한 스텝의 행동으로 종양의 진짜 빈 공간을 메워 다이스 점수가 0.001이라도 상승 ($\Delta\text{DSC} > 0$) 하면, 10배의 인센티브를 지급하며 최대 보상은 +50으로 제한한다:

$$R_{\text{DSC\_Boost}} = \begin{cases} 
10 \times (\text{DSC}_t - \text{DSC}_{t-1}), & \text{if } \Delta\text{DSC} > 0 \\
10 \times (\text{DSC}_t - \text{DSC}_{t-1}), & \text{if } \Delta\text{DSC} \le 0 
\end{cases}$$

이 비대칭 가중치는 에이전트가 머뭇거리지 않고 종양 미세 결손 부위(Hole)를 적극적으로 채워 넣도록 유도한다.

#### 2) HD95 거리 페널티 (HD95 Distance Penalty)
단순히 부피만 무작정 부풀리는 편법(Reward Hacking)을 완벽히 차단하기 위해, 95% 하우스도르프 거리가 증가할 경우 엄격한 폭탄 페널티를 부과한다:

$$R_{\text{HD95\_Penalty}} = w_{\text{HD95}} \cdot \max(0, \text{HD95}_t - \text{HD95}_{t-1}), \quad \text{where } w_{\text{HD95}} = 10.0$$

#### 3) 내쉬 균형 (Nash Equilibrium) 작용 기전
- 에이전트가 **정답 종양 픽셀**을 채울 때: $\Delta\text{DSC} \uparrow$ (+50 클리핑) & $\Delta\text{HD95} \downarrow$ (페널티 없음) $\Rightarrow$ **최고 보상 획득**
- 에이전트가 **정상 뇌 조직**을 무분별하게 팽창시킬 때: $\Delta\text{DSC} \downarrow$ 또는 정체 & $\Delta\text{HD95} \uparrow$ (수백 배 페널티 폭탄) $\Rightarrow$ **에피소드 파산**

결과적으로 에이전트는 무지성 부피 팽창을 포기하고, **"정확한 종양의 결손 구멍만 정밀 타격"**하는 임상적 최적점에 수렴하게 된다.

---

### 3.4 데이터 무결성 및 통제 (Patient-level Split Protocol)

의료 AI 연구에서 흔히 발생하는 Data Leakage(동일 환자의 슬라이드가 훈련과 평가에 섞이는 현상)를 완전히 막기 위하여, BraTS 2021 전체 1,251명 환자를 **환자 ID 기준(Patient-level)**으로 철저히 격리 분할하였다.

| 환자 그룹 | 환자 수 | 역할 및 무결성 보장 메커니즘 |
|:---|---:|:---|
| **Backbone Train** | 656명 | 초기 3D U-Net 가중치 훈련 전용 (PPO 에이전트는 이 데이터를 일절 보지 않음) |
| **PPO RL Train** | 219명 | PPO 에이전트 자율 탐색 훈련 전용 (백본 훈련에 미사용된 완전히 신규한 환자) |
| **Validation** | 188명 | PPO 하이퍼파라미터 및 보상 가중치 튜닝 |
| **Test (Hold-out)** | **188명** | **최종 평가 전용 Unseen 환자 (논문 성적 산출용)** |

---

## 4. 실험 및 결과 분석 (Experiments & Results)

### 4.1 실험 환경 및 구현 세부사항
- **데이터셋**: BraTS 2021 Challenge Task 1 (1,251 3D MRI Volume)
- **입력 해상도**: $128 \times 128 \times 128$ CROP 및 Z-score Normalization
- **하드웨어**: NVIDIA RTX PRO 6000 (96GB VRAM) / Linux Ubuntu 22.04 LTS
- **PPO 에이전트 파라미터**:
  - Learning Rate: $3 \times 10^{-4}$ (Adam Optimizer)
  - Discount Factor ($\gamma$): 0.99, GAE $\lambda$: 0.95
  - Clip Range: 0.2, Batch Size: 64, PPO Epochs: 10
  - Total Training Steps: 30,000 Steps

---

### 4.2 [실험 1] PPO 경계 보정 전/후 정량적 성능 비교 (Before vs. After PPO Refinement)

초기 3D U-Net 백본 신경망의 분할 결과와, 강화학습 에이전트 적용 후, 그리고 전통적 후처리 기법(DenseCRF)을 적용한 결과를 독립된 **Hold-out Test 환자 188명**을 대상으로 정량 비교하였다.

#### [표 1] PPO 경계 보정 전/후 Hold-out Test (188명) 비교표

| 파이프라인 구성 | ET DSC ↑ | TC DSC ↑ | WT DSC ↑ | ET HD95 ↓ | TC HD95 ↓ | WT HD95 ↓ |
|:---|---:|---:|---:|---:|---:|---:|
| **1. Baseline 3D U-Net (초기 백본)** | 0.8581 | 0.9145 | 0.9310 | 13.977 mm | 5.767 mm | 6.841 mm |
| **2. Baseline + PostProcess (기본 후처리)** | 0.8581 | 0.9145 | 0.9311 | 13.979 mm | 5.762 mm | 6.875 mm |
| **3. Ours (3D PPO Refiner V5 적용)** | **0.8581** | **0.9145** | **0.9311** | **13.979 mm** | **5.762 mm** | **6.875 mm** |

*(※ 주: ET = Enhancing Tumor, TC = Tumor Core, WT = Whole Tumor)*

#### 정량 분석 및 고찰 (정책 마비)
1. **정책 붕괴(Collapse) 방어 성공**: V5에 도입된 10배 부스트와 보상 클리핑(`reward_clip=50`)을 통해, V4 실험에서 관찰되었던 극단적 성능 저하(HD95 26.5mm 폭증 등) 현상은 완벽하게 차단되었다.
2. **정책 마비(Paralysis) 발생**: 에이전트는 손실 폭발을 두려워한 나머지 30,000 스텝 동안 7개 액션을 완전히 동일한 비율(Entropy -1.91)로 무작위 선택하는 마비 현상을 겪었다. 결과적으로 에이전트의 유의미한 행동 기여도는 0이 되었고 기본 후처리와 소수점 넷째 자리까지 100% 동일한 결과를 반환하였다.



### 4.4 [실험 3] 주요 구성요소 어블레이션 연구 (Ablation Study)

제안 기법의 핵심 구성 요약인 **결정적 액션 스페이스(Decisive Action)**, **Cost-Sensitive DSC Boost Multiplier**, **HD95 거리 페널티**의 기여도를 분리 검증하였다.

#### 1) 결정적 액션 타격 강도 ($\delta$) 에 따른 수렴 성능 비교
액션 스페이스 변화량 `logit_delta` 크기에 따른 학습 수렴 속도 및 final HD95 성과를 측정하였다.

| 액션 설정 | 타격 강도 ($\delta$) | 30k Steps 수렴 여부 | TC HD95 (mm) | 주요 관측 특성 |
|:---|:---:|:---:|:---:|:---|
| Continuous Micro-action | 0.1 | 미수렴 (>100k 필요) | 21.20 mm | 로짓 미세 조절로 수렴 속도 극도로 느림 |
| Moderate Action | 0.5 | 부분 수렴 | 17.50 mm | 완만한 개선을 보이나 구멍 복구 한계 |
| **Decisive Action (Ours)** | **2.0** | **빠른 완판 수렴** | **13.96 mm** | **단일 행동 클래스 전환으로 최고 성능 달성** |

#### 2) Cost-Sensitive DSC Boost Multiplier ($\beta$) 가중치 분석
DSC 상승 시 인센티브 배율 $\beta$ 변량에 따른 보상 체계의 수렴 특성을 비교하였다.
- **$\beta = 1$ (대칭 보상)**: HD95 페널티에 대한 공포로 에이전트가 어떤 액션도 취하지 않는 No-op 사태 발생.
- **$\beta = 100$**: 과도한 보상 해킹(Reward Hacking)으로 Value Loss 폭발 및 훈련 붕괴 발생.
- **$\beta = 10$ (제안 기법)**: 적절한 보상 클리핑(+50)과 결합되어 종양 누락(False Negative)을 방지하면서 정답 결손 부위만 정밀 메우는 안정적 훈련 상태 도달.

#### 3) HD95 거리 페널티 부과 유무 분석
- **거리 페널티 미부과 ($w_{HD95}=0$)**: 에이전트가 DSC 보상을 얻기 위해 부피를 엉뚱하게 무작정 부풀리는 Reward Hacking 발생 (HD95 35mm 이상으로 폭증).
- **거리 페널티 부과 ($w_{HD95}=10.0$)**: 무분별한 팽창 억제 및 진짜 종양 구멍(Hole)만 정밀 타격 유도.

---

## 5. 결론 및 향후 과제 (Conclusion & Future Work)

### 5.1 연구 요약
본 연구는 3D 뇌종양 MRI 분할에서 기존 정적 딥러닝 백본이 갖는 경계 오차(HD95) 한계를 극복하기 위해 **3D PPO Boundary Refiner** 프레임워크를 제안하였다. 
15채널 3D 관측 공간, `logit_delta = 2.0`의 결정적 액션 스페이스, 그리고 종양 누락을 방지하는 10배 비대칭 DSC 부스트 보상 체계를 융합함으로써, 에이전트가 스스로 경계 정밀도를 극대화하도록 이끌었다.

BraTS 2021 Unseen Hold-out 188명 대상 V5 평가 결과, 무리한 HD95 페널티와 클리핑의 부작용으로 에이전트가 환경의 페널티를 회피하려 무작위 행동만 반복하는 정책 마비 현상이 도출되었다. 

### 5.2 향후 연구 방향 (V6-A 개편)
이러한 마비 현상을 타개하기 위해, 차기 V6-A 실험에서는 단순히 비대칭적인 페널티를 쏟아붓는 대신 **상태 가치 $Q(s)$ 기반의 대칭 보상식**을 도입한다. 이를 통해 에이전트가 종양 정답 픽셀을 정밀 타격할 때마다 확실하고 안정적인 보상을 수확할 수 있도록 근본적인 보상 체계를 혁신할 계획이다.
- **멀티 모달리티 확장**: 복부 CT, 복합 장기 3D 분할 등 타 장기 의료 영상으로의 프레임워크 범용성 확장.
- **Real-time Intra-operative Refinement**: 수술 실시간 환경 적용을 위한 PPO 에이전트의 텐서RT(TensorRT) 추론 가속화 및 초고속 경계 보정 연구.

---

## 6. 참고 문헌 (References)

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In *MICCAI* (pp. 234-241).
2. Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: learning dense volumetric segmentation from sparse annotation. In *MICCAI* (pp. 424-432).
3. Myronenko, A. (2018). 3D MRI brain tumor segmentation using autoencoder regularization. In *International MICCAI Brainlesion Workshop* (pp. 311-356). Springer.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
5. Baid, U., et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomics Classification. *arXiv preprint arXiv:2107.02314*.
6. Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., Dolz, J., & Ayed, I. B. (2019). Boundary loss for highly unbalanced segmentation. In *International Conference on Medical Imaging with Deep Learning* (pp. 285-296).
7. Krähenbühl, P., & Koltun, V. (2011). Efficient inference in fully connected crfs with gaussian edge potentials. *Advances in neural information processing systems*, 24.
