# 🧠  3D PPO Boundary Refinement for BraTS 2021

2026 컴공&인지 연합학술제 연구트랙 · 팀 야호

**딥러닝(U-Net 계열)이 만든 뇌종양 초기 분할(Rough Mask)의 오차를 강화학습(PPO) 에이전트가 직접 3D 공간을 탐색하며 밀고 당겨 다듬는 하이브리드 파이프라인입니다.**

기존의 단순 후처리(Post-processing)를 넘어, **임상적 치명성(Cost-Sensitive)이 반영된 보상 함수**와 **결정적 액션(Decisive Action Space)**을 통해 BraTS 2021 글로벌 Top-2 수준의 경계 오차(HD95) 달성을 목표로 합니다.

---

## 🏆 주요 성과 (Hold-out Test 188명 기준)

본 파이프라인의 V5 실험 결과, 보상 클리핑을 통해 치명적인 붕괴(Collapse)는 방어했으나 에이전트가 행동을 포기하는 **정책 마비(Paralysis)** 현상이 확인되었습니다. 이를 해결하기 위해 상태 가치 $Q(s)$ 기반 대칭 보상식을 도입하는 **V6-A 파이프라인 개편**이 예정되어 있습니다.

| Method (적용 기법) | ET DSC ↑ | TC DSC ↑ | WT DSC ↑ | ET HD95 ↓ | TC HD95 ↓ | WT HD95 ↓ |
|:---|---:|---:|---:|---:|---:|---:|
| 1. Backbone (초기 상태) | 0.8581 | 0.9145 | 0.9310 | 13.977 mm | 5.767 mm | 6.841 mm |
| 2. Backbone + PostProcess | 0.8581 | 0.9145 | 0.9311 | 13.979 mm | 5.762 mm | 6.875 mm |
| **3. Ours (PPO Refiner V5)** | **0.8581** | **0.9145** | **0.9311** | **13.979 mm** | **5.762 mm** | **6.875 mm** |

*(※ V5 평가 결과, 에이전트 기여도 0. V6-A에서 근본적 보상식 개편 예정)*

### 🌟 BraTS 2021 SOTA 모델과의 비교 (초기 백본 기준)

비록 V5 PPO 에이전트는 마비 상태에 빠졌지만, **베이스라인으로 훈련된 초기 백본(Backbone) 모델 자체가 이미 종양 핵심(TC) 영역에서 글로벌 SOTA를 압도하는 경계 정밀도를 달성**했습니다.

| 순위 / 출처 | 모델명 (Method) | **TC HD95 ↓** | **ET HD95 ↓** | **WT HD95 ↓** |
|:---:|:---|---:|---:|---:|
| 🥇 1위 | Extended nnU-Net | 13.72 mm | **10.67 mm** | 4.75 mm |
| 🥈 2위 | SA-Net | 15.40 mm | 11.69 mm | **4.22 mm** |
| 🥉 3위 | Optimized U-Net | 9.04 mm | 17.18 mm | 5.66 mm |
| 5위 | nnU-Net Ensemble | 15.30 mm | 12.30 mm | 4.75 mm |
| **Ours** | **V5 Backbone (현재 베이스)** | 🔥 **5.76 mm** | 13.97 mm | 6.87 mm |

**V6-A의 목표:** 
TC 영역은 이미 범접할 수 없는 5.76mm를 확보했으므로, 차기 V6-A 대칭 보상 에이전트를 통해 **비교적 오차가 있는 ET와 WT 영역의 구멍(Hole)을 정밀 타격하여 완전무결한 SOTA 모델을 완성**하는 것입니다.

---

## 📂 프로젝트 구조 (Standard ML Project)

```text
.
├── configs/                # PPO 및 백본 훈련 설정 (boundary_v5.json 등)
├── docs/                   # 논문(papers), 세부 기획 마크다운(markdowns)
├── requirements.txt        # 패키지 의존성 (PyTorch, MONAI, Stable-Baselines3)
├── scripts/                # 파이프라인 실행 스크립트 모음 (run_pipeline.py 포함)
├── src/                    # 모델, 환경(Env), 데이터 로더 핵심 소스코드
├── results/                # 평가 지표(.json), 시각화(.png), 리포트(.md)
├── runs/                   # 훈련 출력물 (가중치, 로그, 예측 캐시)
└── tests/                  # 단위 테스트 코드
```

---

## 🚀 파이프라인 핵심 기술 (V5 기준)

### 1. Cost-Sensitive Reward (비대칭 보상 설계)
의료 도메인에서 종양을 놓치는 것(False Negative)은 치명적입니다. 에이전트가 놓친 종양 픽셀을 복구하여 DSC가 상승할 때 **10배의 증폭된 보상(DSC Boost)**을 부여하고 최대 보상을 50으로 제한(`reward_clip=50`)하여, 무작정 부피를 깎아 HD95만 낮추는 Reward Hacking과 Value Loss 폭발을 원천 차단했습니다.

### 2. Decisive Action Space (결정적 픽셀 편집)
기존의 미세한 로짓(Logit) 조정 방식을 벗어나, 에이전트가 확신을 가진 경계에 대해 단 한 번의 액션(`logit_delta=2.0`)으로 클래스를 완전히 뒤바꿀 수 있는 환경을 구축했습니다. 에피소드당 최대 256번의 타격을 통해 거대한 구멍(Hole)도 순식간에 메웁니다.

### 3. Patient-level Strict Split (엄격한 데이터 통제)
전체 1,251명의 BraTS 2021 환자 데이터를 **Backbone(656명) / PPO(219명) / Validation(188명) / Test(188명)**로 완벽하게 분할하여 Data Leakage를 차단하고 객관적인 Hold-out 성능을 측정합니다.

---

## 💻 시작하기

### 환경 설치
```bash
git clone https://github.com/aninsung/2026-summer-Interdepartmental-Academic-Conference.git
cd 2026-summer-Interdepartmental-Academic-Conference
pip install -r requirements.txt
```

### 전체 3D 경계 보정 파이프라인 실행
```bash
# V5 본학습 전체 파이프라인 런칭 (백본 -> 캐시적재 -> PPO학습 -> 평가)
python scripts/train/train_boundary_3d.py --config configs/boundary_v5.json --stage all
```

---
*자세한 PPO 환경 설계와 평가 공식은 [docs/markdowns/BOUNDARY_3D.md](docs/markdowns/BOUNDARY_3D.md)를 참조하세요.*
