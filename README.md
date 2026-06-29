# 🧠 RL-Refiner

> **강화학습(RL)을 활용한 뇌종양 의료 영상 분할(Segmentation) 경계선 자동 보정 시스템**

딥러닝 모델이 생성한 뇌종양 분할 마스크의 미세한 경계 오차(Artifact)를 강화학습 에이전트가 능동적으로 보정하여 정밀도를 극대화하는 프로젝트입니다.

---

## 💡 프로젝트 배경 및 필요성

의료 영상 분할에서 U-Net과 같은 딥러닝 모델은 종양의 대략적인 위치는 잘 파악하지만, 종양의 미세한 경계면에서 울퉁불퉁한 오차를 자주 발생시킵니다.

* **정밀도의 중요성:** 방사선 수술(감마나이프 등)과 같이 정밀 타겟팅이 필요한 분야에서는 1mm의 오차도 매우 치명적일 수 있습니다.
* **비용 문제:** 이를 보정하기 위해 전문의가 수작업으로 마스크를 수정하는 과정은 막대한 시간과 비용을 소모합니다.
* **해결책:** 딥러닝의 초기 출력물(Rough Mask)을 강화학습 에이전트가 픽셀 단위로 미세 조정하는 **'Human-in-the-loop' 기반 자동 보정 시스템**을 제안합니다.

---

## 🎯 주요 목표 및 성능 지표

* **최종 목표:** 초기 마스크를 입력받아 정답(Ground Truth)과의 일치도를 최대화하도록 경계선을 수정하는 RL 에이전트 개발.
* **핵심 평가지표 (KPI):**
  * **DSC (Dice Similarity Coefficient):** 보정 전후 유사도 향상도 측정
  * **HD95 (Hausdorff Distance 95%):** 보정 전후 경계면 오차(거리) 감소율 측정

---

## ⚙️ 파이프라인 및 구현 단계

RL-Refiner는 다음 4단계의 파이프라인을 거쳐 학습 및 검증됩니다.

| 단계 | 명칭 | 세부 내용 |
| :---: | :--- | :--- |
| **Step 1** | **초기 모델 구축** | BraTS 데이터셋을 이용한 Light-weight 2D U-Net 학습 (초기 울퉁불퉁한 마스크 생성) |
| **Step 2** | **RL 환경 설계** | `Gymnasium` 기반 커스텀 환경 구축 <br> *(State: 현재 마스크/이미지, Action: 경계 픽셀 팽창/수축/유지)* |
| **Step 3** | **에이전트 학습** | `Stable-Baselines3`의 **PPO 알고리즘**을 활용하여 정답 마스크에 수렴하도록 보상(Reward) 기반 학습 |
| **Step 4** | **성능 검증** | 경계 보정 전/후 지표 비교 및 전통적 영상 처리 기법과의 성능 벤치마크 |

---

## 🛠 기술 스택 (Tech Stack)

* **Language:** Python
* **Deep Learning Framework:** PyTorch, MONAI (의료 영상 특화 라이브러리)
* **Reinforcement Learning:** Gymnasium, Stable-Baselines3
* **Dataset:** BraTS (Multimodal Brain Tumor Segmentation Challenge)

---

## 🚀 기대 효과

* **학술적 가치:** 단순 신경망 학습만으로는 도달하기 어려운 정밀한 픽셀 단위의 경계 제어 가능성을 강화학습과의 결합으로 증명합니다.
* **실무적 가치:** 방사선 수술 계획 단계 등 임상 현장에서 의사의 마스크 수정 수작업 시간을 획기적으로 단축하고, 오진 리스크를 줄이는 강력한 보조 도구로 활용될 수 있습니다.

---

## 📦 시작하기 (Getting Started)

*(프로젝트 진행 상황에 따라 실제 실행 스크립트로 수정하여 사용하세요.)*

```bash
# 저장소 클론
git clone [https://github.com/your-username/RL-Refiner.git](https://github.com/your-username/RL-Refiner.git)
cd RL-Refiner

# 필요 패키지 설치
pip install -r requirements.txt

# 에이전트 학습 실행 예시
python train_agent.py --config configs/ppo_brats.yaml
