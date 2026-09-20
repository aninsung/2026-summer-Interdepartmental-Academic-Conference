# 🧠 TRIO: 3D PPO Boundary Refiner 실험 규약

이 문서는 고정된 초기 딥러닝 분할(Rough Mask)을 3D 공간 상에서 강화학습(PPO) 에이전트가 직접 수정하여 **DSC 증가 및 HD95 감소**를 달성하는 하이브리드 파이프라인의 명세서입니다.

## 🚀 파이프라인 실행

저장소 루트에서 실행합니다.

```bash
# 전체 본학습 자동화 파이프라인 실행 (Backbone -> 3D 캐싱 -> PPO 학습 -> 평가)
python scripts/train/train_boundary_3d.py --config configs/boundary_v5.json --stage all
```

`--stage`는 `backbone`, `predict`, `oracle`, `ppo`, `evaluate`, `all`을 지원합니다.
완료된 모델 훈련이나 캐시 적재 작업은 똑똑하게 건너뛰며, 설정이 달라지면 새로운 output 경로(`runs/`)를 사용해야 합니다.

## 📊 데이터 분할과 무결성 (Patient-level Split)

BraTS 2021 전체 1,251명의 데이터를 **환자 단위(Patient-level)**로 엄격하게 분할하여 Data Leakage를 차단합니다.
- **Backbone 학습**: 656명 (초기 로짓 생성용)
- **PPO (RL) 학습**: 219명 (에이전트 훈련용, Backbone 학습 데이터와 완벽 분리)
- **Validation**: 188명 (검증용)
- **Test**: 188명 (최종 Hold-out 평가용)

*주의: PPO는 Backbone이 학습하지 않은 완전히 낯선 환자 데이터를 통해 학습하므로 과적합(Overfitting)을 방지합니다.*

## 🎮 PPO 환경 및 액션 설계 (V5 기준)

- **상태 관측(Observation)**: 3D MRI 4채널(T1, T1ce, T2, FLAIR) + 초기 예측 확률 4채널 + 현재 예측 확률 4채널 + 거리 등 총 15채널. (정답 GT는 에이전트에게 제공되지 않음)
- **결정적 액션(Decisive Action Space)**: 클래스별 로짓을 단순 증감시키는 것이 아니라, `logit_delta = 2.0`의 강력한 타격을 통해 **단 한 번의 행동으로 클래스를 완전히 전환**시킬 수 있습니다.
- **작전 시간(Max Steps)**: 에피소드당 최대 256번의 타격을 통해 거대한 종양 구멍(Hole)을 적극적으로 메우거나 가시를 깎아냅니다.

## ⚖️ 임상적 보상 함수 (Cost-Sensitive Reward)

에이전트의 목표는 의료 도메인의 특성을 반영하여 설계되었습니다.

1. **안정화된 DSC 부스트 및 보상 클리핑**:
   - 종양을 놓치는 것(False Negative)을 방지하기 위해 DSC 상승 시 **10배의 부스트 보상**을 부여합니다.
   - 단, V4에서 관찰된 과도한 보상 해킹 및 Value Loss 폭발을 방지하기 위해 최대 보상을 `+50`으로 엄격히 클리핑(`reward_clip=50.0`)합니다.
2. **HD95 거리 페널티 및 조기 종료 안전장치**:
   - 무작정 부피를 부풀리는 현상을 억제하기 위해 HD95(경계 오차)에 대한 강한 페널티를 유지합니다.
   - 훈련 시 초기 DSC 대비 90% 이하로 성능이 붕괴(`dsc_floor_ratio=0.9`)하면 에피소드를 즉시 강제 종료하여 정책망 망가짐을 방지합니다.

## 📁 주요 산출물 (Artifacts)

훈련이 진행되는 `runs/boundary_v5/` 내부에 다음 파일들이 생성됩니다.

| 파일 | 내용 |
|---|---|
| `config.json`, `split.json` | 훈련 설정 및 1,251명의 환자 분할 맵 |
| `backbone.pt` | 학습된 3D U-Net (또는 SegResNet) 백본 가중치 |
| `predictions/*.npz` | 초기 로짓, MRI, 정답 캐시 (PPO 훈련 가속용) |
| `ppo_final.zip` | 최종 훈련을 마친 PPO 에이전트 모델 |
| `TEST_RESULTS.md` | Hold-out 188명 대상의 최종 성적표 (논문 제출용) |
