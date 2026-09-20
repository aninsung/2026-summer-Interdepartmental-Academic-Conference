# 🧪 3D PPO Boundary Refiner Ablation Study Plan

본 문서는 **작전 시간(Max Steps) 절제 연구(Ablation Study)**의 목적과 내일(또는 추후) 실행할 수 있는 가이드를 정리한 문서입니다.

---

## 1. 실험 목적 (Objective)
PPO 에이전트가 3D 공간을 보정할 때 주어지는 **에피소드당 최대 스텝(max_steps)**이 성능(DSC, HD95)에 미치는 영향을 분석하여, 본 연구에서 제안한 `max_steps = 256`이 최적의 골디락스(Goldilocks) 지점임을 증명합니다.

## 2. 실험 설정 (Configurations)

| 실험명 | 환경 설정 파일 | `max_steps` | 예상 시나리오 (Hypothesis) |
|---|---|---|---|
| **Ablation 1 (Short)** | `configs/ablation_max_steps_64.json` | 64 | 시간이 너무 부족하여 거대한 구멍(Hole)을 다 메우지 못하고 에피소드가 종료되어 성능 개선이 미미함. |
| **Ours (V5, 메인 실험)** | `configs/boundary_v5.json` | 256 | 짧은 시간 안에 핵심적인 타격점만 골라 타격하여 최적의 오차 감소와 정밀도 유지를 보여줌. |
| **Ablation 2 (Long)** | `configs/ablation_max_steps_1024.json` | 1024 | 시간이 너무 남아돌아 오히려 정상 픽셀까지 과도하게 건드리거나 깎아먹어(Over-exploration) DSC가 하락하는 현상 발생. |

---

## 3. 실행 가이드 (Execution Guide)

실험에 필요한 설정 파일과 자동화 스크립트는 이미 준비되어 있습니다. 퇴근 전이나 주무시기 전에 터미널에서 아래의 **실행 스크립트 단 한 줄**만 입력하시면 두 개의 실험이 약 3시간 30분에 걸쳐 연속으로 실행됩니다.

### 🚀 실행 명령어
```bash
# 스크립트 실행 권한 부여 (최초 1회)
chmod +x scripts/run_ablation_study.sh

# Ablation 실험(64 스텝, 1024 스텝 연속 훈련 및 평가) 백그라운드 실행
./scripts/run_ablation_study.sh
```

### 📁 결과 확인
실험이 종료된 후 다음 경로에서 결과를 확인하실 수 있습니다.
* `runs/ablation_steps_64/` 내의 테스트 결과 및 로그
* `runs/ablation_steps_1024/` 내의 테스트 결과 및 로그

위 두 결과와 기존 **V5 메인 실험 결과**를 모아서 표(Table)로 정리하시면 논문의 훌륭한 Ablation Study 챕터가 완성됩니다!
