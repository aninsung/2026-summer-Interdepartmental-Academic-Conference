# ai-5070 서버 학습/평가 결과 요약 (UNet3+ 백본)

- 실행 위치: `ai-5070` (Tailscale, RTX 5070 12GB)
- 실행 일시: 2026-07-14
- 데이터: BraTS2021, 실제 데이터 사용 (`use_real_data: true`), `max_train_patients: 150`
- 분할 모델: UNet3+ (`checkpoints/unet3plus_best.pt`)
- RL 알고리즘: PPO (Stable-Baselines3), `configs/ppo_brats_unet3plus.yaml` 그대로 사용

> Attention U-Net 버전 결과는 [`../results_ai5070/SUMMARY.md`](../results_ai5070/SUMMARY.md)에 별도로 정리되어 있습니다.

## 0. 사전 학습 (train_unet.py, Step 1)

| 항목 | 값 |
|---|---|
| 모델 | UNet3+ |
| epochs | 20 |
| 학습 슬라이스 | 8,974 (150명) |
| 검증 슬라이스 | 1,735 (30명) |
| **Best val DSC** | **0.8973** (epoch 20) |
| 소요 시간 | 약 5분 30초 (GPU, 에폭당 ~20초) |

전체 로그: [`train_unet3plus.log`](./train_unet3plus.log)

## 1. 학습 (train_agent.py, Step 2)

| 항목 | 값 |
|---|---|
| total_timesteps | 200,000 (실제 200,704 스텝) |
| n_envs | 4 |
| 소요 시간 | 2분 51초 |
| 최종 평가 보상 (200k 시점) | episode_reward = 1.27 ± 2.51 |
| 중간 평가 보상 (160k 시점) | episode_reward = 0.36 ± 2.61 |
| 에러 | 없음 |

전체 로그: [`train_agent_unet3plus.log`](./train_agent_unet3plus.log)

## 2. 평가 (evaluate.py)

인자: `--model_name unet3plus --unet_path checkpoints/unet3plus_best.pt --agent_path checkpoints/ppo_refiner_unet3plus --output_dir results_unet3plus` (나머지는 기본값: `--num_eval 50 --max_steps 20 --data_dir src/data/archive`)

평가 슬라이스 수: 2,902개 (50명 환자, Attention U-Net 평가와 동일 세트)

| 방법 | DSC (mean ± std) | HD95 (mean ± std) |
|---|---|---|
| **rough (UNet3+ 초기 마스크)** | **0.8756 ± 0.1048** | **1.65 ± 1.86** |
| morpho (전통적 형태학 보정) | 0.8539 ± 0.1610 | 1.96 ± 2.12 |
| rl (PPO 보정) | 0.7408 ± 0.2022 | 2.75 ± 2.29 |

- **주의: RL 보정이 오히려 성능을 떨어뜨림** (rough 0.8756 → RL 0.7408). Attention U-Net 실험과 정반대 결과.
- UNet3+ 자체의 초기 분할 품질이 이미 매우 높아(rough DSC 0.88), PPO 에이전트가 개선할 여지가 거의 없거나 과보정(over-refinement)으로 오히려 마스크를 훼손하는 것으로 보임.
- 형태학적 보정(morpho)도 rough 대비 소폭 하락 — 동일한 과보정 경향.
- **해석**: PPO 보상/환경 설계가 "낮은 품질의 rough mask를 개선"하는 상황(Attention U-Net, rough DSC ~0.26)에 맞춰져 있어서, 이미 우수한 rough mask(UNet3+, DSC ~0.88)에 대해서는 불필요한 액션을 계속 취해 품질을 낮추는 것으로 추정됨. 에이전트가 "언제 멈춰야 하는지"를 학습하지 못했을 가능성 — target_dsc 조기 종료 조건이나 보상 함수 재검토 필요.

전체 로그: [`evaluate.log`](./evaluate.log)

## 3. 산출물

| 파일 | 설명 |
|---|---|
| `sample_comparison.png` | 샘플 4개 슬라이스에 대한 GT/Rough/Morpho/RL 마스크 윤곽선 비교 |
| `dsc_boxplot.png` | 방법별 DSC 분포 박스플롯 |
| `train_unet3plus.log` | UNet3+ 사전학습 전체 로그 |
| `train_agent_unet3plus.log` | PPO 학습 전체 로그 |
| `evaluate.log` | 평가 전체 로그 |

## 4. 원격 서버에 남아있는 모델 체크포인트 (로컬 미다운로드)

용량 문제로 로컬로 가져오지 않았습니다. 필요하시면 말씀해 주세요.

경로: `ai-5070:~/projects/brats-rl-refiner/checkpoints/`

| 파일 | 크기 |
|---|---|
| `unet3plus_best.pt` | 2.0 MB |
| `ppo_refiner_unet3plus.zip` | 약 195 MB |
