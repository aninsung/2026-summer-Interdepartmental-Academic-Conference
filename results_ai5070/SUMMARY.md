# ai-5070 서버 학습/평가 결과 요약

- 실행 위치: `ai-5070` (Tailscale, RTX 5070 12GB)
- 실행 일시: 2026-07-13
- 데이터: BraTS2021, 실제 데이터 사용 (`use_real_data: true`), `max_train_patients: 150`
- 분할 모델: Attention U-Net (`checkpoints/attention_unet_best.pt`)
- RL 알고리즘: PPO (Stable-Baselines3), `configs/ppo_brats.yaml` 그대로 사용

## 1. 학습 (train_agent.py)

| 항목 | 값 |
|---|---|
| total_timesteps | 200,000 (실제 200,704 스텝) |
| n_envs | 4 |
| 소요 시간 | 2분 39초 |
| 최종 평가 보상 (200k 시점) | episode_reward = 1.24 ± 2.14 |
| 중간 평가 보상 (160k 시점) | episode_reward = 0.90 ± 1.95 |
| 평균 에피소드 길이 | 14.7 스텝 (최대 20) |
| 에러 | 없음 |

**참고**: `net_arch: [256, 256]`의 MLP 정책을 사용하는 구조라 GPU 활용률이 낮다는 경고가 발생함(`stable_baselines3` 경고). 이미지 기반 관찰에 CNN 정책을 쓰면 GPU를 더 활용할 수 있으나, 현재 설계(9~2채널 flatten 벡터 입력) 특성상 의도된 부분일 수 있음 — 검토 필요.

전체 로그: [`train_agent.log`](./train_agent.log)

## 2. 평가 (evaluate.py)

인자: `--model_name attention_unet --unet_path checkpoints/attention_unet_best.pt --agent_path checkpoints/ppo_refiner` (나머지는 기본값: `--num_eval 50 --max_steps 20 --data_dir src/data/archive`)

평가 슬라이스 수: 2,902개 (50명 환자)

| 방법 | DSC (mean ± std) | HD95 (mean ± std) |
|---|---|---|
| rough (Attention U-Net 초기 마스크) | 0.2609 ± 0.1342 | 22.50 ± 9.44 |
| morpho (전통적 형태학 보정) | 0.2646 ± 0.1351 | 22.04 ± 9.27 |
| **rl (PPO 보정)** | **0.5412 ± 0.2471** | **9.44 ± 7.23** |

- DSC: U-Net 단독 대비 RL 보정 후 **2배 이상 향상** (0.26 → 0.54)
- HD95: 경계 오차가 **절반 이하로 감소** (22.5 → 9.44)
- 전통적 형태학 보정(morpho)은 rough 대비 거의 개선 없음 → RL 보정의 효과가 형태학적 후처리로는 대체되지 않음을 시사
- 일부 샘플([41/50] rough 0.082 → RL 0.103)은 개선 폭이 작아, 실패 케이스에 대한 추가 분석 여지 있음 (boxplot 하단 이상치 참고)

전체 로그: [`evaluate.log`](./evaluate.log)

## 3. 산출물

| 파일 | 설명 |
|---|---|
| `sample_comparison.png` | 샘플 4개 슬라이스에 대한 GT/Rough/Morpho/RL 마스크 윤곽선 비교 |
| `dsc_boxplot.png` | 방법별 DSC 분포 박스플롯 |
| `train_agent.log` | PPO 학습 전체 로그 (진행률 바 노이즈 제거) |
| `evaluate.log` | 평가 전체 로그 (진행률 바 노이즈 제거) |

## 4. 원격 서버에 남아있는 모델 체크포인트 (로컬 미다운로드)

용량이 커서(약 200MB씩) 로컬로 가져오지 않았습니다. 필요하시면 말씀해 주세요.

경로: `ai-5070:~/projects/brats-rl-refiner/checkpoints/`

| 파일 | 크기 |
|---|---|
| `attention_unet_best.pt` | 2.0 MB |
| `ppo_refiner.zip` (최종 모델) | 194.6 MB |
| `best_model.zip` (평가 기준 최고 성능) | 194.6 MB |
| `ppo_refiner_80000_steps.zip` | 194.6 MB |
| `ppo_refiner_160000_steps.zip` | 194.6 MB |
