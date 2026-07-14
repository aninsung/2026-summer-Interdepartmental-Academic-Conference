# ai-5070 서버 학습/평가 결과 요약 (UNet++ 백본)

- 실행 위치: `ai-5070` (Tailscale, RTX 5070 12GB)
- 실행 일시: 2026-07-14
- 데이터: BraTS2021, 실제 데이터 사용 (`use_real_data: true`), `max_train_patients: 150`
- 분할 모델: UNet++ / Nested U-Net (`checkpoints/unetplusplus_best.pt`), 신규 구현 (`src/models/unet.py`의 `UNetPlusPlus` 클래스)
- RL 알고리즘: PPO (Stable-Baselines3), `configs/ppo_brats_unetplusplus.yaml` 그대로 사용

> Attention U-Net 결과: [`../results_ai5070/SUMMARY.md`](../results_ai5070/SUMMARY.md)
> UNet3+ 결과: [`../results_unet3plus/SUMMARY.md`](../results_unet3plus/SUMMARY.md)

## 0. 사전 학습 (train_unet.py, Step 1)

| 항목 | 값 |
|---|---|
| 모델 | UNet++ (Zhou et al., 2018 Nested skip connection, deep supervision 미사용) |
| epochs | 20 |
| 학습 슬라이스 | 8,974 (150명) |
| 검증 슬라이스 | 1,735 (30명) |
| **Best val DSC** | **0.8743** (epoch 20) |
| 소요 시간 | 약 4분 20초 (GPU, 에폭당 ~14초) |

전체 로그: [`train_unetplusplus.log`](./train_unetplusplus.log)

## 1. 학습 (train_agent.py, Step 2)

| 항목 | 값 |
|---|---|
| total_timesteps | 200,000 (실제 200,704 스텝) |
| n_envs | 4 |
| 소요 시간 | 3분 9초 |
| 최종 평가 보상 (200k 시점) | episode_reward = 2.30 ± 1.59 (세 백본 중 최고) |
| 평균 에피소드 길이 (200k 시점) | 9.6 스텝 (최대 20) |
| 에러 | 없음 |

전체 로그: [`train_agent_unetplusplus.log`](./train_agent_unetplusplus.log)

## 2. 평가 (evaluate.py)

인자: `--model_name unetplusplus --unet_path checkpoints/unetplusplus_best.pt --agent_path checkpoints/ppo_refiner_unetplusplus --output_dir results_unetplusplus` (나머지는 기본값: `--num_eval 50 --max_steps 20 --data_dir src/data/archive`)

평가 슬라이스 수: 2,902개 (50명 환자, 다른 두 실험과 동일 세트)

| 방법 | DSC (mean ± std) | HD95 (mean ± std) |
|---|---|---|
| **rough (UNet++ 초기 마스크)** | **0.8447 ± 0.1280** | **2.22 ± 2.06** |
| morpho (전통적 형태학 보정) | 0.8328 ± 0.1497 | 2.40 ± 2.24 |
| rl (PPO 보정) | 0.7500 ± 0.1844 | 2.86 ± 1.88 |

- **UNet3+와 동일한 패턴**: RL 보정이 오히려 DSC를 낮춤 (0.8447 → 0.7500).
- morpho도 rough 대비 소폭 하락 — 과보정 경향이 형태학적 후처리에서도 나타남.

## 3. 세 백본 비교 (rough → RL DSC)

| 백본 | 사전학습 val DSC | 평가 rough DSC | 평가 RL DSC | RL 효과 |
|---|---|---|---|---|
| Attention U-Net | (미기록, `results_ai5070` 참고) | 0.2609 | 0.5412 | ✅ +0.28 (2배 이상 개선) |
| UNet3+ | 0.8973 | 0.8756 | 0.7408 | ❌ -0.13 (악화) |
| **UNet++** | 0.8743 | 0.8447 | 0.7500 | ❌ -0.09 (악화) |

**결론**: rough mask 품질이 낮을 때(Attention U-Net, DSC ~0.26)만 RL 보정이 유의미하게 도움이 되고, 이미 rough mask 품질이 높은 두 백본(UNet3+, UNet++, DSC 0.84~0.88)에서는 오히려 RL이 마스크를 훼손합니다. 현재 PPO 보상/환경 설계가 "저품질 마스크 개선" 상황에만 맞춰져 있고, "고품질 마스크에서 조기 정지"를 학습하지 못한 것으로 보입니다 — `target_dsc` 조기 종료 조건, 보상 함수(현재 스텝마다 DSC 변화량 기반으로 추정)의 재검토가 다음 단계로 필요해 보입니다.

## 4. 산출물

| 파일 | 설명 |
|---|---|
| `sample_comparison.png` | 샘플 4개 슬라이스에 대한 GT/Rough/Morpho/RL 마스크 윤곽선 비교 |
| `dsc_boxplot.png` | 방법별 DSC 분포 박스플롯 |
| `train_unetplusplus.log` | UNet++ 사전학습 전체 로그 |
| `train_agent_unetplusplus.log` | PPO 학습 전체 로그 |
| `evaluate.log` | 평가 전체 로그 |

## 5. 원격 서버에 남아있는 모델 체크포인트 (로컬 미다운로드)

용량 문제로 로컬로 가져오지 않았습니다. 필요하시면 말씀해 주세요.

경로: `ai-5070:~/projects/brats-rl-refiner/checkpoints/`

| 파일 | 크기 |
|---|---|
| `unetplusplus_best.pt` | 2.2 MB |
| `ppo_refiner_unetplusplus.zip` | 약 195 MB |
