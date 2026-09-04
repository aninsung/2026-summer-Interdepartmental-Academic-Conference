# Hybrid: Supervised Fix → PPO 시뮬레이션

메인 파이프라인 미변경. 합성 데이터.

1. Dual-head 지도 학습: candidate(고칠 곳, GT-error+경계로 학습) + fix(후보 가중 GT)
2. 추론: candidate 안에서만 rough 덮어쓰기 (밖은 유지)
3. Hybrid: Fix된 마스크를 초기값으로 PPO 학습·평가

| Size | Method | DSC | note |
|---|---|---:|---|
| small | FixOnly | 0.7851 | cand_frac=0.108 |
| small | PPOOnly | 0.7670 | |
| small | Hybrid Fix→PPO | **0.7898** | Fix 위로 PPO |
| medium | FixOnly | 0.9137 | cand_frac=0.088 |
| medium | PPOOnly | 0.9074 | |
| medium | Hybrid Fix→PPO | **0.9164** | |
| large | FixOnly | 0.9285 | cand_frac=0.018 |
| large | PPOOnly | 0.9486 | |
| large | Hybrid Fix→PPO | **0.9550** | |

## Overall 해석

- **Small / Medium**: Hybrid ≥ FixOnly > PPOOnly (PPO만 쓰는 것보다 Fix가 핵심, PPO 가산은 작음)
- **Large**: Hybrid > PPOOnly > FixOnly (여기서는 PPO 추가 이득이 보임)
- 학습 시간: Fix ~1s, PPO/Hybrid ~20-40s/클래스

> candidate 학습 타깃은 GT를 쓰지만 추론 덮어쓰기는 예측 candidate만 사용.
> 합성·짧은 학습이므로 BraTS 절대수치와 비교하지 말 것.
