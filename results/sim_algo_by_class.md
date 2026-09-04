# 클래스별 알고리즘 시뮬레이션 (합성 데이터, 단기 학습)

- 메인 파이프라인/체크포인트 **미변경**
- GT 단조 게이트 **없음** (raw RL DSC)
- 합성 blob + 노이즈 rough, 학습 3k step / 평가 24 에피소드
- CnnPolicy + CPU (메인 코드 미수정)

| Class | Algo | Init DSC | Final DSC | ΔDSC | MeanRew | Train(s) | Note |
|---|---|---:|---:|---:|---:|---:|---|
| small | PPO | 0.8036 | 0.8005 | -0.0030 | 22.6222 | 24.7 |  |
| small | SAC | 0.8036 | 0.3040 | -0.4995 | -2493.3024 | 206.2 |  |
| medium | PPO | 0.9252 | 0.9254 | 0.0002 | -32.2608 | 46.3 |  |
| medium | A2C | 0.9252 | 0.9254 | 0.0002 | -32.2562 | 40.1 |  |
| large | PPO | 0.9587 | 0.9589 | 0.0002 | 562.1360 | 43.3 |  |
| medium | SAC | — | — | — | — | 0.0 | 이산 MultiDiscrete → SAC 불가 (시뮬 스킵) |
| large | SAC | — | — | — | — | 0.0 | 이산 MultiDiscrete → SAC 불가 (시뮬 스킵) |

## 해석 (이 시뮬 한정)
- Small에서 SAC vs PPO는 짧은 스텝이라 승패가 뒤집힐 수 있음 → 경향만 참고.
- Medium/Large는 SAC 사용 불가; A2C는 참고용이며 보통 PPO가 더 안정적.
- 실제 BraTS·300k step 결과와 수치를 직접 비교하면 안 됨.
