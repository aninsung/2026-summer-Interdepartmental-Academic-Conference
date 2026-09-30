# 실험 문서 안내

현재 모델은 크기별 Expert 세 개와 경계 띠 PPO 하나다.

| 단계 | 체크포인트 |
|---|---|
| 분류 | `checkpoints/shape_classifier_best.pt` |
| Small / Medium / Large | `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt` |
| Stage 3 | `checkpoints/band_ppo.pt` |

논문 수치는 **2026-09-30** 851명 환자 평균이다. DSC 0.8359→0.8607, HD95 4.824→4.557 px. 재현은 `python scripts/eval/evaluate_band_ppo_locked.py`.

→ **[파이프라인](PIPELINE.md)**  
→ **[실험 결과](EXPERIMENT_RESULTS.md)** §0.11

`ppo_v4`의 `ppo_{small,medium,large}.zip`과 08-20 TRIO는 이전 Stage 3다. 현재 평가에 쓰지 않는다.
