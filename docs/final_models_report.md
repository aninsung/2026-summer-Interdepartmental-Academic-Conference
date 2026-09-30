# 현재 모델

크기별 Expert 세 개와 경계 띠 PPO 하나다. 논문 본문은 [paper_draft_ko.md](paper_draft_ko.md)다.

| 단계 | 체크포인트 |
|---|---|
| 분류 | `checkpoints/shape_classifier_best.pt` |
| Small / Medium / Large | `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt` |
| Stage 3 | `checkpoints/band_ppo.pt` |

851명 환자 평균이다. 고정 임계값 Stage 2 대비 DSC 0.8337→0.8604 (짝 차이 +0.0267, 95% CI 0.0256–0.0278). 검증에서 고른 0.45/0.80/0.20 Stage 2(0.8411) 대비 +0.0193 (0.0184–0.0202). HD95 평균 4.888→4.609 px는 빈 마스크 제외가 달라 짝비교가 아니다.

```bash
python scripts/eval/evaluate_band_ppo_locked.py
python scripts/eval/evaluate_stage2_retuned.py
```

KAIST·NVAUTO는 [EXPERIMENT_RESULTS.md §0.12](EXPERIMENT_RESULTS.md)다. 임계값 재선택은 §0.13이다.

단일 백본 다섯 개와 각각의 섹터 PPO는 `checkpoints/single_backbone/`에 있다. TRIO 전문가와 `band_ppo.pt`와 다른 가중치다. 48명 표본(슬라이스 2,754장)의 슬라이스 평균 DSC는 Small 기준 TRIO 0.812, UNet++ 0.778, Attention U-Net 0.761, SegResNet 0.676, UNet+++ 0.543, U-Net 0.537이다. 이 집계는 851명 환자 평균이 아니다. 표는 [EXPERIMENT_RESULTS.md §0.14](EXPERIMENT_RESULTS.md), 그림은 `results/single_backbone_ppo_grid.png`다. Stage 2와 경계 띠 PPO의 여섯 슬라이스는 `results/band_ppo_delta_matched/delta_matched_comparison.png`다.

`ppo_v4`의 `ppo_{small,medium,large}.zip`과 08-20 TRIO는 이전 Stage 3다. 기록은 [history_2026-08-20.md](history_2026-08-20.md)에 있고 현재 평가에 쓰지 않는다.
