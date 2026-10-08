# 현재 모델

크기별 Expert 세 개와 경계 띠 PPO 하나다. 논문 본문은 [paper_draft_ko.md](paper_draft_ko.md)다.

| 단계 | 체크포인트 |
|---|---|
| 분류 | `checkpoints/shape_classifier_best.pt` |
| Small / Medium / Large | `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt` |
| Stage 3 | `checkpoints/band_ppo.pt` |

논문에 쓰는 851명 환자 평균은 2026-10-08 실행이다. 임계값 0.45/0.80/0.20의 Stage 2 대비 DSC 0.8419→0.8595 (짝 차이 +0.0176, 95% CI 0.0166–0.0185). small·medium·large 슬라이스는 19,077, 19,907, 11,026장이고 DSC는 0.7693→0.7922, 0.8780→0.8932, 0.9030→0.9156이다. 빈 마스크를 뺀 HD95 평균 4.747→4.547 px는 짝비교가 아니다. 파일은 `results/threshold_045_080_020/band_ppo_locked.json`이다. 2026-09-30 잠금은 고정 임계값 Stage 2 대비 0.8337→0.8604이고, 그때의 재선택 Stage 2(0.8411) 대비 +0.0193이다.

2026-10-07 재학습 여섯 가중치는 `checkpoints/seeds/{42,7,123}/`이다. 시드 123 학습이 끝난 뒤 작업 디렉터리의 `checkpoints/band_ppo.pt`는 시드 123 배우다. 종양 슬라이스 DSC 환자 평균은 시드 42에서 Stage 2 0.8339, PPO 0.8597, 대형 전 슬라이스 0.8503이다. 시드 7은 0.8370, 0.8620, 0.8540이고 시드 123은 0.8352, 0.8609, 0.8500이다. PPO 평균은 0.8609, 범위는 0.0023이다. 시드 42의 빈 슬라이스 거짓 양성은 0.567, 0.521, 0.930이고 3D Dice는 0.779, 0.813, 0.799이다. 시드 7은 0.532, 0.497, 0.821과 0.786, 0.811, 0.796이다. 시드 123은 0.601, 0.571, 0.829와 0.771, 0.809, 0.759이다. PPO 공통 HD95는 10.73 mm, 10.35 mm, 10.73 mm이다.

같은 시드 42 가중치의 추론 ablation은 5스텝 0.8597, 1스텝 0.8460, 중형 지도학습 1스텝 0.8383, 대형+경계 띠 PPO 0.8676, 정답 크기 라우팅 0.8698, 독립 703명 0.8579, 반경 0은 0.8427, 스텝·가드·띠 폭은 0.8575–0.8601이다.

단일 U-Net과 바닐라 PPO는 `checkpoints/single_backbone/unet.pt`, `ppo_unet_vanilla.zip`이다. 851명 DSC는 0.8156(95% CI 0.8067–0.8243)과 0.8133(0.8046–0.8220)이고 HD95는 6.939 px, 6.955 px이다. 짝 차이는 −0.0023(95% CI −0.0029–−0.0017)이다. 크기별 DSC는 소형 0.7215→0.7197, 중형 0.8758→0.8736, 대형 0.8973→0.8939이다.

```bash
python scripts/eval/evaluate_band_ppo_locked.py --stage2_thresholds 0.45,0.80,0.20
```

논문 표는 [EXPERIMENT_RESULTS.md §0.16](EXPERIMENT_RESULTS.md), 2026-09-30 임계값 재선택은 §0.13, 2026-10-07 재학습은 §0.15이다.

단일 백본 다섯 개와 각각의 섹터 PPO는 `checkpoints/single_backbone/`에 있다. 전역 바닐라 PPO(`ppo_unet_vanilla.zip`)와 다른 정책이다. TRIO 전문가와 `band_ppo.pt`와 다른 가중치다. 48명 표본(슬라이스 2,754장)의 슬라이스 평균 DSC는 Small 기준 TRIO 0.812, UNet++ 0.778, Attention U-Net 0.761, SegResNet 0.676, UNet+++ 0.543, U-Net 0.537이다. 이 집계는 851명 환자 평균이 아니다. 표는 [EXPERIMENT_RESULTS.md §0.14](EXPERIMENT_RESULTS.md), 그림은 `results/single_backbone_ppo_grid.png`다. 여섯 슬라이스 `results/band_ppo_delta_matched/delta_matched_comparison.png`는 이전 잠금 가중치다.

`ppo_v4`의 `ppo_{small,medium,large}.zip`과 08-20 TRIO는 이전 Stage 3다. 기록은 [history_2026-08-20.md](history_2026-08-20.md)에 있고 현재 평가에 쓰지 않는다.
