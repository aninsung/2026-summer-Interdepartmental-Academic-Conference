# 기술 문서 안내

구현 설명은 [PIPELINE.md](PIPELINE.md)다. Stage 3는 경계 띠 PPO(`checkpoints/band_ppo.pt`)다. 논문 본문은 [paper_draft_ko.md](paper_draft_ko.md)다.

851명 환자 평균에서 2026-09-30 잠금 가중치의 고정 임계값 Stage 2 대비 DSC는 0.8337→0.8604 (짝 차이 +0.0267, 95% CI 0.0256–0.0278)이다. 검증에서 고른 임계값 Stage 2(0.8411) 대비 짝 차이는 +0.0193 (0.0184–0.0202)이다. HD95 평균 4.888→4.609 px는 빈 마스크 제외가 달라 짝비교가 아니다.

2026-10-07 재학습은 같은 분할의 시드 42·7·123이다. 경계 띠 PPO DSC는 0.8597, 0.8620, 0.8609(평균 0.8609, 범위 0.0023)이다. Stage 2는 0.8339, 0.8370, 0.8352이고 대형 전 슬라이스는 0.8503, 0.8540, 0.8500이다. 시드 42 프로토콜은 빈 슬라이스 거짓 양성 Stage 2 0.567 / PPO 0.521 / 대형 0.930, 3D Dice 0.779 / 0.813 / 0.799, 공통 HD95 11.54 mm / 10.73 mm이다. 시드 7의 Stage 2·PPO·대형은 거짓 양성 0.532 / 0.497 / 0.821, 3D Dice 0.786 / 0.811 / 0.796이다. 시드 123은 0.601 / 0.571 / 0.829, 0.771 / 0.809 / 0.759이다. 시드 7·123 PPO 공통 HD95는 10.35 mm, 10.73 mm이다. 추론 ablation은 5스텝 0.8597, 1스텝 0.8460, 지도학습 1스텝 0.8383, 대형+띠 0.8676, 정답 라우팅 0.8698, 독립 703명 0.8579, 반경 0은 0.8427, 스텝·가드·띠 폭은 0.8575–0.8601이다. 단일 U-Net은 DSC 0.8156, HD95 6.939 px이고 바닐라 PPO는 DSC 0.8133, HD95 6.955 px, 짝 차이 −0.0023이다. 크기별 DSC는 0.7215→0.7197, 0.8758→0.8736, 0.8973→0.8939이다. 상세는 [EXPERIMENT_RESULTS.md §0.15](EXPERIMENT_RESULTS.md)다.

→ **[실험 결과](EXPERIMENT_RESULTS.md)** §0.11–0.15

구조 그림: [`results/fig2_ppo_internal_route.jpg`](../results/fig2_ppo_internal_route.jpg)

2026-08-20의 8방위 SDF와 TRIO 그림(`results/archive_2026-08-20/method_comparison_3x6.png`)은 이전 Stage 3다. 숫자는 [history_2026-08-20.md](history_2026-08-20.md)와 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) §1에 둔다. Stage 2와 경계 띠 PPO를 맞춘 여섯 슬라이스는 `results/band_ppo_delta_matched/delta_matched_comparison.png`다. 파이프라인 개요는 `results/pipeline_overview.jpg`다.

단일 백본+섹터 PPO 다섯 세트는 2026-09-30 실행이다. 학습은 train 280명, 이진 Whole Tumor, 임계값 0.5, 15스텝·10만 스텝 PPO다. 보고 수치는 미사용 48명, 슬라이스 2,754장의 슬라이스 평균이다. Small DSC는 TRIO 0.812, UNet++ 0.778, Attention U-Net 0.761, SegResNet 0.676, UNet+++ 0.543, U-Net 0.537이다. 851명 환자 평균과 집계가 다르다. 상세는 [EXPERIMENT_RESULTS.md §0.14](EXPERIMENT_RESULTS.md), 그림은 `results/single_backbone_ppo_grid.png`다.
