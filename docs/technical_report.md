# 기술 문서 안내

구현 설명은 [PIPELINE.md](PIPELINE.md)다. Stage 3는 경계 띠 PPO(`checkpoints/band_ppo.pt`)다. 논문 본문은 [paper_draft_ko.md](paper_draft_ko.md)다.

851명 환자 평균에서 고정 임계값 Stage 2 대비 DSC는 0.8337→0.8604 (짝 차이 +0.0267, 95% CI 0.0256–0.0278)이다. 검증에서 고른 임계값 Stage 2(0.8411) 대비 짝 차이는 +0.0193 (0.0184–0.0202)이다. HD95 평균 4.888→4.609 px는 빈 마스크 제외가 달라 짝비교가 아니다. 같은 집합의 DSC는 NVAUTO 0.8576, KAIST 0.8463이고 두 주변 구간은 TRIO와 NVAUTO가 겹친다.

→ **[실험 결과](EXPERIMENT_RESULTS.md)** §0.11–0.13

구조 그림: [`results/band_ppo_agent_internals.png`](../results/band_ppo_agent_internals.png)

2026-08-20의 8방위 SDF와 TRIO 그림(`results/archive_2026-08-20/method_comparison_3x6.png`)은 이전 Stage 3다. 숫자는 [history_2026-08-20.md](history_2026-08-20.md)와 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) §1에 둔다. 현재 Stage 3 비교 그림은 `results/method_comparison_current_3x6.png`다. Stage 2와 경계 띠 PPO를 맞춘 여섯 슬라이스는 `results/band_ppo_delta_matched/delta_matched_comparison.png`다. 파이프라인 개요는 `results/pipeline_overview.jpg`다.

단일 백본+섹터 PPO 다섯 세트는 2026-09-30 실행이다. 학습은 train 280명, 이진 Whole Tumor, 임계값 0.5, 15스텝·10만 스텝 PPO다. 보고 수치는 미사용 48명, 슬라이스 2,754장의 슬라이스 평균이다. Small DSC는 TRIO 0.812, UNet++ 0.778, Attention U-Net 0.761, SegResNet 0.676, UNet+++ 0.543, U-Net 0.537이다. 851명 환자 평균과 집계가 다르다. 상세는 [EXPERIMENT_RESULTS.md §0.14](EXPERIMENT_RESULTS.md), 그림은 `results/single_backbone_ppo_grid.png`다.
