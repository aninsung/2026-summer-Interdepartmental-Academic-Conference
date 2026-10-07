# 실험 문서 안내

논문 본문은 [paper_draft_ko.md](paper_draft_ko.md)다. 코드 설명은 [PIPELINE.md](PIPELINE.md). 숫자 로그는 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) §0.11–0.13이다.

**851명 환자 평균 (2026-09-30 잠금).** 고정 임계값 0.80/0.80/0.50 Stage 2 대비 DSC 0.8337→0.8604 (짝 차이 +0.0267, 95% CI 0.0256–0.0278). 검증 60명에서 고른 0.45/0.80/0.20 Stage 2는 0.8411이고, PPO와의 짝 차이는 +0.0193 (0.0184–0.0202)이다. HD95 평균 4.888→4.609 px는 빈 마스크 제외가 달라 짝비교가 아니다.

**2026-10-07 재학습.** 같은 분할, 시드 42·7·123. 경계 띠 PPO DSC는 0.8597, 0.8620, 0.8609(평균 0.8609, 범위 0.0023)이다. Stage 2는 0.8339, 0.8370, 0.8352이고, 대형 전문가를 전 슬라이스에 쓰면 0.8503, 0.8540, 0.8500이다. 시드 42의 빈 슬라이스 거짓 양성은 Stage 2 0.567, PPO 0.521, 대형 0.930이다. 3D Dice는 0.779, 0.813, 0.799이다. 공통 HD95는 Stage 2 11.54 mm, PPO 10.73 mm이다. 시드 7의 Stage 2·PPO·대형은 거짓 양성 0.532, 0.497, 0.821, 3D Dice 0.786, 0.811, 0.796이다. 시드 123은 0.601, 0.571, 0.829와 0.771, 0.809, 0.759이다. 시드 7·123 PPO 공통 HD95는 10.35 mm, 10.73 mm이다. 추론 ablation은 5스텝 0.8597, 1스텝 0.8460, 지도학습 1스텝 0.8383, 대형+띠 PPO 0.8676, 정답 라우팅 0.8698, 독립 703명 0.8579, 반경 0은 0.8427, 스텝·가드·띠 폭은 0.8575–0.8601이다. 단일 U-Net DSC 0.8156, HD95 6.939 px. 바닐라 PPO DSC 0.8133, HD95 6.955 px, 짝 차이 −0.0023(95% CI −0.0029–−0.0017). 크기별 0.7215→0.7197, 0.8758→0.8736, 0.8973→0.8939. 가중치는 `checkpoints/seeds/{42,7,123}/`, `checkpoints/single_backbone/unet.pt`, `ppo_unet_vanilla.zip`이다. 파일은 `results/protocol_gaps/three_seeds.json`, `results/vanilla_unet_851.json`, `results/ablation_review/summary.json`이다. 로그는 [EXPERIMENT_RESULTS.md §0.15](EXPERIMENT_RESULTS.md)다.

파이프라인: [`results/pipeline_overview.jpg`](../results/pipeline_overview.jpg)  
구조: [`results/fig2_ppo_internal_route.jpg`](../results/fig2_ppo_internal_route.jpg)  
Stage 2와 경계 띠 PPO: [`results/band_ppo_delta_matched/delta_matched_comparison.png`](../results/band_ppo_delta_matched/delta_matched_comparison.png)  
**단일 백본 + 섹터 PPO (2026-09-30).** 학습 280명, 이진 Whole Tumor, 임계값 0.5, PPO 15스텝·10만 스텝. 숫자는 미사용 환자 48명(seed 7), 슬라이스 2,754장의 슬라이스 평균이다. 851명 환자 평균이 아니다. Small DSC는 TRIO 0.812, UNet++ 0.778, Attention U-Net 0.761, SegResNet 0.676, UNet+++ 0.543, U-Net 0.537. 로그는 [EXPERIMENT_RESULTS.md §0.14](EXPERIMENT_RESULTS.md)다. 그림은 [`results/single_backbone_ppo_grid.png`](../results/single_backbone_ppo_grid.png).

`ppo_v4`(DSC 0.8345)는 이전 GT-free val 기록이다. `ppo_v5`는 폐기했다. 2026-08-20 TRIO(210명 풀, val 42명, 단조 DSC 게이트)는 [history_2026-08-20.md](history_2026-08-20.md)와 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) §1이다. 그 비교 그림 `results/archive_2026-08-20/method_comparison_3x6.png`의 TRIO는 현재 Stage 3가 아니다. 그 풀의 148명은 851명 평가에 들어 있다. 그때 정한 값은 고정 임계값 0.80/0.80/0.50과 연결요소 기준이다. 띠 설계는 이번 방법 선택 60명에서 정했다.

이전 섹터 PPO의 품질 gate: [QUALITY_GATE.md](QUALITY_GATE.md)
