# 실험 문서 안내

논문 본문은 [paper_draft_ko.md](paper_draft_ko.md)다. 코드 설명은 [PIPELINE.md](PIPELINE.md). 숫자 로그는 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) §0.11–0.13이다.

**851명 환자 평균 (2026-09-30).** 고정 임계값 0.80/0.80/0.50 Stage 2 대비 DSC 0.8337→0.8604 (짝 차이 +0.0267, 95% CI 0.0256–0.0278). 검증 60명에서 고른 0.45/0.80/0.20 Stage 2는 0.8411이고, PPO와의 짝 차이는 +0.0193 (0.0184–0.0202)이다. HD95 평균 4.888→4.609 px는 빈 마스크 제외가 달라 짝비교가 아니다. 같은 집합의 DSC는 NVAUTO 0.8576, KAIST 0.8463이며 TRIO와 NVAUTO의 주변 구간은 겹친다.

파이프라인: [`results/pipeline_overview.jpg`](../results/pipeline_overview.jpg)  
구조: [`results/band_ppo_agent_internals.png`](../results/band_ppo_agent_internals.png)  
Stage 2와 경계 띠 PPO: [`results/band_ppo_delta_matched/delta_matched_comparison.png`](../results/band_ppo_delta_matched/delta_matched_comparison.png)  
현재 Stage 3와 KAIST·NVAUTO: [`results/method_comparison_current_3x6.png`](../results/method_comparison_current_3x6.png)

**단일 백본 + 섹터 PPO (2026-09-30).** 학습 280명, 이진 Whole Tumor, 임계값 0.5, PPO 15스텝·10만 스텝. 숫자는 미사용 환자 48명(seed 7), 슬라이스 2,754장의 슬라이스 평균이다. 851명 환자 평균이 아니다. Small DSC는 TRIO 0.812, UNet++ 0.778, Attention U-Net 0.761, SegResNet 0.676, UNet+++ 0.543, U-Net 0.537. 로그는 [EXPERIMENT_RESULTS.md §0.14](EXPERIMENT_RESULTS.md)다. 그림은 [`results/single_backbone_ppo_grid.png`](../results/single_backbone_ppo_grid.png).

`ppo_v4`(DSC 0.8345)는 이전 GT-free val 기록이다. `ppo_v5`는 폐기했다. 2026-08-20 TRIO(210명 풀, val 42명, 단조 DSC 게이트)는 [history_2026-08-20.md](history_2026-08-20.md)와 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) §1이다. 그 비교 그림 `results/archive_2026-08-20/method_comparison_3x6.png`의 TRIO는 현재 Stage 3가 아니다. 그 풀의 148명은 851명 평가에 들어 있다. 그때 정한 값은 고정 임계값 0.80/0.80/0.50과 연결요소 기준이다. 띠 설계는 이번 방법 선택 60명에서 정했다.

이전 섹터 PPO의 품질 gate: [QUALITY_GATE.md](QUALITY_GATE.md)
