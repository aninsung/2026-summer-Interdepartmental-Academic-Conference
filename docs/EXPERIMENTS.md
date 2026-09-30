# 실험 문서 안내

현재 파이프라인은 분류 → 크기별 Expert → **경계 띠 PPO 하나**다. 코드 설명은 [PIPELINE.md](PIPELINE.md).

**논문 수치 (2026-09-30):** BraTS 2021 1251명 중 개발 400명(train 280 / val 60 / 방법 선택 test 60)을 뺀 **851명** 환자 평균. DSC 0.8359→0.8607, HD95 4.824→4.557 px.

→ **[실험 결과](EXPERIMENT_RESULTS.md)** §0.11  
→ **[논문 초안](paper_draft_ko.md)**

구조 그림: [`results/band_ppo_agent_internals.png`](../results/band_ppo_agent_internals.png)  
851명 정성 비교: [`results/band_ppo_heldout_samples/band_ppo_heldout_comparison.png`](../results/band_ppo_heldout_samples/band_ppo_heldout_comparison.png)

`ppo_v4`(DSC 0.8345)는 이전 GT-free val 기록이다. `ppo_v5`는 폐기했다. 2026-08-20 TRIO(210명 풀, val 42명, 단조 DSC 게이트)는 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) §1이다. 그 비교 그림 `results/method_comparison_3x6.png`의 TRIO는 현재 Stage 3가 아니다.

이전 섹터 PPO의 품질 gate: [QUALITY_GATE.md](QUALITY_GATE.md)
