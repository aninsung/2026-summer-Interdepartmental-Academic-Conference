# 실험 문서 안내

기술 설명은 현재 파이프라인 기준으로 [PIPELINE.md](PIPELINE.md)에 있다. Stage 3는 경계 띠 PPO(`checkpoints/band_ppo.pt`)다.

논문 수치는 **2026-09-30** 851명 환자 평균이다. DSC 0.8359→0.8607, HD95 4.824→4.557 px.

→ **[실험 결과](EXPERIMENT_RESULTS.md)** §0.11  
→ **[논문 초안](paper_draft_ko.md)**

구조 그림: [`results/band_ppo_agent_internals.png`](../results/band_ppo_agent_internals.png)

2026-08-20의 8방위 SDF와 TRIO 그림(`results/method_comparison_3x6.png`)은 이전 Stage 3다. 숫자는 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) §1에만 둔다.
