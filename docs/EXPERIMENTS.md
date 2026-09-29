# 실험 문서 안내

실험 이력과 최종 벤치마크는 아래 문서로 통합했습니다.

**최신 기준 실행 (2026-09-29~30):** BraTS 환자 **1251명** (train 875 / val 188 / test 188, seed 42),
GT-free Stage 4 heuristic 평가, `t1ce+flair`, 종양 슬라이스.

→ **[실험 결과 보고서](EXPERIMENT_RESULTS.md)** (§0 최신 1251명 결과)

이전 참고 실행: 2026-08-20 val hold-out(210명 풀, val 42명, 2,434 슬라이스) — 같은 문서 §1.

파이프라인 코드 기준 설명: [PIPELINE.md](PIPELINE.md)  
품질/프로필 안내: [QUALITY_GATE.md](QUALITY_GATE.md)  
베이스라인 비교: [baselines/README.md](../baselines/README.md)
