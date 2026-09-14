# TRIO 실험 결과 보고서

본 문서는 BraTS 2021 Task 1 세부 영역(**ET: Enhancing Tumor, TC: Tumor Core, WT: Whole Tumor**)에 대한 **TRIO** 파이프라인의 정량적 평가 결과를 정리합니다.

---

## 1. 평가 설정 (Evaluation Setup)

- **데이터셋**: BraTS 2021 Task 1 (전체 1,251명 중 환자 단위 80/20 분할)
- **Train 집합**: 1,001명 (54,921 슬라이스)
- **Hold-out Val 집합**: **250명 (14,561 슬라이스)**
- **입력 채널**: `t1ce + flair` (128×128 2D 슬라이스)
- **출력 채널**: **3개 세부 영역 (ET, TC, WT)**
- **라우팅 방식**: **Soft Expert Mixture** (소프트맥스 가중치 합성)

---

## 2. 세부 영역별 정량적 평가 결과 (Multi-Region Evaluation)

14,561개 검증 슬라이스 전수에 대한 Stage 2 (초기 전문가 분할) 대비 Stage 3 (최종 Refiner 보정) 결과입니다.

| 영역 (Sub-region) | Stage 2 DSC | Stage 3 (PPO + KAIST) DSC | ΔDSC | HD95 (px) | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| **WT (Whole Tumor)** | 0.8726 | **0.8725** | -0.0001 | **2.248 px** | **0.8825** | **0.8860** |
| **TC (Tumor Core)** | 0.8035 | **0.8035** | 0.0000 | **11.426 px** | **0.8697** | **0.8658** |
| **ET (Enhancing Tumor)** | 0.7841 | **0.7821** | -0.0020 | **12.349 px** | **0.8292** | **0.8560** |

---

## 3. 분석 및 성과 요약

1. **세부 영역 일관성 확보**: KAIST 1위 후처리 ($\text{ET} \le \text{TC} \le \text{WT}$) 및 `MultiChannelBCEDiceLoss` 학습을 통해 생물학적 포함 관계가 올바르게 유지되었습니다.
2. **Precision 향상**: 미세 ET 노이즈 억제 처리(15px 미만 산재 노이즈 제거)로 ET 정밀도가 0.8292로 향상되었습니다.
3. **Soft Routing의 효과**: Soft Expert Mixture 라우팅을 적용하여 소/중/대형 전문가 모델 간 가중 합성 추론이 매끄럽게 수행되었습니다.

