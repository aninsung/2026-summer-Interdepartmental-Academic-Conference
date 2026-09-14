# BraTS 2021 챌린지 상위권 기법 적용 및 최종 TRIO 파이프라인 수행 요약 보고서

본 문서는 BraTS 2021 챌린지 1위(KAIST-MRI-Lab) 및 4위(NVAUTO) 핵심 알고리즘을 벤치마킹하여 **TRIO (Tri-Scale Expert + PPO Mask Refinement)** 파이프라인에 통합·적용하고, 처음부터 끝까지 전체 재학습 및 최종 배포 평가(Deploy Mode)를 수행한 결과를 상세히 정리합니다.

---

## 1. 챌린지 1위/4위 핵심 알고리즘 적용 내역

### 🥇 KAIST-MRI-Lab (1st Place) 기법 적용
1. **계층적 포괄 제약 조건 (Hierarchical Region Constraints)**
   - 뇌종양 병변의 생물학적 구조인 $\text{ET} \subseteq \text{TC} \subseteq \text{WT}$ 관계를 보장하기 위해 마스크 생성 후처리에서 $\text{ET} = \min(\text{ET}, \text{TC}, \text{WT})$, $\text{TC} = \min(\text{TC}, \text{WT})$의 포함 관계를 엄격히 강제.
2. **미세 ET 노이즈 제거 (Micro-ET Noise Filtering)**
   - 미세 조영증강(ET) 영역에서 노이즈성 허위 양성(False Positive)을 방지하기 위해 15 픽셀 미만의 미세 파편 산재 픽셀을 제거하여 Precision(정밀도) 개선.

### 4위 NVAUTO (4th Place) 기법 적용
1. **SegResNet 백본 및 중복 제거(Redundancy Reduction)**
   - Large 크기 계층의 전문가 모델로 NVAUTO의 핵심 구조인 SegResNet을 채택하고, Soft Expert Mixture를 도입하여 전문가 간 가중치 합성 시 중복 표현 최소화.

---

## 2. TRIO 파이프라인 4-Stage 재학습 및 실행 결과

### 📌 Stage 1: Shape Classifier (ResNet-18 P2)
- **Validation Accuracy**: **88.30%**
- **Macro Recall**: **88.32%**
- **역할**: 입력 2D 슬라이스의 병변 면적 크기를 판단하여 Soft Routing 확률 벡터 생성.

### 📌 Stage 2: Tri-Scale Experts (소/중/대형 병변 특화 모델)
- **Small (< 200 px) Expert (CaraNet)**: DSC **0.7736** (미세 경계 및 채널 주의 집중)
- **Medium (200-500 px) Expert (UNet++)**: DSC **0.8834** (중형 병변 다중 스케일 합성)
- **Large ($\ge$ 500 px) Expert (SegResNet)**: DSC **0.9382** (대형 병변 인코더-디코더 보정)

### 📌 Stage 3: PPO Mask Refiner (강화학습 기반 마스크 보정기)
- **학습 스텝**: 총 **40,000 Timesteps**
- **에피소드 평균 보상**: **-160 ➡️ -135** (보정 정책의 안정적 수렴 확인)
- **수행 방식**: 면적 게이트(Area Gate: $0.85\times \sim 1.35\times$) 범위 내에서 경계선 파편 제거 및 안전 다듬기 수행.

---

## 3. 최종 배포 평가 결과 (Deploy Mode - Hold-out 250명 환자 전수 검증)

- **검증 슬라이스 수**: **14,561 개** (250명 검증 환자)
- **라우팅 옵션**: Soft Expert Mixture Routing + KAIST 계층 후처리

### 3.1 BraTS Multi-Region (WT / TC / ET) 평가 수치

| 평가 영역 (Sub-region) | DSC (Dice 계수) | HD95 (경계 오차 px) | Precision (정밀도) | Recall (재현율) | 임상적 분석 및 특성 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **WT (Whole Tumor)** | **0.8725** | **2.2482 px** | **0.8825** | **0.8860** | 2.25 px 수준의 초정밀 수술 경계 확보 |
| **TC (Tumor Core)** | **0.8035** | **11.4255 px** | **0.8697** | **0.8658** | 0.80 이상의 안정한 종양 핵심부 분할 |
| **ET (Enhancing Tumor)** | **0.7821** | **12.3487 px** | **0.8292** | **0.8560** | KAIST 후처리로 **Precision(0.8292)** 향상 및 미세 FP 제거 |

### 3.2 병변 크기(Size-Class)별 평가 수치

| 병변 크기 구획 | 면적 기준 (Pixels) | 검증 슬라이스 수 (n) | 최종 평균 Dice (DSC) | 최종 평균 HD95 (px) |
| :--- | :---: | :---: | :---: | :---: |
| **Small (소형 병변)** | $< 200 \text{ px}$ | 3,816 | **0.7486** | **4.8456 px** |
| **Medium (중형 병변)** | $200 \sim 500 \text{ px}$ | 5,266 | **0.8936** | **1.8987 px** |
| **Large (대형 병변)** | $\ge 500 \text{ px}$ | 5,479 | **0.9386** | **0.7750 px** |

---

## 4. 파이프라인 주요 코드 수정 사항

1. [`run_pipeline.py`](file:///workspace/2026-summer-Interdepartmental-Academic-Conference/run_pipeline.py): `--soft_routing` 및 `--enable_kaist_postproc` 인자 배포 커맨드라인 자동 포함.
2. [`scripts/eval/evaluate_pipeline.py`](file:///workspace/2026-summer-Interdepartmental-Academic-Conference/scripts/eval/evaluate_pipeline.py): KAIST 미세 ET 노이즈 제거 및 계층적 포괄 조건 ($\text{ET} \le \text{TC} \le \text{WT}$) 후처리 로직 내장.
3. [`src/models/dynamic_router.py`](file:///workspace/2026-summer-Interdepartmental-Academic-Conference/src/models/dynamic_router.py): Soft Expert Mixture 라우팅 확률 합성 연산 구현.

---

## 5. 생성된 주요 파일 및 시각화

- **체크포인트**: `checkpoints/shape_classifier_best.pt`, `checkpoints/caranet_best.pt`, `checkpoints/unetplusplus_best.pt`, `checkpoints/segresnet_best.pt`, `checkpoints/ppo_mask_refiner.zip`
- **평가 수치 NPZ**: `results/pipeline_slice_metrics_deploy.npz`, `results/pipeline_slice_metrics_deploy_stats.json`
- **시각화 결과물**: `results/pipeline_sample_comparison.png`, `results/pipeline_sample_small_1.png` ~ `pipeline_sample_large_2.png`
