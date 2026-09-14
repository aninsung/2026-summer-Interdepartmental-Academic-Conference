# TRIO 파이프라인 상세 문서

본 문서는 **TRIO** 파이프라인의 전체 구조, Soft Expert Mixture 라우팅, BraTS Multi-Region (ET, TC, WT) 3채널 직접 예측, 손실 함수, 그리고 자동 가중치 관리 메커니즘을 상세히 다룹니다.

---

## 1. 한눈에 보는 파이프라인 구조

입력은 BraTS 2021 환자의 **T1ce + FLAIR** 2채널 2D 슬라이스(128×128)입니다. 출력은 **BraTS Challenge 공식 세부 영역 (ET: Enhancing Tumor, TC: Tumor Core, WT: Whole Tumor)** 3채널 확률 맵 및 이진 마스크입니다.

| 단계 | 역할 | 산출물 |
|:---:|---|---|
| **Stage 1** | Shape Classifier (종양 크기 분류 및 소프트맥스 확률 산출) | `checkpoints/shape_classifier_best.pt` |
| **Stage 2** | Size Experts (CaraNet 2.5D, UNet++, SegResNet) + Soft Expert Mixture 합성 | `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt` |
| **Stage 3** | Multi-Region SL / PPO Refiner (세부 영역 경계 및 내부 정밀 보정) | `sl_refiner_*.pt`, `ppo_*.zip` |
| **Stage 4** | Deploy Evaluation (GT-Free 면적 안전 게이트 & ET/TC/WT 지표 산출) | `results/pipeline_slice_metrics_deploy.npz` |

---

## 2. 크기 클래스 및 Soft Expert Mixture

### 2.1 크기 클래스 정의
종양 면적(픽셀 수)에 따라 3개 구간으로 분류됩니다:
- **Small**: `< 200 px` (CaraNet 2.5D 백본 + Zoom-Crop)
- **Medium**: `200 ≤ area < 500 px` (UNet++ 백본)
- **Large**: `area ≥ 500 px` (SegResNet 백본)

### 2.2 Soft Expert Mixture 라우팅
하드 분류로 인한 오분류 단층(boundary drop-off)을 방지하기 위해 Stage 1 분류기의 소프트맥스 가중치 $w_c = P(\text{class}=c \mid X)$를 계산합니다. 최종 예측 확률 맵 $P_{\text{final}}$은 각 전문가 모델의 예측 $P_{\text{expert}_c}(X)$의 연속 가중합으로 생성됩니다:

$$P_{\text{final}}(X) = \sum_{c \in \{\text{Small}, \text{Medium}, \text{Large}\}} w_c \cdot P_{\text{expert}_c}(X)$$

---

## 3. BraTS Challenge Multi-Region 3채널 세구조

본 파이프라인은 1채널 단일 경계 밴드제약을 배제하고, 공식 BraTS 2021 대회 규약인 3개 세부 영역을 직접 동시 예측합니다:
1. **ET (Enhancing Tumor)**: 조영 증강 종양 (Label 4)
2. **TC (Tumor Core)**: 종양 핵 (ET + 괴사 및 비증강 종양 Label 1)
3. **WT (Whole Tumor)**: 전체 종양 (TC + 주변 부종 Label 2)

### 다중 영역 손실 함수 (`MultiChannelBCEDiceLoss`)
세부 영역 간 생물학적 포괄 관계 ($\text{ET} \subseteq \text{TC} \subseteq \text{WT}$)를 보장하기 위해 3개 채널의 BCE 손실과 Dice 손실의 가중 합으로 학습합니다.

---

## 4. 환자 단위 분할 및 가중치 관리 메커니즘

- **환자 단위 분할**: `checkpoints/patient_split.json` (Train 1,001명 / Val 250명, seed 42)
- **체크포인트 즉시 삭제 정책**: 파이프라인 학습 재실행 시, 아카이브 폴더 생성으로 인한 디스크 누적을 방지하고 `checkpoints/` 폴더 내의 기존 가중치 파일들을 직접 `os.remove`로 안전 삭제합니다.

---

## 5. 정량적 검증 성과 (Hold-out 250명 / 14,561 슬라이스)

| 영역 | Stage 2 (Expert) DSC | Stage 3 (Refiner) DSC | HD95 (px) | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **WT** | 0.8721 | **0.8758** | **2.108 px** | 0.8945 | 0.8776 |
| **TC** | 0.8045 | **0.8120** | **11.532 px** | 0.8842 | 0.8601 |
| **ET** | 0.7792 | **0.7873** | **12.347 px** | 0.8317 | 0.8558 |
