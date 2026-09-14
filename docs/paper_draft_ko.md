# TRIO: 소프트 동적 라우팅 기반 크기별 전문가 분할과 강화학습 증류 보정을 결합한 다중 영역 뇌종양 MRI 분할

**영문 제목:** TRIO: Soft Dynamic Routing of Size-Matched Experts with Distilled RL Refinement for Multi-Region Brain Tumor Segmentation

**저자:** 안인성(컴퓨터공학과), 한수진(인공지능학과)  
**소속:** 2026 컴공&인지 연합학술제 연구트랙 · 팀 야호

---

## 초록

뇌종양 자기공명영상(MRI) 분할은 조영 증강 종양(ET), 종양 핵(TC), 전체 종양(WT) 등 세부 영역별 이질성과 크기 분산이 극심하여 단일 딥러닝 백본만으로 일관된 정밀도를 확보하기 어렵다. 본 연구는 T1ce와 FLAIR 2채널 입력에서 종양 크기에 따른 소프트 동적 라우팅(Soft Dynamic Routing)으로 크기별 전문가 백본(CaraNet 2.5D, UNet++, SegResNet)의 예측을 가중 합성하고, SL Refiner(PPO teacher 증류)로 세부 영역 경계를 다듬는 파이프라인 **TRIO**를 제안한다.

Soft Routing을 통해 분류기 경계 모호성에 의한 성능 저하를 방지하였으며, BraTS 2021 규약에 맞춘 다중 영역 손실 함수(`MultiChannelBCEDiceLoss`)로 생물학적 포괄 관계($\text{ET} \subseteq \text{TC} \subseteq \text{WT}$)를 유지하였다. BraTS 2021 환자 단위 hold-out 검증 집합(250명 / 14,561 슬라이스) 평가 결과, **WT DSC 0.8758 (HD95 2.108 px), TC DSC 0.8120 (HD95 11.532 px), ET DSC 0.7873 (HD95 12.347 px)**를 달성하여 적응형 구조의 유효성을 정량적으로 입증하였다.

**주요어:** 뇌종양 분할, 다중 영역(ET/TC/WT), 강화학습, PPO 증류, Soft Dynamic Routing, TRIO

---

## 1. 서론

뇌종양 분할은 수술 범위 설정 및 방사선 치료 계획 수립의 핵심 기반 기술이다. 그러나 종양은 픽셀 수가 극소한 미세 병변부터 뇌 넓은 부위를 차지하는 대형 침윤 병변까지 크기 분산이 매우 크고, 세부 영역별(ET/TC/WT) 경계 특성이 상이하다.

본 연구는 이러한 한계를 극복하기 위해:
1. 입력 종양 크기에 따른 **Soft Dynamic Routing** 기법을 적용하여 전문가 백본의 확률 맵을 소프트맥스 가중 합산하고,
2. **PPO teacher 교대 증류**를 거친 SL Refiner를 통해 세부 영역의 국소 오차를 다듬는 적응형 4단계 프레임워크 **TRIO**를 개발하였다.

---

## 2. 연구 방법론

### 2.1 Soft Dynamic Routing 및 백본 구성
- **Small (< 200 px)**: CaraNet 2.5D (인접 슬라이스 문맥 보강 + Zoom-Crop)
- **Medium (200 ~ 500 px)**: UNet++ (Nested Dense Skip Connections)
- **Large (≥ 500 px)**: SegResNet

소프트 라우팅 공식:
$$P_{\text{final}}(X) = \sum_{c \in \{\text{Small}, \text{Medium}, \text{Large}\}} w_c \cdot P_{\text{expert}_c}(X)$$

### 2.2 BraTS Multi-Region 세부 영역 예측
- **ET (Enhancing Tumor)**, **TC (Tumor Core)**, **WT (Whole Tumor)** 3개 채널 직접 예측
- `MultiChannelBCEDiceLoss` 적용으로 계층적 포함 구조 보장

---

## 3. 실험 및 결과

### BraTS 2021 Hold-out 검증 집합 (250명 / 14,561 슬라이스)

| 세부 영역 | Stage 2 (Expert) DSC | Stage 3 (Refiner) DSC | HD95 (px) | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **WT (Whole Tumor)** | 0.8721 | **0.8758** | **2.108 px** | 0.8945 | 0.8776 |
| **TC (Tumor Core)** | 0.8045 | **0.8120** | **11.532 px** | 0.8842 | 0.8601 |
| **ET (Enhancing Tumor)** | 0.7792 | **0.7873** | **12.347 px** | 0.8317 | 0.8558 |
