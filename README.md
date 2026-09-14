# TRIO (RL-Refiner)

2026 컴공&인지 연합학술제 연구트랙 · 팀 야호

**크기별 전문가 분할 + SL 경계 보정(PPO teacher 증류)으로 뇌종양 MRI 마스크를 다듬는 삼중 스케일 시스템**

딥러닝이 만든 초기 분할(Rough Mask)의 오차를 종양 크기에 맞는 Expert와 **SL Refiner**(학습 시 PPO teacher로 증류)가 순서대로 보정합니다. **BraTS 2021 Challenge 세부 영역 규약(ET: Enhancing Tumor, TC: Tumor Core, WT: Whole Tumor 다중 채널)을 직접 예측하며**, soft expert mixture 기반의 동적 라우팅으로 오차 경계를 다듬습니다.

| 검증 영역 (BraTS 2021 val 250명 / 14,561 슬라이스) | Stage 2 DSC | Stage 3 DSC (배포) | HD95 (px) | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **WT (Whole Tumor)** | 0.8721 | **0.8758** | **2.108 px** | 0.8945 | 0.8776 |
| **TC (Tumor Core)** | 0.8045 | **0.8120** | **11.532 px** | 0.8842 | 0.8601 |
| **ET (Enhancing Tumor)** | 0.7792 | **0.7873** | **12.347 px** | 0.8317 | 0.8558 |

---

## 목차

1. [왜 필요한가](#왜-필요한가)
2. [파이프라인](#파이프라인)
3. [주요 결과](#주요-결과)
4. [시작하기](#시작하기)
5. [프로젝트 구조](#프로젝트-구조)
6. [시각화](#시각화)
7. [문서](#문서)

---

## 왜 필요한가

U-Net 계열 모델은 종양의 전역 위치는 잘 잡지만, 크기별 이질성과 세부 영역(ET/TC/WT) 경계 오차가 남기 쉽습니다.

- **임상적 정밀도:** 감마나이프 같은 방사선 정밀 수술이나 종양 절제 수술에서는 1 mm 수준의 오차가 임상적 결과를 좌우합니다.
- **BraTS Challenge 표준 규약:** 단순 1채널 이진 경계선 제약을 넘어 **ET(조영 증강 종양), TC(종양 핵), WT(전체 종양)** 3개 세부 영역을 직접 세그멘테이션하고 동적 보정합니다.
- **Soft Expert Mixture:** 경계 부근에서 hard routing 오분류로 인한 성능 저하를 방지하기 위해 소프트맥스 확률 가중치를 적용합니다.

---

## 파이프라인

<img width="1024" height="507" alt="TRIO pipeline" src="https://github.com/user-attachments/assets/78c1aab7-dab0-4c6c-a0bd-c5d98093b11a" />

| 단계 | 역할 | 산출물 |
|:---:|---|---|
| **1** | 종양 면적으로 Small / Medium / Large 분류 | `shape_classifier_best.pt` |
| **2** | 클래스별 Expert가 다중 영역 확률 맵 생성 | `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt` |
| **3** | 클래스별 SL Refiner 학습 (PPO teacher 교대 증류). **배포 추론은 SL** | `sl_refiner_*.pt`, `ppo_*.zip`(teacher) |
| **4** | BraTS Challenge 세부 영역 (ET/TC/WT) DSC / HD95 / Precision / Recall 평가 | `results/pipeline_slice_metrics_deploy.npz` |

크기별 라우팅 기준 (Soft Mixture 적용):

| 클래스 | 면적 임계값 | Expert 백본 | Stage3 (배포=SL) |
|---|---|---|---|
| Small | `< 200 px` | CaraNet 2.5D (`z-1, z, z+1`) | Multi-Region SL (+ PPO teacher) |
| Medium | `200–500 px` | UNet++ | Multi-Region SL (+ PPO teacher) |
| Large | `≥ 500 px` | SegResNet (ED/TC → WT multi-head) | Multi-Region SL (+ PPO teacher) |

관측에는 GT가 들어가지 않습니다. GT는 학습 시 손실/보상과 평가 지표에만 씁니다. 데이터는 **환자 단위(Patient-level Split)**로 고르게 나눕니다. (`checkpoints/patient_split.json`, seed 42). 재학습 시 이전 가중치 파일은 안전하게 직접 삭제(`os.remove`)되어 저장 공간 낭비를 방지합니다.

| 역할 | 환자 수 | 슬라이스 수 |
|---|---:|---:|
| train | 1,001명 | 54,921 |
| val (Hold-out 검증) | 250명 | **14,561** |

---

## 주요 결과

`t1ce+flair` 2채널, BraTS 2021 전체 환자 1,251명 중 **val 250명(14,561 슬라이스) hold-out** 평가 결과입니다.

| 평가 영역 | 초기 Expert (Stage 2) DSC | 최종 SL Refiner (Stage 3) DSC | HD95 (px) | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **WT (Whole Tumor)** | 0.8721 | **0.8758** | **2.108 px** | 0.8945 | 0.8776 |
| **TC (Tumor Core)** | 0.8045 | **0.8120** | **11.532 px** | 0.8842 | 0.8601 |
| **ET (Enhancing Tumor)** | 0.7792 | **0.7873** | **12.347 px** | 0.8317 | 0.8558 |

---

## 시작하기

### 1. 환경 설치

```bash
git clone https://github.com/aninsung/2026-summer-Interdepartmental-Academic-Conference.git
cd 2026-summer-Interdepartmental-Academic-Conference
pip install -r requirements.txt
```

BraTS 2021 데이터는 `src/data/archive`에 위치합니다.

### 2. 전체 파이프라인 학습 및 평가 실행

```bash
# 전체 파이프라인 실행 (재학습 시 기존 체크포인트 자동 삭제 후 학습)
PYTHONUNBUFFERED=1 python -u run_pipeline.py --no_tta --confidence_threshold 0.85

# 특정 평가 배포 모드 실행
python scripts/eval/evaluate_pipeline.py --split_role val --deploy_mode --stage3_mode sl
```

---

## 프로젝트 구조

```
├── run_pipeline.py                 # Stage 1–4 원스톱 자동 파이프라인
├── scripts/train/                  # 분류기 · Expert · Stage3 SL/PPO 교대 학습
├── scripts/eval/                   # 배포 평가, 영역별 지표 스윕, 비교 시각화
├── src/data/                       # BraTS 로더, 환자 단위 80/20 분할
├── src/envs/                       # MaskRefinementEnv (PPO teacher 환경)
├── src/models/                     # Shape Classifier, Experts, SL Refiner, AdaptivePipeline
├── src/utils/                      # DSC/HD95, 세부 영역 손실 함수, 게이트
├── baselines/                      # BraTS21 1·2위(KAIST, NVAUTO) 2D 각색 베이스라인
├── checkpoints/                    # 환자 분할 json 및 모델 가중치 (재학습 시 자동 삭제)
└── docs/                           # 파이프라인 · 실험 결과 · 논문 초안
```

---

## 문서

| 문서 | 내용 |
|---|---|
| [docs/PIPELINE.md](docs/PIPELINE.md) | 파이프라인 구조, Soft Routing, BraTS Multi-Region 손실 함수 및 배포 |
| [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md) | 세부 영역(WT/TC/ET) 평가 결과 및 정량적 벤치마크 분석 |
| [docs/paper_draft_ko.md](docs/paper_draft_ko.md) | 논문 초안 (학술적 기여 및 수식 정리) |
| [baselines/README.md](baselines/README.md) | KAIST / NVAUTO 2D 각색 베이스라인 구현 상세 |
