# BraTS 2021 상위 입상 방법 베이스라인

적응형 파이프라인의 성능을 대회 상위 입상 방법과 비교하기 위한 독립 실험 폴더입니다.
기존 `src/`, `scripts/` 는 건드리지 않고, 데이터 로딩과 지표 계산만 재사용합니다.

논문은 [docs/paper_draft_ko.md](../docs/paper_draft_ko.md), 최신 수치는 [README.md](../README.md)와 [docs/EXPERIMENT_RESULTS.md](../docs/EXPERIMENT_RESULTS.md) §0.12입니다. 851명 환자 평균 DSC는 TRIO **0.8604**, NVAUTO 0.8576(임계값 0.30, 탐색 하한), KAIST 0.8463(임계값 0.40)입니다. TRIO와 NVAUTO의 DSC 주변 구간은 겹치고, 반전 증강은 TRIO에만 있습니다. 정의된 슬라이스만의 HD95는 NVAUTO 4.184, KAIST 4.224, TRIO 4.608 px입니다. 아래 TRIO 행(DSC 0.9031)은 2026-08-20 단조 게이트 상한이며 [docs/history_2026-08-20.md](../docs/history_2026-08-20.md)에 정리되어 있습니다. `results/archive_2026-08-20/method_comparison_3x6.png`의 TRIO는 그 과거 Stage 3이고, 현재 그림은 `results/method_comparison_current_3x6.png`입니다. Stage 2와 경계 띠 PPO를 맞춘 그림은 `results/band_ppo_delta_matched/delta_matched_comparison.png`입니다.

기준 정량 평가: 2026-08-19 `python baselines/run_comparison.py --epochs 20 --batch_size 32 --sweep` (seed 42, 2,373 슬라이스).
같은 슬라이스 그리드용 재학습: 2026-08-20, 학습 중 val DSC KAIST **0.9021** / NVAUTO **0.8999** (원본 체크포인트 부재).

## 구현한 두 방법

| 폴더 내 이름 | 원 논문 | 대회 성적 |
|---|---|---|
| `kaist` | Luu & Park, *Extending nn-UNet for Brain Tumor Segmentation* ([arXiv:2112.04653](https://arxiv.org/abs/2112.04653)) | **최종 테스트 1위** |
| `nvauto` | Myronenko et al., *Redundancy Reduction in Semantic Segmentation of 3D Brain Tumor MRIs* ([arXiv:2111.00742](https://arxiv.org/abs/2111.00742)) | **최종 순위 2위** (1위와 통계적 유의차 없음) |

### KAIST (1위) — `models/kaist_nnunet.py`

nnU-Net 대비 논문이 제시한 세 가지 변경을 그대로 옮겼습니다.

1. **비대칭 인코더 확장.** 인코더 필터 수만 2배로 늘리고 디코더는 nnU-Net 원본
   값을 유지합니다. 인코더 최대 필터 수는 논문대로 512로 제한했습니다.
   구현값은 인코더 `(64, 128, 256, 512, 512)`, 디코더 `(256, 128, 64, 32)` 입니다.
2. **GroupNorm(32그룹).** 모든 정규화를 GroupNorm 으로 교체했습니다.
3. **Axial attention 디코더.** 행 방향과 열 방향에 각각 self-attention 을 적용해
   비용을 `(HW)²` 에서 `H·W² + W·H²` 로 낮췄습니다.

손실은 nnU-Net BraTS 설정인 **BCE + batch Dice** 에 deep supervision 가중 합을
씁니다 (`losses.py`의 `DeepSupervisionLoss`, `BceBatchDiceLoss`). 최적화는 nnU-Net 그대로
SGD(nesterov, momentum 0.99) + poly 스케줄입니다.

### NVAUTO (2위) — `models/nvauto_segresnet.py`

1. **SegResNet 백본.** MONAI 의 SegResNet 을 그대로 사용합니다. 디코더는 인코더
   특징을 concat 이 아니라 **덧셈**으로 합칩니다.
2. **InstanceNorm.** 논문이 GroupNorm 대비 메모리가 낮고 성능은 동등하다고 밝혀
   InstanceNorm 으로 바꾼 부분을 따랐습니다.
3. **Barlow Twins 중복 감소.** 같은 슬라이스를 독립적으로 두 번 교란해 두 뷰를
   만들고, 최종 정규화 직전 특징을 16배 average pooling 한 뒤 공간 영역들을
   배치로 펼쳐 3층 MLP 에 통과시킵니다. 교차상관 행렬의 대각은 1로(불변성),
   비대각은 0으로(중복 감소) 밀며 `λ=0.005` 를 씁니다. 프로젝션 브랜치는
   추론 시 사용하지 않습니다.
4. **Confidence 앙상블.** "분할 영역의 평균 확률값" 이라는 논문의 heuristic 으로
   상위 N/2 확률맵만 동일 가중 평균합니다. `eval_baseline.py` 에 체크포인트를
   여러 개 넘기면 적용됩니다. 이번 실행은 **단일 시드 모델**입니다.

최적화는 논문대로 AdamW(lr 1e-4, weight decay 1e-5, 드롭아웃 없음) + cosine 입니다.

## 원 논문과 다른 점 (중요)

**이 구현은 원 논문의 재현이 아니라 2D 각색입니다.** 아래 결과를 대회
리더보드 점수(예: NVAUTO 의 WT DSC 0.9265)와 직접 비교하면 안 됩니다.

| 항목 | 원 논문 | 본 구현 |
|---|---|---|
| 입력 차원 | 3D 패치 128³ | **2D 슬라이스 128×128** |
| 모달리티 | 4채널 (T1, T1ce, T2, FLAIR) | **2채널 (T1ce + FLAIR)** |
| 출력 | 3개 중첩 영역 (WT / TC / ET) | **이진 전체종양(WT) 1채널** |
| 학습 데이터 | 1,251명 전체 | **168명** (파이프라인과 동일 분할) |
| 교차검증 | 5-fold + fold 앙상블 | **단일 분할, 단일 모델** |
| 학습 자원 | A100 다수, 수일 | **단일 GPU, 수십 분** |
| 후처리 | BraTS 규약 변환, 면적 200 임계 | **없음** |

각색한 이유는 하나입니다. 적응형 파이프라인이 2D 이진 분할이므로, **같은 데이터와
같은 지표에서 비교해야 의미가 있기** 때문입니다. 3D 원본을 그대로 돌리면 성능
차이가 방법의 차이인지 문제 설정의 차이인지 구분할 수 없습니다.

논문에 쓸 때는 "BraTS 2021 상위 입상 방법의 핵심 설계를 본 실험 조건에 맞춰
재구현한 베이스라인" 으로 명시하고, 위 표를 함께 제시하는 것을 권장합니다.

추가로 밝혀 둘 점이 있습니다.

- **NVAUTO 와 Auto3DSeg 는 다릅니다.** BraTS 2021 당시 NVAUTO 제출물은
  SegResNet + 중복 감소 학습이었고, Auto3DSeg 프레임워크는 이후에 나왔습니다.
- **Barlow Twins 항의 가중치는 논문에 없습니다.** 논문은 `λ=0.005`(BT 내부 비율)만
  밝히고 Dice 와의 균형비는 명시하지 않아, `--bt_weight` 기본값 0.01 로 두었습니다.
  두 항의 크기가 비슷해지도록 잡은 값이므로 조정 대상입니다.
- **Axial attention 은 저해상도 단계에만 넣었습니다.** 기본값은 특징 크기 32 이하
  (즉 16×16, 32×32 단계)입니다. `build_kaist_nnunet(attention_max_size=...)` 로
  조정할 수 있습니다.
- **KAIST 모델은 결정적 모드에서도 완전히 재현되지 않습니다.** PyTorch 의 Flash
  Attention 역전파가 비결정적이기 때문입니다(실행 시 경고가 출력됩니다). NVAUTO
  모델에는 attention 이 없어 해당하지 않습니다.

## 공정성을 위해 맞춘 것

두 베이스라인과 적응형 파이프라인이 공유하는 항목입니다.

- **환자 분할**: `checkpoints/patient_split.json` 을 그대로 재사용합니다.
  `common.py`의 `check_split_compatibility`가 설정이 어긋나 분할을 덮어쓸 상황이면
  실행을 중단합니다. 세 방법 모두 같은 168명으로 학습하고 같은 42명으로 검증합니다.
- **데이터 로딩**: `src.data.patient_split.load_split_brats_datasets`
- **지표 구현**: `src/utils/metrics.py` 의 `dice`, `hd95`, `precision`, `recall`
- **크기 구간 정의**: `gt_size_class` (면적 300 / 700 기준)
- **증강**: 면적을 보존하는 플립과 90° 회전만 사용 (`--no_augment`로 해제)
- **시드**: 기본 42. `train_baseline.py`는 결정적 모드가 기본이며 `--no_deterministic`으로 해제

베이스라인은 크기 구간으로 나누지 않고 단일 모델이 전 범위를 담당합니다.
이것이 파이프라인의 라우팅 구조와 대비되는 핵심 차이입니다.

베이스라인은 **Monotonic DSC 게이트를 쓰지 않습니다.** 파이프라인의 최종 DSC는
GT를 보고 Stage 2로 되돌린 값이므로, 공정 비교의 1차 대상은 파이프라인
**Stage 2 초기 DSC**입니다.

## 사용법

```bash
# 두 베이스라인을 모두 학습하고 평가한 뒤 비교 표 생성
python baselines/run_comparison.py --epochs 20 --batch_size 32 --sweep

# 이미 학습했다면 평가만
python baselines/run_comparison.py --skip_train --sweep

# 개별 학습
python baselines/train_baseline.py --method kaist  --epochs 20 --batch_size 32
python baselines/train_baseline.py --method nvauto --epochs 20 --batch_size 32

# 개별 평가 (임계값 스윕 포함)
python baselines/eval_baseline.py --checkpoint baselines/checkpoints/kaist_best.pt --sweep

# NVAUTO confidence 앙상블 (시드를 바꿔 학습한 여러 모델)
python baselines/eval_baseline.py \
    --checkpoint baselines/checkpoints/nvauto_seed42.pt \
                 baselines/checkpoints/nvauto_seed43.pt \
                 baselines/checkpoints/nvauto_seed44.pt

# 정량 그림 (JSON만 읽음, 재추론 없음)
python baselines/plot_results.py

# 정성 표본 (기본: 크기 구간별 DSC 중앙값 부근)
python baselines/plot_samples.py --pick median
python baselines/plot_samples.py --pick worst

# TRIO와 같은 슬라이스에서 비교 (파이프라인 평가 스크립트)
python scripts/eval/plot_three_method_grid.py --indices 113,2286,1121,2295,95,1480 --out results/archive_2026-08-20/method_comparison_3x6.png
```

### 임계값에 대한 주의

파이프라인은 클래스별 이진화 임계값을 검증셋에서 조정했습니다(현재 0.80/0.80/0.50).
베이스라인을 기본값 0.5 로만 평가하면 파이프라인에만 튜닝 이점을 준 셈이 되어
불공정합니다. `--sweep` 으로 베이스라인의 최적 임계값도 함께 보고하고, 논문에는
**양쪽 모두 튜닝한 결과**를 싣는 것을 권장합니다.

이번 실행에서 베이스라인의 임계값 이득은 거의 없습니다.

| 방법 | 보고 임계값 | DSC | 최적 임계값 | 최적 DSC | 이득 |
|---|---:|---:|---:|---:|---:|
| KAIST | 0.5 | 0.8923 | 0.30 | 0.8932 | +0.0009 |
| NVAUTO | 0.5 | 0.8971 | 0.40 | 0.8972 | +0.0001 |

## 확정 비교 (2026-09-30, 851명)

개발 400명(학습 280 / 검증 60 / 방법 선택 60)을 뺀 851명, 종양 슬라이스 50,010장입니다. 집계는 슬라이스 지표의 환자 평균이고, 임계값은 검증 60명에서만 골랐습니다. TRIO는 분류기 라우팅, 임계값 0.80/0.80/0.50, `checkpoints/band_ppo.pt`입니다.

| 방법 | 임계값 | DSC | HD95 (px) | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **TRIO** | 0.80 / 0.80 / 0.50 | **0.8604** | 4.608 | 0.8918 | 0.8600 |
| NVAUTO | 0.30 | 0.8576 | **4.184** | 0.8836 | 0.8633 |
| KAIST | 0.40 | 0.8463 | 4.224 | 0.8635 | 0.8642 |

크기 행은 그 정답 면적의 슬라이스가 있는 환자 수입니다. 소형 851, 중형 757, 대형 408이며 같은 환자가 두 행 이상에 들어갑니다.

| 클래스 | TRIO DSC | KAIST DSC | NVAUTO DSC |
|---|---:|---:|---:|
| 300px 미만 | **0.7933** | 0.7664 | 0.7821 |
| 300–700px | 0.8952 | 0.8886 | **0.8980** |
| 700px 이상 | 0.9170 | 0.9160 | **0.9198** |

같은 학습 분할로 돌린 단일 백본+섹터 PPO는 이 851명 표와 집계가 다릅니다. 미사용 48명, 슬라이스 2,754장의 슬라이스 평균 Small DSC는 TRIO 0.812, UNet++ 0.778, Attention U-Net 0.761, SegResNet 0.676, UNet+++ 0.543, U-Net 0.537입니다. 상세는 [docs/EXPERIMENT_RESULTS.md §0.14](../docs/EXPERIMENT_RESULTS.md)이고 그림은 `results/single_backbone_ppo_grid.png`입니다.

전체 DSC의 점추정은 TRIO가 조금 높지만 NVAUTO와 95% 구간이 겹칩니다. 300px 미만 DSC는 TRIO가 높고, 중형·대형 DSC와 전체 HD95 평균은 NVAUTO가 더 낮습니다. 방법 사이 짝 차이의 구간은 없습니다. 재현:

```bash
python baselines/eval_heldout.py --checkpoint baselines/checkpoints/kaist_best.pt --out baselines/results/kaist_heldout.json
python baselines/eval_heldout.py --checkpoint baselines/checkpoints/nvauto_best.pt --out baselines/results/nvauto_heldout.json
python scripts/eval/eval_three_by_size.py
python scripts/eval/plot_current_three_method_grid.py --out results/method_comparison_current_3x6.png
```

## 과거 결과 (2026-08-19, val 42명)

베이스라인은 2026-08-19, val 42명 **2,373 슬라이스**. 파이프라인은 2026-08-20, 같은 42명 **2,434 슬라이스**(분류기 라우팅). 슬라이스 수가 완전히 같지는 않습니다.

체크포인트에 기록된 학습 중 val DSC: KAIST 0.8923, NVAUTO 0.8971.

| 방법 | DSC | HD95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Extending nnU-Net (KAIST) | 0.8923 | 1.959 | 0.9215 | 0.8893 |
| SegResNet + 중복 감소 (NVAUTO) | 0.8971 | 1.902 | 0.9077 | 0.9029 |
| 파이프라인 Stage 2 | 0.8948 | 1.734 | — | — |
| **TRIO** Stage 3 (GT 게이트) | **0.9031** | **1.606** | — | — |

크기 구간(베이스라인은 GT 면적 n=942 / 847 / 584, 파이프라인은 분류기 라우팅 n=897 / 1,152 / 385):

| 클래스 | KAIST DSC | NVAUTO DSC | 파이프라인 초기 → 최종 |
|---|---:|---:|---|
| Small | 0.8278 | 0.8377 | 0.8286 → **0.8429** |
| Medium | 0.9165 | 0.9200 | 0.9264 → **0.9314** |
| Large | **0.9614** | 0.9594 | 0.9546 → 0.9582 |

**Stage 2(0.8948)는 KAIST를 넘고 NVAUTO에 근접합니다.** Stage 3(TRIO)는 GT 게이트 상한입니다. Large 격차는 약 0.003입니다.

### 같은 슬라이스 정성 비교

`results/archive_2026-08-20/method_comparison_3x6.png`는 TRIO / KAIST / NVAUTO를 같은 val 슬라이스 6장에서 비교합니다. 행 이름은 TRIO, 제목은 영어입니다. 고정 인덱스 `113,2286,1121,2295,95,1480`.

```bash
python scripts/eval/plot_three_method_grid.py --indices 113,2286,1121,2295,95,1480 --out results/archive_2026-08-20/method_comparison_3x6.png
```

그리드용 가중치는 원본이 없어 2026-08-20에 다시 학습했습니다. 정량 JSON(0.8923 / 0.8971)과 그림의 윤곽이 같은 모델이 아닙니다. 재학습 명령:

```bash
python baselines/train_baseline.py --method kaist  --epochs 20 --batch_size 32 --seed 42 --no_deterministic
python baselines/train_baseline.py --method nvauto --epochs 20 --batch_size 32 --seed 42 --no_deterministic
```

## 폴더 구조

```
baselines/
├── common.py                 # 분할 호환성 검사, collate
├── losses.py                 # DeepSupervision / BCE+batch Dice / Barlow Twins
├── models/kaist_nnunet.py
├── models/nvauto_segresnet.py
├── train_baseline.py
├── eval_baseline.py
├── eval_heldout.py           # 851명 환자 평균
├── run_comparison.py
├── plot_results.py           # 크기별 막대, 임계값 곡선
├── plot_samples.py           # 중앙값/최악 정성 표본
├── checkpoints/{method}_best.pt
└── results/
    ├── comparison.md
    ├── {method}_metrics.json
    ├── {method}_heldout.json     # 851명 환자 평균
    ├── baseline_by_size.png
    ├── baseline_threshold_sweep.png
    ├── baseline_samples_median.png
    └── baseline_samples_worst.png
```

## 산출물

- `baselines/checkpoints/{method}_best.pt` — 가중치와 메타데이터(`val_dsc` 포함)
- `baselines/results/{method}_metrics.json` — 구간별 지표, 임계값 스윕, `best_threshold`
- `baselines/results/comparison.md` — 비교 표
- `baselines/results/baseline_*.png` — 정량·정성 그림
- `results/method_comparison_current_3x6.png` — 현재 Stage 3 / KAIST / NVAUTO, 851명 표본
- `results/archive_2026-08-20/method_comparison_3x6.png` — 2026-08-20 TRIO / KAIST / NVAUTO 같은 슬라이스 그리드
