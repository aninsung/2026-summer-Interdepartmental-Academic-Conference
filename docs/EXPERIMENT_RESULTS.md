# RL-Refiner 실험 결과

> **논문:** [paper_draft_ko.md](paper_draft_ko.md). 환자 풀 **1251명** 중 개발 400명 제외, 평가 **851명** 환자 평균. Stage 3는 **경계 띠 PPO** (`checkpoints/band_ppo.pt`). 고정 임계값 Stage 2 대비 DSC 0.8337→**0.8604** (짝 차이 +0.0267, 95% CI 0.0256–0.0278). 검증에서 고른 임계값 0.45/0.80/0.20의 Stage 2(0.8411) 대비 **+0.0193** (0.0184–0.0202). HD95 평균 4.888→4.609 px는 빈 마스크 제외가 방법마다 달라 짝비교가 아니다. 상세는 [§0.11](#011-확정-평가--파이프라인-전체-미사용-851명-2026-09-30)–[§0.13](#013-임계값-재선택-2026-09-30). 2026-10-07 재학습·프로토콜·U-Net 바닐라 PPO는 [§0.15](#015-2026-10-07-재학습)다. 단일 백본+섹터 PPO 다섯 세트는 [§0.14](#014-단일-백본--단일-섹터-ppo-2026-09-30)이고, 48명 표본의 슬라이스 평균이라 851명 표와 집계가 다르다. 2026-08-20 표는 [history_2026-08-20.md](history_2026-08-20.md)다.
> **`ppo_v4`**(DSC 0.8345)는 그 이전 GT-free val 기록이고, **`ppo_v5`는 폐기**(0.7614)입니다. §0.1–0.4의 train 875 / val 188 분할은 그 기록의 설정이며 이번 400/851 분할과 다릅니다.

> 아래 §1 이후의 08-20(210명) 수치는 과거 GT 게이트 포함 실험이며, 1251명 GT-free 결과와 직접 비교하지 마세요.

- **데이터셋**: BraTS 2021 Task 1
- **입력**: `t1ce+flair`, 128×128 2D 슬라이스
- **지표**: DSC (↑), HD95 px (↓), Precision, Recall — 구현은 `src/utils/metrics.py`
- **파이프라인 구조**: [PIPELINE.md](PIPELINE.md) / **프로필·게이트**: [QUALITY_GATE.md](QUALITY_GATE.md)

---

## 0. 2026-09-29~30 기록

논문에 쓰는 표는 [§0.11](#011-확정-평가--파이프라인-전체-미사용-851명-2026-09-30)이다. §0.1–0.4는 그 이전 섹터 PPO(`ppo_v2`/`ppo_v4`)의 1251명 val 기록이다. 현재 Stage 3는 §0.9 이후의 경계 띠 PPO다.

### 0.1 설정

| 항목 | 값 |
|---|---|
| 환자 풀 / 분할 | 1251 → train **875** / val **188** / test **188** (`checkpoints/patient_split.json`, seed 42) |
| 평가 | val, `slice_selection=tumor`, 슬라이스 **11,073** |
| Stage 1 | ResNet18 2.5D, margin 80, val acc **0.8602** (clear **0.9272**) |
| Stage 2 Expert val DSC | CaraNet **0.7857** / UNet++ **0.8672** / SegResNet **0.8550** |
| Stage 2 임계값 / CC | 0.80,0.80,0.50 / 0,15,25 |
| Stage 3 PPO | 100k steps, n_envs=8, max_steps=15, `ent_coef=0.01` |
| 평가 모드 | classifier routing + `heuristic` (GT-free). Confidence bypass OFF |
| 산출물 | `results/eval_1251_val_tumor` (v2), `results/eval_1251_val_tumor_ppo_v4` (v4) |

라우팅 분포(val): Small 4401 / Medium 4603 / Large 2069.

### 0.2 종합 비교 (전체 평균 DSC)

| 방법 | 전체 DSC | Small | Medium | Large | 비고 |
|---|---:|---:|---:|---:|---|
| Stage 2 only | **0.8344** | 0.7530 | 0.8792 | 0.9081 | Expert 이진화+CC |
| PPO **v2** | 0.8306 | 0.7517 | 0.8804 | 0.8879 | Large 과소분할(Recall 0.942→0.894) |
| PPO **v4** (이전 기록) | **0.8345** | 0.7517 | 0.8805 | **0.9086** | Large 유지, Stage 2와 사실상 동일. 현재 Stage 3 아님 |
| PPO **v5** (폐기) | 0.7614 | 0.7372 | 0.7690 | 0.7957 | 게이트 해제 후 전 컴포넌트 자유 수정 → 붕괴 |

전체 HD95 / Small HD95는 한쪽만 비는 슬라이스 때문에 `nan`이 섞일 수 있습니다(평가 `np.mean`).

### 0.3 v4 클래스별 (GT-free heuristic)

| 클래스 | n | Stage 2 DSC | v4 DSC | Stage 2→v4 P | Stage 2→v4 R | Medium/Large HD95 (px) |
|---|---:|---:|---:|---|---|---|
| Small | 4401 | 0.7530 | 0.7517 | 0.808→0.835 | 0.751→0.725 | nan |
| Medium | 4603 | 0.8792 | 0.8805 | 0.902→0.900 | 0.871→0.876 | 3.92→4.03 |
| Large | 2069 | 0.9081 | 0.9086 | 0.887→0.886 | 0.942→0.943 | 4.64→4.75 |

### 0.4 프로필 요약

| 프로필 | 요지 | 결과 |
|---|---|---|
| **ppo_v2** | 컴포넌트 단위, 8섹터, TTA 정렬 확률, selective shrink/expand | Stage 2 대비 −0.0038. Large 과수축 |
| **ppo_v4** | MRI 전채널 관측, 섹터 16/24/32, 학습=추론 게이트(불확실 섹터·단방향), 보상=크기보정 ΔDSC+surface HD95 | Stage 2와 동등(0.8345). Large 과수축 완화 |
| **ppo_v5** (폐기) | 게이트 없음, 부채꼴 자유 expand/shrink, 수정 부채꼴만 로컬 보상, 200k steps | 학습 중 ΔDSC 한 번도 + 아님. Medium/Large harm≈100%. 체크포인트·결과·설정 삭제함 |

### 0.5 진단에서 확인한 한계 (v4 이후)

컴포넌트 오라클(GT로 부채꼴별 최적 SDF shift)은 같은 제한 안에서도 DSC 상한이 높음(예: Large ~0.96).  
실제 PPO는 부채꼴마다 늘릴/줄일 곳을 구분하지 못해 상한에 못 미침.  
선택기(밝기 기반 shrink/expand) 정확도는 ~50%.  
**목적 변경 (2026-09-30):** Stage 3는 외곽선 평행 이동이 아니라 경계 띠의 영역·경계 보정이다. 편집은 마스크 ±수 픽셀 또는 확률 0.35–0.65로 제한하고, 띠 픽셀을 켜거나 끈다. 채택은 HD95가 Stage 2보다 줄 때다. 윤곽선 이동과 전역 재분할(v5)은 이 목적의 구현이 아니다.

### 0.8 윤곽 표현 타당성 (MARL-MambaContour 선검증, 2026-09-30)

- 환자 400, seed 42. Medium/Large만, GT 면적 라우팅, TTA. 모델은 컴포넌트당 128점 닫힌 윤곽을 3회 이동(순환 1D conv, 1.14M). 학습 val 컴포넌트 DSC 0.8864→**0.8942**.
- test 60명 / 2170슬라이스 (`results/contour_probe.json`):

| | Stage 2 | 윤곽 회귀 | 윤곽 상한 |
|---|---:|---:|---:|
| DSC | 0.9102 | **0.9178** | 0.9890 |
| HD95 px | **3.847** | 3.957 | 1.571 |

- Medium 0.9035→0.9117, Large 0.9223→0.9290. 슬라이스의 80%에서 윤곽 DSC가 더 높음. Recall 0.898→0.910.
- 호 길이 점 대응 목표는 마스크를 못 바꿈(val 0.8858 < 초기 0.8864). 최근접 경계점 목표만 유효.
- 판정: 컴포넌트별 닫힌 윤곽은 Medium/Large에서 Stage 2를 DSC로 넘김. HD95는 아직 Stage 2보다 나쁨. 상한(DSC 0.989, HD95 1.57)까지 여유가 있어 MARL은 HD95를 직접 노릴 때만 의미 있음. Small은 다중 컴포넌트 비율이 높아 단일 윤곽으로는 상한이 0.94대.
- 표면 손실을 처음부터 가중치 0.5로 학습하면 DSC가 깨지고 HD95도 악화되어 저장하지 않음.
- 최근접점 모델을 HD95 상위 5% 양방향 거리로 12 epoch 미세조정(`contour_evolve_surface.pt`). test 2170장 (`results/contour_probe_surface.json`): DSC 0.9102→**0.9180**, HD95 3.847→**3.882**. 최근접점만 썼을 때(0.9178 / 3.957)보다 HD95는 줄었지만 Stage 2보다는 크다. Medium HD95만 3.592→3.544로 개선, Large는 4.312→4.500으로 악화.

### 0.9 경계 띠 보정 (Medium, 2026-09-30)

확률 0.35–0.65 또는 마스크 경계 ±2px만 켜기/유지/끄기. 손실은 전체 Dice + 띠 경계거리. 추론 가드는 띠 안 FLAIR 상대 밝기. 체크포인트 `checkpoints/band_refine_medium.pt`. test 60명 / Medium 1402장 (`results/band_refine_medium.json`):

| | Stage 2 | raw | FLAIR 가드 |
|---|---:|---:|---:|
| DSC | 0.9035 | **0.9053** | 0.9048 |
| HD95 px | 3.592 | 3.312 | **3.311** |

가드는 슬라이스의 99.7%를 바꾸고, DSC가 더 높은 비율은 54.8%. Precision 0.920→0.915, Recall 0.896→0.903. val 가드 DSC 0.9177→0.9186, HD95 2.535→2.424. HD95는 윤곽 surface(Medium 3.544)보다 낮다. DSC 이득은 0.0013.

### 0.10 클래스 공통 경계 띠 PPO (2026-09-30)

정책은 하나(`checkpoints/band_ppo.pt`). 시작 가중치는 Medium 지도학습. 5스텝, 마지막 스텝만 FLAIR 가드. 보상은 뒤집힌 픽셀의 GT 일치와 HD95 감소. Small Expert는 이번 400명 분할에서 학습한 CaraNet(val DSC 0.8159). test 60명 / 3495장, GT 면적 라우팅 (`results/band_ppo.json`):

| | Stage 2 DSC | PPO DSC | Stage 2 HD95 | PPO HD95 |
|---|---:|---:|---:|---:|
| Small | 0.7500 | **0.7846** | 6.640 | **6.393** |
| Medium | 0.9035 | **0.9215** | 3.592 | **2.885** |
| Large | 0.9223 | **0.9330** | 4.312 | **4.073** |
| 전체 | 0.8494 | **0.8721** | 4.905 | **4.474** |

세 클래스 모두 DSC·HD95가 Stage 2보다 나아서 전부 채택. 저장된 검증 모델은 5 epoch, macro DSC 0.8972 / HD95 3.558 (Stage 2 0.8741 / 3.838). 이 표는 GT 면적 라우팅의 개발 test이며 논문 확정 수치가 아니다. 확정 수치는 §0.11이다.

### 0.11 확정 평가 — 파이프라인 전체, 미사용 851명 (2026-09-30)

1251명 중 개발 400명(train 280 / val 60 / 방법 선택 test 60)은 제외. 남은 851명, 종양 슬라이스 50010장. 분류기(val acc 0.8151, 이 집합 슬라이스 정확도 0.801)가 Expert를 고르고 TTA·임계값 0.80/0.80/0.50 후 `band_ppo.pt`가 보정. 집계는 슬라이스 DSC의 환자 평균이다. 결과는 `results/band_ppo_locked.json`. 재현은 `python scripts/eval/evaluate_band_ppo_locked.py`.

크기 행은 환자군이 아니다. 그 정답 면적의 슬라이스가 있는 환자 수이며, 같은 환자가 두 행 이상에 들어간다. 슬라이스 수는 소형 19077, 중형 19907, 대형 11026이다.

| 정답 크기 구간 | 해당 슬라이스가 있는 환자 | Stage 2 DSC | PPO DSC | 환자 짝 차이 (95% CI) |
|---|---:|---:|---:|---|
| 전체 | 851 | 0.8337 | **0.8604** | +0.0267 (0.0256–0.0278) |
| 300px 미만 | 851 | 0.7579 | **0.7933** | +0.0354 (0.0338–0.0370) |
| 300–700px | 757 | 0.8700 | **0.8952** | +0.0252 (0.0239–0.0265) |
| 700px 이상 | 408 | 0.8981 | **0.9169** | +0.0188 (0.0173–0.0205) |

DSC 환자 짝 차이 **+0.0267**의 95% CI는 0을 포함하지 않는다. 이 표는 고정 임계값 Stage 2와의 짝비교다. 임계값을 다시 고른 대조는 §0.13이다. 앞의 400명 test 표(정답 면적 라우팅, DSC 0.8721)는 방법 선택 결과다. 고정 임계값 0.80/0.80/0.50과 연결요소 기준은 이전 210명 풀에서 정했고, 그 풀의 148명이 이 851명에 들어 있다. 띠 설계는 이번 방법 선택 60명에서 정했다. 가중치는 이번 400명 분할로 다시 학습했다. Stage 2와 PPO를 맞춘 여섯 슬라이스는 `results/band_ppo_delta_matched/delta_matched_comparison.png`다. 열 제목의 뒤는 분류기 라우팅이고, 중형 첫째는 CaraNet이다. 칸의 HD95는 환자 평균이 아니며, 그 중형 슬라이스는 2.00에서 3.38 px로 늘었다. 재현은 `python scripts/eval/plot_delta_matched.py`다.

HD95는 128px 경계 거리이고, 예측과 정답 중 한쪽이 비면 그 슬라이스를 평균에서 뺀다. Stage 2와 PPO가 서로 다른 슬라이스를 뺄 수 있어 아래는 짝비교가 아니다.

| 정답 크기 구간 | Stage 2 HD95 | PPO HD95 | 기록된 평균 차이 (95% CI) |
|---|---:|---:|---|
| 전체 | 4.888 | 4.609 | −0.279 (−0.345–−0.219) |
| 300px 미만 | 5.801 | 5.697 | −0.104 (−0.192–−0.019) |
| 300–700px | 4.364 | 3.887 | −0.477 (−0.563–−0.390) |
| 700px 이상 | 4.484 | 3.924 | −0.560 (−0.658–−0.461) |

`run_pipeline.py --refinement_profile band_ppo --max_train_patients 1251 --split_role all --slice_selection tumor` 결과(`results/band_ppo_pipeline_1251`)는 개발 400명을 포함한 슬라이스 평균이다. DSC 0.8503→0.8758. HD95는 한쪽이 빈 슬라이스를 빼면 4.441→4.085 px (Small 6.348→5.904, Medium 3.467→3.109, Large 2.929→2.730). 논문 확정 표는 위 851명 환자 평균이다.

### 0.13 임계값 재선택 (2026-09-30)

검증 60명에서 분류기 라우팅 기준 크기별 임계값 세 개를 환자 평균 DSC로 함께 골랐다. 탐색은 0.10–0.90, 0.05 간격이고 연결요소 기준은 그대로다. 고른 값은 소형 0.45, 중형 0.80, 대형 0.20이다. 검증 DSC는 0.8588에서 0.8650이 되었다. 그 임계값을 851명에 한 번 적용했다. PPO 가중치는 고정 임계값 마스크로 학습한 것을 그대로 썼다. 파일은 `results/stage2_retuned.json`. 재현은 `python scripts/eval/evaluate_stage2_retuned.py`.

| 정답 면적 구간 | 고정 Stage 2 | 재선택 Stage 2 | PPO | PPO − 재선택 (95% CI) |
|---|---:|---:|---:|---|
| 전체 | 0.8337 | 0.8411 | **0.8604** | +0.0193 (0.0184–0.0202) |
| 300px 미만 | 0.7579 | 0.7682 | **0.7933** | +0.0251 (0.0237–0.0265) |
| 300–700px | 0.8700 | 0.8790 | **0.8952** | +0.0162 (0.0154–0.0170) |
| 700px 이상 | 0.8981 | 0.9014 | **0.9169** | +0.0156 (0.0143–0.0169) |

재선택만의 짝 차이는 +0.0074 (0.0066–0.0082)로, 고정 임계값 대비 +0.0267의 약 28%이다. 전체 정밀도/재현율은 고정 Stage 2 0.883/0.820, 재선택 Stage 2 0.863/0.849, PPO 0.892/0.860이다. 재선택 마스크에서 PPO를 시작해도 전체 DSC는 0.8604이다.

### 0.15 2026-10-07 재학습

9월 30일 잠금 표(0.8337→0.8604)와 다른 가중치다. 환자 분할은 `checkpoints/patient_split.json` 그대로 두고 시드 42, 7, 123만 바꿨다. 집계는 종양 슬라이스 DSC의 환자 평균이다. 파일은 `results/protocol_gaps/three_seeds.json`이다.

| 시드 | Stage 2 | 경계 띠 PPO | 대형 전문가 전 슬라이스 |
|---|---:|---:|---:|
| 42 | 0.8339 | 0.8597 | 0.8503 |
| 7 | 0.8370 | 0.8620 | 0.8540 |
| 123 | 0.8352 | 0.8609 | 0.8500 |

PPO DSC 평균은 0.8609이고 범위는 0.0023(0.8597–0.8620)이다. 가중치는 `checkpoints/seeds/{42,7,123}/`이다. 시드 123 학습이 끝난 뒤 작업 디렉터리의 `checkpoints/band_ppo.pt`는 시드 123 배우다.

빈 슬라이스를 포함한 851×155=131,905장. 종양 비율 0.2% 미만이면서 예측이 있는 비율, z를 쌓은 3D Dice 환자 평균, 양쪽 마스크가 있는 슬라이스의 HD95 환자 평균이다. mm는 픽셀×1.9이다. HD95는 양쪽이 비면 0으로 넣지 않고, 한쪽만 비면 정의하지 않는다. Stage 2와 PPO의 HD95는 둘 다 마스크가 있는 슬라이스 집합이다. 대형 행의 HD95는 Stage 2·PPO·대형이 모두 있는 슬라이스 집합이라 위 두 행과 분모가 다르다. 원본은 `results/protocol_gaps/seed_{42,7,123}.json`이다.

| 시드 | 방법 | 빈 슬라이스 거짓 양성 | 3D Dice | 공통 HD95 (px) | 공통 HD95 (mm) |
|---|---|---:|---:|---:|---:|
| 42 | Stage 2 | 0.567 | 0.779 | 6.072 | 11.536 |
| 42 | 경계 띠 PPO | 0.521 | 0.813 | 5.649 | 10.734 |
| 42 | 대형 전 슬라이스 | 0.930 | 0.799 | 5.569 | 10.581 |
| 7 | Stage 2 | 0.532 | 0.786 | 5.779 | 10.979 |
| 7 | 경계 띠 PPO | 0.497 | 0.811 | 5.448 | 10.351 |
| 7 | 대형 전 슬라이스 | 0.821 | 0.796 | 5.491 | 10.433 |
| 123 | Stage 2 | 0.601 | 0.771 | 6.061 | 11.516 |
| 123 | 경계 띠 PPO | 0.571 | 0.809 | 5.648 | 10.731 |
| 123 | 대형 전 슬라이스 | 0.829 | 0.759 | 5.994 | 11.388 |

같은 시드 42 가중치의 추론 ablation(`results/ablation_review/summary.json`)에서 5스텝 PPO는 0.8597, 1스텝 0.8460, 중형 지도학습 초기화 1스텝 0.8383, 대형 전문가에 경계 띠 PPO 0.8676, 정답 크기 라우팅 PPO 0.8698이다. 독립 703명에서 5스텝 PPO는 0.8579이다. 반경 0은 0.8427이고 스텝 3, 가드 해제, 띠 0.45–0.55, 띠 0.25–0.75는 0.8575–0.8601이다.

단일 U-Net(`checkpoints/single_backbone/unet.pt`)을 같은 280명으로 학습하고 임계값 0.5, TTA 없이 851명 종양 슬라이스 50,010장에 적용하면 DSC 0.8156(95% CI 0.8067–0.8243), HD95 6.939 px이다. 전역 1픽셀 수축·유지·팽창 바닐라 PPO 15스텝(`ppo_unet_vanilla.zip`)은 DSC 0.8133(0.8046–0.8220), HD95 6.955 px이다. 환자 짝 차이는 −0.0023(95% CI −0.0029–−0.0017)이다. 정답 크기별 DSC는 소형 0.7215→0.7197(851명), 중형 0.8758→0.8736(757명), 대형 0.8973→0.8939(408명)이다. 파일은 `results/vanilla_unet_851.json`이다. 재현은 `python scripts/eval/evaluate_vanilla_unet.py`이다.

### 0.14 단일 백본 + 단일 섹터 PPO (2026-09-30)

크기별 전문가와 경계 띠를 쓰지 않는 대조다. 학습은 `checkpoints/patient_split.json`의 train 280명, 종양 슬라이스 16,500장, 20 epoch, 배치 32, 이진 Whole Tumor다. SegResNet은 `--no_multi_region`이라 대형 전문가의 ED+TC 합성이 아니다. 초기 마스크 임계값은 0.5다. U-Net 검증 DSC는 0.8834다. 나머지 네 백본의 검증 DSC는 이 실행 로그에 남아 있지 않다.

PPO는 백본마다 하나다. `MaskRefinementEnv` `refinement_mode=medium`, 관측 3채널, 5×8 섹터, `max_steps=15`, `total_timesteps=100000`, 환경 8개, CnnPolicy, `n_steps=512`, `batch_size=64`, `n_epochs=4`, 학습률 `3e-4`, `ent_coef=0.01`. 가중치는 `checkpoints/single_backbone/`에 있고 TRIO 체크포인트를 덮지 않는다.

아래 표는 개발 400명을 뺀 환자 48명(seed 7), 종양 슬라이스 2,754장의 **슬라이스 평균**이다. 851명 환자 평균이 아니다. HD95는 양쪽이 비지 않은 슬라이스만의 슬라이스 평균이다. TRIO 열은 같은 48명에 현재 `band_ppo.pt` 파이프라인을 적용한 값이다. 파일은 `results/single_backbone_ppo_grid_metrics.json`, 그림은 `results/single_backbone_ppo_grid.png`. 재현은 `python scripts/eval/plot_single_backbone_grid.py`.

| 방법 | Small DSC | Medium DSC | Large DSC | Small HD95 | Medium HD95 | Large HD95 |
|---|---:|---:|---:|---:|---:|---:|
| TRIO | 0.812 | 0.904 | 0.916 | 4.877 | 3.432 | 3.761 |
| U-Net + PPO | 0.537 | 0.813 | 0.866 | 12.126 | 6.518 | 6.527 |
| UNet++ + PPO | 0.778 | 0.894 | 0.926 | 5.786 | 3.617 | 3.773 |
| UNet+++ + PPO | 0.543 | 0.828 | 0.900 | 10.308 | 5.087 | 4.345 |
| Attention U-Net + PPO | 0.761 | 0.884 | 0.904 | 6.660 | 3.821 | 4.606 |
| SegResNet + PPO | 0.676 | 0.871 | 0.912 | 8.821 | 4.240 | 4.319 |

이 표본의 Small DSC는 TRIO 0.812, UNet++ 0.778, Attention U-Net 0.761이다. U-Net과 UNet+++은 0.537, 0.543이다. Medium·Large DSC는 UNet++가 0.894, 0.926으로 이 표본의 TRIO와 같은 자리다.

### 0.6 Stage 1 변경 요약 (1251)

- `ShapeDataset`: margin=80으로 300/700 경계 ±80px clear 샘플만 학습, 2.5D 6채널, augment
- 분류기: label smoothing 0.05, Clear Acc 모니터링, early stop
- 캐시: `BRATS_SKIP_CACHE_SAVE=1`로 대용량 pickle 저장 생략 가능

### 0.7 재현 명령

현재 파이프라인. 논문 표는 앞의 세 평가 명령이다. `run_pipeline.py`는 개발 400명이 포함된 슬라이스 평균이다.

```bash
python scripts/eval/evaluate_band_ppo_locked.py
python scripts/eval/evaluate_stage2_retuned.py
python scripts/eval/eval_three_by_size.py
python scripts/eval/plot_single_backbone_grid.py

python run_pipeline.py --refinement_profile band_ppo \
  --max_train_patients 1251 --split_role all --slice_selection tumor \
  --no_deterministic --output_dir results/band_ppo_pipeline_1251 --no_plots
```

체크포인트: `checkpoints/shape_classifier_best.pt`, `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt`, `band_ppo.pt`.

이전 `ppo_v4` 기록(§0.1–0.4)을 다시 만들 때만 아래를 쓴다. 논문 표가 아니다.

```bash
python run_pipeline.py --config configs/ppo_brats_v4.yaml \
  --max_train_patients 1251 --num_workers 8 \
  --slice_selection tumor --no_deterministic \
  --output_dir results/eval_1251_val_tumor_ppo_v4
```

그때의 에이전트는 `checkpoints/ppo_v4/ppo_{small,medium,large}.zip`이다.

---

## 1. 과거 결과 (2026-08-20, val hold-out, 210명 풀)

### 1.1 평가 설정

| 항목 | 값 |
|---|---|
| 실행 명령 | `python run_pipeline.py batch_size=64` |
| 환자 풀 | 210명 → **train 168명 / val 42명** (`checkpoints/patient_split.json`, seed 42) |
| 학습 슬라이스 | 9,868 |
| **평가 슬라이스** | **2,434** (val 환자 전수, 분류기 라우팅) |
| 클래스 라우팅 | **Stage 1 분류기 예측** (Oracle 아님) |
| Stage 2 이진화 임계값 | **0.80 / 0.80 / 0.50** (Small / Medium / Large) |
| CC filter | **0 / 15 / 25** (Small 끔) |
| Small Expert | CaraNet **2.5D** + `<50px` ×4 오버샘플 + zoom-crop, BCEDice |
| Large Expert | 전 구간 학습 + **ED/TC** 2채널 → WT 합집합 |
| Stage 3 | 세 클래스 **모두** PPO `predict()` 15스텝. 초안 마스크는 평가와 같은 임계값으로 이진화 |
| 안전장치 | **GT-free 면적**(0.2×–4×, 배포 가능) + **Monotonic DSC Gate**(GT, 상한) |
| Confidence skip | 기본 **OFF** (`--confidence_threshold` 미지정) |
| 시드 / 결정적 모드 | 42 / ON |

08-20 기록에서는 면적 게이트와 GT DSC 단조 게이트를 사용했습니다. 그 게이트는 아래 섹터 PPO 절차에 남아 있다. 현재 `band_ppo` 평가는 면적·에지 gate와 GT 단조 게이트를 쓰지 않고, 마지막 스텝의 FLAIR 가드만 둔다. 상세는 [QUALITY_GATE.md](QUALITY_GATE.md)다.

### 1.2 파이프라인 종합

| 지표 | Stage 2 초기 | Stage 3 보정 후 | 변화 |
|---|---:|---:|---:|
| 평균 DSC | 0.8948 | **0.9031** | **+0.0083 (+0.83%p)** |
| 평균 HD95 | 1.7336 px | **1.6060 px** | **−0.1276 px** |

세 클래스 모두 DSC가 올랐고 HD95가 줄었습니다.

### 1.3 크기별 성능

분류기 라우팅 결과 분포: Small 897 / Medium 1,152 / Large 385.

| 클래스 | Expert | n | 초기 DSC | 최종 DSC | ΔDSC | 초기 HD95 | 최종 HD95 | ΔHD95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Small (`<300px`)** | CaraNet 2.5D | 897 | 0.8286 | **0.8429** | +0.0143 | 3.1618 | **2.9482** | −0.2136 |
| **Medium (`300–700px`)** | UNet++ | 1,152 | 0.9264 | **0.9314** | +0.0050 | 1.0790 | **1.0000** | −0.0790 |
| **Large (`≥700px`)** | SegResNet ED/TC | 385 | 0.9546 | **0.9582** | +0.0036 | 0.3650 | **0.2919** | −0.0731 |

### 1.4 Precision / Recall

`P < R`이면 과분할(경계가 GT 밖으로 번짐), `P > R`이면 과소분할(종양을 놓침)입니다.

| 클래스 | 초기 P | 최종 P | 초기 R | 최종 R | 해석 |
|---|---:|---:|---:|---:|---|
| Small | 0.9005 | **0.9195** | 0.7958 | 0.8008 | **과소분할**. PPO가 Precision을 더 올림 |
| Medium | 0.9529 | 0.9545 | 0.9089 | **0.9165** | 약한 과소분할. PPO가 Recall을 올림 |
| Large | 0.9596 | 0.9613 | 0.9515 | **0.9568** | 거의 균형. 임계값 0.50이 Recall을 회복 |

Large 임계값을 0.70에서 0.50으로 내린 뒤 P/R이 맞춰졌습니다.

### 1.5 Small 층화

| 구간 | n | 초기 DSC | 최종 DSC | 초기 HD95 | 최종 HD95 |
|---|---:|---:|---:|---:|---:|
| Active Tumor (`≥50px`) | 828 | 0.8427 | **0.8529** | 2.8934 | **2.7456** |
| Micro Fragment (`<50px`) | 69 | 0.6595 | **0.7239** | 6.3828 | **5.3794** |

미세 파편 DSC +6.44%p, HD95도 줄었습니다.

### 1.6 안전 가드와 Monotonic DSC 발동률

기본 평가 순서: PPO 15스텝 → 면적 게이트(0.2×–4×) → 단조 DSC 게이트. Confidence skip은 끄고 돌렸습니다.

| 항목 | 값 |
|---|---:|
| 평가 슬라이스 | 2,434 |
| 단조 게이트 발동 (최종 DSC &lt; 초기 → Stage 2 유지) | **858 (35.3%)** |

PPO는 슬라이스 3분의 1 이상에서 DSC를 떨어뜨렸고, 보고된 최종 점수는 그 손실을 **GT로 걸러낸** 값입니다. “클래스별 점수 하락 0%”는 이 게이트 이후의 이야기이며, 게이트 없는 Stage 2가 배포에 가까운 숫자입니다.

### 1.7 시각화 샘플 (클래스 평균 Final DSC에 가깝고 Final > Initial)

| 파일 | Rough DSC | RL DSC | ΔDSC |
|---|---:|---:|---:|
| `pipeline_sample_small_1.png` | 0.8322 | 0.8429 | +0.0106 |
| `pipeline_sample_small_2.png` | 0.8127 | 0.8432 | +0.0305 |
| `pipeline_sample_medium_1.png` | 0.9003 | 0.9315 | +0.0313 |
| `pipeline_sample_medium_2.png` | 0.9209 | 0.9316 | +0.0107 |
| `pipeline_sample_large_1.png` | 0.9535 | 0.9582 | +0.0047 |
| `pipeline_sample_large_2.png` | 0.9520 | 0.9583 | +0.0063 |

통합 그림: [`results/archive_2026-08-20/pipeline_sample_comparison.png`](../results/archive_2026-08-20/pipeline_sample_comparison.png). 원본 MRI는 `*_original.png`.

---

## 3. 2026-08-20 실행의 단계별 학습

이 절부터 §7까지는 210명 풀·8방위 SDF PPO·단조 DSC 게이트 기록이다. 현재 파이프라인(경계 띠 PPO, 851명)이 아니다. 현재 설명은 [PIPELINE.md](PIPELINE.md)다.

| 단계 | 환자 | 학습 슬라이스 | 검증 슬라이스 |
|---|---:|---:|---:|
| Stage 1 Shape Classifier | 168 / 42 | 9,868 | 2,373 |
| Stage 2 Expert 3종 | 168 / 42 | 크기별 필터 (아래) | 크기별 필터 |
| Stage 3 PPO 3종 | 168 / 42 | 크기별 필터 + 50% 믹스업 | 크기별 필터 (hold-out) |
| Stage 4 평가 | 42 | — | 2,434 전수 |

GT 크기 구간 분포:

| 역할 | 슬라이스 | Small | Medium | Large |
|---|---:|---:|---:|---:|
| train | 9,868 | 3,561 | 4,184 | 2,123 |
| val | 2,373 | 942 | 847 | 584 |

### 3.1 Stage 1 — Shape Classifier

| 항목 | 값 |
|---|---|
| 모델 | ResNet18, `conv1` 2채널, `fc` 3클래스 |
| 초기화 | **ImageNet 사전학습** (기본). `--no_pretrained`로 해제 |
| 옵티마이저 | **AdamW** (lr 1e-3, weight decay 1e-4) + **CosineAnnealingLR** |
| 에폭 / 배치 | 15 / 64 |
| 분할 | **환자 단위** |
| Best Val Acc | **0.8512** |
| 저장 | `checkpoints/shape_classifier_best.pt` |

사전학습 ablation:

| 초기화 | Best Val Acc | 체크포인트 |
|---|---:|---|
| ImageNet 사전학습 | **0.8512** | `shape_classifier_best.pt` |
| 무작위 (scratch) | 0.8424 | `shape_classifier_scratch_backup.pt` |

이득이 +0.9%p에 그쳤고 Train Acc는 0.96–0.99까지 올라 과적합이 남습니다. 병목은 특징 품질이 아니라 **면적 300 / 700 경계 부근의 레이블 모호성**입니다. 몇 픽셀 차이로 클래스가 갈리므로 분류 문제로는 줄일 수 없는 잡음이 있습니다.

> **과거 수치 정정:** 이전 문서의 Val Acc **0.9383**은 **슬라이스 단위 80/20 분할**로 측정한 값입니다. 같은 환자의 인접 슬라이스가 train과 val에 함께 들어가 누출이 있었으므로 현재 환자 단위 수치와 비교할 수 없습니다.

### 3.2 Stage 2 — 크기별 Expert

공통: 20 epoch, batch 64, Adam lr 3e-4, CosineAnnealingLR, AMP(FP16), 환자 단위 분할.

| Expert | 손실 | 학습 범위 | 저장 경로 |
|---|---|---|---|
| Small CaraNet | BCEDice 0.5+0.5, 2.5D, `<50px` ×4 | Small만 | `caranet_best.pt` |
| Medium UNet++ | BCEDice 0.5+0.5 | Medium만 | `unetplusplus_best.pt` |
| Large SegResNet | MultiChannelBCEDice (ED+TC) | **전 구간** | `segresnet_best.pt` |

Expert 단독 성능은 이진화 임계값에 좌우되므로, val에서의 실측치는 §1.3의 **초기 DSC**(Small 0.8286 / Medium 0.9264 / Large 0.9546)를 봅니다. 이 값은 임계값 0.80/0.80/0.50과 TTA를 적용한 뒤의 수치입니다.

> **과거 수치 정정:** 이전 문서의 Expert Val DSC 0.8262 / 0.9197 / 0.9381은 크기별로 필터한 뒤 **슬라이스 단위 80/20**으로 나눈 학습 중 최고값입니다. 환자 단위 hold-out 수치와 직접 비교할 수 없습니다.

### 3.3 Stage 3 — 크기별 PPO

공통 하이퍼파라미터 (`configs/ppo_brats.yaml`):

| 항목 | 값 |
|---|---|
| max_train_patients | **210** |
| total_timesteps | 300,000 (실제 303,104) |
| n_envs / n_steps | 8 / 1,024 |
| batch_size / n_epochs | 256 / 10 |
| lr / clip / ent_coef | 1e-4 / 0.2 / 0.01 |
| gamma / GAE λ | 0.99 / 0.95 |
| net_arch | [512, 256, 128] |
| max_steps / target_dsc | 30 / 1.0 |
| step_penalty | 0.001 |
| 학습 데이터 | 실제 Expert 예측 50% + 형태학 노이즈 마스크 50% |
| 평가 환경 | **val 환자 hold-out**, 믹스업 없음 |

조기 종료 장치:

| 장치 | 설정 | 이유 |
|---|---|---|
| `stop_dsc_target` | **0 (비활성화)** | Medium/Large rough DSC가 이미 기본 임계값 0.92를 넘어 첫 평가(2,048 스텝)에서 즉시 중단됐다 |
| `stop_plateau` | **false (비활성화)** | 세 에이전트 모두 완주시킨다 |
| Milestone Snapshot | 25 / 50 / 75% | `checkpoints/snapshots/{small,medium,large}/`에 **클래스별 하위 폴더** 저장 |

과거에는 스냅샷이 클래스와 무관한 한 폴더에 저장돼 마지막 에이전트가 앞선 것을 덮어썼습니다. 지금은 `refinement_mode`별 하위 폴더로 분리됩니다.

| 에이전트 | 백본 | 클래스 필터 (train) | 믹스업 후 | val hold-out | 저장 |
|---|---|---:|---:|---:|---|
| Small | CaraNet | 3,561 | 7,122 | 942 | `ppo_small.zip` |
| Medium | UNet++ | 4,184 | 8,368 | 847 | `ppo_medium.zip` |
| Large | SegResNet | 2,123 | 4,246 | 584 | `ppo_large.zip` |

Large 에이전트 상세: 27m56s, eval 보상 120K **606.76** → 240K **639.43**, 에피소드 길이 30.0 고정. `target_dsc=1.0`이라 조기 종료가 거의 없습니다.

최종 모델은 eval 보상이 가장 좋았던 `checkpoints/best_{mode}/best_model.zip`을 복사한 것입니다.

---

## 4. 크기별 PPO가 다르게 동작하는 방식

환경: `src/envs/mask_refinement_env.py`

세 에이전트 공통:

1. 가장 큰 연결 요소의 중심으로 8개 방위 섹터를 나눔
2. 각 섹터의 shift를 마스크 SDF에 더함 (`SDF + shift ≥ 0`이 새 마스크)
3. Closing/Opening 후, 초기 rough mask ±8px 밖으로 나가지 못하게 클립

| 항목 | Small | Medium | Large |
|---|---|---|---|
| 관측 | 4ch 64×64 zoom (영상, 마스크, 확률, Sobel) | 3ch 128×128 | 3ch 128×128 |
| 행동 | 연속 `Box(-2, 2)` 8차원 | 이산 5단계×8 (`-1.0 / -0.4 / 0 / +0.4 / +1.0` px) | Medium과 동일 |
| DSC 목표 보너스 | ≥ 0.85 → +50 | ≥ 0.95 → +50 | ≥ 0.95 → +50 |
| HD95 보상 가중치 | 0.2 | 0.1 | **0.5** |
| 보상 스케일 | ×30 × size_scale | ×30 × size_scale | × size_scale |

관측은 영상·마스크·확률·에지로만 구성되며 **GT를 포함하지 않습니다.** GT는 보상 계산에만 쓰입니다.

보상 공통 안전장치:

- DSC / 경계 DSC(GT ±3px 밴드) / HD95가 나빠지면 개선분의 2배 감점
- 에피소드 시작 DSC보다 떨어지면 −5.0
- Keep이면서 DSC ≥ 0.85이면 +0.05
- DSC 목표 보너스는 조건을 만족하는 **매 스텝** 부여

평가 단계에서는 **세 클래스 모두** PPO `predict()`를 15스텝 돌립니다. Small은 마스크가 35px 미만이면 음수(수축) 행동을 0으로 자릅니다.

> **과거 서술 정정:** 이전 문서에는 "평가에서 Small만 PPO를 돌리고 Medium/Large는 TTA·임계값·형태학 경로를 쓴다"고 적혀 있었습니다. 현재 코드는 세 클래스 모두 PPO를 사용합니다.

---

## 5. 실험 이력 요약

초기에는 단일 U-Net + 전역 erode/dilate PPO였고, 이후 백본 교체 → 크기별 라우팅 → SDF/줌인 PPO → 환자 단위 분할·임계값 조정으로 바뀌었습니다.

### 5.1 단일 모델 + 전역 PPO (실험 1–5)

| 실험 | 핵심 변경 | RL DSC | 메모 |
|---|---|---:|---|
| 1 | Discrete(3), 1,251명, 200K | 0.6208 | Rough 0.8357보다 크게 하락 |
| 2 | Discrete(5) + HD95 페널티 | 0.6208 | 변화 없음. 샘플당 학습 횟수 부족 |
| 3 | 100명으로 축소, 보상 DSC only, 500K | 0.7327 | RL이 아직 Rough를 밑돎 |
| 4 | Step 1을 SegResNet으로 | Rough 0.8491 | RL은 U-Net 기준으로 학습되어 OOD |
| 5 | U-Net / UNet++ / AttUNet / UNet 3+ 비교 | UNet 3+ RL **0.8327** | 단일 백본 벤치마크 SOTA |

실험 5 단일 백본 비교 (50명, 2,902 슬라이스):

| 백본 | Rough DSC | Morpho DSC | RL DSC | RL HD95 |
|---|---:|---:|---:|---:|
| UNet 3+ (32ch+BN) | 0.8279 | 0.8284 | **0.8327** | 3.46 px |
| Attention U-Net | 0.8246 | 0.8275 | 0.8297 | **2.53 px** |
| UNet++ | 0.8264 | 0.8248 | 0.8292 | 2.63 px |
| SegResNet (당시 단일) | 0.8111 | 0.8126 | 0.8210 | 2.84 px |
| U-Net | 0.8090 | 0.8113 | 0.8134 | 2.03 px |

당시 교훈: 백본 Rough 품질이 상한을 정하고, 전역 PPO는 데이터를 줄여 샘플당 반복을 늘려야 학습이 됩니다.

### 5.2 3-Stage 라우팅 (실험 6–14)

| 실험 | 날짜 | 설정 요약 | 평균 초기 → 최종 DSC | 평균 HD95 |
|---|---|---|---|---|
| 6 | — | Small 연속 PPO, 4채널, 단기 10K | 0.7860 → 0.7938 | — |
| 7 | 08-16 | 1,251명, AttUNet/UNet++/SegResNet | 0.8129 → 0.8198 | 3.81 (단위 혼재) |
| 8 | 08-17 | `t1ce+flair`, Zoom-Refiner | 0.8480 → 0.8486 | 4.0440 px |
| 9 | 08-18 | SDF, 컴포넌트 독립 보정, 단조 가드 | 0.8999 → 0.9000 | 1.3340 px |
| 10 | 08-18 | Dual Gate, CaraNet Small | 0.9030 → 0.9031 | 1.1906 → 1.1874 px |
| 11 | 08-19 | 클래스당 100장 균형 평가 | 0.9031 → 0.9113 | 1.1662 → 1.0500 px |
| 12 | 08-19 | 210명 **전수** 12,241장, Oracle 라우팅, Dual Gate | 0.8775 → 0.8880 | 1.8416 → 1.6843 px |
| 13 | 08-19 | val 42명 hold-out 2,373장, 분류기 라우팅, 임계값 0.8/0.8/0.6 | 0.8672 → 0.8752 | 2.3365 → 2.2897 px |
| **14 (현재 정량)** | **08-20** | **2.5D CaraNet, Large ED/TC, 임계값 0.8/0.8/0.5, val 2,434장** | **0.8948 → 0.9031** | **1.7336 → 1.6060 px** |
| 15 | 08-20 | 같은 슬라이스 3×6 그리드. 행 이름 TRIO, 제목 영어 | 그림만 | — |

**실험 12와 13은 직접 비교할 수 없습니다.** 프로토콜이 세 군데에서 바뀌었습니다.

| 항목 | 실험 12 | 실험 13 |
|---|---|---|
| 평가 집합 | 210명 **전수** (학습 데이터 포함) | **val 42명만** (hold-out) |
| 라우팅 | **Oracle** (GT 면적) | **분류기 예측** |
| 게이트 | Dual (DSC + HD95) + Medium/Large GT 형태학 | Monotonic **DSC만** |

실험 13의 점수가 낮은 것은 성능 퇴행이 아니라 **누출과 Oracle 이점을 제거한 결과**입니다. 실험 7–8·12는 슬라이스 전수 평균, 실험 9–11은 클래스당 100장 균형 평균입니다.

실험 13과 14는 같은 val 42명·분류기 라우팅이지만 슬라이스 수(2,373 vs 2,434)와 Expert 구조가 달라 1:1은 아닙니다. 14에서 Stage 2가 0.8672 → 0.8948로 오른 것은 2.5D CaraNet, Large ED/TC, Large 임계값 0.50이 겹친 결과입니다.

### 5.3 실험 13–14에서 고친 것

| 항목 | 이전 | 실험 13 | 실험 14 (현재) |
|---|---|---|---|
| 분할 | 슬라이스 단위 80/20 (같은 환자가 양쪽에) | **환자 단위** 168 / 42 | 동일 |
| 데이터 로딩 | `max_patients`로 "정렬 순 앞 N명" 로드 | 분할을 먼저 확정한 뒤 해당 환자 ID만 로드 | 동일 |
| 라우팅 | Oracle (GT 면적) | 분류기 예측 | 동일 |
| 이진화 임계값 | 고정 0.50 | 0.80 / 0.80 / 0.60 | **0.80 / 0.80 / 0.50** |
| CC filter | — | 15 / 15 / 25 | **0 / 15 / 25** (Small 끔) |
| Small Expert | FocalTversky, 2D | CaraNet 2D | **2.5D + `<50px` ×4 + zoom-crop, BCEDice** |
| Large Expert | Large만 학습, WT 1채널 | 동일 | **전 구간 학습 + ED/TC 2채널** |
| 미세 파편 처리 | 고정 후보 + 휴리스틱 | 임계값 사다리 | 동일 |
| 게이트 | Dual (DSC + HD95) | Monotonic DSC만 | 동일 |
| Medium/Large 평가 | TTA + GT 형태학 | PPO `predict()` 15스텝 | 동일 |
| 지표 | DSC, HD95 | + Precision, Recall | 동일 |
| PPO 스냅샷 | 한 폴더 공유 → 덮어씀 | `snapshots/{mode}/` 분리 | 동일 |
| PPO 조기 종료 | `stop_dsc_target 0.92`로 즉시 중단 | 비활성화 | 동일 |
| 재현성 | 시드 미고정 | seed 42 + 결정적 모드 기본 ON | 동일 |
| 분류기 | scratch ResNet18 + Adam | ImageNet 사전학습 + AdamW + Cosine | 동일 |
| 시각화 표본 | ΔDSC 최대 | — | **클래스 평균 Final DSC + Final > Initial** |
| 방법 비교 그리드 | 없음 | — | **같은 슬라이스 3×6, 제목 영어** |

---

## 6. 해석

1. **Stage 3 이득은 +0.83%p DSC, HD95 −0.13 px.** 게이트가 858장(35.3%)을 Stage 2로 되돌리므로 최종 DSC는 상한입니다.
2. **게이트 없는 Stage 2 DSC는 0.8948이다.**
3. **Large 최종 DSC는 0.9582이다.** ED/TC 헤드와 임계값 0.50이 과소분할(이전 P 0.97 / R 0.93)을 균형으로 돌렸습니다.
4. **Small은 2.5D 이후 0.83까지 올랐지만 과소분할이 남습니다** (P 0.92 / R 0.80).
5. **분류기가 라우팅 상한을 제한한다.** Val Acc 0.8512이므로 슬라이스 약 15%가 엉뚱한 Expert로 갑니다.
6. **시각화 표본은 클래스 평균 Final DSC에 가깝고 실제로 개선된 슬라이스**입니다. 예전처럼 ΔDSC 최대 이상치를 고르지 않습니다.
7. **비교 그림의 방법 이름은 TRIO**입니다. Stage 3 + 단조 DSC 게이트와 같습니다.
8. **Medium UNet++ 학습 스크립트의 val 로더는 train 증강을 그대로 씁니다** (`train_unetplusplus.py`의 `collate_fn=augment_batch`). 학습 중 val DSC만 흔들릴 수 있고, Stage 4 평가 로더는 증강이 없습니다. Medium을 다시 학습하지 않는 한 보고된 0.9264 → 0.9314는 그대로입니다.

---

## 7. 재현

```bash
python run_pipeline.py batch_size=64
```

기본 환자 풀은 210명(train 168 / val 42)이고, seed 42 · 결정적 모드가 기본입니다.

```bash
# 평가만
python scripts/eval/evaluate_pipeline.py --split_role val

# Stage 2 단독 (베이스라인과 비교용)
python scripts/eval/evaluate_pipeline.py --split_role val --skip_ppo

# 임계값 스윕
python scripts/eval/sweep_threshold.py --split_role val --plot results/archive_2026-08-20/threshold_sweep.png

# 베이스라인
python baselines/run_comparison.py --epochs 20 --batch_size 32 --sweep

# 같은 슬라이스 그리드
python scripts/eval/plot_three_method_grid.py --indices 113,2286,1121,2295,95,1480 --out results/archive_2026-08-20/method_comparison_3x6.png
```

주요 산출물:

| 경로 | 내용 |
|---|---|
| `checkpoints/patient_split.json` | 환자 분할 (train 168 / val 42, seed 42) |
| `checkpoints/shape_classifier_best.pt` | Stage 1 (Val Acc 0.8512) |
| `checkpoints/shape_classifier_scratch_backup.pt` | Stage 1 사전학습 없음 ablation (0.8424) |
| `checkpoints/caranet_best.pt` | Small Expert |
| `checkpoints/unetplusplus_best.pt` | Medium Expert |
| `checkpoints/segresnet_best.pt` | Large Expert |
| `checkpoints/ppo_{small,medium,large}.zip` | PPO 303,104 steps |
| `checkpoints/snapshots/{mode}/` | 25 / 50 / 75% 마일스톤 |
| `results/archive_2026-08-20/pipeline_sample_comparison.png` | 6장 통합 시각화 |
| `results/archive_2026-08-20/pipeline_sample_{small,medium,large}_{1,2}.png` | 클래스 평균에 가까운 개선 샘플 |
| `results/archive_2026-08-20/pipeline_sample_*_original.png` | 해당 슬라이스 원본 MRI |
| `results/archive_2026-08-20/method_comparison_3x6.png` | 같은 슬라이스 3×6 그리드 |
| `results/archive_2026-08-20/threshold_sweep.png` | 임계값 스윕 곡선 |
