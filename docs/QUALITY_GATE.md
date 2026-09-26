# 실측 평가와 GT-free 품질 gate

8월 20일 학습 설정과 초기 임계값은 유지합니다. 새로운 Stage 3 성능은 재평가 대상입니다. 과거 GT gate 수치와 새 결과를 섞지 않습니다.

## 평가 모드

| `--eval_mode` | 최종 출력 |
|---|---|
| `stage2` | 초기 분할, flip/PPO 없음 (`--skip_ppo`와 동일) |
| `augmentation` | flip 평균 확률 + 컴포넌트 재이진화 + closing, PPO 없음 |
| `ppo_raw` | 같은 augmentation에서 PPO + closing, gate 없음 |
| `heuristic` | PPO 결과에 컴포넌트·슬라이스 면적/에지 가드 (기본) |
| `quality` | gate 없는 PPO 후보와 초기 마스크를 학습된 품질 gate + 면적 가드로 선택 |
| `oracle` | gate 없는 PPO 후보와 초기 마스크를 GT DSC로 선택 (연구용 상한) |
| `compare` | 같은 초기 예측과 PPO 후보를 공유해 위 결과를 기록. quality는 모델 제공 시, oracle은 `--allow_oracle_gate` 지정 시 포함 |

`--energy_gate`를 켜면 TTA와 원본 확률의 불일치, 경계 확률 엔트로피, MRI 에지 불일치를 합친 GT-free `boundary_energy`를 계산합니다. energy가 임계값보다 낮은 컴포넌트는 PPO를 실행하지 않고 TTA 결과를 사용합니다. 이 버전은 학습된 Energy Model이 아니라 결정론적 1차 선택기이므로, `--energy_threshold`를 고정한 뒤 독립 검증에서 선택률과 DSC 변화를 보고해야 합니다.

`augmentation`과 `ppo_raw`의 차이가 PPO 효과입니다. `quality`는 보정 후 **수락/거절** 모듈입니다. 학습된 Energy Model을 함께 사용하면 PPO 전후 energy가 증가한 후보를 거절하고, `--quality_gate`를 함께 주면 TEGDA-style 품질 예측도 통과한 후보만 수락합니다. 판정 입력은 영상, 모델 확률, flip 평균 확률, 보정 전후 마스크입니다. 실제 GT나 GT 기반 지표는 추론 특징에 포함하지 않습니다.

체크포인트가 없으면 평가를 중단합니다. 무작위 모델/부분 로드 모델로 결과를 만들지 않습니다. 필수 파일은 `checkpoints/shape_classifier_best.pt`, `caranet_best.pt`, `unetplusplus_best.pt`, `segresnet_best.pt`, PPO 모드에서는 `ppo_small.zip`, `ppo_medium.zip`, `ppo_large.zip`입니다.

## 학습된 Energy Model

고정 수식 energy 대신 train 환자의 GT perturbation으로 학습한 모델을 사용할 수 있습니다. 학습은 train 환자만 사용하고, val/test에서는 `--energy_model`로 가중치만 읽습니다.

```bash
python scripts/train/train_energy_model.py \
  --train_root src/data/archive \
  --patient_split checkpoints/patient_split.json \
  --output checkpoints/energy_model.pt

python run_pipeline.py --skip_classifier --skip_experts --skip_agents \
  --refinement_profile ppo_v2 --agent_dir checkpoints/ppo_v2 \
  --eval_mode compare --energy_gate \
  --energy_model checkpoints/energy_model.pt \
  --energy_threshold 0.5 --slice_selection all --no_plots \
  --output_dir results/learned_energy_compare
```

`--energy_model`을 주지 않으면 기존의 결정론적 energy 수식을 사용합니다. 학습된 모델의 임계값 범위는 0–1이며 calibration 환자에서 고정해야 합니다. 이 모델은 현재 PPO 가중치나 Expert 가중치를 test-time에 업데이트하지 않고 PPO 실행 영역과 결과 수락 여부를 선택합니다.

## 실행 순서

저장소 루트에서 실행합니다. `requirements.txt`의 의존성이 필요합니다. 각 출력 디렉터리는 새 경로여야 합니다.

```bash
# 1. 기존 val은 개발용 비교. 전체 슬라이스로 종양 없는 슬라이스 오탐까지 측정
python run_pipeline.py --skip_classifier --skip_experts --skip_agents \
  --eval_mode compare --slice_selection all --no_plots \
  --output_dir results/compare_val

# 2. train 환자에서 gate 학습용 실제 PPO 후보와 DSC 차이 수집
python scripts/eval/evaluate_pipeline.py --split_role train \
  --eval_mode compare --slice_selection all --no_plots \
  --output_dir results/gate_train

# 3. train 기록 내부에서 환자별 fit/calibration 분리 (기본 75/25)
python scripts/train/train_quality_gate.py --records results/gate_train \
  --output checkpoints/quality_gate.json

# 4. 같은 checkpoint/설정으로 개발용 val 비교
python run_pipeline.py --skip_classifier --skip_experts --skip_agents \
  --eval_mode compare --quality_gate checkpoints/quality_gate.json \
  --slice_selection all --no_plots --output_dir results/quality_val

# 5. 실제 환자 단위 통계
python scripts/eval/patient_level_stats.py --records results/quality_val \
  --output results/quality_val/patient_stats.json
```

독립 test 평가는 `evaluate_pipeline.py --split_role test --patient_split <분할 JSON>`로 실행합니다. JSON의 `test` 목록은 비어 있으면 안 되고 `train`/`val`과 겹치면 중단합니다. 이 명령은 test 분할을 자동 생성하지 않습니다. test 환자는 backbone/PPO 학습에도 쓰지 않은 환자여야 합니다. 기존에 반복 튜닝한 val 42명을 독립 test라고 부르지 않습니다.

품질 모델은 fit/calibration 환자와 평가 환자의 중복, checkpoint SHA-256 차이, 주요 후보 생성 설정 차이를 검사합니다. 별도 환자 데이터로 일반화 성능을 검증해야 합니다. 현재 간단한 학습 경로의 train 예측은 backbone/PPO에 대해 in-sample일 수 있어 성능이 낙관적일 수 있습니다. 논문용 확정 실험에서는 별도 gate 개발 환자 또는 모델 학습부터 분리한 nested/out-of-fold 설계가 필요합니다.

## 품질 모델

12개 특징: 전후 면적, 로그 면적비, 변경 픽셀 비율, 전후 평균 확률, 경계 엔트로피, flip 불일치, 에지 정합입니다. 표준화 통계와 ridge 회귀 계수는 fit 환자에서만 학습하며, 환자별 총 가중치를 같게 둡니다. 목표값은 실제 `DSC(PPO 후보) − DSC(Stage 2)`입니다.

임계값은 별도 calibration 환자에서 선택합니다. 기본적으로 평균 DSC가 악화된 calibration 환자 비율을 10% 이하로 제한하면서 환자 평균 개선량을 최대화합니다. 개선 후보가 없으면 전부 거절하도록 설정합니다. 이는 경험적 calibration 기준이며 test DSC의 개선이나 하락 방지를 보장하지 않습니다. 상관계수만으로 gate 효과를 주장하지 않습니다.

## 결과 파일과 통계 정의

- `metadata.json`: 설정, 환자 목록, checkpoint 해시, 사용한 gate, 실행 완료 여부.
- `slices.jsonl`: 환자 ID, 원래 z 인덱스, 방법별 DSC/HD95/Precision/Recall, 빈 GT 오탐, PPO 호출 수, 공유 추론 시간, 품질 특징·예측·수락 여부.
- `--save_masks`: 슬라이스별 NPZ에 평가 대상 예측과 GT 저장. 디스크 사용량이 증가합니다.
- 시각화는 실행 디렉터리의 `plots/`에 저장합니다. GT로 대표 사례를 고르는 그림은 사후 분석이며 추론 정책의 일부가 아닙니다.

DSC는 슬라이스 지표의 **환자별 평균을 다시 동일 가중 평균**합니다. 이는 3D 볼륨 DSC가 아닙니다. 초기 품질별 poor/medium/good 구간도 환자 평균 DSC 기준 사후 분석입니다.

새 HD95는 마스크 **경계** 사이 양방향 거리의 95백분위수이며 128×128 평가 좌표의 픽셀 단위입니다. 과거 내부 전체 픽셀 기반 HD95와 직접 비교하지 않습니다. 두 마스크가 모두 비면 0, 한쪽만 비면 `null`로 기록하고 별도 실패 건수를 보고합니다. 정의된 거리만의 평균을 단독으로 성능 근거로 사용하면 안 됩니다. mm 거리와 3D 지표는 원본 geometry 복원 후 별도로 구현해야 합니다.

통계 스크립트는 환자별 paired DSC 변화, 환자 bootstrap 95% CI, 악화 비율, 0.02 초과 하락 비율, 최악 환자 변화, 빈 GT 슬라이스 오탐률을 계산합니다. 불완전 실행/중복 슬라이스/방법별 샘플 불일치는 거절합니다. 합성 점수 및 고정된 유의성 결론은 제거했습니다.

`compare`의 시간은 공유 후보 생성 비용이므로 방법별 독립 지연시간이 아닙니다. 각 모드를 별도 실행해 비교해야 하며, 전처리·모델 로드·지표 계산·파일 저장 시간은 슬라이스 추론 시간에서 제외됩니다.

전체 슬라이스 선택은 MRI의 z 범위를 사용합니다. GT 파일은 사후 지표 계산에 여전히 필요하지만 슬라이스 선택에는 쓰지 않습니다. `--slice_selection tumor`는 과거 비교를 위해 남겨 둔 GT 선택 종양 슬라이스 평가입니다.

## 검증

```bash
python -m unittest discover -s tests -v
```

테스트의 합성 마스크와 대체 모델은 기능 검증 전용이며 실측 성능 자료로 저장소에 포함하지 않습니다. 학습된 checkpoint가 없는 환경에서는 실제 성능 재평가와 실제 품질 모델 학습을 완료할 수 없습니다.


## PPO v3: 전체 슬라이스 보정 실험

`configs/ppo_brats_v3.yaml`은 성능 향상이 검증된 모델이 아니라 새 학습 프로필입니다.
`run_pipeline.py`는 이제 알 수 없는 CLI 옵션을 무시하지 않고 오류로 알립니다.
개발 중 test 반복 조회를 피하기 위해 실행기의 기본 평가 분할은 `val`입니다.
최종 평가를 할 때만 `--split_role test`를 명시하세요.

- 관측값: 모든 MRI 모달리티(기본 T1ce, FLAIR), 현재 마스크, TTA 확률,
  누적 SDF, 남은 스텝 비율, 최소/최대 허용 마스크. 모두 전체 슬라이스 좌표입니다.
  Small의 이전 64×64 crop도 사용하지 않습니다.
- 학습/추론 단위: Stage 1이 예측한 클래스에 따라 전체 슬라이스를 한 에이전트가 보정합니다.
  연결 성분별 GT를 잘라서 학습하거나 보정 결과들을 합치지 않습니다.
- 보상: 전체 슬라이스 DSC 변화 × 100 − 행동 비용.
  마지막에는 초기 마스크 대비 DSC 변화 × 100을 추가합니다.
  에피소드는 15 step 고정이고, 학습 GT는 보상 계산에만 쓰입니다.
- Small: 35픽셀 미만 수축 금지를 해제하여 작은 오탐을 지울 수 있습니다.
  5픽셀 미만 성분도 전체 마스크에 포함되어 함께 보정됩니다.
- 마스크 후처리: v3는 PPO 이후 closing을 하지 않습니다. KEEP이면 초기 마스크를 유지합니다.
- 데이터: v3 PPO 학습은 모든 z 슬라이스를 사용하여 빈 GT/정상 슬라이스도 포함합니다.
- 체크포인트 선택: 해당 에이전트에 라우팅된 모든 validation 슬라이스를 평가하고,
  환자별 평균 개선량을 환자 간 동일 가중 평균합니다. `selection.json`에
  `patient_mean_delta_dsc`, `improves_baseline`을 저장합니다.
  음수 중 최선인 모델도 진단을 위해 저장하므로 파일 존재 자체가 개선을 뜻하지 않습니다.

`ppo_v3`는 기존 legacy/v2 체크포인트와 관측 공간이 다르므로 새로 학습해야 합니다.
기본 저장 경로는 `checkpoints/ppo_v3`입니다. 기존 Quality/Energy/면적·에지 gate는
새 프로필용으로 보정되지 않았기 때문에 v3에서는 `quality`, `heuristic`,
`--auto_tegda`, confidence/energy bypass를 지원하지 않습니다.
먼저 `ppo_raw` 또는 `compare`로 PPO 자체를 측정하세요.
`compare`는 Stage 2와 PPO raw를 기록하며, `augmentation` 마스크는 Stage 2와 동일합니다.
v3의 TTA는 관측 확률에만 쓰고 초기 이진 마스크를 바꾸지 않기 때문입니다.
이는 이전 프로필의 TTA 재이진화 + closing 방식과 다릅니다.
`ppo_slice_calls`는 슬라이스 호출 수이고 `ppo_component_calls`는 v3에서 0입니다.

Stage 1/2 체크포인트가 현재 400명 분할의 train 환자로 학습되었음을 확인한 경우:

```bash
# 세 에이전트를 새로 학습하고 validation에서 비교 (기존 Stage 1/2는 사용)
python run_pipeline.py --config configs/ppo_brats_v3.yaml \
  --max_train_patients 400 --skip_classifier --skip_experts \
  --eval_mode compare --split_role val --no_plots \
  --output_dir results/ppo_v3_val

# 환자 단위 변화와 bootstrap CI
python scripts/eval/patient_level_stats.py --records results/ppo_v3_val \
  --output results/ppo_v3_val/patient_stats.json
```

동일 분할의 Stage 1/2 체크포인트가 없다면 먼저 해당 모델을 재학습해야 합니다.
검증 결과로 설정을 확정한 뒤 `--skip_agents --split_role test`로 별도 test 평가를 합니다.
평가 출력 디렉터리는 매번 새 경로여야 합니다.

남은 한계: 이 단계는 OOF Stage 2 예측 생성까지 구현하지 않습니다. 기존 학습 환자의
in-sample 초기 예측을 사용하는 문제는 남아 있습니다. 또한 행동은 여전히 8방향
수축/팽창이며 초기 마스크의 ±8픽셀 범위에 제한됩니다. 초기 예측이 완전히 빈 경우
새 병변을 생성할 수 없습니다. 환자 수 증가나 이 프로필 자체가 성능 개선을 보장하지 않습니다.
테스트의 합성 예시는 기능 검증이며 논문용 성능 자료가 아닙니다.
