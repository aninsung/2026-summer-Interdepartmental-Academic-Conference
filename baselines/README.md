# BraTS 2021 상위 입상 방법 베이스라인 (2D 각색)

적응형 파이프라인(TRIO)의 성능을 BraTS 2021 대회 상위 입상 알고리즘과 동일 조건에서 비교하기 위한 베이스라인 폴더입니다.

---

## 구현한 두 베이스라인

| 폴더 내 이름 | 원 논문 | 대회 성적 | 2D 각색 설명 |
|---|---|---|---|
| `kaist` | Luu & Park, *Extending nn-UNet for Brain Tumor Segmentation* | **최종 테스트 1위** | Asymmetric Encoder + GroupNorm(32) + Axial Attention (2D slice-level) |
| `nvauto` | Myronenko et al., *Redundancy Reduction in Semantic Segmentation* | **최종 순위 2위** | SegResNet + InstanceNorm + Barlow Twins representation learning |

---

## 공정 비교를 위해 고정한 설정

- **환자 단위 분할**: `checkpoints/patient_split.json` 공유 (Train 1,001명 / Val 250명)
- **입력 채널**: `t1ce + flair` 2D (128×128)
- **출력 규약**: BraTS Multi-Region 세부 영역 (**ET, TC, WT**) 3채널 직접 예측
- **평가 지표**: `src/utils/metrics.py` (DSC, HD95, Precision, Recall)
