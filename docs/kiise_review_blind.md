# 크기별 전문가와 경계 띠 정책 보정을 이용한 뇌종양 MRI 분할

Size-Specific Experts and Boundary-Band Policy Refinement for Brain Tumor MRI

## 요 약

뇌종양 MRI의 whole tumor는 단면 크기와 경계 양상이 달라 하나의 분할 모델로 맞추기 어렵다. 본 연구는 T1ce와 FLAIR를 128×128로 줄인 축상 단면에서 종양 크기를 분류하고, 크기별 전문가로 초기 마스크를 만든 뒤, 하나의 PPO가 정해진 경계 띠 안의 픽셀만 켜거나 끈다. 300픽셀 미만은 2.5D CaraNet, 300 이상 700 미만은 UNet++, 700 이상은 부종과 종양핵을 나눠 예측하는 SegResNet이다. BraTS 2021에서 개발에 쓴 400명을 뺀 851명, 종양 슬라이스 50,010장을 2차원 슬라이스 DSC의 환자 평균으로 평가했다. 분류기가 전문가를 고르고 임계값을 0.45/0.80/0.20으로 둔 초기 분할 대비 DSC는 0.8419에서 0.8595로 올랐다(환자 짝 차이 +0.0176, 95% CI 0.0166–0.0185).

![그림 1 파이프라인. 크기 분류, 크기별 전문가, 경계 띠 PPO, 확정 평가.](../results/pipeline_overview.jpg)

그림 1 파이프라인. 크기 분류, 크기별 전문가, 경계 띠 PPO, 확정 평가.

## 1. 서 론

뇌종양 영역 분할은 종양 부담을 정량화하고 치료 범위를 정하는 데 쓰인다. MRI에서 종양의 크기와 부종 경계는 슬라이스마다 달라, 단일 백본이 아주 작은 단면과 큰 단면을 함께 맞추기 어렵다. U-Net 계열은 의료영상 분할의 기본 구조이지만[1], 크기별로 오차 형태가 다르면 초기 마스크의 경계에 과소분할이 남는다.

본 연구는 초기 분할과 경계 보정을 나눈다. 분류기가 슬라이스 크기를 고르면 해당 전문가가 확률 맵을 만들고, 하나의 PPO가 애매한 확률 구간과 마스크 경계 주변만 수정한다. 이 세 단계를 TRIO라 부른다. TRIO는 크기 분류, 크기별 전문가 분할, 경계 띠 PPO를 가리킨다. 기여는 크기별 라우팅과, 띠 안의 픽셀을 켜고 끄는 PPO 하나이다. 중형 슬라이스만으로 학습한 띠 보정 네트워크는 이 PPO의 초기 가중치이며, 그 지도학습 보정 자체는 기여가 아니다.

## 2. 방 법

그림 1은 전체 파이프라인이다. 입력은 BraTS 2021의 T1ce와 FLAIR이다[6]. 슬라이스는 128×128로 맞추고, 뇌 영역 안에서 z-score 정규화한 뒤 1–99 퍼센타일로 잘라 0–1로 맞춘다. 종양 픽셀 비율이 0.2% 이상인 단면만 학습과 평가에 넣는다. Whole tumor는 분할 라벨이 0이 아닌 픽셀이다. 크기는 정답 면적으로 나눈다. 300픽셀 미만은 소형, 300 이상 700 미만은 중형, 700 이상은 대형이다.

분류기는 ImageNet으로 초기화한 ResNet-18이다. 추론에서 분류기가 전문가를 고른다. 소형은 여섯 채널 2.5D CaraNet[3]이고, 중형은 두 채널 UNet++[2]이며, 대형은 SegResNet[4]이 부종과 종양핵을 따로 예측한 뒤 1−(1−p_ED)(1−p_TC)로 합친다. 확률은 원본과 좌우·상하 반전의 평균이다. 이진화 임계값은 0.45, 0.80, 0.20이고 연결요소 최소 크기는 0, 15, 25픽셀이다.

보정 정책은 하나다. 폭 32의 소형 U-Net이 T1ce, FLAIR, 현재 마스크, 전문가 확률을 보고 픽셀마다 끄기, 유지, 켜기 중 하나를 고른다. 수정 범위는 전문가 확률 0.35–0.65인 픽셀이거나 마스크 경계 ±2픽셀이다. 다섯 스텝 동안 확률 맵은 고정하고 마스크만 넘긴다. FLAIR 상대 밝기 제약은 마지막 스텝에만 둔다. 정답은 학습 보상에만 쓰고 추론 관측에는 넣지 않는다. 뒤집은 픽셀이 정답과 같으면 +1, 다르면 −1이며, HD95가 줄면 그 감소에 0.25를 곱해 더한다. 학습은 PPO[5]이고 클립은 0.1이다. BraTS의 다중 기관 영상과 전문가 라벨은 Menze 등[7]과 Bakas 등[8]이 정리했고, nnU-Net[9]은 그 계열에서 전처리와 모델 선택을 데이터에 맞추는 기준이다. 본 연구는 그 3차원 프로토콜 대신 2차원 단면의 경계 띠만 수정한다.

## 3. 실 험

환자 단위로 나눈다. 1,251명 중 시드 42로 고른 개발 400명은 학습 280, 검증 60, 방법 선택 60이다. 나머지 851명은 가중치 학습과 방법 선택에 쓰지 않았다. 이진화 임계값 0.45/0.80/0.20은 이 분할의 검증 60명에서 골랐다. 연결요소 최소 크기는 이전 프로토콜의 0, 15, 25픽셀이고, 그 기준을 정한 210명 풀 가운데 148명이 이 851명에 들어 있다. 학습과 에폭 선택은 정답 면적으로 전문가를 고르고, 표 1의 평가는 분류기 라우팅이다. 평가 슬라이스의 16.5%는 정답 크기와 다른 전문가의 마스크를 보정한다.

주 지표는 2차원 종양 슬라이스 DSC의 환자 평균이다. 환자 안과 환자 사이 모두 같은 가중을 쓴다. 표 1은 임계값 0.45/0.80/0.20의 초기 분할과 경계 띠 PPO를 같은 마스크에서 짝비교한 결과이다. 환자 열은 그 짝 차이의 표본 수이다. 슬라이스 열은 그 정답 면적의 종양 슬라이스 수이고, small·medium·large를 더하면 50,010이다. 같은 환자가 여러 행에 들어가므로 환자 수를 더하면 851을 넘는다.

표 1 미사용 851명. 분류기 라우팅, 임계값 0.45/0.80/0.20. 슬라이스 DSC의 환자 평균.

| 구간 | 환자 | 슬라이스 | Stage 2 | PPO | 짝 차이 (95% CI) |
|---|---:|---:|---:|---:|---|
| 전체 | 851 | 50,010 | 0.8419 | 0.8595 | +0.0176 (0.0166–0.0185) |
| small | 851 | 19,077 | 0.7693 | 0.7922 | +0.0229 (0.0215–0.0243) |
| medium | 757 | 19,907 | 0.8780 | 0.8932 | +0.0152 (0.0144–0.0160) |
| large | 408 | 11,026 | 0.9030 | 0.9156 | +0.0125 (0.0112–0.0138) |

전체 환자 짝 차이 +0.0176의 95% 신뢰구간은 0을 포함하지 않는다. 차이는 small에서 +0.0229로 가장 크고, large에서 +0.0125로 가장 작다. small 행의 851명은 소형 단면이 한 장이라도 있는 환자 전부다. large의 초기 DSC가 이미 0.9030이라 띠 안에서 더 올릴 여지가 작다. 같은 평가에서 빈 마스크를 뺀 HD95 평균은 4.747픽셀에서 4.547픽셀이다. 빠지는 슬라이스가 서로 달라 이 HD95는 짝비교로 쓰지 않는다.

![그림 2 개발 집합에 없는 단면. 위는 정답, 가운데는 PPO 이전, 아래는 PPO 이후.](../results/band_ppo_delta_matched/delta_matched_comparison.png)

그림 2는 이전 잠금 가중치로 그린, 개발 집합에 없는 단면이다. 위는 정답, 가운데는 PPO 이전, 아래는 PPO 이후. 칸의 DSC와 HD95는 그 슬라이스만의 값이며 표 1의 환자 평균이 아니다. 정량 주장은 표 1이다.

## 4. 결 론

크기별 전문가와 하나의 경계 띠 PPO를 미사용 851명에 적용하면, 임계값 0.45/0.80/0.20의 초기 분할보다 2차원 슬라이스 DSC의 환자 평균이 0.8419에서 0.8595로 오른다(환자 짝 차이 +0.0176, 95% CI 0.0166–0.0185). 수정 범위는 경계 ±2픽셀과 확률 0.35–0.65인 픽셀이다. 지표는 2차원 슬라이스 DSC의 환자 평균이고, 빈 슬라이스는 평가에 넣지 않았다. 학습은 시드 하나의 실행이다. HD95는 4.547픽셀이며, 한쪽 마스크가 비면 그 슬라이스를 평균에서 뺀다.

## 참 고 문 헌

[1] O. Ronneberger, P. Fischer, T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, pp. 234–241, 2015.

[2] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, J. Liang, “UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation,” IEEE Trans. Med. Imaging, vol. 39, no. 6, pp. 1856–1867, 2020.

[3] A. Lou, S. Guan, M. Loew, “CaraNet: Context Axial Reverse Attention Network for Segmentation of Small Medical Objects,” J. Med. Imaging, vol. 10, no. 1, 014005, 2023.

[4] A. Myronenko, “3D MRI Brain Tumor Segmentation Using Autoencoder Regularization,” BrainLes, MICCAI, pp. 311–320, 2018.

[5] J. Schulman et al., “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017.

[6] U. Baid et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification,” arXiv:2107.02314, 2021.

[7] B. H. Menze et al., “The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS),” IEEE Trans. Med. Imaging, vol. 34, no. 10, pp. 1993–2024, 2015.

[8] S. Bakas et al., “Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features,” Scientific Data, vol. 4, 170117, 2017.

[9] F. Isensee et al., “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203–211, 2021.
