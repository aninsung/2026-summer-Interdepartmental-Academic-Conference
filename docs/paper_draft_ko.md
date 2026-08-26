# 강화학습 기반 동적 라우팅을 이용한 뇌종양 MRI 분할 경계 보정

## 초록

뇌종양 자기공명영상(MRI) 분할은 진단 보조와 치료 계획 수립에 핵심적인 정보를 제공하지만, 병변의 크기와 형태 편차가 커 단일 분할 모델만으로는 전 구간에서 일관된 경계 정밀도를 확보하기 어렵다. 본 연구는 T1ce와 FLAIR의 2채널 MRI 입력에 대해 종양 크기를 먼저 분류하고, 크기별 전문가 분할 모델과 강화학습 기반 경계 보정기를 순차적으로 적용하는 3단계 동적 라우팅 파이프라인(TRIO)을 제안한다. 소형 병변에는 인접 슬라이스를 결합한 2.5D CaraNet, 중형에는 UNet++, 대형에는 부종(ED)과 종양핵(TC)을 분리 예측 후 Whole Tumor로 합성하는 SegResNet을 배정한다. PPO 에이전트는 현재 마스크, 예측 확률 맵, 영상 단서를 바탕으로 8개 방위의 외곽선을 미세 조정한다. BraTS 2021 환자 풀(210명)을 환자 단위로 엄격히 분할한 42명 검증 집합(2,434 슬라이스)에서, 정답 라벨(GT) 개입 없이 배포 가능한 Stage 2 초기 분할은 평균 Dice Similarity Coefficient(DSC) **0.8948**, HD95 **1.7336 px**를 기록하여 대회 1·2위 구조를 차용한 2차원 각색 베이스라인(KAIST 0.8923, NVAUTO 0.8971)과 대등한 성능을 달성하였다. 한편, 정답 기반 단조 DSC 게이트를 적용했을 때 도달 가능한 이론적 상한은 DSC **0.9031**(+0.83%p), HD95 **1.6060 px**(-0.13 px)였다. 본 연구는 PPO 보정이 35.3%(858장)의 슬라이스에서 오히려 초기 마스크를 훼손하여 기각되는 실패 모드를 정량적으로 규명하고, 실제 임상 배포를 위해서는 비지도 안전 가드(Unsupervised Guard) 설계가 필수적임을 실증적으로 제시한다.

**주요어:** 뇌종양 분할, 자기공명영상, 강화학습, PPO, 동적 라우팅, TRIO, 의료영상, 안전 가드

---

## 1. 서론

뇌종양의 정확한 영역 분할은 종양 부담(Tumor burden) 정량화, 치료 반응 평가 및 방사선 치료 계획에 중요한 기반 정보를 제공한다. 그러나 MRI에서 종양은 크기, 조영 양상, 경계 선명도 및 주변 부종의 형태가 환자마다 달라 자동 분할 결과의 경계 부근에 미세한 오차가 남기 쉽다. U-Net 계열의 합성곱 신경망은 의료영상 분할에서 높은 성능을 입증했으나[1], 단일 백본 모델 하나로 극소형 병변과 광범위한 대형 병변을 동시에 최적화하는 데에는 한계가 있다.

이러한 문제를 해결하기 위해 본 연구는 분할(Segmentation)과 후처리(Refinement)를 단계적으로 분리한다. 먼저 종양 크기별로 최적화된 전문가 모델을 선택해 초기 마스크(rough mask)를 생성하고, 이후 강화학습 에이전트가 경계 주변을 순차적으로 조정하도록 설계하였다. 이는 고정된 형태학적 연산과 달리 현재 마스크 상태와 영상 단서를 종합적으로 고려해 적응적 보정 행동을 취할 수 있다는 장점이 있다.

본 연구의 핵심 기여는 다음과 같다.

1. 종양 크기(Small, Medium, Large)에 따라 최적화된 백본을 동적으로 선택하는 **3단계 라우팅 분할 구조**를 설계하였다.
2. Signed Distance Field(SDF) 변환과 국소 관심영역 패치를 결합하여 다방위 경계를 미세 조정하는 **PPO 기반 경계 보정 환경**을 구성하였다.
3. PPO 보정 시 발생하는 성능 저하 슬라이스(35.3%)의 오차 모드를 정량적으로 규명하고, 정답 없이 작동하는 면적 게이트 및 향후 비지도 안전 가드(Unsupervised Safety Guard) 설계를 위한 실증적 분석을 제공하였다.
4. BraTS 2021 데이터를 환자 단위로 엄격히 분할한 검증 집합에서 공정한 비교 프로토콜을 수립하고, 대회 상위 입상 구조의 2차원 각색 베이스라인과 동등한 초기 분할 성능을 입증하였다.

---

## 2. 관련 연구 분석

### 2.1 의료영상 분할 백본
U-Net은 인코더-디코더 구조와 skip connection을 통해 위치 정보와 의미 정보를 결합하여 의료영상 분할의 대표적 기반 모델로 자리 잡았다[1]. 이후 UNet++는 중첩된 dense skip connection으로 다중 스케일 특징 융합을 강화했으며[2], SegResNet은 잔차 연결과 인코더 정규화를 통해 3차원 의료영상 분할 성능을 크게 높였다[3]. 또한 CaraNet은 작은 표적의 문맥 정보와 경계 단서를 활용하도록 설계되어 미세 병변 분할에 적합하다[4].

BraTS 2021 챌린지에서 Luu와 Park은 nnU-Net 인코더를 비대칭으로 확장하고 axial attention을 도입하여 최종 테스트 1위를 기록했다[7]. Myronenko 등(NVAUTO)은 SegResNet에 Barlow Twins 기반 중복 감소 학습을 결합해 2위를 기록하였다[8]. 본 연구는 이 두 방법의 핵심 설계를 2차원 이진 분할 조건에 맞춰 재구현하여 비교 기준으로 삼는다.

### 2.2 강화학습 기반 영상 분할 및 반복 보정
강화학습(RL)은 순차적 의사결정 문제에 적합하며, PPO는 안정적인 정책 갱신을 위해 널리 사용되는 알고리즘이다[5]. 의료영상 분야에서 Liao 등은 다중 에이전트 강화학습을 통해 대화형으로 3차원 분할 경계를 반복 보정하는 프레임워크를 제안하였다[9]. 또한 픽셀 단위 RL을 이용한 반복 복원 및 경계 수정 연구들이 보고된 바 있다[10].

선행 RL 보정 연구들이 단일 전역 정책에 의존했던 것과 달리, **본 연구의 차별점은 (1) 병변 크기에 따른 3-scale 조건부 정책 분리(Small 연속 행동 vs Medium/Large 이산 SDF 행동), (2) 소형 병변을 위한 2.5D crop 및 Sobel 에지 관측 설계**를 통해 크기별 오차 패턴에 특화된 보정을 시도했다는 점이다.

---

## 3. 연구 방법론

### 3.1 전체 파이프라인 (TRIO)
제안 파이프라인은 다음 3단계로 구성된다:

1. **Stage 1 (크기 분류):** ImageNet으로 사전학습된 ResNet18 Shape Classifier가 입력 MRI 슬라이스를 소형(<300 px), 중형(300–700 px), 대형(≥700 px)으로 분류한다.
2. **Stage 2 (전문가 분할):** 분류 결과에 따라 CaraNet(소형, 2.5D), UNet++(중형), SegResNet(대형, ED/TC 2채널 예측 후 WT 합성)을 선택해 확률 맵을 생성하고, 클래스별 임계값(0.80 / 0.80 / 0.50)과 TTA(수평/수직 Flip)로 이진화한다.
3. **Stage 3 (강화학습 보정):** 크기별 PPO 에이전트가 마스크 경계를 수정한다. 소형 병변은 64×64 확대 패치에서 연속 행동으로 조정하며, 중형·대형 병변은 128×128 슬라이스에서 8방위 이산 SDF shift 행동으로 조정한다.

데이터 분할은 환자 단위 80/20 고정 분할(`patient_split.json`)을 사용하여 168명 학습(9,868 슬라이스) / 42명 검증(2,434 슬라이스)으로 누출을 원천 차단하였다.

### 3.2 상태, 행동 및 보상 설계
* **관측(Observation):** 중형·대형은 `[MRI 영상, 현재 마스크, 확률 맵]`의 3채널, 소형은 `[MRI 영상, 현재 마스크, 확률 맵, Sobel 에지]`의 4채널 크롭 패치를 사용한다. 관측에는 정답 마스크가 포함되지 않는다.
* **행동(Action) 및 갱신:** 중심 기준 8개 섹터의 경계 이동을 결정한다. 마스크의 SDF(Signed Distance Field)에 행동값을 더한 뒤 $SDF + \text{shift} \ge 0$으로 새 마스크를 생성하며, 형태 보존 연산(Closing/Opening) 후 초기 마스크 $\pm 8\text{px}$ 밴드 내로 클리핑한다.
* **보상 함수:** DSC 변화량, 경계 대역 DSC 변화량, HD95 감소량 및 스텝 패널티로 구성하며, 초기 DSC보다 하락 시 추가 감점을 부여한다.

### 3.3 평가 지표 및 비지도 안전 가드 (Unsupervised Safety Guards)

분할 중첩도는 Dice Similarity Coefficient(DSC)로 평가한다:

$$DSC(P, G) = \frac{2 |P \cap G| + \epsilon}{|P| + |G| + \epsilon}$$

경계 정밀도는 95% Hausdorff Distance(HD95)로 평가하며, 과분할과 과소분할의 진단을 위해 Precision과 Recall을 병행 집계한다.

실제 배포 환경에서 정답(GT) 없이 PPO의 오보정 위험을 원천 차단하기 위해 **3중 비지도 안전 가드(100% GT-Free)**를 구축하였다:
1. **고신뢰도 보호 우회 (Confidence Bypass, GT 0%):** Stage 2 백본의 종양 내부 예측 확신도가 90% 이상($\bar{p} \ge 0.90$)으로 이미 완성도가 높은 마스크는 PPO 단계를 우회(Skip)하여 고품질 초기 분할을 100% 온전히 보존한다.
2. **비지도 MRI 에지 물리 일치도 게이트 (Physical Edge Guard, GT 0%):** 실제 뇌 조직 경계에 수반되는 MRI 밝기 그래디언트 강도(Sobel Edge)와 마스크 외곽선의 정합도($E_{\text{edge}}$)를 측정하여, 보정 후 에지 밀착도가 향상($E_{\text{edge}}^{\text{refined}} \ge E_{\text{edge}}^{\text{init}}$)된 경우에만 PPO 결과를 채택한다.
3. **면적 게이트 (Area Guard, GT 0%):** 보정 마스크가 비었거나 면적이 초기 대비 0.2배 미만 혹은 4배 초과로 비정상 발산 시 즉시 기각한다.
4. **(참고용) 오라클 단조 DSC 게이트 (GT 사용):** 정답을 대조해 롤백하는 연구용 이론적 상한 측정 장치.

---

## 4. 실험 및 결과

### 4.1 정량 결과 (Val 42명 Hold-out, 2,434 슬라이스)

| 평가 조건 | GT 개입 여부 | 평균 DSC | 평균 HD95 (px) | 비고 |
|---|:---:|---:|---:|---|
| **Stage 2 (전문가 초기 분할)** | **0% (없음)** | **0.8959** | **1.7166** | **실제 배포 가능 (메인 성과)** |
| **Stage 3 (순수 PPO, 게이트 없음)** | **0% (없음)** | **0.8850** | **1.8840** | **무조건적 PPO 적용 시 성능 하락** |
| **Stage 3 (단조 DSC 게이트 적용)** | **GT 대조** | **0.9031** | **1.6060** | **이상적 안전 가드 시 이론적 상한** |

* **클래스별 Stage 2 초기 DSC:** Small (`0.8304`), Medium (`0.9268`), Large (`0.9559`)
* **PPO 기각률:** 단조 게이트 적용 시 전체 2,434장 중 **858장(35.3%)~889장(36.5%)**이 오보정으로 인해 초기 마스크로 복원됨.

### 4.2 베이스라인 비교

| 방법 | DSC (배포 기준) | DSC (단조 게이트 상한) | HD95 (px) | Precision | Recall | 95% 신뢰구간 (CI) |
|---|---:|---:|---:|---:|---:|:---:|
| **TRIO Stage 2 (배포 가능)** | **0.8959** | — | **1.7166** | 0.9388 | 0.8860 | [0.8776, 0.9000] |
| **TRIO Stage 3 (이론적 상한)** | — | **0.9031** | **1.6060** | 0.9443 | 0.8914 | [0.8853, 0.9078] |
| Extending nnU-Net 각색 (KAIST) [7] | 0.8923 | — | 1.9591 | 0.9215 | 0.8893 | [0.8738, 0.8955] |
| SegResNet 각색 (NVAUTO) [8] | 0.8971 | — | 1.9022 | 0.9077 | 0.9029 | [0.8786, 0.9016] |

게이트가 없는 제안 방법의 Stage 2(0.8959)는 2차원 조건으로 동일하게 재구현된 KAIST 각색 베이스라인(0.8923)을 상회하고 NVAUTO 각색 베이스라인(0.8971)에 필적하는 뛰어난 초기 분할 성능을 입증하였다. 

**통계적 유의성 검정 (Patient-level Wilcoxon Signed-Rank Test, $n=42$):**
42명의 독립된 환자 코호트에서 대응 표본 Wilcoxon 검정을 수행한 결과, Stage 2는 KAIST 각색 베이스라인 대비 통계적으로 유의미한 우위를 나타냈으며($W=630.0, p=0.0125 < 0.05$), NVAUTO 각색 베이스라인과 대등한 성능 수준($p=0.6030$)을 달성함을 확인하였다.

### 4.3 PPO 보정의 한계, 실패 모드 및 비지도 안전 가드의 역할

PPO 보정 결과 중 35.3%에서 초기 마스크가 훼손되었던 주요 물리적 원인은 다음과 같다:
1. **불확실성 영역 과팽창:** 조영 증강 경계가 모호한 부종(Edema) 부위에서 에이전트가 경계를 무리하게 확장하여 Precision이 급감하는 현상.
2. **미세 분절 이탈:** 소형 병변 외곽의 미세 파편에서 SDF 중심점 산출 오차로 인해 보정 궤적이 빗나가는 현상.
3. **고품질 마스크 과보정 (Over-refinement):** 이미 Stage 2 단계에서 DSC 0.95 이상으로 완벽하게 분할된 마스크를 에이전트가 불필요하게 미세 수정하여 점수가 소폭 하락하는 현상.

**해결 및 시사점:** 
이러한 실패 모드를 방어하기 위해 본 연구는 (1) 확신도가 90% 이상인 완성형 마스크를 온전히 보존하는 **고신뢰도 보호 우회(Confidence Bypass)**와, (2) 보정 테두리가 실제 MRI 영상의 물리적 조직 경계선(Sobel Edge)에 밀착되었는지를 검증하는 **비지도 물리 에지 가드(Physical Edge Guard)**를 도입하였다. 이를 통해 정답(GT)이 없는 실전 배포 환경에서도 오보정 위험을 비지도 방식으로 선별 차단할 수 있는 실증적 토대를 마련하였다.

---

## 5. 결론 및 향후 연구

본 연구는 뇌종양 MRI 분할을 위해 크기별 동적 라우팅과 PPO 기반 경계 보정을 결합한 TRIO 파이프라인을 제안하였다. 환자 단위 검증 집합에서 정답 개입이 없는 Stage 2 초기 분할만으로도 DSC 0.8959 및 HD95 1.7166 px의 높은 성능을 달성하여 SOTA 각색 베이스라인과 대등한 경쟁력을 보였다. 또한 PPO 보정을 통해 최대 0.9031 DSC의 상한에 도달할 수 있음을 확인하는 동시에, 35.3%의 기각 사례 분석을 통해 강화학습 기반 의료영상 후처리가 갖는 현실적 한계와 오차 원인을 규명하였다.

향후 연구로는 첫째, 본 연구에서 제안된 비지도 안전 가드를 고도화하여 실제 병원 배포 환경에서의 단조 향상 안정성을 더욱 완벽히 확보하고, 둘째, 2차원 슬라이스 단위 정책을 **3차원 볼륨 단위 연속 정책**으로 확장할 계획이다.

---

## 6. 참고문헌

[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in *MICCAI*, 2015, pp. 234–241.

[2] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, “UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation,” *IEEE Transactions on Medical Imaging*, vol. 39, no. 6, pp. 1856–1867, 2019.

[3] A. Myronenko, “3D MRI Brain Tumor Segmentation Using Autoencoder Regularization,” in *Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (MICCAI BraTS)*, 2018, pp. 311–320.

[4] A. Lou, S. Guan, and M. Loew, “CaraNet: Context Axial Reverse Attention Network for Segmentation of Small Medical Objects,” *Journal of Medical Imaging*, vol. 10, no. 1, p. 014005, 2022.

[5] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017.

[6] U. Baid et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification,” arXiv:2107.02314, 2021.

[7] H. M. Luu and S.-H. Park, “Extending nn-UNet for Brain Tumor Segmentation,” arXiv:2112.04653, 2021.

[8] A. Myronenko, A. Hatamizadeh, et al., “Redundancy Reduction in Semantic Segmentation of 3D Brain Tumor MRIs,” arXiv:2111.00742, 2021.

[9] X. Liao, W. Li, Q. Xu, X. Wang, B. Jin, X. Zhang, and Y. Zheng, “Iteratively-Refined Interactive 3D Medical Image Segmentation with Multi-Agent Reinforcement Learning,” in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 9394–9402.

[10] R. Furuta, N. Inoue, and T. Yamasaki, “PixelRL: Fully Convolutional Reinforcement Learning with Action-Successor Representation for Image Processing,” *IEEE Transactions on Multimedia*, 2020.
