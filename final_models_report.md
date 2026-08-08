# 🔬 RL-Refiner 최종 실험 결과 보고서 (전체 백본 모델 비교)

> **실험일**: 2026-08-08  
> **데이터셋**: BraTS 2021 Task 1 (전체 1,251명 학습 / 50명 평가)  
> **목적**: 기초 백본 모델(U-Net, SegResNet, UNet++, Attention U-Net, UNet 3+)별 초기 분할 성능과 RL-Refiner(강화학습 보정) 적용 후의 성능 향상도 비교  

---

## 📊 1. 정량적 성능 비교 (Quantitative Results)

백본 모델들에 대해 초기 분할(Rough)과 형태학적 보정(Morpho), 그리고 강화학습 기반 보정(RL Refined)을 거친 후의 **DSC(Dice Similarity Coefficient)**와 **HD95(Hausdorff Distance 95%)**를 비교한 결과입니다.

| 백본 모델 | 방법 | DSC (Mean ± Std) ↑ | HD95 (Mean ± Std px) ↓ | 비고 / RL 보정 효과 (Rough 대비) |
|:---|:---|:---:|:---:|:---|
| **UNet 3+**<br>*(32ch + BatchNorm)* 🌟 | Rough | 0.8279 ± 0.1278 | 3.51 ± 7.10 px | 기준선 |
| | Morpho | 0.8284 ± 0.1308 | 3.65 ± 7.15 px | |
| | **RL Refined** | **0.8327** ± 0.1273 | **3.46 ± 7.11 px** | 🥇 **종합 1위 (SOTA), DSC +0.48%p** |
| **Attention U-Net**<br>*(MONAI 16ch)* | Rough | 0.8246 ± 0.1179 | 2.58 ± 2.28 px | 기준선 |
| | Morpho | 0.8275 ± 0.1179 | 2.78 ± 2.56 px | |
| | **RL Refined** | **0.8297** ± 0.1171 | **2.53 ± 2.31 px** | 🥈 **종합 2위, DSC +0.51%p** |
| **UNet++**<br>*(MONAI Basic)* | Rough | 0.8264 ± 0.1392 | 2.67 ± 2.47 px | 기준선 |
| | Morpho | 0.8248 ± 0.1412 | 2.72 ± 2.53 px | |
| | **RL Refined** | **0.8292** ± 0.1390 | **2.63 ± 2.49 px** | 🥉 **종합 3위, DSC +0.28%p** |
| **SegResNet**<br>*(MONAI ResNet)* ⚡ | Rough | 0.8111 ± 0.1356 | 3.42 ± 2.87 px | 기준선 |
| | Morpho | 0.8126 ± 0.1366 | 3.33 ± 2.69 px | |
| | **RL Refined** | **0.8210** ± 0.1331 | **2.84 ± 2.44 px** | 🎉 **DSC +0.99%p 향상, HD95 -17% 감소** |
| **U-Net**<br>*(커스텀)* | Rough | 0.8090 ± 0.1928 | 3.93 ± 9.17 px | 기준선 |
| | Morpho | 0.8113 ± 0.1926 | 2.06 ± 1.91 px | |
| | **RL Refined** | **0.8134** ± 0.1929 | **2.03 ± 1.92 px** | DSC +0.44%p, HD95 -48% |

> [!TIP]
> **성능 해석 및 SegResNet 보정 효과**
> 1. **UNet 3+ 하이브리드 SOTA**: 32채널 + BatchNorm 조합의 **UNet 3+ 모델**이 RL 보정 후 **최고 DSC 0.8327**로 전체 1위를 기록했습니다.
> 2. **SegResNet 파이프라인 성능 상승**: SegResNet 파이프라인 전용 RL 에이전트 학습 시, 초기 Rough 마스크(`0.8111`) 대비 **RL Refined 마스크(`0.8210`)로 +0.99%p의 유의미한 DSC 상승**과 **HD95 거리 감소(3.42px ➔ 2.84px)**를 달성했습니다.
> 3. **RL 보정의 보편적 유효성**: 5개 백본 모델 전체에서 RL-Refiner 적용 시 **예외 없이 DSC 향상 및 경계 정밀화**가 검증되었습니다.

---

## 📈 2. 모델별 성능 분포 시각화 (DSC Boxplots)

각 모델별로 전체 슬라이스의 DSC 성능 분포가 보정 전/후 어떻게 달라졌는지 보여주는 결과 파일 시각화입니다.

### 🥇 UNet 3+ (SOTA 1위) Boxplot
![UNet 3+ Boxplot](results/dsc_boxplot_unet3plus.png)

### 🥈 Attention U-Net Boxplot
![Attention U-Net Boxplot](results/dsc_boxplot_attention_unet.png)

### 🥉 UNet++ Boxplot
![UNet++ Boxplot](results/dsc_boxplot_unetplusplus.png)

### ⚡ SegResNet Boxplot
![SegResNet Boxplot](results/dsc_boxplot_segresnet.png)

### 4️⃣ U-Net Boxplot
![U-Net Boxplot](results/dsc_boxplot_unet.png)

---

## 🖼️ 3. 정성적 시각화 비교 (Sample Comparison)

최신 시각화 개선사항(가로형 전치 와이드 뷰, 18pt/15pt 확대 폰트 레이아웃)이 적용된 5개 백본 모델별 샘플 비교 이미지입니다.

* **빨간색(Rough)**: 초기 백본이 예측한 마스크
* **주황색(Morpho)**: 형태학적 연산으로 다듬은 마스크
* **하늘색(RL)**: PPO 에이전트가 픽셀 단위로 미세 조정한 마스크

### 🥇 UNet 3+ (SOTA 1위) Sample Comparison
![UNet 3+ Sample Comparison](results/sample_comparison_unet3plus.png)

### 🥈 Attention U-Net Sample Comparison
![Attention U-Net Sample Comparison](results/sample_comparison_attention_unet.png)

### 🥉 UNet++ Sample Comparison
![UNet++ Sample Comparison](results/sample_comparison_unetplusplus.png)

### ⚡ SegResNet Sample Comparison
![SegResNet Sample Comparison](results/sample_comparison_segresnet.png)

### 4️⃣ U-Net Sample Comparison
![U-Net Sample Comparison](results/sample_comparison_unet.png)

---

## 🎯 4. 결론 (Conclusion)

1. **최고 성능 조합**: **UNet 3+ (32ch + BatchNorm) + RL-Refiner** 조합이 최종 DSC **0.8327**을 달성하여 RL 보정 파이프라인 1위를 차지했습니다.
2. **SegResNet RL 보정 성과**: SegResNet 모델 역시 파이프라인 학습 시 Rough `0.8111` ➔ RL `0.8210`으로 **+0.99%p 향상**과 함께 HD95 오차를 **2.84px**까지 줄이는 뛰어난 경계 정밀화 성능을 검증했습니다.
3. **전통적 기법 한계 vs RL 유효성**: 모폴로지 보정은 종종 외곽 경계를 과하게 뭉개어 HD95를 악화시켰으나, RL-Refiner는 원본 MRI 경계를 추종하여 지능적으로 정밀화했습니다.
