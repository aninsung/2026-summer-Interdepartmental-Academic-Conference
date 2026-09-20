"""
논문 제출용 시각화 스크립트 (Qualitative Results Visualization)
BraTS 2021 데이터에 대하여 [MRI 원본, 정답(GT), Baseline, 제안기법(PPO)] 을 나란히 비교합니다.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# BraTS 컬러 맵핑 정의 (투명도 지원을 위해 RGBA 사용)
# 0: Background (투명)
# 1: NCR / NET (Red)
# 2: ED (Green)
# 3: ET (Blue)
CMAP_BRATS = ListedColormap([
    (0.0, 0.0, 0.0, 0.0),      # BG
    (1.0, 0.0, 0.0, 0.6),      # 1: NCR/NET (Red)
    (0.0, 1.0, 0.0, 0.5),      # 2: ED (Green)
    (0.0, 0.3, 1.0, 0.8),      # 3: ET (Blue)
])


def load_data(run_dir: str, patient_id: str):
    """지정된 환자의 캐시된 원본 이미지와 최종 평가 마스크를 불러옵니다."""
    run_path = Path(run_dir)
    
    # 1. 원본 이미지 (전처리/크롭 완료본) 불러오기
    pred_path = run_path / "predictions" / f"{patient_id}.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction cache not found: {pred_path}")
    
    with np.load(pred_path, allow_pickle=True) as data:
        # data["image"] shape: (4, H, W, D) -> 0:T1, 1:T1ce, 2:T2, 3:FLAIR
        image = data["image"][1] # T1ce (조영증강 채널)가 종양 경계 관찰에 가장 좋습니다.
        
    # 2. 결과 마스크 (Initial vs PPO) 불러오기
    # test 세트인 경우 test_masks, val 세트인 경우 validation_masks
    mask_path = run_path / f"test_masks_{patient_id}.npz"
    if not mask_path.exists():
        mask_path = run_path / f"validation_masks_{patient_id}.npz"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask results not found for {patient_id}")
            
    with np.load(mask_path, allow_pickle=True) as data:
        target = data["target"]
        initial = data["initial"]
        ppo = data["ppo"]
        
    return image, target, initial, ppo


def find_best_slice(target_mask):
    """종양(ET, TC)이 가장 잘 보이는 Z축 슬라이스 인덱스를 찾습니다."""
    # ET(3) 영역이 가장 넓은 슬라이스 우선 탐색
    et_area = np.sum(target_mask == 3, axis=(0, 1))
    if et_area.max() > 10:
        return int(np.argmax(et_area))
    
    # ET가 작으면 전체 종양 영역이 가장 넓은 슬라이스 선택
    wt_area = np.sum(target_mask > 0, axis=(0, 1))
    return int(np.argmax(wt_area))


def plot_comparison(run_dir: str, patient_ids: list, output_filename: str):
    """여러 환자의 비교 이미지를 하나의 표(Grid) 형태로 그립니다."""
    num_cases = len(patient_ids)
    fig, axes = plt.subplots(num_cases, 4, figsize=(16, 4 * num_cases), facecolor='white')
    
    if num_cases == 1:
        axes = [axes]
        
    col_titles = ["Image (T1ce)", "Ground Truth", "Baseline (SegResNet)", "Ours (SegResNet + PPO)"]
    
    for row_idx, patient_id in enumerate(patient_ids):
        image, target, initial, ppo = load_data(run_dir, patient_id)
        z_slice = find_best_slice(target)
        
        # 2D 슬라이스 추출 및 회전 (보통 90도 회전 시 뇌 모양이 똑바로 보임)
        img_2d = np.rot90(image[:, :, z_slice])
        gt_2d = np.rot90(target[:, :, z_slice])
        init_2d = np.rot90(initial[:, :, z_slice])
        ppo_2d = np.rot90(ppo[:, :, z_slice])
        
        # 대비 조절 (Vmin, Vmax)
        vmin, vmax = np.percentile(img_2d[img_2d > 0], [1, 99]) if np.any(img_2d > 0) else (0, 1)
        
        masks = [None, gt_2d, init_2d, ppo_2d]
        
        for col_idx in range(4):
            ax = axes[row_idx][col_idx]
            ax.axis('off')
            
            # 배경 MRI 출력
            ax.imshow(img_2d, cmap='gray', vmin=vmin, vmax=vmax)
            
            # 마스크 오버레이
            if col_idx > 0:
                mask_display = np.ma.masked_where(masks[col_idx] == 0, masks[col_idx])
                ax.imshow(mask_display, cmap=CMAP_BRATS, interpolation='nearest')
                
            # 타이틀 (첫 번째 행에만)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=16, fontweight='bold', pad=15)
                
            # y축 라벨 (환자 ID)
            if col_idx == 0:
                ax.text(-0.1, 0.5, f"{patient_id}\n(z={z_slice})", va='center', ha='right', 
                        rotation=90, transform=ax.transAxes, fontsize=14, fontweight='bold')
                
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    output_path = Path(run_dir) / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 논문용 시각화 이미지가 성공적으로 저장되었습니다: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Segmentation Results")
    parser.add_argument("--run_dir", type=str, default="runs/boundary_hybrid_fix", help="학습 결과 폴더")
    parser.add_argument("--patients", type=str, nargs="+", required=True, help="시각화할 환자 ID (공백으로 구분)")
    parser.add_argument("--out", type=str, default="qualitative_comparison_fig.png", help="저장할 파일명")
    args = parser.parse_args()
    
    plot_comparison(args.run_dir, args.patients, args.out)
