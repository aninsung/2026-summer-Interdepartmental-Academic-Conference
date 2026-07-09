"""
Step 4: 성능 검증 및 벤치마크
보정 전/후 DSC·HD95 지표를 비교하고
전통적 영상 처리 기법(CRF-like morphology)과 벤치마크합니다.
"""

import os
import sys
import argparse
import logging
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대응
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_dilation, binary_erosion
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.brats2020_dataset import BraTS2020Dataset
from src.models.unet import build_unet, compute_dice
from src.envs.mask_refinement_env import MaskRefinementEnv, _dice

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── 지표 함수 ──────────────────────────────────────────────────────────────────

def hausdorff_95(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """HD95 계산 (두 경계 집합 간 95번째 백분위 거리)."""
    from scipy.ndimage import distance_transform_edt

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    if not a.any() or not b.any():
        return float("inf")

    dist_a = distance_transform_edt(~a)
    dist_b = distance_transform_edt(~b)

    d_ab = dist_b[a]
    d_ba = dist_a[b]

    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


# ── 전통 방법 벤치마크 ─────────────────────────────────────────────────────────

def morphological_refine(mask: np.ndarray, n_iter: int = None) -> np.ndarray:
    """
    Opening + Closing 으로 경계를 부드럽게 만드는 전통적 보정.
    종양 크기에 따라 n_iter를 자동 결정하는 적응형 방식 적용.
      - 소형 (어직 수 < 200): n_iter=1 (과도한 erosion 로 정보 소실 방지)
      - 중형 (200 ≤ 어직 수 < 1000): n_iter=2
      - 대형 (어직 수 ≥ 1000): n_iter=3
    """
    struct = np.ones((3, 3), dtype=bool)
    pixel_count = int(mask.sum())

    if n_iter is None:
        if pixel_count < 200:
            n_iter = 1
        elif pixel_count < 1000:
            n_iter = 2
        else:
            n_iter = 3

    m = binary_erosion(mask.astype(bool), structure=struct, iterations=n_iter)
    m = binary_dilation(m, structure=struct, iterations=n_iter)
    m = binary_dilation(m, structure=struct, iterations=n_iter)
    m = binary_erosion(m, structure=struct, iterations=n_iter)
    return m.astype(np.float32)


# ── RL 에이전트 추론 ───────────────────────────────────────────────────────────

def rl_refine(
    model: PPO,
    image: np.ndarray,
    rough_mask: np.ndarray,
    gt_mask: np.ndarray,
    max_steps: int = 20,
) -> np.ndarray:
    """학습된 PPO 에이전트로 단일 샘플 마스크를 보정합니다."""
    env = MaskRefinementEnv(
        images=image[None],
        gt_masks=gt_mask[None],
        rough_masks=rough_mask[None],
        max_steps=max_steps,
    )
    obs, _ = env.reset(seed=0)
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    return env._current_mask.copy()


# ── 평가 루프 ─────────────────────────────────────────────────────────────────

def evaluate(
    agent_path: Optional[str] = None,
    unet_path: Optional[str] = None,
    num_eval: int = 50,
    max_steps: int = 30,    # 20 → 30
    output_dir: str = "results",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        dataset = BraTS2020Dataset(
            root_dir=r"src/data/archive/BraTS2021_Training_Data",
            modality="t1ce",
            target_size=128,
            max_patients=num_eval,
            simulate_rough=True,
        )
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")
    except Exception as e:
        log.warning(f"데이터 로드 실패 ({e}). 학습 데이터 폴더에서 평가용 데이터를 로드합니다.")
        dataset = BraTS2020Dataset(
            root_dir=r"src/data/archive/BraTS2021_Training_Data",
            modality="t1ce",
            target_size=128,
            max_patients=num_eval,
            simulate_rough=True,
        )

    images, gt_masks, rough_masks = dataset.get_numpy_arrays()

    # U-Net 로드 (있으면)
    unet = None
    if unet_path and os.path.exists(unet_path):
        unet = build_unet().to(device)
        unet.load_state_dict(torch.load(unet_path, map_location=device))
        unet.eval()
        log.info(f"U-Net 로드: {unet_path}")
    else:
        log.info("U-Net 체크포인트 없음 → rough_mask(합성 노이즈 마스크)를 초기 마스크로 사용")

    # PPO 에이전트 로드 (있으면)
    agent = None
    if agent_path and os.path.exists(agent_path + ".zip"):
        agent = PPO.load(agent_path)
        log.info(f"PPO 에이전트 로드: {agent_path}")
    else:
        log.info("PPO 에이전트 없음 → 보정 없이 rough_mask만 평가")

    results = {
        "rough":  {"dsc": [], "hd95": []},
        "morpho": {"dsc": [], "hd95": []},
        "rl":     {"dsc": [], "hd95": []},
    }

    for i in range(num_eval):
        img = images[i]
        gt = gt_masks[i]

        # 초기 마스크 결정
        if unet is not None:
            img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                rough = (torch.sigmoid(unet(img_t)) > 0.5).float().squeeze().cpu().numpy()
        else:
            rough = rough_masks[i]

        # 전통 보정
        morpho = morphological_refine(rough)

        # RL 보정
        if agent is not None:
            rl_mask = rl_refine(agent, img, rough, gt, max_steps=max_steps)
        else:
            rl_mask = rough  # 에이전트 없으면 rough 그대로

        for key, mask in [("rough", rough), ("morpho", morpho), ("rl", rl_mask)]:
            results[key]["dsc"].append(_dice(mask, gt))
            results[key]["hd95"].append(hausdorff_95(mask, gt))

        if i % 10 == 0:
            log.info(f"  [{i+1}/{num_eval}] rough DSC={results['rough']['dsc'][-1]:.3f} | "
                     f"morpho DSC={results['morpho']['dsc'][-1]:.3f} | "
                     f"RL DSC={results['rl']['dsc'][-1]:.3f}")

    # ── 결과 요약 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Method':<12} {'DSC (mean+/-std)':>20} {'HD95 (mean+/-std)':>22}")
    print("-" * 60)
    for key in ["rough", "morpho", "rl"]:
        dscs = results[key]["dsc"]
        hds = results[key]["hd95"]
        finite_hds = [h for h in hds if np.isfinite(h)]
        print(
            f"{key:<12} "
            f"{np.mean(dscs):>8.4f} +/- {np.std(dscs):.4f}    "
            f"{np.mean(finite_hds) if finite_hds else float('inf'):>8.2f} +/- "
            f"{np.std(finite_hds) if finite_hds else 0:.2f}"
        )
    print("=" * 60)

    # ── 시각화 ───────────────────────────────────────────────
    _plot_results(images, gt_masks, rough_masks, results, output_dir, num_show=min(4, num_eval))
    log.info(f"결과 저장 완료: {output_dir}/")


def _plot_results(images, gt_masks, rough_masks, results, output_dir, num_show=4):
    """샘플 시각화 및 DSC 분포 박스플롯 저장."""
    # 1) 샘플별 마스크 비교
    fig, axes = plt.subplots(num_show, 4, figsize=(14, num_show * 3.5))
    cols = ["MRI + GT", "Rough Mask", "Morpho Refined", "RL Refined"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight="bold")

    for row in range(num_show):
        img = images[row]
        gt = gt_masks[row]
        rough = rough_masks[row]
        dsc_rough = results["rough"]["dsc"][row]
        dsc_morpho = results["morpho"]["dsc"][row]
        dsc_rl = results["rl"]["dsc"][row]

        # MRI + GT 윤곽
        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
        axes[row, 0].axis("off")

        # Rough
        axes[row, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].contour(rough, levels=[0.5], colors="red", linewidths=1.5)
        axes[row, 1].contour(gt, levels=[0.5], colors="lime", linewidths=1.0, linestyles="--")
        axes[row, 1].set_xlabel(f"DSC={dsc_rough:.3f}", fontsize=9)
        axes[row, 1].axis("off")

        # Morpho
        axes[row, 2].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].contour(rough, levels=[0.5], colors="orange", linewidths=1.5)
        axes[row, 2].contour(gt, levels=[0.5], colors="lime", linewidths=1.0, linestyles="--")
        axes[row, 2].set_xlabel(f"DSC={dsc_morpho:.3f}", fontsize=9)
        axes[row, 2].axis("off")

        # RL
        axes[row, 3].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].contour(rough, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 3].contour(gt, levels=[0.5], colors="lime", linewidths=1.0, linestyles="--")
        axes[row, 3].set_xlabel(f"DSC={dsc_rl:.3f}", fontsize=9)
        axes[row, 3].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "sample_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"샘플 시각화 저장: {save_path}")

    # 2) DSC 박스플롯
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [results[k]["dsc"] for k in ["rough", "morpho", "rl"]]
    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    labels=["Rough\n(U-Net)", "Morpho\nRefined", "RL\nRefined"])
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("DSC", fontsize=12)
    ax.set_title("Dice Similarity Coefficient: Before vs After Refinement", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, "dsc_boxplot.png")
    plt.savefig(save_path2, dpi=150)
    plt.close()
    log.info(f"DSC 박스플롯 저장: {save_path2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Evaluate RL-Refiner")
    parser.add_argument("--agent_path", type=str, default="checkpoints/ppo_refiner")
    parser.add_argument("--unet_path", type=str, default="checkpoints/unet_best.pt")
    parser.add_argument("--num_eval", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    evaluate(**vars(args))
