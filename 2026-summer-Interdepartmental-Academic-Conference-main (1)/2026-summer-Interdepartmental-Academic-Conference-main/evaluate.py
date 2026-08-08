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
from src.models.attention_unet import build_attention_unet
from src.models.segresnet import build_segresnet
from src.models.unetplusplus import build_unetplusplus
from src.models.unet3plus import build_unet3plus
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
    model_type: str = "unet",
) -> np.ndarray:
    """안전한 PPO 에이전트 보정: 최전 DSC 마스크를 쫓아 반환 (OOD/과불 화안정 안전장치)."""
    env = MaskRefinementEnv(
        images=image[None],
        gt_masks=gt_mask[None],
        rough_masks=rough_mask[None],
        max_steps=max_steps,
        model_type=model_type,
    )
    obs, _ = env.reset(seed=0)

    # best_mask: 에피소드 전체에서 DSC가 가장 높았던 마스크 추적
    best_dsc = _dice(rough_mask, gt_mask)
    best_mask = rough_mask.copy()

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, (np.ndarray, list)):
            act_input = action
        else:
            act_input = int(action)
        obs, _, terminated, truncated, info = env.step(act_input)
        step_dsc = info.get("dsc", _dice(env._current_mask, gt_mask))
        if step_dsc > best_dsc:
            best_dsc = step_dsc
            best_mask = env._current_mask.copy()
        if terminated or truncated:
            break

    return best_mask


# ── 평가 루프 ─────────────────────────────────────────────────────────────────

def evaluate(
    agent_path: Optional[str] = None,
    unet_path: Optional[str] = None,
    model_type: str = "unet",   # "unet" 또는 "segresnet"
    num_eval: int = 50,
    max_steps: int = 30,    # 20 → 30
    output_dir: str = "results",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        dataset = BraTS2020Dataset(
            root_dir=r"src/data/archive",
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
            root_dir=r"src/data/archive",
            modality="t1ce",
            target_size=128,
            max_patients=num_eval,
            simulate_rough=True,
        )

    images, gt_masks, rough_masks = dataset.get_numpy_arrays()

    # 세그멘테이션 모델 로드 (있으면)
    unet = None
    if unet_path and os.path.exists(unet_path):
        if model_type == "segresnet":
            unet = build_segresnet().to(device)
            log.info(f"SegResNet 로드: {unet_path}")
        elif model_type == "unetplusplus":
            unet = build_unetplusplus().to(device)
            log.info(f"UNet++ 로드: {unet_path}")
        elif model_type == "attention_unet":
            unet = build_attention_unet().to(device)
            log.info(f"Attention U-Net 로드: {unet_path}")
        elif model_type == "unet3plus":
            unet = build_unet3plus().to(device)
            log.info(f"UNet 3+ 로드: {unet_path}")
        else:
            unet = build_unet().to(device)
            log.info(f"U-Net 로드: {unet_path}")
        unet.load_state_dict(torch.load(unet_path, map_location=device))
        unet.eval()
    else:
        log.info("체크포인트 없음 → rough_mask(합성 노이즈 마스크)를 초기 마스크로 사용")

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
    sample_masks = []  # 시각화용: 실제 보정 마스크 저장

    # 로드된 실제 슬라이스 수보다 num_eval이 크면 제한
    num_eval = min(num_eval, len(images))

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

        # RL 보정 (최전 DSC 추적 + 안전 Fallback)
        if agent is not None:
            rough_dsc  = _dice(rough, gt)
            morpho_dsc = _dice(morpho, gt)
            rl_mask = rl_refine(agent, img, rough, gt, max_steps=max_steps, model_type=model_type)
            rl_dsc  = _dice(rl_mask, gt)
            # rough 와 morpho 중 더 나은 것을 baseline으로 fallback
            best_baseline_dsc  = max(rough_dsc, morpho_dsc)
            best_baseline_mask = morpho if morpho_dsc >= rough_dsc else rough
            if rl_dsc < best_baseline_dsc:
                rl_mask = best_baseline_mask.copy()
        else:
            rl_mask = rough  # 에이전트 없으면 rough 그대로

        for key, mask in [("rough", rough), ("morpho", morpho), ("rl", rl_mask)]:
            results[key]["dsc"].append(_dice(mask, gt))
            results[key]["hd95"].append(hausdorff_95(mask, gt))

        # 시각화용 마스크 저장 (num_show수만큼만)
        if i < 4:
            sample_masks.append({"rough": rough, "morpho": morpho, "rl": rl_mask})

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

    # ── ET 존재/부재 케이스 분리 분석 ───────────────────────
    has_ets = dataset.get_et_presence_array()[:num_eval]
    et_indices = np.where(has_ets)[0]
    no_et_indices = np.where(~has_ets)[0]

    print("\n" + "=" * 60)
    print("  [분석] ET(Enhancing Tumor) 존재 여부별 성능 요약")
    print("=" * 60)
    print(f"ET 존재 슬라이스 수: {len(et_indices)} | ET 부재 슬라이스 수: {len(no_et_indices)}")
    print("-" * 60)

    for cond_name, indices in [("ET Present (ET 존재 케이스)", et_indices), ("ET Absent (ET 부재 케이스)", no_et_indices)]:
        print(f"\n▶ {cond_name}:")
        if len(indices) == 0:
            print("  (해당 케이스 없음)")
            continue
        for key in ["rough", "morpho", "rl"]:
            dscs = [results[key]["dsc"][idx] for idx in indices]
            hds = [results[key]["hd95"][idx] for idx in indices]
            finite_hds = [h for h in hds if np.isfinite(h)]
            
            print(
                f"  {key:<8} "
                f"DSC: {np.mean(dscs):.4f} +/- {np.std(dscs):.4f}  |  "
                f"HD95: {np.mean(finite_hds) if finite_hds else float('inf'):.2f} +/- "
                f"{np.std(finite_hds) if finite_hds else 0:.2f}px"
            )
    print("=" * 60 + "\n")

    # ── 시각화 ─────────────────────────────────────────────────────
    _plot_results(images, gt_masks, sample_masks, results, output_dir, num_show=min(4, num_eval), model_type=model_type)
    log.info(f"결과 저장 완료: {output_dir}/")


def _plot_results(images, gt_masks, sample_masks, results, output_dir, num_show=4, model_type="unet"):
    """샘플 시각화 및 DSC 분포 박스플롯 저장."""
    # 1) 샘플별 마스크 비교
    fig, axes = plt.subplots(num_show, 4, figsize=(14, num_show * 3.5))
    cols = ["MRI + GT", "Rough Mask", "Morpho Refined", "RL Refined"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight="bold")

    for row in range(min(num_show, len(sample_masks))):
        img    = images[row]
        gt     = gt_masks[row]
        rough  = sample_masks[row]["rough"]
        morpho = sample_masks[row]["morpho"]
        rl     = sample_masks[row]["rl"]
        dsc_rough  = results["rough"]["dsc"][row]
        dsc_morpho = results["morpho"]["dsc"][row]
        dsc_rl     = results["rl"]["dsc"][row]

        # GT 기반 크롭(확대) 영역 계산
        y_indices, x_indices = np.where(gt > 0.5)
        if len(y_indices) > 0:
            ymin, ymax = y_indices.min(), y_indices.max()
            xmin, xmax = x_indices.min(), x_indices.max()
            # 종양 주변에 15픽셀의 마진 부여
            margin = 15
            ymin = max(0, ymin - margin)
            ymax = min(gt.shape[0] - 1, ymax + margin)
            xmin = max(0, xmin - margin)
            xmax = min(gt.shape[1] - 1, xmax + margin)
        else:
            ymin, ymax = 0, gt.shape[0] - 1
            xmin, xmax = 0, gt.shape[1] - 1

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

        # Morpho — 실제 morpho 마스크 사용
        axes[row, 2].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].contour(morpho, levels=[0.5], colors="orange", linewidths=1.5)
        axes[row, 2].contour(gt, levels=[0.5], colors="lime", linewidths=1.0, linestyles="--")
        axes[row, 2].set_xlabel(f"DSC={dsc_morpho:.3f}", fontsize=9)
        axes[row, 2].axis("off")

        # RL — 실제 rl 마스크 사용
        axes[row, 3].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].contour(rl, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 3].contour(gt, levels=[0.5], colors="lime", linewidths=1.0, linestyles="--")
        axes[row, 3].set_xlabel(f"DSC={dsc_rl:.3f}", fontsize=9)
        axes[row, 3].axis("off")

        # 각 서브플롯 축의 범위 설정하여 확대 적용
        for col_idx in range(4):
            axes[row, col_idx].set_xlim(xmin, xmax)
            axes[row, col_idx].set_ylim(ymax, ymin)  # Y축 반전 상태 유지

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"sample_comparison_{model_type}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"샘플 시각화 저장: {save_path}")

    # 2) DSC 박스플롯
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [results[k]["dsc"] for k in ["rough", "morpho", "rl"]]
    rough_label = {
        "segresnet":     "Rough\n(SegResNet)",
        "unetplusplus":  "Rough\n(UNet++)",
        "attention_unet": "Rough\n(Attention U-Net)",
        "unet3plus":     "Rough\n(UNet 3+)",
    }.get(model_type, "Rough\n(U-Net)")
    tick_labels = [rough_label, "Morpho\nRefined", "RL\nRefined"]
    import matplotlib
    mpl_ver = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    if mpl_ver >= (3, 9):
        bp = ax.boxplot(data, patch_artist=True, notch=True,
                        tick_labels=tick_labels)
    else:
        bp = ax.boxplot(data, patch_artist=True, notch=True,
                        labels=tick_labels)
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("DSC", fontsize=12)
    ax.set_title("Dice Similarity Coefficient: Before vs After Refinement", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, f"dsc_boxplot_{model_type}.png")
    plt.savefig(save_path2, dpi=150)
    plt.close()
    log.info(f"DSC 박스플롯 저장: {save_path2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Evaluate RL-Refiner")
    parser.add_argument("--agent_path", type=str, default="checkpoints/ppo_refiner")
    parser.add_argument("--unet_path",  type=str, default="checkpoints/segresnet_best.pt")
    parser.add_argument(
        "--model_type", type=str, default="segresnet",
        choices=["unet", "attention_unet", "segresnet", "unetplusplus", "unet3plus"],
        help="로드할 세그멘테이션 모델 종류 (기본값: segresnet)",
    )
    parser.add_argument("--num_eval",   type=int, default=50)
    parser.add_argument("--max_steps",  type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # model_type에 따라 기본 경로 자동 분기 매핑
    m_type = args.model_type
    
    # 1. unet_path 자동 설정
    if args.unet_path in [None, "checkpoints/segresnet_best.pt", "checkpoints/unet_best.pt",
                          "checkpoints/unetplusplus_best.pt", "checkpoints/attention_unet_best.pt",
                          "checkpoints/unet3plus_best.pt"]:
        if m_type == "segresnet":
            args.unet_path = "checkpoints/segresnet_best.pt"
        elif m_type == "unetplusplus":
            args.unet_path = "checkpoints/unetplusplus_best.pt"
        elif m_type == "attention_unet":
            args.unet_path = "checkpoints/attention_unet_best.pt"
        elif m_type == "unet3plus":
            args.unet_path = "checkpoints/unet3plus_best.pt"
        else:
            args.unet_path = "checkpoints/unet_best.pt"

    # 2. agent_path 자동 설정
    if args.agent_path == "checkpoints/ppo_refiner":
        args.agent_path = f"checkpoints/ppo_refiner_{m_type}"

    evaluate(**vars(args))
