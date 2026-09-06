import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import os
import torch
import numpy as np
from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import load_or_create_patient_split, DEFAULT_SPLIT_PATH
from src.models.dynamic_router import AdaptivePipeline
from src.utils.metrics import apply_monotonic_dsc_gate, dice, filter_small_components, hd95, precision, recall
from skimage.segmentation import morphological_chan_vese
from src.envs.mask_refinement_env import MaskRefinementEnv
from stable_baselines3 import PPO
from scipy.ndimage import sobel, binary_dilation, binary_erosion, binary_closing, binary_opening

def _compute_edge_map(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img_2d = np.mean(img, axis=0)
    else:
        img_2d = img.copy()
    edge_x = sobel(img_2d, axis=0)
    edge_y = sobel(img_2d, axis=1)
    edge = np.sqrt(edge_x**2 + edge_y**2)
    e_min, e_max = edge.min(), edge.max()
    if e_max > e_min:
        edge = (edge - e_min) / (e_max - e_min)
    return edge.astype(np.float32)

def _get_boundary_mask(mask: np.ndarray) -> np.ndarray:
    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(mask > 0.5, structure=struct)
    eroded = binary_erosion(mask > 0.5, structure=struct)
    return dilated ^ eroded

def _average_edge_intensity(boundary_mask: np.ndarray, edge_map: np.ndarray) -> float:
    if np.sum(boundary_mask) == 0:
        return 0.0
    return float(np.mean(edge_map[boundary_mask]))


def _tta_probability(pipeline, img_t, rough_mask_t, class_pred):
    with torch.no_grad():
        img_hf = torch.flip(img_t, dims=[3])
        out_hf, _ = pipeline(img_hf, true_class_preds=class_pred)
        out_hf = torch.flip(out_hf, dims=[3])
        img_vf = torch.flip(img_t, dims=[2])
        out_vf, _ = pipeline(img_vf, true_class_preds=class_pred)
        out_vf = torch.flip(out_vf, dims=[2])
        return ((rough_mask_t + out_hf + out_vf) / 3.0).squeeze().cpu().numpy()


def _gt_free_accept(
    rough: np.ndarray,
    refined: np.ndarray,
    lo: float = 0.85,
    hi: float = 1.2,
) -> bool:
    """Area gate (GT-free). Reject empty or extreme area change vs rough."""
    if float(np.sum(refined)) < 1.0:
        return False
    r = max(1.0, float(np.sum(rough)))
    f = float(np.sum(refined))
    return (float(lo) * r) <= f <= (float(hi) * r)


def _shape_energy(mask_np: np.ndarray) -> float:
    """Shape energy (compactness): perimeter^2 / (4 * pi * area). Higher = more irregular/noisy."""
    area = float(np.sum(mask_np > 0.5))
    if area <= 1.0:
        return 0.0
    er = binary_erosion(mask_np > 0.5, structure=np.ones((3, 3)))
    peri = float(np.sum((mask_np > 0.5) ^ er))
    return (peri ** 2) / (4.0 * np.pi * area)


def _sdf_displacement_95(init_mask: np.ndarray, refined_mask: np.ndarray) -> float:
    """Stage 2 초기 마스크 대비 PPO 보정 마스크의 경계 변위(Displacement) 95th percentile (GT-free)."""
    from scipy.ndimage import distance_transform_edt
    init_b = (init_mask > 0.5)
    ref_b = (refined_mask > 0.5)
    if not init_b.any() or not ref_b.any():
        return 0.0
    dist_from_init = distance_transform_edt(~init_b)
    ref_boundary = ref_b ^ binary_erosion(ref_b, iterations=1)
    if not ref_boundary.any():
        return 0.0
    dists = dist_from_init[ref_boundary]
    return float(np.percentile(dists, 95))


def _soft_prob_consistency(refined_mask: np.ndarray, soft_prob: np.ndarray) -> bool:
    """Soft probability confidence 준수 검사 (GT-free).
    High confidence(>0.85) 보존율 및 Low confidence(<0.10) 침범율 평가."""
    ref_b = (refined_mask > 0.5)
    high_conf = (soft_prob > 0.85)
    low_conf = (soft_prob < 0.10)
    
    if high_conf.any():
        retained = np.logical_and(ref_b, high_conf).sum() / high_conf.sum()
        if retained < 0.80:
            return False
            
    if ref_b.any():
        low_ratio = np.logical_and(ref_b, low_conf).sum() / ref_b.sum()
        if low_ratio > 0.20:
            return False
            
    return True


def _boundary_dsc_gt(mask: np.ndarray, band: np.ndarray, gt: np.ndarray) -> float:
    """[오프라인 상한 분석 전용] GT 경계 band 내 DSC (Kervadec et al.)"""
    m = (mask > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    mb = m * band
    gb = g * band
    inter = float((mb * gb).sum())
    denom = float(mb.sum() + gb.sum())
    return (2.0 * inter / denom) if denom > 0 else 1.0



def _policy_expects_stop(agent) -> bool:
    """SB3 정책 action space가 STOP 채널을 포함하는지 판별."""
    space = agent.policy.action_space
    if hasattr(space, "nvec"):
        return len(space.nvec) >= 9 and int(space.nvec[-1]) == 2
    shape = getattr(space, "shape", None)
    return bool(shape is not None and len(shape) == 1 and int(shape[0]) >= 9)


def _assert_agent_stop_compat(agent, enable_stop: bool, name: str):
    expects = _policy_expects_stop(agent)
    if enable_stop and not expects:
        raise RuntimeError(
            f"{name}: --enable_stop 인데 체크포인트 action space에 STOP이 없습니다. "
            "STOP으로 재학습한 zip을 쓰거나 --enable_stop 을 끄세요."
        )
    if (not enable_stop) and expects:
        print(
            f"WARNING: {name}: 체크포인트는 STOP 정책인데 --enable_stop 이 꺼져 있습니다. "
            "(TEGDA Entropy Gating 작동을 위해 강제 허용합니다.)"
        )


def _refine_with_ppo(
    agent,
    image,
    gt,
    init_mask,
    prob_map,
    refinement_mode,
    n_steps=15,
    clip_shrink=False,
    select_best=True,
    apply_monotonic=True,
    enable_stop=False,
):
    """PPO 보정.

    select_best=True  → 15스텝 중 GT DSC 최고 마스크 (상한, 배포 불가)
    select_best=False → 마지막(또는 STOP) 마스크 (배포에 가깝음)
    apply_monotonic   → 보정 DSC < 초기면 초기 유지 (GT 필요)
    enable_stop       → STOP 행동으로 학습된 에이전트일 때 True
    """
    env = MaskRefinementEnv(
        image[None, ...],
        gt[None, ...],
        np.expand_dims(init_mask, 0),
        uncertainty_maps=np.expand_dims(prob_map, 0),
        max_steps=n_steps,
        target_dsc=1.0,
        refinement_mode=refinement_mode,
        enable_stop=enable_stop,
    )
    obs, _ = env.reset(seed=0)
    best_mask = init_mask.copy()
    best_dsc = dice(init_mask, gt)
    last_mask = init_mask.copy()
    for _ in range(n_steps):
        action, _ = agent.predict(obs, deterministic=True)
        if clip_shrink and np.sum(env._current_mask) < 35:
            action = np.asarray(action, dtype=np.float32).copy()
            action[:8] = np.maximum(0.0, action[:8])
        obs, _, terminated, truncated, info = env.step(action)
        last_mask = env._current_mask.copy()
        step_dsc = float(info.get("dsc", dice(last_mask, gt)))
        if select_best and step_dsc >= best_dsc:
            best_dsc = step_dsc
            best_mask = last_mask.copy()
        if terminated or truncated:
            break
    out = best_mask if select_best else last_mask
    if apply_monotonic:
        return apply_monotonic_dsc_gate(init_mask, out, gt)
    return out

def main():
    import argparse
    parser = argparse.ArgumentParser(description="3-Stage Dynamic Routing Pipeline Evaluation")
    parser.add_argument("--train_root", type=str, default="src/data/archive", help="데이터셋 경로")
    parser.add_argument("--modality", type=str, default="t1ce+flair", help="MRI 모달리티 ('t1ce', 't1ce+flair' 등)")
    parser.add_argument("--max_patients", type=int, default=1251, help="평가 풀 환자 수 (split 생성 기준)")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="클래스당 최대 샘플 수 (None이면 제한 없음)")
    parser.add_argument("--patient_split", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split_role", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--oracle_routing", action="store_true", help="GT 면적으로 Expert를 고르는 상한 평가")
    parser.add_argument("--confidence_threshold", type=float, default=None, help="이 값 이상 평균 확률이면 PPO 생략")
    parser.add_argument("--stage2_thresholds", type=str, default="0.1,0.1,0.1",
                        help="클래스별(Small,Medium,Large) Stage 2 이진화 임계값.")
    parser.add_argument("--skip_ppo", action="store_true", help="Stage 3 생략 (Stage 2 단독 베이스라인 측정)")
    parser.add_argument(
        "--stage3_mode",
        type=str,
        default="ppo",
        choices=["ppo", "skip"],
        help="Stage3: ppo | skip",
    )
    parser.add_argument("--micro_area_floor", type=float, default=80.0,
                        help="Small 마스크 면적이 이 값 미만이면 임계값을 단계적으로 낮춘다.")
    parser.add_argument("--micro_thr_floor", type=float, default=0.15,
                        help="마이크로 조각 임계값 완화의 하한.")
    parser.add_argument("--cc_min_sizes", type=str, default="80,80,20",
                        help="클래스별 CC filter 최소 크기 (Small,Medium,Large)")
    parser.add_argument(
        "--stage2_erode_classes",
        type=str,
        default="1,2",
        help="Stage2 이진화 후 erosion 적용 클래스. 기본 1,2=Medium+Large under-seg",
    )
    parser.add_argument(
        "--stage2_erode_px",
        type=int,
        default=0,
        help="stage2_erode_classes에 적용할 erosion 반복(px). 0이면 비활성.",
    )
    parser.add_argument("--skip_plot", action="store_true", help="시각화 PNG 생략")
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="results/pipeline_slice_metrics.npz",
        help="슬라이스별 Init/Final 지표 저장 경로 (Wilcoxon·CI용)",
    )
    parser.add_argument(
        "--deploy_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="배포형 평가(기본): 마지막(또는 STOP) 마스크 + 면적 게이트만. 끄기: --no-deploy_mode",
    )
    parser.add_argument(
        "--gt_upper_bound",
        action="store_true",
        help="논문 금지용 상한: GT best-of-N + 단조 DSC 게이트 (배포 아님·메인 수치 금지)",
    )
    parser.add_argument(
        "--last_mask",
        action="store_true",
        help="15스텝 중 GT best 대신 마지막 마스크 사용",
    )
    parser.add_argument(
        "--no_monotonic_gate",
        action="store_true",
        help="단조 DSC 게이트 비활성화",
    )

    parser.add_argument(
        "--stage3_skip_classes",
        type=str,
        default="",
        help="Stage3를 건너뛸 클래스 (0=Small,1=Medium,2=Large). 기본: 없음(Medium/Large expand 보정)",
    )
    parser.add_argument(
        "--area_gate_lo",
        type=float,
        default=0.85,
        help="면적 게이트 하한 (refined/rough). 배포 기본 0.85",
    )
    parser.add_argument(
        "--area_gate_hi",
        type=float,
        default=1.2,
        help="면적 게이트 상한 (refined/rough). 배포 기본 1.2",
    )
    parser.add_argument(
        "--area_gate_hi_medium",
        type=float,
        default=1.5,
        help="Medium(class=1) 전용 면적 게이트 상한. <=0 이면 --area_gate_hi 사용",
    )
    parser.add_argument(
        "--area_gate_hi_large",
        type=float,
        default=1.35,
        help="Large(class=2) 전용 면적 게이트 상한. <=0 이면 --area_gate_hi 사용",
    )
    parser.add_argument(
        "--medium_active_max_area",
        type=float,
        default=0.0,
        help="Medium(class=1): rough 성분 면적 < 이 값이면 적극 refine(band 없음). "
        "0이면 비활성. 권장 450",
    )
    parser.add_argument(
        "--medium_skip_min_area",
        type=float,
        default=0.0,
        help="Medium: rough 성분 면적 ≥ 이 값이면 Stage3 스킵(Stage2 유지). "
        "0이면 비활성. 권장 700 (active~skip 사이는 shrink)",
    )
    parser.add_argument(
        "--enable_stop",
        action="store_true",
        help="STOP 행동으로 학습된 PPO 체크포인트 평가",
    )
    parser.add_argument(
        "--active_contour",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CVPR 2019 Active Contour 방식 기반 경계 후처리 스무딩 (HD95 감축용)",
    )
    parser.add_argument(
        "--ppo_action_tta",
        action="store_true",
        help="PPO 평가 시 Horizontal Flip Action TTA 앙상블 적용",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.3,
        help="Gate 1: Boundary Shannon entropy threshold to trigger PPO (default 0.3).",
    )
    parser.add_argument(
        "--fast_mode",
        action="store_true",
        default=True,
        help="Use single backbone forward for probability instead of 3x TTA forward (much faster).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataset dataloader workers (default 4 to avoid thread starvation)",
    )
    args = parser.parse_args()
    if args.skip_ppo:
        args.stage3_mode = "skip"

    # 기본은 배포 프로토콜. --gt_upper_bound 만 옛 GT 상한을 켠다.
    if args.gt_upper_bound:
        args.deploy_mode = False
        args.last_mask = False
        args.no_monotonic_gate = False
        args.enable_stop = False
        print(
            "[gt_upper_bound] best-of-N + 단조 DSC 게이트 ON — "
            "오프라인 상한만. 논문/배포 메인 수치·'성능 보장'으로 쓰지 말 것"
        )
    elif args.deploy_mode:
        args.last_mask = True
        args.no_monotonic_gate = True
        # PPO 모드일 때만 STOP 강제 -> TEGDA Entropy Gating을 위해 강제 비활성
        if args.stage3_mode in ("ppo", "hybrid"):
            args.enable_stop = False
            print("[deploy_mode] --enable_stop 강제 비활성화 (TEGDA 가동)")
        print(
            "[deploy_mode] GT monotonic gate OFF — 면적 게이트만 사용 "
            f"(lo={args.area_gate_lo}, hi={args.area_gate_hi}); "
            "hybrid/ppo uses gt_free last-mask (no GT cherry-pick)"
        )

    select_best = not args.last_mask
    apply_monotonic = not args.no_monotonic_gate
    enable_stop = bool(args.enable_stop)
    area_lo = float(args.area_gate_lo)
    area_hi = float(args.area_gate_hi)
    if not (0.0 < area_lo <= area_hi):
        parser.error("--area_gate_lo/--area_gate_hi 는 0 < lo <= hi 이어야 합니다.")
    area_hi_by_class = {0: area_hi, 1: area_hi, 2: area_hi}
    med_hi = float(args.area_gate_hi_medium)
    large_hi = float(args.area_gate_hi_large)
    if med_hi > 0:
        if med_hi < area_lo:
            parser.error("--area_gate_hi_medium 는 area_gate_lo 이상이어야 합니다.")
        area_hi_by_class[1] = med_hi
    if large_hi > 0:
        if large_hi < area_lo:
            parser.error("--area_gate_hi_large 는 area_gate_lo 이상이어야 합니다.")
        area_hi_by_class[2] = large_hi
    print(
        f"Area gate: lo={area_lo:.2f}× | hi Small={area_hi_by_class[0]:.2f} "
        f"Medium={area_hi_by_class[1]:.2f} Large={area_hi_by_class[2]:.2f}"
    )
    stage2_thr = [float(t) for t in args.stage2_thresholds.split(",")]
    if len(stage2_thr) != 3:
        parser.error("--stage2_thresholds 는 쉼표로 구분된 3개 값이어야 합니다.")
    cc_min = [int(t) for t in args.cc_min_sizes.split(",")]
    if len(cc_min) != 3:
        parser.error("--cc_min_sizes 는 쉼표로 구분된 3개 정수여야 합니다.")
    try:
        erode_classes = {
            int(x.strip())
            for x in str(args.stage2_erode_classes).split(",")
            if x.strip() != ""
        }
    except ValueError:
        parser.error("--stage2_erode_classes 는 쉼표로 구분된 정수여야 합니다.")
    stage2_erode_px = max(0, int(args.stage2_erode_px))
    try:
        skip_classes = {
            int(x.strip())
            for x in str(args.stage3_skip_classes).split(",")
            if x.strip() != ""
        }
    except ValueError:
        parser.error("--stage3_skip_classes 는 쉼표로 구분된 정수여야 합니다.")
    if skip_classes:
        print(f"[stage3-skip] classes={sorted(skip_classes)} (Stage2 mask 유지)")
    if float(args.medium_active_max_area) > 0 or float(args.medium_skip_min_area) > 0:
        print(
            f"[medium-area] active_refine if rough_area < {float(args.medium_active_max_area):.0f} | "
            f"shrink if mid | skip Stage3 if rough_area >= {float(args.medium_skip_min_area):.0f} "
            f"(0=해당 구간 비활성)"
        )
    print(f"Stage 2 이진화 임계값: Small={stage2_thr[0]}, Medium={stage2_thr[1]}, Large={stage2_thr[2]}")
    print(f"CC filter min_size: Small={cc_min[0]}, Medium={cc_min[1]}, Large={cc_min[2]}")
    if stage2_erode_px > 0 and erode_classes:
        print(f"Stage2 erode: {stage2_erode_px}px classes={sorted(erode_classes)} (under-seg bias)")
    print(
        f"Eval mode: select_best={select_best}, monotonic_gate={apply_monotonic}, "
        f"enable_stop={enable_stop}"
        + (" [deploy_mode]" if args.deploy_mode else "")
    )

    from src.utils.device import require_cuda_device
    device = require_cuda_device()
    print(f"Device: {device}")
    
    patient_ids = None
    if args.split_role != "all":
        split = load_or_create_patient_split(args.train_root, args.max_patients, args.patient_split)
        patient_ids = split[args.split_role]
        print(f"Eval split: {args.split_role} ({len(patient_ids)} patients) from {args.patient_split}")

    dataset = BraTS2020Dataset(
        root_dir=args.train_root,
        modality=args.modality,
        target_size=128,
        max_patients=None if patient_ids is not None else args.max_patients,
        patient_ids=patient_ids,
        simulate_rough=False,
        num_workers=args.num_workers,
    )
    
    # Extract arrays
    images, gt_masks, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    
    print("Loading 3-Stage Pipeline Models...")
    img_in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=img_in_ch)
    
    # 3. Stage 3: SL Refiner and/or PPO
    print(f"Loading Stage3 refiners (mode={args.stage3_mode})...")
    CLASS_NAMES = {0: "Small", 1: "Medium", 2: "Large"}
    agents = {}
    agent_name_map = {
        "small": "ppo_small.zip",
        "medium": "ppo_medium.zip",
        "large": "ppo_large.zip",
    }

    missing_stage3 = []
    for mode, class_idx in zip(["small", "medium", "large"], [0, 1, 2]):
        agents[class_idx] = None
        if args.stage3_mode == "skip" or class_idx in skip_classes:
            if class_idx in skip_classes:
                print(f"Skipping Stage3 load for class {class_idx} ({mode})")
            continue
        agent_path = f"checkpoints/{agent_name_map[mode]}"
        if os.path.exists(agent_path):
            print(f"Loading PPO Agent: {agent_path}")
            agent = PPO.load(agent_path, device=device)
            _assert_agent_stop_compat(agent, enable_stop, agent_path)
            agents[class_idx] = agent
        else:
            missing_stage3.append(agent_path)

    if missing_stage3:
        raise FileNotFoundError(
            f"stage3_mode={args.stage3_mode} 인데 체크포인트가 없습니다: {missing_stage3}. "
            "Stage3를 학습하거나 --stage3_mode skip / --stage3_skip_classes 로 해당 클래스를 빼세요."
        )

    final_dsc_list = []
    final_hd95_list = []
    initial_dsc_list = []
    initial_hd95_list = []
    
    class_initial_dsc = {0: [], 1: [], 2: []}
    class_initial_hd95 = {0: [], 1: [], 2: []}
    class_initial_prec = {0: [], 1: [], 2: []}
    class_initial_rec = {0: [], 1: [], 2: []}
    class_final_dsc = {0: [], 1: [], 2: []}
    class_final_hd95 = {0: [], 1: [], 2: []}
    class_final_prec = {0: [], 1: [], 2: []}
    class_final_rec = {0: [], 1: [], 2: []}
    
    class_counts = {0:0, 1:0, 2:0}
    monotonic_reverts = 0
    all_candidates = []
    slice_pids = []
    slice_cls = []
    
    small_active_init, small_active_fin, small_active_init_hd, small_active_hd = [], [], [], []
    small_micro_init, small_micro_fin, small_micro_init_hd, small_micro_hd = [], [], [], []
    
    print("\nStarting Evaluation...")
    for i in range(len(images)):
        img_np = images_25d[i]
        gt_np = gt_masks[i]
        center_np = images[i]
        
        if img_np.ndim == 2:
            img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        else:
            img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
        
        # Stage 1: 분류기 라우팅 (기본). --oracle_routing 이면 GT 면적.
        if args.oracle_routing:
            area = np.sum(gt_np)
            if area < 300:
                true_c = 0
            elif area < 700:
                true_c = 1
            else:
                true_c = 2
            route_cls = torch.tensor([true_c], dtype=torch.long, device=device)
        else:
            route_cls = None
        
        # Stage 2
        with torch.no_grad():
            rough_mask_t, class_pred = pipeline(img_t, true_class_preds=route_cls)
            
        c = class_pred.item()
        if args.max_samples_per_class and class_counts[c] >= args.max_samples_per_class:
            continue
        class_counts[c] += 1
        
        rough_prob_np = rough_mask_t.squeeze().cpu().numpy()
        
        # ── Micro Fragment 임계값 완화 (Small 전용) ──
        # 클래스 임계값에서 조각이 지나치게 작아지면 소실을 막기 위해 임계값을
        # 단계적으로 낮춘다. 면적 하한(상수)만 보며 GT는 참조하지 않는다.
        rough_mask_np = (rough_prob_np > stage2_thr[c]).astype(np.float32)
        if c == 0 and np.sum(rough_mask_np) < args.micro_area_floor:
            for thr in np.arange(stage2_thr[c] - 0.05, args.micro_thr_floor - 1e-9, -0.05):
                cand_mask = (rough_prob_np > thr).astype(np.float32)
                rough_mask_np = cand_mask
                if np.sum(cand_mask) >= args.micro_area_floor:
                    break
        rough_mask_np = filter_small_components(rough_mask_np, cc_min[c])
        # Medium/Large: 1px erosion → 과소분할 편향 (외곽 FP 제거, Recall↓)
        if stage2_erode_px > 0 and c in erode_classes and np.sum(rough_mask_np) > 0:
            from scipy.ndimage import binary_erosion
            eroded = rough_mask_np > 0.5
            for _ in range(stage2_erode_px):
                eroded = binary_erosion(eroded, iterations=1)
            if np.any(eroded):
                rough_mask_np = eroded.astype(np.float32)

        init_dsc = dice(rough_mask_np, gt_np)
        init_hd95 = hd95(rough_mask_np, gt_np)
        initial_dsc_list.append(init_dsc)
        initial_hd95_list.append(init_hd95)
        class_initial_dsc[c].append(init_dsc)
        class_initial_hd95[c].append(init_hd95)
        class_initial_prec[c].append(precision(rough_mask_np, gt_np))
        class_initial_rec[c].append(recall(rough_mask_np, gt_np))

        slice_pids.append(dataset._sample_pids[i])
        slice_cls.append(c)
        
        # Stage 3: Component-wise Independent Refinement
        from scipy.ndimage import label as sp_label
        lbl, num_feats = sp_label(rough_mask_np > 0.2)
        
        valid_comp_indices = [k for k in range(1, num_feats + 1) if np.sum(lbl == k) >= 5]
        
        final_mask_np = np.zeros_like(rough_mask_np)
        for k in range(1, num_feats + 1):
            if k not in valid_comp_indices:
                final_mask_np = np.maximum(final_mask_np, (lbl == k).astype(np.float32))

        if args.fast_mode:
            prob_tta_np = rough_prob_np
        else:
            prob_tta_np = _tta_probability(pipeline, img_t, rough_mask_t, class_pred)
                
        # 유효 컴포넌트들을 각각 독립적으로 보정하여 합산 (분리된 종양들의 독립 미세 조정 지원)
        # Stage3 시작 마스크 = Stage2 컴포넌트(non-TTA). Init DSC와 동일한 기준이라
        # Final−Init Δ에 TTA 이득이 섞이지 않는다. TTA는 soft probability 채널에만 사용.
        for k in valid_comp_indices:
            comp_mask_k = (lbl == k).astype(np.float32)
            comp_area = float(np.sum(comp_mask_k))

            # Stage3 라우팅: 슬라이스 분류기 클래스 (학습 class_filter=classifier와 정합)
            ck = int(c)
            agent_k = agents[ck]
            ref_mode_k = {0: "small", 1: "medium", 2: "large"}[ck]

            if ck in skip_classes or agent_k is None:
                final_mask_np = np.maximum(final_mask_np, comp_mask_k)
                continue

            # Medium: rough 성분 면적 기반 분기 (GT 미사용, deploy 가능)
            med_active = float(args.medium_active_max_area)
            med_skip = float(args.medium_skip_min_area)
            if ck == 1 and med_skip > 0 and comp_area >= med_skip:
                final_mask_np = np.maximum(final_mask_np, comp_mask_k)
                continue

            struct_k = np.ones((3, 3))
            dilate_iter = 2 if ck == 0 else 3
            comp_dilated = binary_dilation(comp_mask_k, struct_k, iterations=dilate_iter)
            comp_init = comp_mask_k.copy()
            soft_prob = (prob_tta_np * comp_dilated).astype(np.float32)
            if float(np.sum(soft_prob)) == 0.0:
                soft_prob = (rough_prob_np * comp_dilated).astype(np.float32)

            if args.confidence_threshold is not None:
                nz = comp_init > 0.5
                mean_p = float(np.mean(prob_tta_np[nz])) if np.any(nz) else 0.0
                if mean_p >= args.confidence_threshold:
                    final_mask_np = np.maximum(final_mask_np, comp_init)
                    continue

            # ================================================================
            # TEGDA: Test-time Evaluation-Guided Dynamic Adaptation
            # ================================================================

            # --- Gate 1: Boundary-region Entropy (Jungo et al., MICCAI) ---
            # 전체 마스크 대신 ±band_px morphological boundary band에서만 entropy 계산
            # → 경계에서 불확실한 경우에만 PPO 적용 (내부 안정 영역 터치 방지)
            from scipy.ndimage import binary_dilation as _sp_dil, binary_erosion as _sp_ero
            _comp_b = comp_init > 0.5
            _band_px = {0: 2, 1: 2, 2: 2}[ck]
            if _comp_b.any():
                _dil = _sp_dil(_comp_b, iterations=_band_px)
                _ero_iter = _band_px if _comp_b.sum() > (_band_px * 2 + 1) ** 2 else 1
                _ero = _sp_ero(_comp_b, iterations=_ero_iter)
                _boundary_band_mask = np.logical_xor(_dil, _ero)
            else:
                _boundary_band_mask = np.zeros_like(_comp_b)

            _p_raw = soft_prob.astype(np.float64)
            _p = np.clip(_p_raw, 1e-7, 1.0 - 1e-7)
            _entropy_map = -_p * np.log2(_p) - (1.0 - _p) * np.log2(1.0 - _p)
            # 경계 band 내 픽셀만 사용 (없으면 soft prob 경계 영역으로 fallback)
            _gate_region = _boundary_band_mask if _boundary_band_mask.any() else ((_p_raw > 0.1) & (_p_raw < 0.9))
            mean_entropy = float(np.mean(_entropy_map[_gate_region])) if _gate_region.any() else 0.0

            # 클래스별 entropy 임계: Small은 작아 불확실성이 더 큼 → 낮게, Medium/Large는 높게
            _ent_thr_map = {0: float(args.entropy_threshold) * 0.85,
                           1: float(args.entropy_threshold),
                           2: float(args.entropy_threshold) * 1.1}
            ENTROPY_THRESHOLD = _ent_thr_map[ck]

            if mean_entropy < ENTROPY_THRESHOLD:
                # Gate 1 Reject: 경계 확신 → PPO 생략, Stage 2 유지
                refined_k_mask = comp_init.copy()
            else:
                # Gate 1 Accept: 경계 불확실 → PPO 실행
                from src.envs.zoom_ppo_refine import refine_zoom_ppo
                is_gt_free = not args.gt_upper_bound  # deploy_mode/기본 평가는 GT-free=True!
                refined_k_mask = refine_zoom_ppo(
                    agent_k,
                    images[i],
                    gt_masks[i],
                    comp_init,
                    soft_prob,
                    ref_mode_k,
                    device=str(device),
                    enable_stop=enable_stop,
                    seed=(i * 1009 + k),
                    gt_free=is_gt_free,     # deploy: last-mask, gt_upper_bound: oracle best
                    use_action_tta=bool(args.ppo_action_tta),
                )

                if args.gt_upper_bound:
                    # [오프라인 상한 전용] GT 경계 DSC 및 오라클 게이트
                    _gt_b = gt_np > 0.5
                    if _gt_b.any():
                        _gt_dil = _sp_dil(_gt_b, iterations=_band_px)
                        _gt_ero = _sp_ero(_gt_b, iterations=1) if _gt_b.sum() > 4 else np.zeros_like(_gt_b)
                        _gt_band = np.logical_xor(_gt_dil, _gt_ero).astype(np.float32)
                    else:
                        _gt_band = np.ones_like(gt_np, dtype=np.float32)
                    
                    init_bdsc = _boundary_dsc_gt(comp_init, _gt_band, gt_np)
                    ppo_bdsc = _boundary_dsc_gt(refined_k_mask, _gt_band, gt_np)
                    gate2a_pass = (_shape_energy(refined_k_mask) <= _shape_energy(comp_init) * 1.05)
                    gate2b_pass = (ppo_bdsc >= init_bdsc - 0.005)
                    gate2c_pass = True
                else:
                    # [배포 전용 GT-free 게이트 Master Recipe]
                    # Gate 2a: Shape Energy (Compactness 1.05x)
                    gate2a_pass = (_shape_energy(refined_k_mask) <= _shape_energy(comp_init) * 1.05)
                    # Gate 2b: Soft-Probability Confidence Consistency
                    gate2b_pass = _soft_prob_consistency(refined_k_mask, soft_prob)
                    # Gate 2c: Boundary SDF Displacement 95th Percentile Proxy
                    max_disp_allowed = max(3.0, float(_band_px) * 1.8)
                    gate2c_pass = (_sdf_displacement_95(comp_init, refined_k_mask) <= max_disp_allowed)
                    # Gate 2d: Asymmetric Expansion Control (과도 팽창 하락 억제 1.03x)
                    init_area = float(np.sum(comp_init > 0.5))
                    ref_area = float(np.sum(refined_k_mask > 0.5))
                    gate2d_pass = True
                    if init_area > 0 and ref_area > init_area * 1.03:
                        # 3% 초과 팽창 시 soft prob 0.70 미만 영역으로 과도 이탈하면 reject
                        low_prob_expansion = np.logical_and(refined_k_mask > 0.5, soft_prob < 0.70).sum()
                        if low_prob_expansion > init_area * 0.02:
                            gate2d_pass = False

                if not (gate2a_pass and gate2b_pass and gate2c_pass and gate2d_pass):
                    # 어느 하나라도 실패 → Stage 2 마스크 유지 (100% 하한 방어)
                    refined_k_mask = comp_init.copy()

            if np.sum(refined_k_mask) > 0:
                # Deep Snake-style Contour Anti-Aliasing (1px 톱니 노이즈 제어)
                closed = binary_closing(refined_k_mask, struct_k).astype(np.float32)
                refined_k_mask = closed
            if np.sum(refined_k_mask) == 0 or not _gt_free_accept(
                comp_init, refined_k_mask, lo=area_lo, hi=area_hi_by_class[int(c)]
            ):
                refined_k_mask = comp_init
            if apply_monotonic:
                refined_k_mask = apply_monotonic_dsc_gate(comp_init, refined_k_mask, gt_np)
            final_mask_np = np.maximum(final_mask_np, refined_k_mask)

        # 전체 슬라이스 레벨 면적 게이트 (컴포넌트 합산 시 극단 이탈만 유연하게 보정)
        if not _gt_free_accept(
            rough_mask_np, final_mask_np, lo=min(area_lo, 0.70), hi=max(area_hi_by_class[int(c)], 1.50)
        ):
            final_mask_np = rough_mask_np

        if apply_monotonic:
            gated_mask = apply_monotonic_dsc_gate(rough_mask_np, final_mask_np, gt_np)
            if not np.array_equal(gated_mask, final_mask_np):
                monotonic_reverts += 1
            final_mask_np = gated_mask

        # ---------------------------------------------------------
        # 단계 4. 추론 후단 Active Contour 및 형태학적(CC) 정제
        # ---------------------------------------------------------
        if args.active_contour and final_mask_np.max() > 0:
            final_mask_np = morphological_chan_vese(
                center_np[0] if center_np.ndim == 3 else center_np,
                iterations=5,
                init_level_set=final_mask_np.copy(),
                smoothing=1,
            )
            final_mask_np = (final_mask_np > 0).astype(np.float32)
            
        # CC 찌꺼기 필터링 (가장 큰 종양과 연관없는 파편 제거)
        final_mask_np = filter_small_components(final_mask_np, cc_min[c])
        # ---------------------------------------------------------

        fin_dsc = dice(final_mask_np, gt_np)
        fin_hd95 = hd95(final_mask_np, gt_np)

        final_dsc_list.append(fin_dsc)
        final_hd95_list.append(fin_hd95)
        
        class_final_dsc[c].append(fin_dsc)
        class_final_hd95[c].append(fin_hd95)
        class_final_prec[c].append(precision(final_mask_np, gt_np))
        class_final_rec[c].append(recall(final_mask_np, gt_np))
        
        # Save representative samples per class for visualization later
        all_candidates.append({
            "class": c,
            "img": center_np[0] if center_np.ndim == 3 else center_np,
            "gt": gt_np,
            "rough": rough_mask_np,
            "final": final_mask_np,
            "init_dsc": init_dsc,
            "fin_dsc": fin_dsc,
            "delta_dsc": fin_dsc - init_dsc,
        })

        if c == 0:
            gt_area = np.sum(gt_np)
            if gt_area >= 50:
                small_active_init.append(init_dsc)
                small_active_fin.append(fin_dsc)
                small_active_init_hd.append(init_hd95)
                small_active_hd.append(fin_hd95)
            else:
                small_micro_init.append(init_dsc)
                small_micro_fin.append(fin_dsc)
                small_micro_init_hd.append(init_hd95)
                small_micro_hd.append(fin_hd95)

        if (i + 1) % 25 == 0 or (i + 1) == len(images):
            cur_init_dsc = float(np.mean(initial_dsc_list))
            cur_fin_dsc = float(np.mean(final_dsc_list))
            delta = cur_fin_dsc - cur_init_dsc
            pct = ((i + 1) / len(images)) * 100.0
            print(f"[Stage 3 Gated Eval] Progress: {i+1:5d}/{len(images)} ({pct:5.1f}%) | "
                  f"Class={CLASS_NAMES.get(int(c), str(c)):<6} | "
                  f"Rough DSC: {cur_init_dsc:.4f} → Gated PPO DSC: {cur_fin_dsc:.4f} (Δ={delta:+.4f})", flush=True)
            
    mode_tag = "deploy (last mask, area gate only)" if args.deploy_mode else (
        f"select_best={select_best}, monotonic={apply_monotonic}"
    )
    print("\n--- Pipeline Evaluation ({routing}, Stage3={s3}, {mode}) ---".format(
        routing="oracle routing" if args.oracle_routing else "classifier routing",
        s3=args.stage3_mode,
        mode=mode_tag,
    ))
    print(f"Total Slices Evaluated: {len(final_dsc_list)}")
    if skip_classes:
        print(f"Stage3 skipped classes: {sorted(skip_classes)} → Final≡Init for those slices")
    if apply_monotonic:
        print(f"Monotonic DSC reverts (final < initial → keep Stage 2): {monotonic_reverts}")
    else:
        print("Monotonic DSC gate: OFF (deploy — not a performance guarantee)")
    print(f"Class Distribution: Small: {class_counts[0]}, Medium: {class_counts[1]}, Large: {class_counts[2]}")
    print(f"Average Initial DSC  (Stage 2):        {np.mean(initial_dsc_list):.4f}")
    print(f"Average Final   DSC  (Stage 3*):       {np.mean(final_dsc_list):.4f}")
    print(f"Average Initial HD95 (px):             {np.mean(initial_hd95_list):.4f}")
    print(f"Average Final   HD95 (px):             {np.mean(final_hd95_list):.4f}")
    # Stage3가 실제로 돌아가는 클래스만 (skip 시 Large Δ=0이 전체 평균을 왜곡하지 않게)
    active_idx = [i for i, c in enumerate(slice_cls) if int(c) not in skip_classes]
    if active_idx and skip_classes:
        act_init = np.mean([initial_dsc_list[i] for i in active_idx])
        act_fin = np.mean([final_dsc_list[i] for i in active_idx])
        act_ih = np.mean([initial_hd95_list[i] for i in active_idx])
        act_fh = np.mean([final_hd95_list[i] for i in active_idx])
        print(
            f"Stage3-active only (excl. skip {sorted(skip_classes)}): "
            f"n={len(active_idx)} | DSC {act_init:.4f} → {act_fin:.4f} "
            f"| HD95 {act_ih:.4f} → {act_fh:.4f}"
        )
    
    print("\n--- Class-wise Performance Breakdown ---")
    names = {0: "Small (CaraNet)", 1: "Medium (UNet++)", 2: "Large (SegResNet)"}
    for c in [0, 1, 2]:
        if len(class_final_dsc[c]) > 0:
            init_dsc_avg = np.mean(class_initial_dsc[c])
            fin_dsc_avg  = np.mean(class_final_dsc[c])
            init_hd_avg  = np.mean(class_initial_hd95[c])
            fin_hd_avg   = np.mean(class_final_hd95[c])
            skip_tag = " [Stage3 SKIP→Init]" if c in skip_classes else ""
            print(f"[{names[c]}]{skip_tag} count: {len(class_final_dsc[c])} "
                  f"| DSC {init_dsc_avg:.4f} → {fin_dsc_avg:.4f} "
                  f"| HD95 {init_hd_avg:.4f} → {fin_hd_avg:.4f} (px)")
            print(
                f"    └ Precision: {np.mean(class_initial_prec[c]):.4f} → {np.mean(class_final_prec[c]):.4f} "
                f"| Recall: {np.mean(class_initial_rec[c]):.4f} → {np.mean(class_final_rec[c]):.4f}"
            )

    if small_active_init:
        print(
            f"[Small Active ≥50px] n={len(small_active_init)} "
            f"| DSC {np.mean(small_active_init):.4f} → {np.mean(small_active_fin):.4f} "
            f"| HD95 {np.mean(small_active_init_hd):.4f} → {np.mean(small_active_hd):.4f}"
        )
    if small_micro_init:
        print(
            f"[Small Micro <50px] n={len(small_micro_init)} "
            f"| DSC {np.mean(small_micro_init):.4f} → {np.mean(small_micro_fin):.4f} "
            f"| HD95 {np.mean(small_micro_init_hd):.4f} → {np.mean(small_micro_hd):.4f}"
        )

    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    np.savez_compressed(
        args.metrics_out,
        pid=np.array(slice_pids),
        cls=np.array(slice_cls, dtype=np.int8),
        init_dsc=np.array(initial_dsc_list, dtype=np.float64),
        final_dsc=np.array(final_dsc_list, dtype=np.float64),
        init_hd95=np.array(initial_hd95_list, dtype=np.float64),
        final_hd95=np.array(final_hd95_list, dtype=np.float64),
        deploy_mode=np.array([int(args.deploy_mode)]),
        select_best=np.array([int(select_best)]),
        monotonic_gate=np.array([int(apply_monotonic)]),
        stage3_skip_classes=np.array(sorted(skip_classes), dtype=np.int8),
        area_gate_lo=np.array([area_lo]),
        area_gate_hi=np.array([area_hi]),
    )
    print(f"Saved slice metrics: {args.metrics_out}")
    try:
        import importlib.util
        stats_path = os.path.join(os.path.dirname(__file__), "paired_stats.py")
        spec = importlib.util.spec_from_file_location("paired_stats", stats_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.report_from_npz(args.metrics_out)
    except Exception as exc:
        print(f"[stats] skipped: {exc}")

    if not args.skip_plot:
        pipeline_samples = _select_pipeline_samples(all_candidates, class_final_dsc, n=2)
        _plot_pipeline_results(pipeline_samples, output_dir="results")


def _select_pipeline_samples(all_candidates, class_final_dsc, n=2):
    """클래스 평균 Final DSC에 가깝고, PPO가 실제로 올린 슬라이스의 원본을 고른다."""
    names = {0: "Small", 1: "Medium", 2: "Large"}
    pipeline_samples = {}
    for c in [0, 1, 2]:
        class_candidates = [s for s in all_candidates if s["class"] == c]
        if not class_candidates:
            pipeline_samples[c] = []
            continue
        mean_fin = float(np.mean(class_final_dsc[c])) if class_final_dsc[c] else float(
            np.mean([s["fin_dsc"] for s in class_candidates])
        )
        # 클래스 평균 Final에 가까운 표본 (Δ>0 강제 금지: 낙관적 체리피킹)
        pool = sorted(class_candidates, key=lambda s: abs(s["fin_dsc"] - mean_fin))
        selected = pool[:n]
        pipeline_samples[c] = selected
        print(
            f"[Viz] {names[c]}: mean Final DSC={mean_fin:.4f}, "
            f"pool=all n={len(class_candidates)}, "
            f"selected Final={[round(s['fin_dsc'], 4) for s in selected]}"
        )
    return pipeline_samples


def _crop_box(gt: np.ndarray, margin: int = 15):
    y_indices, x_indices = np.where(gt > 0.5)
    if len(y_indices) > 0:
        ymin, ymax = y_indices.min(), y_indices.max()
        xmin, xmax = x_indices.min(), x_indices.max()
        ymin = max(0, ymin - margin)
        ymax = min(gt.shape[0] - 1, ymax + margin)
        xmin = max(0, xmin - margin)
        xmax = min(gt.shape[1] - 1, xmax + margin)
    else:
        ymin, ymax = 0, gt.shape[0] - 1
        xmin, xmax = 0, gt.shape[1] - 1
    return xmin, xmax, ymin, ymax


def _save_original(s: dict, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = s["img"]
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[Viz] original saved {save_path}")


def _plot_one_sample(s: dict, title: str, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = s["img"]
    gt = s["gt"]
    rough = s["rough"]
    final = s["final"]
    init_dsc = s["init_dsc"]
    fin_dsc = s["fin_dsc"]
    delta_dsc = s.get("delta_dsc", fin_dsc - init_dsc)
    xmin, xmax, ymin, ymax = _crop_box(gt)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].contour(gt, levels=[0.5], colors="lime", linewidths=2.0)
    axes[0].set_title("Original + GT", fontsize=11)
    axes[0].set_xlabel("GT", fontsize=11, fontweight="bold", color="lime")

    axes[1].imshow(img, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*rough.shape, 4))
    overlay[rough > 0.5] = [1, 0, 0, 0.4]
    axes[1].imshow(overlay)
    axes[1].contour(gt, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
    axes[1].set_title("Rough Mask", fontsize=11)
    axes[1].set_xlabel(f"Rough DSC={init_dsc:.3f}", fontsize=11, fontweight="bold")

    axes[2].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[2].contour(final, levels=[0.5], colors="cyan", linewidths=2.0)
    axes[2].contour(gt, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
    axes[2].set_title("RL Refined", fontsize=11)
    axes[2].set_xlabel(f"RL DSC={fin_dsc:.3f}  (Δ={delta_dsc:+.3f})", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] saved {save_path}  (Rough={init_dsc:.4f} → RL={fin_dsc:.4f}, Δ={delta_dsc:+.4f})")


def _plot_pipeline_results(pipeline_samples: dict, output_dir: str = "results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    class_names = {0: "small", 1: "medium", 2: "large"}
    class_titles = {0: "Small (CaraNet)", 1: "Medium (UNet++)", 2: "Large (SegResNet)"}

    sample_list = []
    class_labels = []
    saved_files = []

    for c in [0, 1, 2]:
        samples = pipeline_samples.get(c, [])[:2]
        for idx, s in enumerate(samples):
            title = f"{class_titles[c]}  Sample {idx + 1}"
            fname = f"pipeline_sample_{class_names[c]}_{idx + 1}.png"
            save_path = os.path.join(output_dir, fname)
            _plot_one_sample(s, title, save_path)
            orig_path = os.path.join(output_dir, f"pipeline_sample_{class_names[c]}_{idx + 1}_original.png")
            _save_original(s, orig_path)
            saved_files.append(save_path)
            saved_files.append(orig_path)
            sample_list.append(s)
            class_labels.append(f"{class_titles[c]}\nSample {idx + 1}")

    if not sample_list:
        print("[Viz] no samples to plot")
        return

    n_cols = len(sample_list)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 9.5))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, s in enumerate(sample_list):
        img = s["img"]
        gt = s["gt"]
        rough = s["rough"]
        final = s["final"]
        init_dsc = s["init_dsc"]
        fin_dsc = s["fin_dsc"]
        delta_dsc = s.get("delta_dsc", fin_dsc - init_dsc)
        xmin, xmax, ymin, ymax = _crop_box(gt)

        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].contour(gt, levels=[0.5], colors="lime", linewidths=2.0)
        axes[0, col].set_title(class_labels[col], fontsize=13, fontweight="bold", pad=8)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        axes[1, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*rough.shape, 4))
        overlay[rough > 0.5] = [1, 0, 0, 0.4]
        axes[1, col].imshow(overlay)
        axes[1, col].contour(gt, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
        axes[1, col].set_xlabel(f"Rough DSC={init_dsc:.3f}", fontsize=12, fontweight="bold")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        axes[2, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[2, col].contour(final, levels=[0.5], colors="cyan", linewidths=2.0)
        axes[2, col].contour(gt, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
        axes[2, col].set_xlabel(f"RL DSC={fin_dsc:.3f}  (Δ={delta_dsc:+.3f})", fontsize=12, fontweight="bold")
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

        for row_idx in range(3):
            axes[row_idx, col].set_xlim(xmin, xmax)
            axes[row_idx, col].set_ylim(ymax, ymin)

    fig.text(0.01, 0.78, "MRI + GT", va="center", rotation="vertical", fontsize=14, fontweight="bold", color="lime")
    fig.text(0.01, 0.50, "Stage 2 Rough", va="center", rotation="vertical", fontsize=14, fontweight="bold", color="red")
    fig.text(0.01, 0.22, "Stage 3 RL Refined", va="center", rotation="vertical", fontsize=14, fontweight="bold", color="cyan")

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    overview_path = os.path.join(output_dir, "pipeline_sample_comparison.png")
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] overview saved {overview_path}")
    print(f"[Viz] class-wise images ({len(saved_files)}/6): {saved_files}")


if __name__ == "__main__":
    main()
