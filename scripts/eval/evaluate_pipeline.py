import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import os
import torch
import numpy as np
from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import load_or_create_patient_split, DEFAULT_SPLIT_PATH
from src.models.dynamic_router import AdaptivePipeline
from src.utils.metrics import apply_monotonic_dsc_gate, dice, filter_small_components, hd95, precision, recall
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


def _gt_free_accept(rough: np.ndarray, refined: np.ndarray) -> bool:
    if float(np.sum(refined)) < 1.0:
        return False
    r = max(1.0, float(np.sum(rough)))
    f = float(np.sum(refined))
    return (0.2 * r) <= f <= (4.0 * r)


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
        raise RuntimeError(
            f"{name}: 체크포인트는 STOP 정책인데 --enable_stop 이 꺼져 있습니다. "
            "배포 평가에는 --enable_stop --deploy_mode 를 쓰세요."
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
    parser.add_argument("--max_patients", type=int, default=210, help="평가 풀 환자 수 (split 생성 기준)")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="클래스당 최대 샘플 수 (None이면 제한 없음)")
    parser.add_argument("--patient_split", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split_role", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--oracle_routing", action="store_true", help="GT 면적으로 Expert를 고르는 상한 평가")
    parser.add_argument("--confidence_threshold", type=float, default=None, help="이 값 이상 평균 확률이면 PPO 생략")
    parser.add_argument("--stage2_thresholds", type=str, default="0.80,0.80,0.50",
                        help="클래스별(Small,Medium,Large) Stage 2 이진화 임계값. Large는 과소분할이라 0.50.")
    parser.add_argument("--skip_ppo", action="store_true", help="Stage 3 생략 (Stage 2 단독 베이스라인 측정)")
    parser.add_argument(
        "--stage3_mode",
        type=str,
        default="sl",
        choices=["sl", "ppo", "skip"],
        help="Stage3: sl=교대학습 SL refiner(기본), ppo=기존 PPO, skip=생략",
    )
    parser.add_argument("--micro_area_floor", type=float, default=80.0,
                        help="Small 마스크 면적이 이 값 미만이면 임계값을 단계적으로 낮춘다.")
    parser.add_argument("--micro_thr_floor", type=float, default=0.15,
                        help="마이크로 조각 임계값 완화의 하한.")
    parser.add_argument("--cc_min_sizes", type=str, default="0,15,25",
                        help="클래스별(Small,Medium,Large) 연결요소 최소 픽셀. 0이면 비활성.")
    parser.add_argument("--skip_plot", action="store_true", help="시각화 PNG 생략")
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="results/pipeline_slice_metrics.npz",
        help="슬라이스별 Init/Final 지표 저장 경로 (Wilcoxon·CI용)",
    )
    parser.add_argument(
        "--deploy_mode",
        action="store_true",
        default=True,
        help="배포형 평가(기본): 마지막(또는 STOP) 마스크 + 면적 게이트만",
    )
    parser.add_argument(
        "--gt_upper_bound",
        action="store_true",
        help="논문 금지용 상한: GT best-of-N + 단조 DSC 게이트 (배포 아님)",
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
        "--enable_stop",
        action="store_true",
        help="STOP 행동으로 학습된 PPO 체크포인트 평가",
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
        print("[gt_upper_bound] best-of-N + 단조 게이트 ON (논문 메인 수치로 쓰지 말 것)")
    elif args.deploy_mode:
        args.last_mask = True
        args.no_monotonic_gate = True
        # PPO 모드일 때만 STOP 강제
        if args.stage3_mode == "ppo" and (not args.enable_stop):
            args.enable_stop = True
            print("[deploy_mode] --enable_stop 자동 활성화")

    select_best = not args.last_mask
    apply_monotonic = not args.no_monotonic_gate
    enable_stop = bool(args.enable_stop)
    stage2_thr = [float(t) for t in args.stage2_thresholds.split(",")]
    if len(stage2_thr) != 3:
        parser.error("--stage2_thresholds 는 쉼표로 구분된 3개 값이어야 합니다.")
    cc_min = [int(t) for t in args.cc_min_sizes.split(",")]
    if len(cc_min) != 3:
        parser.error("--cc_min_sizes 는 쉼표로 구분된 3개 정수여야 합니다.")
    print(f"Stage 2 이진화 임계값: Small={stage2_thr[0]}, Medium={stage2_thr[1]}, Large={stage2_thr[2]}")
    print(f"CC filter min_size: Small={cc_min[0]}, Medium={cc_min[1]}, Large={cc_min[2]}")
    print(
        f"Eval mode: select_best={select_best}, monotonic_gate={apply_monotonic}, "
        f"enable_stop={enable_stop}"
        + (" [deploy_mode]" if args.deploy_mode else "")
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    )
    
    # Extract arrays
    images, gt_masks, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    
    print("Loading 3-Stage Pipeline Models...")
    img_in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=img_in_ch)
    
    # 3. Stage 3: SL Refiner and/or PPO
    print(f"Loading Stage3 refiners (mode={args.stage3_mode})...")
    agents = {}
    sl_nets = {}
    agent_name_map = {
        "small": "ppo_small.zip",
        "medium": "ppo_medium.zip",
        "large": "ppo_large.zip",
    }
    sl_name_map = {
        "small": "sl_refiner_small.pt",
        "medium": "sl_refiner_medium.pt",
        "large": "sl_refiner_large.pt",
    }
    from src.models.sl_refiner import DualHeadRefiner, apply_sl_refiner

    missing_stage3 = []
    for mode, class_idx in zip(["small", "medium", "large"], [0, 1, 2]):
        agents[class_idx] = None
        sl_nets[class_idx] = None
        if args.stage3_mode == "skip":
            continue
        if args.stage3_mode == "sl":
            sl_path = f"checkpoints/{sl_name_map[mode]}"
            if os.path.exists(sl_path):
                print(f"Loading SL Refiner: {sl_path}")
                ckpt = torch.load(sl_path, map_location=device, weights_only=False)
                sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                if isinstance(ckpt, dict) and "in_ch" in ckpt:
                    sl_in_ch = int(ckpt["in_ch"])
                elif isinstance(sd, dict) and "enc.0.weight" in sd:
                    sl_in_ch = int(sd["enc.0.weight"].shape[1])
                else:
                    sl_in_ch = img_in_ch + 2
                net = DualHeadRefiner(in_ch=sl_in_ch).to(device)
                net.load_state_dict(sd)
                net.eval()
                sl_nets[class_idx] = net
            else:
                missing_stage3.append(sl_path)
        elif args.stage3_mode == "ppo":
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
            "Stage3를 학습하거나 --stage3_mode skip 을 쓰세요."
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

        prob_tta_np = _tta_probability(pipeline, img_t, rough_mask_t, class_pred)
                
        # 유효 컴포넌트들을 각각 독립적으로 보정하여 합산 (분리된 종양들의 독립 미세 조정 지원)
        for k in valid_comp_indices:
            comp_mask_k = (lbl == k).astype(np.float32)
            comp_area = float(np.sum(comp_mask_k))
            
            # Stage3 라우팅: 슬라이스 분류기 클래스와 동일 (학습 GT-class / Expert와 정합)
            # 컴포넌트는 독립 보정하되, Refiner는 슬라이스 클래스 ck=c 를 쓴다.
            ck = int(c)
            agent_k = agents[ck]
            sl_k = sl_nets[ck]
            ref_mode_k = {0: "small", 1: "medium", 2: "large"}[ck]
            
            if agent_k is None and sl_k is None:
                final_mask_np = np.maximum(final_mask_np, comp_mask_k)
                continue

            struct_k = np.ones((3, 3))
            dilate_iter = 2 if ck == 0 else 3
            comp_dilated = binary_dilation(comp_mask_k, struct_k, iterations=dilate_iter)
            is_micro = (ck == 0 and comp_area < 50)
            # TTA 재이진화는 Stage 2와 같은 슬라이스 클래스 임계값을 쓴다.
            tta_thr = 0.30 if is_micro else stage2_thr[c]
            comp_from_tta = (prob_tta_np > tta_thr).astype(np.float32) * comp_dilated
            if np.sum(comp_from_tta) == 0:
                comp_from_tta = comp_mask_k.copy()

            if args.confidence_threshold is not None:
                nz = comp_from_tta > 0.5
                mean_p = float(np.mean(prob_tta_np[nz])) if np.any(nz) else 0.0
                if mean_p >= args.confidence_threshold:
                    final_mask_np = np.maximum(final_mask_np, comp_from_tta)
                    continue

            if sl_k is not None:
                img_2d = images[i]
                refined_k_mask = apply_sl_refiner(
                    sl_k,
                    img_2d,
                    comp_from_tta,
                    (prob_tta_np * comp_from_tta).astype(np.float32),
                    device,
                    morph_small=(ck == 0),
                )
            else:
                refined_k_mask = _refine_with_ppo(
                    agent_k,
                    images[i],
                    gt_masks[i],
                    comp_from_tta,
                    prob_tta_np * comp_from_tta,
                    ref_mode_k,
                    n_steps=15,
                    clip_shrink=(ck == 0),
                    select_best=select_best,
                    apply_monotonic=apply_monotonic,
                    enable_stop=enable_stop,
                )
            if np.sum(refined_k_mask) > 0:
                refined_k_mask = binary_closing(refined_k_mask, struct_k).astype(np.float32)
            if np.sum(refined_k_mask) == 0 or not _gt_free_accept(comp_from_tta, refined_k_mask):
                refined_k_mask = comp_from_tta
            if apply_monotonic:
                refined_k_mask = apply_monotonic_dsc_gate(comp_from_tta, refined_k_mask, gt_np)
            final_mask_np = np.maximum(final_mask_np, refined_k_mask)

        if not _gt_free_accept(rough_mask_np, final_mask_np):
            final_mask_np = rough_mask_np

        if apply_monotonic:
            gated_mask = apply_monotonic_dsc_gate(rough_mask_np, final_mask_np, gt_np)
            if not np.array_equal(gated_mask, final_mask_np):
                monotonic_reverts += 1
            final_mask_np = gated_mask

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

        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(images)} slices...")
            
    mode_tag = "deploy (last mask, area gate only)" if args.deploy_mode else (
        f"select_best={select_best}, monotonic={apply_monotonic}"
    )
    print("\n--- Pipeline Evaluation ({routing}, PPO, {mode}) ---".format(
        routing="oracle routing" if args.oracle_routing else "classifier routing",
        mode=mode_tag,
    ))
    print(f"Total Slices Evaluated: {len(final_dsc_list)}")
    if apply_monotonic:
        print(f"Monotonic DSC reverts (final < initial → keep Stage 2): {monotonic_reverts}")
    else:
        print("Monotonic DSC gate: OFF")
    print(f"Class Distribution: Small: {class_counts[0]}, Medium: {class_counts[1]}, Large: {class_counts[2]}")
    print(f"Average Initial DSC  (Stage 2):        {np.mean(initial_dsc_list):.4f}")
    print(f"Average Final   DSC  (Stage 3 RL):     {np.mean(final_dsc_list):.4f}")
    print(f"Average Initial HD95 (px):             {np.mean(initial_hd95_list):.4f}")
    print(f"Average Final   HD95 (px):             {np.mean(final_hd95_list):.4f}")
    
    print("\n--- Class-wise Performance Breakdown ---")
    names = {0: "Small (CaraNet)", 1: "Medium (UNet++)", 2: "Large (SegResNet)"}
    for c in [0, 1, 2]:
        if len(class_final_dsc[c]) > 0:
            init_dsc_avg = np.mean(class_initial_dsc[c])
            fin_dsc_avg  = np.mean(class_final_dsc[c])
            init_hd_avg  = np.mean(class_initial_hd95[c])
            fin_hd_avg   = np.mean(class_final_hd95[c])
            print(f"[{names[c]}] count: {len(class_final_dsc[c])} "
                  f"| DSC {init_dsc_avg:.4f} → {fin_dsc_avg:.4f} "
                  f"| HD95 {init_hd_avg:.4f} → {fin_hd_avg:.4f} (px)")
            print(
                f"    └ Precision: {np.mean(class_initial_prec[c]):.4f} → {np.mean(class_final_prec[c]):.4f} "
                f"| Recall: {np.mean(class_initial_rec[c]):.4f} → {np.mean(class_final_rec[c]):.4f} "
                f"(P<R: 과분할 / P>R: 과소분할)"
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
