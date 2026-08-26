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


def _edge_alignment_score(image: np.ndarray, mask: np.ndarray) -> float:
    """
    정답(GT) 없이 MRI 밝기 경계선(Sobel Gradient)과 마스크 외곽선 간의 물리적 일치도를 측정하는 비지도 점수
    """
    if float(np.sum(mask)) < 1.0:
        return 0.0
    
    # 2D 슬라이스 추출 (T1ce 또는 FLAIR 평균)
    if image.ndim == 3:
        img_slice = np.mean(image, axis=0).astype(np.float32)
    else:
        img_slice = image.astype(np.float32)
        
    from scipy.ndimage import sobel
    gx = sobel(img_slice, axis=0)
    gy = sobel(img_slice, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    max_g = np.max(grad_mag)
    if max_g > 1e-6:
        grad_mag /= max_g
        
    # 마스크 외곽선(Boundary) 추출 (1-pixel band)
    struct = np.ones((3, 3), dtype=bool)
    mask_b = mask > 0.5
    boundary = binary_dilation(mask_b, structure=struct) ^ binary_erosion(mask_b, structure=struct)
    b_count = float(np.sum(boundary))
    if b_count < 1.0:
        return 0.0
    return float(np.sum(grad_mag * boundary)) / b_count


def _edge_accept(image: np.ndarray, rough: np.ndarray, refined: np.ndarray, margin: float = 0.02) -> bool:
    """
    PPO 보정 마스크의 MRI 에지 일치도가 초기 마스크보다 우수하거나 동등한지 검사하는 비지도 물리 가드
    """
    e_rough = _edge_alignment_score(image, rough)
    e_refined = _edge_alignment_score(image, refined)
    return e_refined >= (e_rough * (1.0 - margin))


def _refine_with_ppo(agent, image, init_mask, prob_map, refinement_mode, n_steps=15, clip_shrink=False, clip_shrink_threshold=150):
    """
    100% GT-Free 순수 자율 추론:
    정답(GT)을 보지 않고 입력 영상, 초기 마스크, 확률 맵만으로 PPO가 15스텝 동안 경계를 보정합니다.
    """
    # 환경 초기화를 위한 가상 더미 GT (추론 관측값 obs에는 GT가 포함되지 않음)
    dummy_gt = init_mask.copy()
    env = MaskRefinementEnv(
        image[None, ...],
        dummy_gt[None, ...],
        np.expand_dims(init_mask, 0),
        uncertainty_maps=np.expand_dims(prob_map, 0),
        max_steps=n_steps,
        target_dsc=1.0,
        refinement_mode=refinement_mode,
    )
    obs, _ = env.reset(seed=0)
    for _ in range(n_steps):
        try:
            action, _ = agent.predict(obs, deterministic=True)
            if clip_shrink and np.sum(env._current_mask) < clip_shrink_threshold:
                action = np.maximum(0.0, action)
            obs, _, _, truncated, _ = env.step(action)
            if truncated:
                break
        except Exception as e:
            print(f"Skipping RL step for component due to: {e}")
            break
    # 15스텝 완주 후의 최종 결과 마스크 반환 (GT 대조 최고점 선택 없음)
    return env._current_mask.copy()

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
    parser.add_argument("--confidence_threshold", type=float, default=0.95, help="이 값 이상 평균 확률이면 PPO 생략 (고신뢰도 보호 우회, 기본값: 0.95)")
    parser.add_argument("--small_confidence_threshold", type=float, default=0.80, help="Small 클래스용 고신뢰도 보호 우회 임계값 (기본값: 0.80)")
    parser.add_argument("--disable_edge_gate", action="store_true", help="비지도 MRI 에지 물리 일치도 게이트 비활성화")
    parser.add_argument("--stage2_thresholds", type=str, default="0.70,0.70,0.50",
                        help="클래스별(Small,Medium,Large) Stage 2 이진화 임계값. Small=0.70, Medium=0.70, Large=0.50.")
    parser.add_argument("--skip_ppo", action="store_true", help="Stage 3 생략 (Stage 2 단독 베이스라인 측정)")
    parser.add_argument("--allow_oracle_gate", action="store_true", help="연구용 오라클 단조 게이트(GT 필요) 활성화")
    parser.add_argument("--micro_area_floor", type=float, default=80.0,
                        help="Small 마스크 면적이 이 값 미만이면 임계값을 단계적으로 낮춘다.")
    parser.add_argument("--micro_thr_floor", type=float, default=0.15,
                        help="마이크로 조각 임계값 완화의 하한.")
    parser.add_argument("--cc_min_sizes", type=str, default="0,15,25",
                        help="클래스별(Small,Medium,Large) 연결요소 최소 픽셀. 0이면 비활성.")
    args = parser.parse_args()

    stage2_thr = [float(t) for t in args.stage2_thresholds.split(",")]
    if len(stage2_thr) != 3:
        parser.error("--stage2_thresholds 는 쉼표로 구분된 3개 값이어야 합니다.")
    cc_min = [int(t) for t in args.cc_min_sizes.split(",")]
    if len(cc_min) != 3:
        parser.error("--cc_min_sizes 는 쉼표로 구분된 3개 정수여야 합니다.")
    print(f"Stage 2 이진화 임계값: Small={stage2_thr[0]}, Medium={stage2_thr[1]}, Large={stage2_thr[2]}")
    print(f"CC filter min_size: Small={cc_min[0]}, Medium={cc_min[1]}, Large={cc_min[2]}")

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
    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch)
    
    # 3. Stage 3: RL Refiner (Multi-Agent)
    print("Loading PPO Refiners...")
    agents = {}
    agent_name_map = {
        "small": "ppo_small.zip",
        "medium": "ppo_medium.zip",
        "large": "ppo_large.zip"
    }
    for mode, class_idx in zip(["small", "medium", "large"], [0, 1, 2]):
        agent_path = f"checkpoints/{agent_name_map[mode]}"
        if args.skip_ppo:
            agents[class_idx] = None
        elif os.path.exists(agent_path):
            print(f"Loading PPO Agent: {agent_path}")
            agents[class_idx] = PPO.load(agent_path, device=device)
        else:
            print(f"[Warning] PPO Agent not found: {agent_path}. S3 Refinement will be skipped for class {class_idx}.")
            agents[class_idx] = None
        
    initial_dsc_list = []
    initial_hd95_list = []
    final_dsc_list = []
    final_hd95_list = []
    
    class_initial_dsc = {0: [], 1: [], 2: []}
    class_initial_hd95 = {0: [], 1: [], 2: []}
    class_final_dsc = {0: [], 1: [], 2: []}
    class_final_hd95 = {0: [], 1: [], 2: []}
    # 과분할(precision↓) / 과소분할(recall↓) 진단용 보조 지표
    class_initial_prec = {0: [], 1: [], 2: []}
    class_initial_rec = {0: [], 1: [], 2: []}
    class_final_prec = {0: [], 1: [], 2: []}
    class_final_rec = {0: [], 1: [], 2: []}
    
    class_counts = {0:0, 1:0, 2:0}
    monotonic_reverts = 0
    all_candidates = []
    
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
            
            # Component-level Size Routing (개별 컴포넌트 크기에 따라 적합한 전문가 에이전트 매핑)
            if comp_area < 300:
                ck = 0
            elif comp_area < 700:
                ck = 1
            else:
                ck = 2
                
            agent_k = agents[ck]
            ref_mode_k = {0: "small", 1: "medium", 2: "large"}[ck]
            
            if agent_k is None:
                final_mask_np = np.maximum(final_mask_np, comp_mask_k)
                continue

            struct_k = np.ones((3, 3))
            dilate_iter = 2 if ck == 0 else 3
            comp_dilated = binary_dilation(comp_mask_k, struct_k, iterations=dilate_iter)
            is_micro = (ck == 0 and comp_area < 50)
            # TTA 재이진화는 Stage 2와 같은 슬라이스 클래스 임계값을 쓴다.
            # 컴포넌트 크기 ck 로 자르면 Large 슬라이스의 작은 덩어리가 Small 0.80으로 다시 잘린다.
            tta_thr = 0.30 if is_micro else stage2_thr[c]
            comp_from_tta = (prob_tta_np > tta_thr).astype(np.float32) * comp_dilated
            if np.sum(comp_from_tta) == 0:
                comp_from_tta = comp_mask_k.copy()

            if args.confidence_threshold is not None:
                nz = comp_from_tta > 0.5
                mean_p = float(np.mean(prob_tta_np[nz])) if np.any(nz) else 0.0
                bypass_thr = args.small_confidence_threshold if ck == 0 else args.confidence_threshold
                if mean_p >= bypass_thr:
                    final_mask_np = np.maximum(final_mask_np, comp_from_tta)
                    continue

            refined_k_mask = _refine_with_ppo(
                agent_k,
                images[i],
                comp_from_tta,
                prob_tta_np * comp_from_tta,
                ref_mode_k,
                n_steps=15,
                clip_shrink=(ck == 0),
                clip_shrink_threshold=150,
            )
            if np.sum(refined_k_mask) > 0:
                refined_k_mask = binary_closing(refined_k_mask, struct_k).astype(np.float32)
            # 1. 면적 가드 검사 (0.2x ~ 4.0x)
            if np.sum(refined_k_mask) == 0 or not _gt_free_accept(comp_from_tta, refined_k_mask):
                refined_k_mask = comp_from_tta
            # 2. 비지도 MRI 에지 물리 일치도 가드 검사 (정답 불필요)
            elif not args.disable_edge_gate:
                if not _edge_accept(images[i], comp_from_tta, refined_k_mask):
                    refined_k_mask = comp_from_tta

            if args.allow_oracle_gate:
                refined_k_mask = apply_monotonic_dsc_gate(comp_from_tta, refined_k_mask, gt_np)
            final_mask_np = np.maximum(final_mask_np, refined_k_mask)

        # 슬라이스 레벨 2중 비지도 안전 가드 검사 (GT 불필요)
        if not _gt_free_accept(rough_mask_np, final_mask_np):
            final_mask_np = rough_mask_np
        elif not args.disable_edge_gate:
            if not _edge_accept(images[i], rough_mask_np, final_mask_np):
                final_mask_np = rough_mask_np

        if args.allow_oracle_gate:
            gated_mask = apply_monotonic_dsc_gate(rough_mask_np, final_mask_np, gt_np)
            if not np.array_equal(gated_mask, final_mask_np):
                monotonic_reverts += 1
            final_mask_np = gated_mask

        fin_dsc = dice(final_mask_np, gt_np)
        fin_hd95 = hd95(final_mask_np, gt_np)

        final_dsc_list.append(fin_dsc)
        final_hd95_list.append(fin_hd95)
        
        class_initial_dsc[c].append(init_dsc)
        class_initial_hd95[c].append(init_hd95)
        class_final_dsc[c].append(fin_dsc)
        class_final_hd95[c].append(fin_hd95)
        class_initial_prec[c].append(precision(rough_mask_np, gt_np))
        class_initial_rec[c].append(recall(rough_mask_np, gt_np))
        class_final_prec[c].append(precision(final_mask_np, gt_np))
        class_final_rec[c].append(recall(final_mask_np, gt_np))
        
        # Stratification for Small Class
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

        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(images)} slices...")
            
    gate_str = "monotonic DSC gate (Oracle upper bound)" if args.allow_oracle_gate else "100% GT-Free Deployable Mode"
    print(f"\n--- Pipeline Evaluation ({'oracle routing' if args.oracle_routing else 'classifier routing'}, PPO all classes, {gate_str}, CC filter) ---")
    print(f"Total Slices Evaluated: {len(initial_dsc_list)}")
    if args.allow_oracle_gate:
        print(f"Monotonic DSC reverts (final < initial → keep Stage 2): {monotonic_reverts}")
    else:
        conf_str = f"Confidence Bypass (Small >= {args.small_confidence_threshold}, Med/Large >= {args.confidence_threshold})" if args.confidence_threshold is not None else "Confidence Bypass: OFF"
        edge_str = "MRI Edge Physical Guard: ON" if not args.disable_edge_gate else "MRI Edge Physical Guard: OFF"
        print(f"Unsupervised Safety Guards: {conf_str} | {edge_str} (100% GT-Free)")
    print(f"Class Distribution: Small: {class_counts[0]}, Medium: {class_counts[1]}, Large: {class_counts[2]}")
    print(f"Average Initial DSC  (Stage 2):        {np.mean(initial_dsc_list):.4f}")
    print(f"Average Final   DSC  (Stage 3 RL):     {np.mean(final_dsc_list):.4f}")
    print(f"Average Initial HD95 (px):             {np.mean(initial_hd95_list):.4f}")
    print(f"Average Final   HD95 (px):             {np.mean(final_hd95_list):.4f}")
    
    print("\n--- Class-wise Performance Breakdown ---")
    names = {0: "Small (CaraNet)", 1: "Medium (UNet++)", 2: "Large (SegResNet)"}
    for c in [0, 1, 2]:
        if len(class_initial_dsc[c]) > 0:
            init_dsc_avg = np.mean(class_initial_dsc[c])
            fin_dsc_avg  = np.mean(class_final_dsc[c])
            init_hd_avg  = np.mean(class_initial_hd95[c])
            fin_hd_avg   = np.mean(class_final_hd95[c])
            print(f"[{names[c]}] count: {len(class_initial_dsc[c])} "
                  f"| Initial DSC: {init_dsc_avg:.4f} -> Final DSC: {fin_dsc_avg:.4f} "
                  f"| Initial HD95: {init_hd_avg:.4f} -> Final HD95: {fin_hd_avg:.4f} (px)")
            print(f"    └ Precision: {np.mean(class_initial_prec[c]):.4f} -> {np.mean(class_final_prec[c]):.4f} "
                  f"| Recall: {np.mean(class_initial_rec[c]):.4f} -> {np.mean(class_final_rec[c]):.4f} "
                  f"(P<R: 과분할 / P>R: 과소분할)")


    print("\n--- Stratified Analysis for Small Class ---")
    if len(small_active_init) > 0:
        print(f"[Small - Active Tumor (>=50px)] count: {len(small_active_init)} "
              f"| Initial DSC: {np.mean(small_active_init):.4f} -> Final DSC: {np.mean(small_active_fin):.4f} "
              f"| Initial HD95: {np.mean(small_active_init_hd):.4f} -> Final HD95: {np.mean(small_active_hd):.4f} (px)")
    if len(small_micro_init) > 0:
        print(f"[Small - Micro Boundary Fragment (<50px)] count: {len(small_micro_init)} "
              f"| Initial DSC: {np.mean(small_micro_init):.4f} -> Final DSC: {np.mean(small_micro_fin):.4f} "
              f"| Initial HD95: {np.mean(small_micro_init_hd):.4f} -> Final HD95: {np.mean(small_micro_hd):.4f} (px)")
    
    # 시각화: 클래스 평균 Final DSC에 가깝고, Final > Initial인 원본 2장 → 총 6장
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
        improved = [s for s in class_candidates if s["fin_dsc"] > s["init_dsc"] + 1e-8]
        used_fallback = False
        if improved:
            pool = improved
        else:
            pool = class_candidates
            used_fallback = True
        pool = sorted(pool, key=lambda s: abs(s["fin_dsc"] - mean_fin))
        selected = pool[:n]
        pipeline_samples[c] = selected
        print(
            f"[Viz] {names[c]}: mean Final DSC={mean_fin:.4f}, "
            f"pool={'final>initial' if not used_fallback else 'fallback(all)'} n={len(pool)}, "
            f"selected Final={[round(s['fin_dsc'], 4) for s in selected]} "
            f"Δ={[round(s['delta_dsc'], 4) for s in selected]}"
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
