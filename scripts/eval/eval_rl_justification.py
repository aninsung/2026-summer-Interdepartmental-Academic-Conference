"""RL 정당화 ablation: 같은 val 슬라이스에서 A/B/C/D/E를 한 번에 측정.

A  Stage 2 only (현재 임계값·CC·micro ladder)
B  Stage 2 확률맵 + 클래스별 임계값/형태학 그리드 (val에서 최적인 조합, GT로 고름)
C  PPO 15스텝 마지막 마스크 (게이트 없음, 스텝 선택 없음)
D  C + 면적 게이트만 (GT 불필요)
E  현재 논문: 15스텝 중 최고 DSC + 면적 게이트 + 단조 DSC 게이트 (GT 상한)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening, label as sp_label

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.eval.evaluate_pipeline import _gt_free_accept, _tta_probability
from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import DEFAULT_SPLIT_PATH, load_or_create_patient_split
from src.envs.mask_refinement_env import MaskRefinementEnv
from src.models.dynamic_router import AdaptivePipeline
from src.utils.metrics import apply_monotonic_dsc_gate, dice, filter_small_components, hd95
from stable_baselines3 import PPO

CLASS_NAMES = {0: "Small", 1: "Medium", 2: "Large"}
STRUCT = np.ones((3, 3), dtype=bool)

MORPH_OPS = {
    "identity": lambda m: m.astype(np.float32),
    "open1": lambda m: binary_opening(m > 0.5, STRUCT).astype(np.float32),
    "close1": lambda m: binary_closing(m > 0.5, STRUCT).astype(np.float32),
    "open2": lambda m: binary_opening(m > 0.5, STRUCT, iterations=2).astype(np.float32),
    "close2": lambda m: binary_closing(m > 0.5, STRUCT, iterations=2).astype(np.float32),
    "dilate1": lambda m: binary_dilation(m > 0.5, STRUCT).astype(np.float32),
    "erode1": lambda m: binary_erosion(m > 0.5, STRUCT).astype(np.float32),
    "open_close": lambda m: binary_closing(
        binary_opening(m > 0.5, STRUCT), STRUCT
    ).astype(np.float32),
    "close_open": lambda m: binary_opening(
        binary_closing(m > 0.5, STRUCT), STRUCT
    ).astype(np.float32),
}


def _binarize_stage2(prob, c, stage2_thr, cc_min, micro_area_floor, micro_thr_floor):
    mask = (prob > stage2_thr[c]).astype(np.float32)
    if c == 0 and np.sum(mask) < micro_area_floor:
        for thr in np.arange(stage2_thr[c] - 0.05, micro_thr_floor - 1e-9, -0.05):
            mask = (prob > thr).astype(np.float32)
            if np.sum(mask) >= micro_area_floor:
                break
    return filter_small_components(mask, cc_min[c])


def _ppo_last_and_best(agent, image, gt, init_mask, prob_map, refinement_mode, n_steps=15, clip_shrink=False):
    env = MaskRefinementEnv(
        image[None, ...],
        gt[None, ...],
        np.expand_dims(init_mask, 0),
        uncertainty_maps=np.expand_dims(prob_map, 0),
        max_steps=n_steps,
        target_dsc=1.0,
        refinement_mode=refinement_mode,
    )
    obs, _ = env.reset(seed=0)
    last_mask = init_mask.copy()
    best_mask = init_mask.copy()
    best_dsc = dice(init_mask, gt)
    for _ in range(n_steps):
        try:
            action, _ = agent.predict(obs, deterministic=True)
            if clip_shrink and np.sum(env._current_mask) < 35:
                action = np.maximum(0.0, action)
            obs, _, _, truncated, info = env.step(action)
            last_mask = env._current_mask.copy()
            step_dsc = float(info.get("dsc", dice(last_mask, gt)))
            if step_dsc >= best_dsc:
                best_dsc = step_dsc
                best_mask = last_mask.copy()
            if truncated:
                break
        except Exception as exc:
            print(f"Skipping RL step: {exc}")
            break
    return last_mask, best_mask


def _maybe_close(mask):
    if np.sum(mask) > 0:
        return binary_closing(mask, STRUCT).astype(np.float32)
    return mask.astype(np.float32)


def _summarize(dscs, hd95s, routed):
    out = {
        "n": int(len(dscs)),
        "dsc": float(np.mean(dscs)) if len(dscs) else float("nan"),
        "hd95": float(np.mean(hd95s)) if len(hd95s) else float("nan"),
        "by_class": {},
    }
    routed = np.asarray(routed)
    for c, name in CLASS_NAMES.items():
        idx = np.where(routed == c)[0]
        if len(idx) == 0:
            continue
        out["by_class"][name] = {
            "n": int(len(idx)),
            "dsc": float(np.mean(dscs[idx])),
            "hd95": float(np.mean(hd95s[idx])),
        }
    return out


def _print_row(label, summary):
    print(f"{label:<28} n={summary['n']:4d}  DSC={summary['dsc']:.4f}  HD95={summary['hd95']:.4f}")
    for name, row in summary["by_class"].items():
        print(f"  {name:<22} n={row['n']:4d}  DSC={row['dsc']:.4f}  HD95={row['hd95']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="A/B/C/D/E RL justification ablation")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--max_patients", type=int, default=210)
    parser.add_argument("--patient_split", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split_role", type=str, default="val")
    parser.add_argument("--stage2_thresholds", type=str, default="0.80,0.80,0.50")
    parser.add_argument("--cc_min_sizes", type=str, default="0,15,25")
    parser.add_argument("--micro_area_floor", type=float, default=80.0)
    parser.add_argument("--micro_thr_floor", type=float, default=0.15)
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--skip_ppo", action="store_true")
    parser.add_argument("--out", type=str, default="results/rl_justification_ablation.json")
    return parser.parse_args()


def main():
    args = parse_args()
    stage2_thr = [float(t) for t in args.stage2_thresholds.split(",")]
    cc_min = [int(t) for t in args.cc_min_sizes.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split = load_or_create_patient_split(args.train_root, args.max_patients, args.patient_split)
    patient_ids = split[args.split_role]
    print(f"Eval split: {args.split_role} ({len(patient_ids)} patients)")

    dataset = BraTS2020Dataset(
        root_dir=args.train_root,
        modality=args.modality,
        target_size=128,
        max_patients=None,
        patient_ids=patient_ids,
        simulate_rough=False,
    )
    images, gt_masks, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    n_all = len(images) if args.max_slices is None else min(len(images), args.max_slices)
    print(f"Slices: {n_all} / {len(images)}")

    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch)

    agents = {0: None, 1: None, 2: None}
    if not args.skip_ppo:
        for mode, idx in zip(["small", "medium", "large"], [0, 1, 2]):
            path = f"checkpoints/ppo_{mode}.zip"
            if os.path.exists(path):
                print(f"Loading {path}")
                agents[idx] = PPO.load(path, device=device)
            else:
                print(f"[Warning] missing {path}")

    dsc = {k: [] for k in ["A", "B", "C", "D", "E"]}
    hd = {k: [] for k in ["A", "B", "C", "D", "E"]}
    routed = []
    probs = []
    stage2_masks = []
    monotonic_reverts = 0
    worse_than_init = 0

    print("\n=== PPO pass (A/C/D/E) ===")
    for i in range(n_all):
        img_np = images_25d[i]
        gt_np = gt_masks[i]
        if img_np.ndim == 2:
            img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        else:
            img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)

        with torch.no_grad():
            rough_mask_t, class_pred = pipeline(img_t, true_class_preds=None)
        c = int(class_pred.item())
        routed.append(c)
        rough_prob_np = rough_mask_t.squeeze().cpu().numpy().astype(np.float32)
        probs.append(rough_prob_np)
        rough_mask_np = _binarize_stage2(
            rough_prob_np, c, stage2_thr, cc_min, args.micro_area_floor, args.micro_thr_floor
        )
        stage2_masks.append(rough_mask_np)

        dsc["A"].append(dice(rough_mask_np, gt_np))
        hd["A"].append(hd95(rough_mask_np, gt_np))

        last_final = np.zeros_like(rough_mask_np)
        last_area = np.zeros_like(rough_mask_np)
        best_gated = np.zeros_like(rough_mask_np)

        lbl, num_feats = sp_label(rough_mask_np > 0.2)
        valid = [k for k in range(1, num_feats + 1) if np.sum(lbl == k) >= 5]
        for k in range(1, num_feats + 1):
            if k not in valid:
                tiny = (lbl == k).astype(np.float32)
                last_final = np.maximum(last_final, tiny)
                last_area = np.maximum(last_area, tiny)
                best_gated = np.maximum(best_gated, tiny)

        if args.skip_ppo or all(a is None for a in agents.values()):
            last_final = rough_mask_np.copy()
            last_area = rough_mask_np.copy()
            best_gated = rough_mask_np.copy()
        else:
            prob_tta_np = _tta_probability(pipeline, img_t, rough_mask_t, class_pred)
            for k in valid:
                comp_mask_k = (lbl == k).astype(np.float32)
                comp_area = float(np.sum(comp_mask_k))
                ck = 0 if comp_area < 300 else (1 if comp_area < 700 else 2)
                agent_k = agents[ck]
                ref_mode_k = {0: "small", 1: "medium", 2: "large"}[ck]
                if agent_k is None:
                    last_final = np.maximum(last_final, comp_mask_k)
                    last_area = np.maximum(last_area, comp_mask_k)
                    best_gated = np.maximum(best_gated, comp_mask_k)
                    continue

                dilate_iter = 2 if ck == 0 else 3
                comp_dilated = binary_dilation(comp_mask_k, STRUCT, iterations=dilate_iter)
                is_micro = ck == 0 and comp_area < 50
                tta_thr = 0.30 if is_micro else stage2_thr[c]
                comp_from_tta = (prob_tta_np > tta_thr).astype(np.float32) * comp_dilated
                if np.sum(comp_from_tta) == 0:
                    comp_from_tta = comp_mask_k.copy()

                last_k, best_k = _ppo_last_and_best(
                    agent_k,
                    images[i],
                    gt_masks[i],
                    comp_from_tta,
                    prob_tta_np * comp_from_tta,
                    ref_mode_k,
                    n_steps=15,
                    clip_shrink=(ck == 0),
                )
                last_k = _maybe_close(last_k)
                best_k = _maybe_close(best_k)
                last_final = np.maximum(last_final, last_k)

                area_k = last_k
                if np.sum(area_k) == 0 or not _gt_free_accept(comp_from_tta, area_k):
                    area_k = comp_from_tta
                last_area = np.maximum(last_area, area_k)

                e_k = best_k
                if np.sum(e_k) == 0 or not _gt_free_accept(comp_from_tta, e_k):
                    e_k = comp_from_tta
                e_k = apply_monotonic_dsc_gate(comp_from_tta, e_k, gt_np)
                best_gated = np.maximum(best_gated, e_k)

        if not _gt_free_accept(rough_mask_np, last_area):
            last_area = rough_mask_np
        if not _gt_free_accept(rough_mask_np, best_gated):
            best_gated = rough_mask_np
        gated = apply_monotonic_dsc_gate(rough_mask_np, best_gated, gt_np)
        if not np.array_equal(gated, best_gated):
            monotonic_reverts += 1

        dsc["C"].append(dice(last_final, gt_np))
        hd["C"].append(hd95(last_final, gt_np))
        dsc["D"].append(dice(last_area, gt_np))
        hd["D"].append(hd95(last_area, gt_np))
        dsc["E"].append(dice(gated, gt_np))
        hd["E"].append(hd95(gated, gt_np))
        if dsc["C"][-1] + 1e-8 < dsc["A"][-1]:
            worse_than_init += 1

        if (i + 1) % 50 == 0 or (i + 1) == n_all:
            print(
                f"[{i+1}/{n_all}] A={np.mean(dsc['A']):.4f}  "
                f"C={np.mean(dsc['C']):.4f}  D={np.mean(dsc['D']):.4f}  E={np.mean(dsc['E']):.4f}"
            )
            if (i + 1) % 200 == 0:
                partial = {
                    "n_done": i + 1,
                    "A": float(np.mean(dsc["A"])),
                    "C": float(np.mean(dsc["C"])),
                    "D": float(np.mean(dsc["D"])),
                    "E": float(np.mean(dsc["E"])),
                    "worse_than_init_C": worse_than_init,
                }
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                with open(args.out.replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
                    json.dump(partial, f, ensure_ascii=False, indent=2)

    routed = np.asarray(routed)
    for k in dsc:
        dsc[k] = np.asarray(dsc[k], dtype=np.float64)
        hd[k] = np.asarray(hd[k], dtype=np.float64)

    print("\n=== B: class-wise threshold + morphology grid on Stage 2 probs ===")
    thr_grid = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
    class_best = {}
    b_masks_dsc = np.zeros(n_all, dtype=np.float64)
    b_masks_hd = np.zeros(n_all, dtype=np.float64)
    for c, name in CLASS_NAMES.items():
        idx = np.where(routed == c)[0]
        if len(idx) == 0:
            continue
        best = {"dsc": -1.0}
        for thr in thr_grid:
            for op_name, op in MORPH_OPS.items():
                vals = []
                for i in idx:
                    pred = (probs[i] > thr).astype(np.float32)
                    if c == 0 and np.sum(pred) < args.micro_area_floor:
                        for t2 in np.arange(thr - 0.05, args.micro_thr_floor - 1e-9, -0.05):
                            pred = (probs[i] > t2).astype(np.float32)
                            if np.sum(pred) >= args.micro_area_floor:
                                break
                    pred = filter_small_components(pred, cc_min[c])
                    pred = op(pred)
                    vals.append(dice(pred, gt_masks[i]))
                mean_dsc = float(np.mean(vals))
                if mean_dsc > best["dsc"]:
                    best = {
                        "class": name,
                        "thr": thr,
                        "op": op_name,
                        "dsc": mean_dsc,
                        "n": int(len(idx)),
                        "dscs": vals,
                    }
        winner_hd = []
        thr, op = best["thr"], MORPH_OPS[best["op"]]
        for i in idx:
            pred = (probs[i] > thr).astype(np.float32)
            if c == 0 and np.sum(pred) < args.micro_area_floor:
                for t2 in np.arange(thr - 0.05, args.micro_thr_floor - 1e-9, -0.05):
                    pred = (probs[i] > t2).astype(np.float32)
                    if np.sum(pred) >= args.micro_area_floor:
                        break
            pred = op(filter_small_components(pred, cc_min[c]))
            winner_hd.append(hd95(pred, gt_masks[i]))
        best["hd95"] = float(np.mean(winner_hd))
        class_best[name] = {k: v for k, v in best.items() if k != "dscs"}
        print(
            f"  {name}: thr={best['thr']:.2f}  op={best['op']}  "
            f"DSC={best['dsc']:.4f}  HD95={best['hd95']:.4f}"
        )
        b_masks_dsc[idx] = np.asarray(best["dscs"])
        b_masks_hd[idx] = np.asarray(winner_hd)

    dsc["B"] = b_masks_dsc
    hd["B"] = b_masks_hd

    labels = {
        "A": "A  Stage 2 only",
        "B": "B  thr+morpho (val-tuned)",
        "C": "C  PPO last-15, no gate",
        "D": "D  PPO last-15 + area gate",
        "E": "E  PPO best-step + GT gate",
    }
    summaries = {k: _summarize(dsc[k], hd[k], routed) for k in labels}
    print("\n=== Ablation (same val slices) ===")
    for k, lab in labels.items():
        _print_row(lab, summaries[k])
        if k != "A":
            print(f"  ΔDSC vs A: {summaries[k]['dsc'] - summaries['A']['dsc']:+.4f}")

    print(f"\nPPO last-15 worse than Stage 2: {worse_than_init}/{n_all} ({100*worse_than_init/n_all:.1f}%)")
    print(f"Monotonic-style reverts vs ungated best: {monotonic_reverts}/{n_all}")

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n": n_all,
        "protocol": {
            "stage2_thresholds": stage2_thr,
            "cc_min_sizes": cc_min,
            "split_role": args.split_role,
            "modality": args.modality,
        },
        "class_best_B": class_best,
        "worse_than_init_C": {"n": worse_than_init, "rate": worse_than_init / n_all},
        "summaries": summaries,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
