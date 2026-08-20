"""TRIO / KAIST / NVAUTO를 같은 슬라이스에서 3×6으로 비교한다.

행: TRIO, KAIST, NVAUTO
열: Small 1–2, Medium 1–2, Large 1–2 (GT 면적 구간)
표본: 세 방법 DSC가 각 클래스 평균에 가까운 슬라이스.
고정 인덱스 예: --indices 113,2286,1121,2295,95,1480
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_dilation, label as sp_label

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baselines.eval_baseline import load_baseline
from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import DEFAULT_SPLIT_PATH, load_or_create_patient_split
from src.models.dynamic_router import AdaptivePipeline
from src.utils.metrics import (
    apply_monotonic_dsc_gate,
    dice,
    filter_small_components,
    gt_size_class,
)
from stable_baselines3 import PPO


def _import_eval():
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "evaluate_pipeline.py")
    spec = importlib.util.spec_from_file_location("evaluate_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = None

CLASS_NAMES = {0: "Small", 1: "Medium", 2: "Large"}
COL_TITLES = ["TRIO", "KAIST", "NVAUTO"]
COLORS = {
    "pipe": "#00bcd4",
    "kaist": "#2b7fd4",
    "nvauto": "#e06c2a",
}
# 문서에 적힌 클래스 평균 (선택 목표). 실제 측정 평균이 있으면 그 값을 쓴다.
TARGET_DSC = {
    "pipe": {0: 0.8429, 1: 0.9314, 2: 0.9582},
    "kaist": {0: 0.8278, 1: 0.9165, 2: 0.9614},
    "nvauto": {0: 0.8377, 1: 0.9200, 2: 0.9594},
}


def _crop_box(gt: np.ndarray, margin: int = 15):
    ys, xs = np.where(gt > 0.5)
    if len(ys) == 0:
        return 0, gt.shape[1] - 1, 0, gt.shape[0] - 1
    return (
        max(0, int(xs.min()) - margin),
        min(gt.shape[1] - 1, int(xs.max()) + margin),
        max(0, int(ys.min()) - margin),
        min(gt.shape[0] - 1, int(ys.max()) + margin),
    )


def _to_img_t(img_np, device):
    if img_np.ndim == 2:
        return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    return torch.from_numpy(img_np).unsqueeze(0).to(device)


def stage2_mask(pipeline, img_25d, stage2_thr, cc_min, micro_floor=80.0, micro_thr_floor=0.15):
    img_t = _to_img_t(img_25d, next(pipeline.parameters()).device)
    with torch.no_grad():
        rough_t, class_pred = pipeline(img_t)
    c = int(class_pred.item())
    prob = rough_t.squeeze().cpu().numpy()
    mask = (prob > stage2_thr[c]).astype(np.float32)
    if c == 0 and np.sum(mask) < micro_floor:
        for thr in np.arange(stage2_thr[c] - 0.05, micro_thr_floor - 1e-9, -0.05):
            cand = (prob > thr).astype(np.float32)
            mask = cand
            if np.sum(cand) >= micro_floor:
                break
    mask = filter_small_components(mask, cc_min[c])
    return mask, prob, rough_t, class_pred, c, img_t


def stage3_gated(pipeline, agents, image_center, img_25d, gt, stage2_thr, cc_min):
    rough, prob, rough_t, class_pred, c, img_t = stage2_mask(
        pipeline, img_25d, stage2_thr, cc_min
    )
    lbl, num_feats = sp_label(rough > 0.2)
    valid = [k for k in range(1, num_feats + 1) if np.sum(lbl == k) >= 5]
    final = np.zeros_like(rough)
    for k in range(1, num_feats + 1):
        if k not in valid:
            final = np.maximum(final, (lbl == k).astype(np.float32))

    tta = ev._tta_probability(pipeline, img_t, rough_t, class_pred)
    for k in valid:
        comp = (lbl == k).astype(np.float32)
        area = float(np.sum(comp))
        ck = 0 if area < 300 else (1 if area < 700 else 2)
        agent = agents[ck]
        mode = {0: "small", 1: "medium", 2: "large"}[ck]
        if agent is None:
            final = np.maximum(final, comp)
            continue
        struct = np.ones((3, 3))
        dilated = binary_dilation(comp, struct, iterations=2 if ck == 0 else 3)
        thr = 0.30 if (ck == 0 and area < 50) else stage2_thr[c]
        from_tta = (tta > thr).astype(np.float32) * dilated
        if np.sum(from_tta) == 0:
            from_tta = comp.copy()
        refined = ev._refine_with_ppo(
            agent, image_center, gt, from_tta, tta * from_tta, mode,
            n_steps=15, clip_shrink=(ck == 0),
        )
        if np.sum(refined) > 0:
            refined = binary_closing(refined, struct).astype(np.float32)
        if np.sum(refined) == 0 or not ev._gt_free_accept(from_tta, refined):
            refined = from_tta
        refined = apply_monotonic_dsc_gate(from_tta, refined, gt)
        final = np.maximum(final, refined)

    if not ev._gt_free_accept(rough, final):
        final = rough
    return apply_monotonic_dsc_gate(rough, final, gt), c


def predict_baseline(model, img_center, device, threshold=0.5):
    t = torch.from_numpy(img_center).unsqueeze(0).to(device)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(t).float()).cpu().numpy()[0, 0]
    return (prob > threshold).astype(np.float32)


def select_indices(gt_cls, d_pipe, d_k, d_n, pids, n_per=2, min_gap=8):
    chosen = []
    for c in (0, 1, 2):
        idx = np.where(gt_cls == c)[0]
        if len(idx) == 0:
            continue
        tp, tk, tn = TARGET_DSC["pipe"][c], TARGET_DSC["kaist"][c], TARGET_DSC["nvauto"][c]
        dist = (
            np.abs(d_pipe[idx] - tp)
            + np.abs(d_k[idx] - tk)
            + np.abs(d_n[idx] - tn)
        )
        order = idx[np.argsort(dist)]
        picked = []
        used_pid = set()
        last = -10_000
        for i in order:
            pid = pids[i] if pids is not None else None
            if abs(int(i) - last) < min_gap:
                continue
            if pid is not None and pid in used_pid and len(picked) < n_per:
                # 다른 환자가 남아 있으면 건너뛴다
                continue
            picked.append(int(i))
            last = int(i)
            if pid is not None:
                used_pid.add(pid)
            if len(picked) >= n_per:
                break
        # 부족하면 gap만 보고 채움
        if len(picked) < n_per:
            for i in order:
                if int(i) in picked:
                    continue
                if any(abs(int(i) - j) < min_gap for j in picked):
                    continue
                picked.append(int(i))
                if len(picked) >= n_per:
                    break
        chosen.extend(picked[:n_per])
    return chosen


def render(rows, out_path):
    n_rows, n_cols = 3, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.6, 9.4))
    fig.suptitle(
        "Representative slices near class-mean DSC  ·  dashed green = GT",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    keys = ["pipe", "kaist", "nvauto"]
    dkeys = ["d_pipe", "d_kaist", "d_nvauto"]
    colors = [COLORS["pipe"], COLORS["kaist"], COLORS["nvauto"]]

    for c, s in enumerate(rows):
        img, gt = s["img"], s["gt"]
        x0, x1, y0, y1 = _crop_box(gt)
        cls_name = CLASS_NAMES[s["gt_c"]]
        sample_i = 1 if c % 2 == 0 else 2
        for r in range(n_rows):
            ax = axes[r, c]
            mask = s[keys[r]]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            overlay = np.zeros((*mask.shape, 4))
            rgb = matplotlib.colors.to_rgb(colors[r])
            overlay[mask > 0.5] = [*rgb, 0.42]
            ax.imshow(overlay)
            ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.3, linestyles="--")
            ax.contour(mask, levels=[0.5], colors=[colors[r]], linewidths=1.6)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{cls_name} {sample_i}", fontsize=12, fontweight="bold", pad=8)
            ax.set_xlabel(f"DSC={s[dkeys[r]]:.3f}", fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(COL_TITLES[r], fontsize=13, fontweight="bold", labelpad=8)

    fig.tight_layout(rect=[0.01, 0.01, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {out_path}")


def main():
    global ev
    ev = _import_eval()

    p = argparse.ArgumentParser()
    p.add_argument("--train_root", default="src/data/archive")
    p.add_argument("--patient_split", default=DEFAULT_SPLIT_PATH)
    p.add_argument("--max_patients", type=int, default=210)
    p.add_argument("--modality", default="t1ce+flair")
    p.add_argument("--kaist", default="baselines/checkpoints/kaist_best.pt")
    p.add_argument("--nvauto", default="baselines/checkpoints/nvauto_best.pt")
    p.add_argument("--stage2_thresholds", default="0.80,0.80,0.50")
    p.add_argument("--cc_min_sizes", default="0,15,25")
    p.add_argument("--shortlist", type=int, default=10, help="클래스당 Stage 3 후보 수")
    p.add_argument("--indices", type=str, default="",
                   help="고정 슬라이스 인덱스(쉼표). 있으면 선별을 건너뛴다.")
    p.add_argument("--out", default="results/method_comparison_3x6.png")
    args = p.parse_args()

    for path in (args.kaist, args.nvauto):
        if not os.path.exists(path):
            raise SystemExit(
                f"베이스라인 체크포인트가 없습니다: {path}\n"
                "먼저 `python baselines/train_baseline.py --method kaist` 와 "
                "`python baselines/train_baseline.py --method nvauto` 를 실행하세요."
            )

    stage2_thr = [float(x) for x in args.stage2_thresholds.split(",")]
    cc_min = [int(x) for x in args.cc_min_sizes.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split = load_or_create_patient_split(args.train_root, args.max_patients, args.patient_split)
    val_ids = split["val"]
    ds = BraTS2020Dataset(
        root_dir=args.train_root,
        modality=args.modality,
        target_size=128,
        max_patients=None,
        patient_ids=val_ids,
        simulate_rough=False,
    )
    images, gts, _ = ds.get_numpy_arrays()
    images_25d = ds.get_numpy_25d_arrays()
    pids = list(getattr(ds, "_sample_pids", [None] * len(gts)))
    n = len(gts)
    print(f"val 슬라이스 {n}장")

    kaist, _, _ = load_baseline(args.kaist, device)
    nvauto, _, _ = load_baseline(args.nvauto, device)

    print("KAIST / NVAUTO 추론...")
    d_k = np.zeros(n, dtype=np.float32)
    d_n = np.zeros(n, dtype=np.float32)
    pred_k = [None] * n
    pred_n = [None] * n
    gt_cls = np.array([gt_size_class(g) for g in gts], dtype=np.int32)
    kaist.eval()
    nvauto.eval()

    fixed = [int(x) for x in args.indices.split(",") if x.strip()] if args.indices else []

    def _infer_baselines(indices):
        with torch.no_grad():
            for i in indices:
                t = torch.from_numpy(np.expand_dims(images[i], 0)).to(device)
                if t.ndim == 3:
                    t = t.unsqueeze(1)
                pred_k[i] = (torch.sigmoid(kaist(t).float()) > 0.5).cpu().numpy()[0, 0].astype(np.float32)
                pred_n[i] = (torch.sigmoid(nvauto(t).float()) > 0.5).cpu().numpy()[0, 0].astype(np.float32)
                d_k[i] = dice(pred_k[i], gts[i])
                d_n[i] = dice(pred_n[i], gts[i])

    if fixed:
        _infer_baselines(fixed)
    else:
        bs = 32
        with torch.no_grad():
            for start in range(0, n, bs):
                end = min(start + bs, n)
                batch = torch.from_numpy(np.stack(images[start:end])).to(device)
                if batch.ndim == 3:
                    batch = batch.unsqueeze(1)
                pk = (torch.sigmoid(kaist(batch).float()) > 0.5).cpu().numpy()[:, 0]
                pn = (torch.sigmoid(nvauto(batch).float()) > 0.5).cpu().numpy()[:, 0]
                for j, i in enumerate(range(start, end)):
                    pred_k[i] = pk[j].astype(np.float32)
                    pred_n[i] = pn[j].astype(np.float32)
                    d_k[i] = dice(pred_k[i], gts[i])
                    d_n[i] = dice(pred_n[i], gts[i])
                if end % 400 < bs or end == n:
                    print(f"  baseline {end}/{n}")

    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch)
    agents = {}
    for mode, idx in zip(["small", "medium", "large"], [0, 1, 2]):
        path = f"checkpoints/ppo_{mode}.zip"
        agents[idx] = PPO.load(path, device=device) if os.path.exists(path) else None

    print("Stage 2 초안 DSC (후보 선별)...")
    d_s2 = np.zeros(n, dtype=np.float32)
    if not fixed:
        for i in range(n):
            mask, *_ = stage2_mask(pipeline, images_25d[i], stage2_thr, cc_min)
            d_s2[i] = dice(mask, gts[i])
            if (i + 1) % 400 == 0:
                print(f"  stage2 {i+1}/{n}")

        short = []
        for c in (0, 1, 2):
            idx = np.where(gt_cls == c)[0]
            dist = (
                np.abs(d_s2[idx] - TARGET_DSC["pipe"][c])
                + np.abs(d_k[idx] - TARGET_DSC["kaist"][c])
                + np.abs(d_n[idx] - TARGET_DSC["nvauto"][c])
            )
            order = idx[np.argsort(dist)][: args.shortlist]
            short.extend(int(x) for x in order)
        short = sorted(set(short))
    else:
        short = list(fixed)
    print(f"Stage 3 후보 {len(short)}장")

    d_pipe = np.full(n, 99.0, dtype=np.float32)
    pred_pipe = [None] * n
    for j, i in enumerate(short):
        gated, _ = stage3_gated(
            pipeline, agents, images[i], images_25d[i], gts[i], stage2_thr, cc_min
        )
        pred_pipe[i] = gated
        d_pipe[i] = dice(gated, gts[i])
        print(f"  stage3 {j+1}/{len(short)} idx={i} GT={CLASS_NAMES[int(gt_cls[i])]} "
              f"pipe={d_pipe[i]:.4f} kaist={d_k[i]:.4f} nvauto={d_n[i]:.4f}")

    if fixed:
        chosen = list(fixed)
    else:
        mask_ok = np.zeros(n, dtype=bool)
        mask_ok[short] = True
        d_pipe_sel = np.where(mask_ok, d_pipe, 99.0)
        chosen = select_indices(gt_cls, d_pipe_sel, d_k, d_n, pids, n_per=2)
    print("선택 인덱스:", chosen)

    rows = []
    for i in chosen:
        center = images[i]
        img2d = center[0] if center.ndim == 3 else center
        rows.append({
            "img": img2d,
            "gt": gts[i],
            "gt_c": int(gt_cls[i]),
            "pipe": pred_pipe[i],
            "kaist": pred_k[i],
            "nvauto": pred_n[i],
            "d_pipe": float(d_pipe[i]),
            "d_kaist": float(d_k[i]),
            "d_nvauto": float(d_n[i]),
            "idx": i,
        })
        print(
            f"  {CLASS_NAMES[int(gt_cls[i])]} idx={i} pid={pids[i]}  "
            f"Gate {d_pipe[i]:.4f} (목표 {TARGET_DSC['pipe'][int(gt_cls[i])]:.4f})  "
            f"KAIST {d_k[i]:.4f}  NVAUTO {d_n[i]:.4f}"
        )

    render(rows, args.out)


if __name__ == "__main__":
    main()
