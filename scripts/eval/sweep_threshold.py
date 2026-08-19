"""Stage 2 이진화 임계값 스윕.

확률맵을 한 번만 계산해 캐시한 뒤 여러 임계값을 재적용하므로
evaluate_pipeline.py 를 임계값마다 돌리는 것보다 훨씬 빠르다.
PPO/TTA 를 거치지 않으므로 순수하게 임계값 효과만 분리해서 본다.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import load_or_create_patient_split, DEFAULT_SPLIT_PATH
from src.models.dynamic_router import AdaptivePipeline
from src.utils.metrics import dice, hd95, precision, recall

CLASS_NAMES = {0: "Small", 1: "Medium", 2: "Large"}


def _compute_prob_maps(pipeline, images, device, batch_size, oracle_routing, gt_masks):
    """전체 슬라이스의 Stage 2 확률맵과 라우팅 클래스를 한 번에 계산."""
    probs = np.zeros((len(images),) + images.shape[-2:], dtype=np.float32)
    routed = np.zeros(len(images), dtype=np.int64)

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch = images[start:end]
            if batch.ndim == 3:
                batch_t = torch.from_numpy(batch).unsqueeze(1).to(device)
            else:
                batch_t = torch.from_numpy(batch).to(device)

            if oracle_routing:
                areas = np.array([np.sum(gt_masks[i]) for i in range(start, end)])
                true_c = np.where(areas < 300, 0, np.where(areas < 700, 1, 2))
                route = torch.from_numpy(true_c).long().to(device)
            else:
                route = None

            out, class_pred = pipeline(batch_t, true_class_preds=route)
            probs[start:end] = out.squeeze(1).cpu().numpy()
            routed[start:end] = class_pred.cpu().numpy()

    return probs, routed


def main():
    parser = argparse.ArgumentParser(description="Stage 2 binarization threshold sweep")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--max_patients", type=int, default=210)
    parser.add_argument("--patient_split", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split_role", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--oracle_routing", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
    )
    parser.add_argument("--skip_hd95", action="store_true", help="HD95 생략 (더 빠름)")
    parser.add_argument("--plot", type=str, default=None, help="DSC/Precision/Recall 곡선 PNG 저장 경로")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    patient_ids = None
    if args.split_role != "all":
        split = load_or_create_patient_split(args.train_root, args.max_patients, args.patient_split)
        patient_ids = split[args.split_role]
        print(f"Eval split: {args.split_role} ({len(patient_ids)} patients)")

    dataset = BraTS2020Dataset(
        root_dir=args.train_root,
        modality=args.modality,
        target_size=128,
        max_patients=None if patient_ids is not None else args.max_patients,
        patient_ids=patient_ids,
        simulate_rough=False,
    )
    images, gt_masks, _ = dataset.get_numpy_arrays()
    print(f"Slices: {len(images)}")

    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch)

    print("Stage 2 확률맵 계산 중...")
    probs, routed = _compute_prob_maps(
        pipeline, images, device, args.batch_size, args.oracle_routing, gt_masks
    )

    # GT 거리맵은 임계값과 무관하므로 한 번만 계산해 재사용
    gt_dists = None
    if not args.skip_hd95:
        print("GT 거리맵 사전 계산 중...")
        gt_dists = np.stack([distance_transform_edt(~gt.astype(bool)) for gt in gt_masks])

    print(f"\n임계값 {len(thresholds)}개 평가 중...")
    results = {}
    for thr in thresholds:
        per_class = {c: {"dsc": [], "prec": [], "rec": [], "hd": []} for c in (0, 1, 2)}
        for i in range(len(images)):
            pred = (probs[i] > thr).astype(np.float32)
            gt = gt_masks[i]
            c = int(routed[i])
            per_class[c]["dsc"].append(dice(pred, gt))
            per_class[c]["prec"].append(precision(pred, gt))
            per_class[c]["rec"].append(recall(pred, gt))
            if gt_dists is not None:
                per_class[c]["hd"].append(hd95(pred, gt, dist_b=gt_dists[i]))
        results[thr] = per_class
        overall = np.mean([v for c in (0, 1, 2) for v in per_class[c]["dsc"]])
        print(f"  thr={thr:.2f}  overall DSC={overall:.4f}")

    def agg(thr, c, key):
        vals = results[thr][c][key]
        return float(np.mean(vals)) if vals else float("nan")

    def overall(thr, key):
        vals = [v for c in (0, 1, 2) for v in results[thr][c][key]]
        return float(np.mean(vals)) if vals else float("nan")

    counts = {c: len(results[thresholds[0]][c]["dsc"]) for c in (0, 1, 2)}
    print("\n" + "=" * 78)
    print(f"임계값 스윕 결과 ({'oracle' if args.oracle_routing else 'classifier'} routing)")
    print(f"슬라이스 분포: Small={counts[0]}, Medium={counts[1]}, Large={counts[2]}")
    print("=" * 78)

    hd_hdr = "     HD95" if gt_dists is not None else ""
    print(f"\n{'thr':>6}  {'overall DSC':>12}{hd_hdr}")
    for thr in thresholds:
        hd_txt = f"  {overall(thr, 'hd'):8.4f}" if gt_dists is not None else ""
        print(f"{thr:6.2f}  {overall(thr, 'dsc'):12.4f}{hd_txt}")

    for c in (0, 1, 2):
        if counts[c] == 0:
            continue
        print(f"\n--- {CLASS_NAMES[c]} (n={counts[c]}) ---")
        hd_hdr = f"  {'HD95':>8}" if gt_dists is not None else ""
        print(f"{'thr':>6}  {'DSC':>8}  {'Prec':>8}  {'Rec':>8}  {'P-R':>8}{hd_hdr}")
        for thr in thresholds:
            p, r = agg(thr, c, "prec"), agg(thr, c, "rec")
            hd_txt = f"  {agg(thr, c, 'hd'):8.4f}" if gt_dists is not None else ""
            print(
                f"{thr:6.2f}  {agg(thr, c, 'dsc'):8.4f}  {p:8.4f}  {r:8.4f}  {p - r:+8.4f}{hd_txt}"
            )

    print("\n" + "=" * 78)
    print("최적 임계값")
    print("=" * 78)
    best_overall = max(thresholds, key=lambda t: overall(t, "dsc"))
    base = 0.50 if 0.50 in results else thresholds[0]
    print(
        f"  전역        : thr={best_overall:.2f}  DSC={overall(best_overall, 'dsc'):.4f}  "
        f"(thr={base:.2f} 대비 {overall(best_overall, 'dsc') - overall(base, 'dsc'):+.4f})"
    )
    for c in (0, 1, 2):
        if counts[c] == 0:
            continue
        best_c = max(thresholds, key=lambda t: agg(t, c, "dsc"))
        print(
            f"  {CLASS_NAMES[c]:<11}: thr={best_c:.2f}  DSC={agg(best_c, c, 'dsc'):.4f}  "
            f"(thr={base:.2f} 대비 {agg(best_c, c, 'dsc') - agg(base, c, 'dsc'):+.4f})"
        )

    per_class_best = {c: max(thresholds, key=lambda t: agg(t, c, "dsc")) for c in (0, 1, 2) if counts[c]}
    mixed = np.mean(
        [v for c, t in per_class_best.items() for v in results[t][c]["dsc"]]
    )
    print(
        f"\n  클래스별 최적 조합 DSC={mixed:.4f}  "
        f"(thr={base:.2f} 전역 대비 {mixed - overall(base, 'dsc'):+.4f})"
    )
    print(f"  조합 = {{{', '.join(f'{CLASS_NAMES[c]}:{t:.2f}' for c, t in per_class_best.items())}}}")

    if args.plot:
        _plot(args.plot, thresholds, results, counts, overall, agg, base)


def _plot(path, thresholds, results, counts, overall, agg, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(None, "Overall")] + [(c, f"{CLASS_NAMES[c]} (n={counts[c]})") for c in (0, 1, 2) if counts[c]]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))

    for ax, (c, title) in zip(np.atleast_1d(axes), panels):
        if c is None:
            dsc = [overall(t, "dsc") for t in thresholds]
            prec = rec = None
        else:
            dsc = [agg(t, c, "dsc") for t in thresholds]
            prec = [agg(t, c, "prec") for t in thresholds]
            rec = [agg(t, c, "rec") for t in thresholds]

        ax.plot(thresholds, dsc, "o-", color="#1f77b4", lw=2.2, label="DSC")
        if prec is not None:
            ax.plot(thresholds, prec, "s--", color="#2ca02c", alpha=0.75, label="Precision")
            ax.plot(thresholds, rec, "^--", color="#d62728", alpha=0.75, label="Recall")

        best = thresholds[int(np.argmax(dsc))]
        ax.axvline(best, color="#ff7f0e", ls=":", lw=1.8)
        ax.axvline(base, color="gray", ls="-", lw=1.0, alpha=0.6)
        ax.annotate(
            f"best {best:.2f}\nDSC {max(dsc):.4f}",
            xy=(best, max(dsc)),
            xytext=(6, -34),
            textcoords="offset points",
            fontsize=9,
            color="#ff7f0e",
        )
        ax.set_title(title)
        ax.set_xlabel("binarization threshold")
        ax.set_ylabel("score")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        f"Stage 2 binarization threshold sweep (gray line = current default {base:.2f})",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"\n[Plot] saved {path}")


if __name__ == "__main__":
    main()
