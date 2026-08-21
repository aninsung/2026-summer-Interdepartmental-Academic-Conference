"""베이스라인 평가 스크립트.

지표 정의를 파이프라인과 완전히 일치시키기 위해 src/utils/metrics.py 를 그대로
가져다 쓴다. 종양 크기 구간(Small/Medium/Large)별 분해도 파이프라인과 같은
gt_size_class 기준(면적 300/700)을 사용하므로 표를 나란히 놓고 비교할 수 있다.

사용 예시
    python baselines/eval_baseline.py --checkpoint baselines/checkpoints/kaist_best.pt
    python baselines/eval_baseline.py --checkpoint baselines/checkpoints/nvauto_best.pt --sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.common import check_split_compatibility, simple_collate
from baselines.losses import confidence_ensemble
from baselines.models import build_kaist_nnunet, build_nvauto_segresnet
from src.utils.metrics import dice, gt_size_class, hd95, precision, recall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = {0: "Small", 1: "Medium", 2: "Large"}


def load_baseline(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    method = ckpt["method"]
    in_channels = ckpt.get("in_channels", 2)
    target_size = ckpt.get("target_size", 128)

    if method == "kaist":
        model = build_kaist_nnunet(
            in_channels=in_channels, out_channels=1, input_size=target_size
        )
    else:
        model = build_nvauto_segresnet(in_channels=in_channels, out_channels=1)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    log.info(f"{method} 로드 완료 ({checkpoint}, 학습 시 val_DSC={ckpt.get('val_dsc', float('nan')):.4f})")
    return model, method, target_size


@torch.no_grad()
def collect_probabilities(
    models: List[torch.nn.Module], loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """확률맵과 정답을 모아 둔다. 임계값 스윕을 재추론 없이 하기 위함."""
    probs_all, gts_all = [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["gt_mask"]
        per_model = [torch.sigmoid(m(img).float()) for m in models]
        prob = confidence_ensemble(per_model) if len(per_model) > 1 else per_model[0]
        probs_all.append(prob.cpu().numpy())
        gts_all.append(gt.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(gts_all, axis=0)


def evaluate_at_threshold(
    probs: np.ndarray, gts: np.ndarray, threshold: float
) -> Dict[str, Dict[str, float]]:
    """전체 및 크기 구간별 DSC / HD95 / Precision / Recall 을 계산한다."""
    buckets: Dict[str, List[List[float]]] = {
        name: [[], [], [], []] for name in ["Overall", "Small", "Medium", "Large"]
    }

    for i in range(probs.shape[0]):
        gt = gts[i, 0]
        pred = (probs[i, 0] > threshold).astype(np.float32)
        scores = [
            dice(pred, gt),
            hd95(pred, gt),
            precision(pred, gt),
            recall(pred, gt),
        ]
        for key in ("Overall", CLASS_NAMES[gt_size_class(gt)]):
            for slot, value in zip(buckets[key], scores):
                slot.append(value)

    results: Dict[str, Dict[str, float]] = {}
    for name, (d, h, p, r) in buckets.items():
        if not d:
            continue
        results[name] = {
            "n": len(d),
            "dsc": float(np.mean(d)),
            "hd95": float(np.mean(h)),
            "precision": float(np.mean(p)),
            "recall": float(np.mean(r)),
        }
    return results


def print_table(title: str, results: Dict[str, Dict[str, float]]) -> None:
    log.info(f"\n── {title} ──")
    log.info(f"{'구간':<8}{'n':>6}{'DSC':>9}{'HD95':>9}{'Prec':>9}{'Recall':>9}")
    for name in ("Overall", "Small", "Medium", "Large"):
        if name not in results:
            continue
        r = results[name]
        log.info(
            f"{name:<8}{r['n']:>6}{r['dsc']:>9.4f}{r['hd95']:>9.3f}"
            f"{r['precision']:>9.4f}{r['recall']:>9.4f}"
        )


def main():
    p = argparse.ArgumentParser(description="BraTS 2021 베이스라인 평가")
    p.add_argument("--checkpoint", type=str, nargs="+", required=True,
                   help="평가할 체크포인트. 여러 개 주면 NVAUTO 의 confidence 앙상블을 적용한다.")
    p.add_argument("--train_root", type=str, default="src/data/archive")
    p.add_argument("--modality", type=str, default="t1ce+flair")
    p.add_argument("--max_patients", type=int, default=210)
    p.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sweep", action="store_true",
                   help="임계값을 훑어 최적값을 함께 보고한다(파이프라인의 임계값 튜닝과 조건을 맞출 때 사용).")
    p.add_argument("--out_json", type=str, default="baselines/results/baseline_metrics.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models, methods = [], []
    target_size = 128
    for ckpt in args.checkpoint:
        model, method, target_size = load_baseline(ckpt, device)
        models.append(model)
        methods.append(method)

    check_split_compatibility(args.train_root, args.max_patients, args.patient_split)

    from src.data.patient_split import load_split_brats_datasets

    _, val_ds = load_split_brats_datasets(
        train_root=args.train_root,
        modality=args.modality,
        target_size=target_size,
        max_patients=args.max_patients,
        patient_split=args.patient_split,
        refinement_mode=None,
        simulate_rough=False,
    )
    log.info(f"검증 슬라이스 {len(val_ds)}개 (파이프라인과 동일 분할)")

    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=simple_collate,
    )

    probs, gts = collect_probabilities(models, loader, device)
    label = "+".join(methods)

    results = evaluate_at_threshold(probs, gts, args.threshold)
    print_table(f"{label} @ threshold={args.threshold}", results)

    payload = {
        "methods": methods,
        "checkpoints": args.checkpoint,
        "threshold": args.threshold,
        "results": results,
    }

    if args.sweep:
        sweep = {}
        for thr in np.arange(0.3, 0.91, 0.05):
            thr = round(float(thr), 2)
            sweep[str(thr)] = evaluate_at_threshold(probs, gts, thr)["Overall"]
        best_thr = max(sweep, key=lambda t: sweep[t]["dsc"])
        log.info("\n── 임계값 스윕 (Overall DSC) ──")
        for thr, r in sweep.items():
            mark = " ←최적" if thr == best_thr else ""
            log.info(f"  thr={thr}: DSC={r['dsc']:.4f} Prec={r['precision']:.4f} "
                     f"Recall={r['recall']:.4f}{mark}")
        payload["sweep"] = sweep
        payload["best_threshold"] = float(best_thr)
        print_table(
            f"{label} @ 최적 threshold={best_thr}",
            evaluate_at_threshold(probs, gts, float(best_thr)),
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info(f"\n결과 저장: {args.out_json}")


if __name__ == "__main__":
    main()
