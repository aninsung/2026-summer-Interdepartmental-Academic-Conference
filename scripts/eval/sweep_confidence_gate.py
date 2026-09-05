"""Post-hoc confidence gate sweep for Medium/Large Stage3.

Computes region mean probability (Expert soft mask interior) once, then
simulates: if conf >= thr → keep Stage2; else keep Stage3 final from a
precomputed metrics npz (typically unconstrained deploy SL).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import DEFAULT_SPLIT_PATH, load_or_create_patient_split
from src.models.dynamic_router import AdaptivePipeline
from src.utils.metrics import filter_small_components


def compute_conf(
    conf_out: Path,
    train_root: str,
    modality: str,
    max_patients: int,
    patient_split: str,
) -> np.ndarray:
    if conf_out.exists():
        conf = np.load(conf_out)["conf"]
        print(f"loaded conf cache: {conf_out} n={len(conf)}")
        return conf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_or_create_patient_split(train_root, max_patients, patient_split)
    ds = BraTS2020Dataset(
        root_dir=train_root,
        modality=modality,
        target_size=128,
        max_patients=None,
        patient_ids=split["val"],
        simulate_rough=False,
    )
    images, _, _ = ds.get_numpy_arrays()
    images_25d = ds.get_numpy_25d_arrays()
    img_in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=img_in_ch)
    pipeline.eval()
    stage2_thr = [0.80, 0.80, 0.50]
    cc_min = [0, 15, 25]
    conf = np.zeros(len(images), dtype=np.float64)
    cls_pred = np.zeros(len(images), dtype=np.int8)

    print(f"Computing region confidence on {len(images)} slices ({device})...")
    with torch.no_grad():
        for i in range(len(images)):
            img_np = images_25d[i]
            if img_np.ndim == 2:
                img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
            else:
                img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
            rough_mask_t, class_pred = pipeline(img_t, true_class_preds=None)
            c = int(class_pred.item())
            cls_pred[i] = c
            prob = rough_mask_t.squeeze().cpu().numpy().astype(np.float64)
            mask = (prob > stage2_thr[c]).astype(np.float32)
            mask = filter_small_components(mask, cc_min[c])
            nz = mask > 0.5
            conf[i] = float(prob[nz].mean()) if np.any(nz) else 0.0
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(images)}")

    conf_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(conf_out, conf=conf, cls_pred=cls_pred)
    print(f"saved {conf_out}")
    return conf


def sweep(metrics_path: Path, conf: np.ndarray, gate_classes=(1, 2)) -> None:
    d = np.load(metrics_path)
    init, final, cls = d["init_dsc"], d["final_dsc"], d["cls"].astype(int)
    assert len(conf) == len(init), (len(conf), len(init))
    delta = final - init
    names = {0: "Small", 1: "Medium", 2: "Large"}

    print(f"\n=== source: {metrics_path.name} ===")
    print(f"baseline overall: init={init.mean():.4f} final={final.mean():.4f} Δ={delta.mean():+.4f}")
    for c in (0, 1, 2):
        m = cls == c
        print(
            f"  {names[c]:6s} n={m.sum():5d} conf mean={conf[m].mean():.4f} "
            f"p10={np.quantile(conf[m],0.1):.3f} p50={np.quantile(conf[m],0.5):.3f} "
            f"p90={np.quantile(conf[m],0.9):.3f} | Δ={delta[m].mean():+.4f}"
        )

    # Correlation: low conf vs SL help
    for c in gate_classes:
        m = cls == c
        print(f"\n--- {names[c]}: Δ by conf quartile ---")
        qs = np.quantile(conf[m], [0.0, 0.25, 0.5, 0.75, 1.0])
        for lo, hi in zip(qs[:-1], qs[1:]):
            b = m & (conf >= lo) & (conf <= hi if hi == qs[-1] else conf < hi)
            if b.sum() == 0:
                continue
            dd = delta[b]
            print(
                f"  conf [{lo:.3f},{hi:.3f}) n={b.sum():4d} "
                f"init={init[b].mean():.4f} Δ={dd.mean():+.4f} "
                f"improve={(dd>1e-4).mean()*100:4.1f}% worsen={(dd<-1e-4).mean()*100:4.1f}%"
            )

    print(f"\n--- Post-hoc gate (classes={gate_classes}): high conf → Stage2 ---")
    print(f"{'thr':>6} {'n_ref':>6} {'n_skip':>6} {'overall':>8} {'Δ':>8} {'Med':>8} {'Large':>8}")
    # no gate
    print(
        f"{'none':>6} {len(init):6d} {0:6d} {final.mean():8.4f} {delta.mean():+8.4f} "
        f"{final[cls==1].mean():8.4f} {final[cls==2].mean():8.4f}"
    )
    # always skip med/large
    always = final.copy()
    for c in gate_classes:
        always[cls == c] = init[cls == c]
    print(
        f"{'skipML':>6} {0:6d} {(np.isin(cls, gate_classes)).sum():6d} "
        f"{always.mean():8.4f} {(always-init).mean():+8.4f} "
        f"{always[cls==1].mean():8.4f} {always[cls==2].mean():8.4f}"
    )

    thrs = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
    best = None
    for thr in thrs:
        gated = final.copy()
        skip = np.zeros(len(init), dtype=bool)
        for c in gate_classes:
            skip |= (cls == c) & (conf >= thr)
        gated[skip] = init[skip]
        n_ref = int(((np.isin(cls, gate_classes)) & ~skip).sum())
        n_skip = int(skip.sum())
        ov = gated.mean()
        row = (
            f"{thr:6.2f} {n_ref:6d} {n_skip:6d} {ov:8.4f} {(gated-init).mean():+8.4f} "
            f"{gated[cls==1].mean():8.4f} {gated[cls==2].mean():8.4f}"
        )
        print(row)
        if best is None or ov > best[0]:
            best = (ov, thr, n_ref, n_skip)
    print(f"best overall={best[0]:.4f} @ thr={best[1]} (refine {best[2]}, skip {best[3]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, default="results/pipeline_slice_metrics_deploy.npz")
    p.add_argument("--conf_cache", type=str, default="results/slice_region_confidence_val.npz")
    p.add_argument("--train_root", type=str, default="src/data/archive")
    p.add_argument("--modality", type=str, default="t1ce+flair")
    p.add_argument("--max_patients", type=int, default=1251)
    p.add_argument("--patient_split", type=str, default=DEFAULT_SPLIT_PATH)
    args = p.parse_args()

    conf = compute_conf(
        Path(args.conf_cache),
        args.train_root,
        args.modality,
        args.max_patients,
        args.patient_split,
    )
    sweep(Path(args.metrics), conf, gate_classes=(1, 2))
    # also shrink baseline for reference
    shrink = Path("results/pipeline_slice_metrics_shrink_stdsl.npz")
    if shrink.exists() and Path(args.metrics).resolve() != shrink.resolve():
        sweep(shrink, conf, gate_classes=(1, 2))


if __name__ == "__main__":
    main()
