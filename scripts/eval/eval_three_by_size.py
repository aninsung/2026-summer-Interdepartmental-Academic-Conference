"""851명 환자 평균. TRIO / KAIST / NVAUTO 를 정답 크기별로 DSC, HD95, P, R."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baselines.eval_baseline import load_baseline
from baselines.eval_heldout import held_out_ids, val_threshold

THRESHOLDS = [0.80, 0.80, 0.50]
CC_SIZES = [0, 15, 25]


def tta_probability(pipeline, image, base, route):
    with torch.no_grad():
        horizontal = torch.flip(pipeline(torch.flip(image, dims=[3]), true_class_preds=route)[0], dims=[3])
        vertical = torch.flip(pipeline(torch.flip(image, dims=[2]), true_class_preds=route)[0], dims=[2])
        return (base + horizontal + vertical) / 3.0
from src.models.band_refine import build_band_refine, refine_batch
from src.utils.evaluation_records import measured_metrics
from src.utils.metrics import gt_size_class
from src.utils.refinement_inputs import stage2_mask

CLASS = {0: "small", 1: "medium", 2: "large"}
ORDER = ("overall", "small", "medium", "large")


def add(rows, pid, gt, metrics, prefix):
    rows.append({
        "patient_id": pid,
        "gt_class": CLASS[gt_size_class(gt)],
        f"{prefix}_dsc": metrics["dsc"],
        f"{prefix}_hd95": metrics["hd95_surface_px"],
        f"{prefix}_p": metrics["precision"],
        f"{prefix}_r": metrics["recall"],
    })


def merge(a, b):
    out = []
    for left, right in zip(a, b):
        row = dict(left)
        row.update(right)
        out.append(row)
    return out


def means(rows, prefix, class_name):
    picked = rows if class_name == "overall" else [r for r in rows if r["gt_class"] == class_name]
    grouped = {}
    for row in picked:
        grouped.setdefault(row["patient_id"], []).append(row)
    dsc, hd, p, r = [], [], [], []
    for group in grouped.values():
        dsc.append(float(np.mean([x[f"{prefix}_dsc"] for x in group])))
        h = [x[f"{prefix}_hd95"] for x in group if x[f"{prefix}_hd95"] is not None]
        if h:
            hd.append(float(np.mean(h)))
        p.append(float(np.mean([x[f"{prefix}_p"] for x in group])))
        r.append(float(np.mean([x[f"{prefix}_r"] for x in group])))
    return {
        "n": len(dsc),
        "dsc": float(np.mean(dsc)),
        "hd95": float(np.mean(hd)) if hd else float("nan"),
        "p": float(np.mean(p)),
        "r": float(np.mean(r)),
    }


def main():
    root = "src/data/archive"
    split_path = "checkpoints/patient_split.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    held = held_out_ids(root, split_path)
    from src.data.brats2020_dataset import BraTS2020Dataset
    dataset = BraTS2020Dataset(
        root_dir=root, modality="t1ce+flair", target_size=128,
        patient_ids=held, slice_selection="tumor", simulate_rough=False, num_workers=8,
    )
    images, gts, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    pids = list(dataset._sample_pids)
    print(f"patients {len(held)} slices {len(gts)}", flush=True)

    base_rows = []
    for name, ckpt in (("kaist", "baselines/checkpoints/kaist_best.pt"), ("nvauto", "baselines/checkpoints/nvauto_best.pt")):
        model, _, target = load_baseline(ckpt, device)
        thr = val_threshold(model, root, split_path, "t1ce+flair", target, device, 16)
        print(f"{name} thr {thr:.2f}", flush=True)
        rows = []
        with torch.no_grad():
            for start in range(0, len(images), 16):
                stop = min(start + 16, len(images))
                batch = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
                prob = torch.sigmoid(model(batch).float()).squeeze(1).cpu().numpy()
                for offset, index in enumerate(range(start, stop)):
                    pred = (prob[offset] > thr).astype(np.float32)
                    add(rows, pids[index], gts[index], measured_metrics(pred, gts[index]), name)
        base_rows.append(rows)
        del model
        torch.cuda.empty_cache()

    from src.models.dynamic_router import AdaptivePipeline
    pipeline = AdaptivePipeline(device, in_channels=images.shape[1], strict_checkpoints=True)
    pipeline.eval()
    ckpt = torch.load("checkpoints/band_ppo.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(ckpt["model"])
    actor.eval()
    trio = []
    for start in range(0, len(images), 8):
        stop = min(start + 8, len(images))
        image_25d = torch.from_numpy(np.ascontiguousarray(images_25d[start:stop])).to(device)
        with torch.no_grad():
            base, route = pipeline(image_25d)
            prob = tta_probability(pipeline, image_25d, base, route)
        prob_np = prob.squeeze(1).detach().cpu().numpy()
        route_np = route.detach().cpu().numpy()
        masks = [stage2_mask(prob_np[i], int(route_np[i]), THRESHOLDS, CC_SIZES) for i in range(stop - start)]
        center = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
        refined = refine_batch(
            actor, center,
            torch.from_numpy(np.ascontiguousarray(prob_np)).unsqueeze(1).to(device),
            torch.from_numpy(np.stack(masks).astype(np.float32)).unsqueeze(1).to(device),
            center[:, cfg["flair_index"]:cfg["flair_index"] + 1],
            int(cfg["n_steps"]), float(cfg["prob_lo"]), float(cfg["prob_hi"]), int(cfg["radius"]),
            guard_last=True,
        ).detach().cpu().numpy()
        for local, index in enumerate(range(start, stop)):
            add(trio, pids[index], gts[index], measured_metrics(refined[local, 0], gts[index]), "trio")
        if start % 4000 == 0:
            print(f"TRIO {stop}/{len(images)}", flush=True)

    rows = merge(merge(trio, base_rows[0]), base_rows[1])
    summary = {
        name: {cls: means(rows, name, cls) for cls in ORDER}
        for name in ("trio", "kaist", "nvauto")
    }
    os.makedirs("results", exist_ok=True)
    with open("results/three_by_size.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("method class n DSC HD95 P R", flush=True)
    for name in ("trio", "kaist", "nvauto"):
        for cls in ORDER:
            m = summary[name][cls]
            print(f"{name} {cls} {m['n']} {m['dsc']:.4f} {m['hd95']:.3f} {m['p']:.4f} {m['r']:.4f}", flush=True)


if __name__ == "__main__":
    main()
