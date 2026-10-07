"""리뷰 ablation. 평가 프로토콜(빈 슬라이스, 3D Dice, mm 환산)은 바꾸지 않는다.

같은 개발 분할(seed 42, 400명)로 학습한 가중치를, 미사용 851명의 종양 슬라이스에서
환자 평균으로 비교한다. 임계값은 검증 60명에서만 고른다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.eval.evaluate_band_ppo_locked import bootstrap_ci, tta_probability
from src.models.band_refine import build_band_refine, constrain_logits
from src.utils.band import edit_band, flair_z_map
from src.utils.evaluation_records import measured_metrics
from src.utils.metrics import dice, gt_size_class
from src.utils.refinement_inputs import stage2_mask

CLASS_NAMES = {0: "small", 1: "medium", 2: "large"}
PAPER_THR = [0.80, 0.80, 0.50]
CC_SIZES = [0, 15, 25]
GRID = [round(float(x), 2) for x in np.arange(0.10, 0.91, 0.05)]


def _hd_block(pred: np.ndarray, gt: np.ndarray) -> list:
    return [measured_metrics(pred[i], gt[i])["hd95_surface_px"] for i in range(len(pred))]


def slice_hd95(pred: np.ndarray, gt: np.ndarray, workers: int) -> np.ndarray:
    if workers <= 1 or len(pred) < 256:
        vals = _hd_block(pred, gt)
    else:
        chunks = np.array_split(np.arange(len(pred)), workers)
        parts = [(pred[idx], gt[idx]) for idx in chunks if len(idx)]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(parts), mp_context=ctx) as pool:
            blocks = list(pool.map(_hd_block, [p for p, _ in parts], [g for _, g in parts]))
        vals = [v for block in blocks for v in block]
    out = np.full(len(pred), np.nan, dtype=np.float32)
    for i, value in enumerate(vals):
        if value is not None:
            out[i] = value
    return out


def slice_dsc(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_b = pred.reshape(len(pred), -1) > 0.5
    gt_b = gt.reshape(len(gt), -1) > 0.5
    inter = (pred_b & gt_b).sum(axis=1)
    total = pred_b.sum(axis=1) + gt_b.sum(axis=1)
    return ((2.0 * inter + 1e-5) / (total + 1e-5)).astype(np.float32)


def patient_values(pids: list[str], values: np.ndarray, keep: np.ndarray | None = None) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for pid, value, ok in zip(pids, values, np.ones(len(values), dtype=bool) if keep is None else keep):
        if not ok or not np.isfinite(value):
            continue
        grouped.setdefault(pid, []).append(float(value))
    return {pid: float(np.mean(xs)) for pid, xs in grouped.items() if xs}


def pack_values(values: dict[str, float], seed: int) -> dict:
    ids = sorted(values)
    arr = np.array([values[i] for i in ids], dtype=float)
    return {
        "n_patients": len(ids),
        "mean": float(arr.mean()) if len(ids) else float("nan"),
        "ci95": bootstrap_ci(arr, seed),
    }


def summarize(pids, dsc, hd, gt_class, seed: int, patients: set[str] | None = None) -> dict:
    if patients is not None:
        keep = np.array([pid in patients for pid in pids])
    else:
        keep = np.ones(len(pids), dtype=bool)

    def block(name: str | None) -> dict:
        cls_keep = keep if name is None else keep & (gt_class == name)
        dsc_pack = pack_values(patient_values(pids, dsc, cls_keep), seed)
        hd_pack = pack_values(patient_values(pids, hd, cls_keep), seed + 1)
        return {"n_slices": int(cls_keep.sum()), "dsc": dsc_pack, "hd95": hd_pack}

    return {
        "overall": block(None),
        "by_gt_class": {name: block(name) for name in ("small", "medium", "large")},
    }


def paired(a: dict[str, float], b: dict[str, float], seed: int) -> dict:
    ids = sorted(set(a) & set(b))
    delta = np.array([a[i] - b[i] for i in ids], dtype=float)
    return {
        "n_patients": len(ids),
        "delta_mean": float(delta.mean()) if len(ids) else float("nan"),
        "delta_ci95": bootstrap_ci(delta, seed),
    }


def score_method(pids, pred, gt, gt_class, seed, cohorts, workers) -> dict:
    dsc = slice_dsc(pred, gt)
    hd = slice_hd95(pred, gt, workers)
    out = {"all_851": summarize(pids, dsc, hd, gt_class, seed)}
    for name, cohort in cohorts.items():
        out[name] = summarize(pids, dsc, hd, gt_class, seed, cohort)
    out["_dsc"] = dsc
    out["_hd"] = hd
    return out


def public_summary(block: dict) -> dict:
    return {k: v for k, v in block.items() if not k.startswith("_")}


@torch.no_grad()
def class_probability(pipeline, images: torch.Tensor, cls: int, use_tta: bool) -> torch.Tensor:
    route = torch.full((images.shape[0],), cls, device=images.device, dtype=torch.long)
    base, _ = pipeline(images, true_class_preds=route)
    if not use_tta:
        return base
    horizontal = torch.flip(pipeline(torch.flip(images, dims=[3]), true_class_preds=route)[0], dims=[3])
    vertical = torch.flip(pipeline(torch.flip(images, dims=[2]), true_class_preds=route)[0], dims=[2])
    return (base + horizontal + vertical) / 3.0


@torch.no_grad()
def baseline_probability(model, images: torch.Tensor, use_tta: bool) -> torch.Tensor:
    def forward(x):
        return torch.sigmoid(model(x).float())

    base = forward(images)
    if not use_tta:
        return base
    horizontal = torch.flip(forward(torch.flip(images, dims=[3])), dims=[3])
    vertical = torch.flip(forward(torch.flip(images, dims=[2])), dims=[2])
    return (base + horizontal + vertical) / 3.0


def collect_split(dataset, pipeline, baselines, device, batch_size: int) -> dict:
    images, gts, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    pids = list(dataset._sample_pids)
    n = len(gts)
    gt = np.stack([(g > 0.5).astype(np.uint8) for g in gts])
    gt_class = np.array([CLASS_NAMES[gt_size_class(g)] for g in gts])
    center = np.ascontiguousarray(images)
    vol = np.ascontiguousarray(images_25d)
    store = {
        "prob": {},
        "route": np.zeros(n, dtype=np.int64),
    }
    keys = [f"{name}_{kind}" for name in ("small", "medium", "large") for kind in ("base", "tta")]
    for key in keys:
        store["prob"][key] = np.empty((n, gt.shape[1], gt.shape[2]), dtype=np.float16)
    for name in baselines:
        store["prob"][f"{name}_base"] = np.empty_like(store["prob"]["large_base"])
        store["prob"][f"{name}_tta"] = np.empty_like(store["prob"]["large_base"])

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        image_25d = torch.from_numpy(vol[start:stop]).to(device)
        image = torch.from_numpy(center[start:stop]).to(device)
        base, route = pipeline(image_25d)
        prob = tta_probability(pipeline, image_25d, base, route)
        if "routed_base" not in store["prob"]:
            store["prob"]["routed_base"] = np.empty_like(store["prob"]["large_base"])
            store["prob"]["routed_tta"] = np.empty_like(store["prob"]["large_base"])
        store["prob"]["routed_base"][start:stop] = base.squeeze(1).detach().cpu().numpy()
        store["prob"]["routed_tta"][start:stop] = prob.squeeze(1).detach().cpu().numpy()
        store["route"][start:stop] = route.detach().cpu().numpy()
        for cls, name in CLASS_NAMES.items():
            got = class_probability(pipeline, image_25d, cls, False)
            store["prob"][f"{name}_base"][start:stop] = got.squeeze(1).detach().cpu().numpy()
            got = class_probability(pipeline, image_25d, cls, True)
            store["prob"][f"{name}_tta"][start:stop] = got.squeeze(1).detach().cpu().numpy()
        for name, model in baselines.items():
            got = baseline_probability(model, image, False)
            store["prob"][f"{name}_base"][start:stop] = got.squeeze(1).detach().cpu().numpy()
            got = baseline_probability(model, image, True)
            store["prob"][f"{name}_tta"][start:stop] = got.squeeze(1).detach().cpu().numpy()
        if start % (batch_size * 20) == 0:
            print(f"  확률맵 {stop}/{n}", flush=True)
    store["pid"] = pids
    store["gt"] = gt
    store["gt_class"] = gt_class
    store["image"] = center.astype(np.float16)
    return store


def best_threshold(prob: np.ndarray, gt: np.ndarray, grid) -> tuple[float, float]:
    best_thr, best_dsc = float(grid[0]), -1.0
    for thr in grid:
        dsc = float(slice_dsc((prob > thr).astype(np.float32), gt).mean())
        if dsc > best_dsc:
            best_thr, best_dsc = float(thr), dsc
    return best_thr, best_dsc


def routed_masks(prob_map: dict, routes: np.ndarray, thresholds, ccs, prefix: str) -> np.ndarray:
    n, height, width = prob_map[f"small_{prefix}"].shape
    out = np.empty((n, height, width), dtype=np.float32)
    for index in range(n):
        route = int(routes[index])
        name = CLASS_NAMES[route]
        out[index] = stage2_mask(prob_map[f"{name}_{prefix}"][index].astype(np.float32), route, thresholds, ccs)
    return out


def single_masks(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob.astype(np.float32) > threshold).astype(np.float32)


def large_stage2_masks(prob: np.ndarray) -> np.ndarray:
    """대형 전문가의 논문 임계값 0.50과 연결요소 25를 모든 슬라이스에 적용한다."""
    out = np.empty(prob.shape, dtype=np.float32)
    for index in range(len(prob)):
        out[index] = stage2_mask(prob[index].astype(np.float32), 2, PAPER_THR, CC_SIZES)
    return out


@torch.no_grad()
def refine_masks(actor, image, prob, mask, n_steps, prob_lo, prob_hi, radius, guard_last, flair_index, batch_size, device):
    current = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(1).to(device)
    prob_t = torch.from_numpy(np.ascontiguousarray(prob)).unsqueeze(1).to(device)
    image_t = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    flips = np.zeros(n_steps, dtype=np.int64)
    band_pixels = np.zeros(n_steps, dtype=np.int64)
    changed_slices = np.zeros(n_steps, dtype=np.int64)
    for start in range(0, current.shape[0], batch_size):
        stop = min(start + batch_size, current.shape[0])
        step_mask = current[start:stop]
        step_prob = prob_t[start:stop]
        step_image = image_t[start:stop]
        flair = step_image[:, flair_index:flair_index + 1]
        for step in range(n_steps):
            logits = actor(torch.cat([step_image, step_mask, step_prob], dim=1))
            bands, zs = [], []
            prob_np = step_prob[:, 0].detach().cpu().numpy()
            mask_np = step_mask[:, 0].detach().cpu().numpy()
            flair_np = flair[:, 0].detach().cpu().numpy()
            for index in range(step_mask.shape[0]):
                band = edit_band(prob_np[index], mask_np[index], prob_lo, prob_hi, radius)
                bands.append(band)
                zs.append(flair_z_map(flair_np[index], band))
            band_t = torch.from_numpy(np.stack(bands)).to(device)
            z_t = torch.from_numpy(np.stack(zs)).to(device)
            logits = constrain_logits(logits, band_t, z_t, guard=guard_last and step == n_steps - 1)
            actions = logits.argmax(dim=1)
            updated = step_mask.clone()
            plane = updated[:, 0]
            editable = band_t.to(dtype=torch.bool)
            plane[editable & (actions == 2)] = 1.0
            plane[editable & (actions == 0)] = 0.0
            delta = (updated - step_mask).abs().sum(dim=(1, 2, 3))
            flips[step] += int((updated - step_mask).abs().sum().item())
            band_pixels[step] += int(editable.sum().item())
            changed_slices[step] += int((delta > 0).sum().item())
            step_mask = updated
        current[start:stop] = step_mask
    return current[:, 0].detach().cpu().numpy().astype(np.float32), {
        "flips_per_step": flips.tolist(),
        "band_pixels_per_step": band_pixels.tolist(),
        "changed_slices_per_step": changed_slices.tolist(),
        "n_slices": int(current.shape[0]),
    }


def load_actor(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(ckpt["model"])
    actor.eval()
    return actor, cfg


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="리뷰 지적 ablation (평가 프로토콜 4.5 제외)")
    parser.add_argument("--train_root", default="src/data/archive")
    parser.add_argument("--patient_split", default="checkpoints/patient_split.json")
    parser.add_argument("--pool210", default="checkpoints/patient_split_210.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/ablation_review/summary.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.patient_split, encoding="utf-8") as f:
        split = json.load(f)
    with open(args.pool210, encoding="utf-8") as f:
        old = json.load(f)
    from src.data.patient_split import list_patient_ids
    all_ids = set(list_patient_ids(args.train_root))
    used = set(split["train"]) | set(split["val"]) | set(split["test"])
    held = sorted(all_ids - used)
    old_pool = set(old["train"]) | set(old["val"]) | set(old.get("test", []))
    overlap = set(held) & old_pool
    independent = set(held) - old_pool
    cohorts = {"independent_703": independent, "overlap_148": overlap}

    from src.data.brats2020_dataset import BraTS2020Dataset
    from src.models.dynamic_router import AdaptivePipeline

    print(f"held {len(held)} overlap {len(overlap)} independent {len(independent)}", flush=True)
    held_ds = BraTS2020Dataset(
        args.train_root, modality="t1ce+flair", target_size=128,
        patient_ids=held, slice_selection="tumor", simulate_rough=False, num_workers=16,
    )
    val_ds = BraTS2020Dataset(
        args.train_root, modality="t1ce+flair", target_size=128,
        patient_ids=split["val"], slice_selection="tumor", simulate_rough=False, num_workers=8,
    )
    pipeline = AdaptivePipeline(device, in_channels=2, strict_checkpoints=True).eval()
    baselines = {}
    unet_path = "checkpoints/unet_best.pt"
    if os.path.exists(unet_path):
        from src.models import build_unet
        state = torch.load(unet_path, map_location=device, weights_only=True)
        model = build_unet(in_channels=2, out_channels=1).to(device)
        model.load_state_dict(state)
        model.eval()
        baselines["unet"] = model
    else:
        print(f"건너뜀: {unet_path}", flush=True)

    print("검증 확률맵", flush=True)
    val = collect_split(val_ds, pipeline, baselines, device, args.batch_size)
    print("851명 확률맵", flush=True)
    test = collect_split(held_ds, pipeline, baselines, device, args.batch_size)
    del held_ds, val_ds
    torch.cuda.empty_cache()

    grids = {}
    chosen = {}
    for key in ("large_base", "large_tta", "unet_base", "unet_tta"):
        if key not in val["prob"]:
            continue
        full_thr, full_dsc = best_threshold(val["prob"][key].astype(np.float32), val["gt"], GRID)
        paper_grid = [t for t in GRID if t >= 0.30]
        paper_thr, paper_dsc = best_threshold(val["prob"][key].astype(np.float32), val["gt"], paper_grid)
        grids[key] = {
            "extended_0.10_0.90": {"threshold": full_thr, "val_slice_dsc": full_dsc},
            "paper_grid_0.30_0.90": {"threshold": paper_thr, "val_slice_dsc": paper_dsc},
            "boundary_hit": bool(full_thr <= 0.10 + 1e-6 or paper_thr <= 0.30 + 1e-6),
        }
        chosen[key] = full_thr

    ppo, ppo_cfg = load_actor("checkpoints/band_ppo.pt", device)
    supervised, _ = load_actor("checkpoints/band_refine_medium.pt", device)
    supervised_ft = None
    if os.path.exists("checkpoints/band_supervised_ft.pt"):
        supervised_ft, _ = load_actor("checkpoints/band_supervised_ft.pt", device)

    methods = {}
    arrays = {}

    def add(name, pred, seed):
        print(f"지표 {name}", flush=True)
        scored = score_method(
            test["pid"], pred, test["gt"], test["gt_class"], seed, cohorts, args.workers,
        )
        arrays[name] = (scored.pop("_dsc"), scored.pop("_hd"))
        methods[name] = scored

    routed_tta = routed_masks(test["prob"], test["route"], PAPER_THR, CC_SIZES, "tta")
    add("trio_stage2_fixed", routed_tta, args.seed)
    oracle_route = np.array([{"small": 0, "medium": 1, "large": 2}[c] for c in test["gt_class"]], dtype=np.int64)
    add("trio_stage2_oracle_route", routed_masks(test["prob"], oracle_route, PAPER_THR, CC_SIZES, "tta"), args.seed + 1)
    add("large_all_paper_thr", large_stage2_masks(test["prob"]["large_tta"]), args.seed + 2)
    if "large_tta" in chosen:
        add("large_all_val_thr", single_masks(test["prob"]["large_tta"], chosen["large_tta"]), args.seed + 3)
        add("large_all_val_thr_no_tta", single_masks(test["prob"]["large_base"], chosen["large_base"]), args.seed + 4)

    for name in ("unet",):
        for kind in ("base", "tta"):
            key = f"{name}_{kind}"
            if key not in chosen:
                continue
            add(f"{name}_{kind}_valthr", single_masks(test["prob"][key], chosen[key]), args.seed + 5)
            if kind == "base":
                paper_thr = grids[key]["paper_grid_0.30_0.90"]["threshold"]
                add(f"{name}_papergrid_no_tta", single_masks(test["prob"][key], paper_thr), args.seed + 6)

    image = test["image"].astype(np.float32)
    prob = test["prob"]["routed_tta"].astype(np.float32)

    def run_refine(actor, init, source_prob, n_steps, prob_lo, prob_hi, radius, guard, tag):
        refined, stats = refine_masks(
            actor, image, source_prob, init, n_steps, prob_lo, prob_hi, radius, guard,
            int(ppo_cfg.get("flair_index", 1)), args.batch_size, device,
        )
        add(tag, refined, args.seed + 7)
        return stats

    flip_stats = {}
    flip_stats["ppo5"] = run_refine(ppo, routed_tta, prob, 5, 0.35, 0.65, 2, True, "trio_ppo_5step")
    flip_stats["ppo1"] = run_refine(ppo, routed_tta, prob, 1, 0.35, 0.65, 2, True, "trio_ppo_1step")
    flip_stats["supervised_medium_1step"] = run_refine(
        supervised, routed_tta, prob, 1, 0.35, 0.65, 2, True, "supervised_medium_1step",
    )
    if supervised_ft is not None:
        flip_stats["supervised_ft_1step"] = run_refine(
            supervised_ft, routed_tta, prob, 1, 0.35, 0.65, 2, True, "supervised_ft_allclass_1step",
        )
        flip_stats["supervised_ft_5step"] = run_refine(
            supervised_ft, routed_tta, prob, 5, 0.35, 0.65, 2, True, "supervised_ft_allclass_5step",
        )
    oracle_init = routed_masks(test["prob"], oracle_route, PAPER_THR, CC_SIZES, "tta")
    oracle_prob = np.empty_like(prob)
    for index, route in enumerate(oracle_route):
        oracle_prob[index] = test["prob"][f"{CLASS_NAMES[int(route)]}_tta"][index]
    flip_stats["oracle_ppo5"] = run_refine(
        ppo, oracle_init, oracle_prob, 5, 0.35, 0.65, 2, True, "oracle_route_ppo_5step",
    )
    if "large_tta" in chosen:
        large_mask = single_masks(test["prob"]["large_tta"], chosen["large_tta"])
        large_prob = test["prob"]["large_tta"].astype(np.float32)
        flip_stats["large_band"] = run_refine(
            ppo, large_mask, large_prob, 5, 0.35, 0.65, 2, True, "large_all_plus_band_ppo",
        )

    design = {}
    for tag, steps, lo, hi, radius, guard in (
        ("steps_3", 3, 0.35, 0.65, 2, True),
        ("guard_off", 5, 0.35, 0.65, 2, False),
        ("radius_0", 5, 0.35, 0.65, 0, True),
        ("radius_4", 5, 0.35, 0.65, 4, True),
        ("band_0.45_0.55", 5, 0.45, 0.55, 2, True),
        ("band_0.25_0.75", 5, 0.25, 0.75, 2, True),
    ):
        design[tag] = run_refine(ppo, routed_tta, prob, steps, lo, hi, radius, guard, f"design_{tag}")

    # 크기 경계는 재학습하지 않고, 같은 예측을 다른 면적 구간으로만 다시 집계한다.
    area = test["gt"].reshape(len(test["gt"]), -1).sum(axis=1)
    boundary_bins = {}
    dsc, hd = arrays["trio_ppo_5step"]
    for label, lo, hi in (("200_600", 200, 600), ("300_700", 300, 700), ("400_800", 400, 800)):
        names = np.array(["small"] * len(area), dtype=object)
        names[(area >= lo) & (area < hi)] = "medium"
        names[area >= hi] = "large"
        boundary_bins[label] = summarize(test["pid"], dsc, hd, names, args.seed)["by_gt_class"]

    def means(name, cohort="all_851"):
        return patient_values(test["pid"], arrays[name][0], None if cohort == "all_851" else np.array([pid in cohorts[cohort] for pid in test["pid"]]))

    comparisons = {}
    left = "trio_ppo_5step"
    rights = [
        "trio_stage2_fixed", "trio_ppo_1step", "supervised_medium_1step",
        "supervised_ft_allclass_1step", "supervised_ft_allclass_5step",
        "large_all_val_thr", "large_all_paper_thr", "large_all_plus_band_ppo",
        "unet_base_valthr", "oracle_route_ppo_5step", "design_guard_off",
    ]
    for right in rights:
        if right not in arrays or left not in arrays:
            continue
        comparisons[f"{left}_minus_{right}"] = paired(means(left), means(right), args.seed)
        both = np.isfinite(arrays[left][1]) & np.isfinite(arrays[right][1])
        hd_left = patient_values(test["pid"], arrays[left][1] - arrays[right][1], both)
        comparisons[f"hd95_{left}_minus_{right}"] = paired(
            {k: v for k, v in hd_left.items()},
            {k: 0.0 for k in hd_left},
            args.seed + 3,
        )

    # 48명 슬라이스 평균. 표 7과 같은 표본 규칙이며 PPO 없는 백본만 둔다.
    rng = np.random.default_rng(7)
    sample = set(rng.choice(held, size=48, replace=False).tolist())
    sample_keep = np.array([pid in sample for pid in test["pid"]])
    backbone48 = {}
    for name in ("trio_stage2_fixed", "trio_ppo_5step", "large_all_val_thr", "unet_base_valthr"):
        if name not in arrays:
            continue
        dsc, hd = arrays[name]
        row = {}
        for cls in ("small", "medium", "large"):
            keep = sample_keep & (test["gt_class"] == cls)
            row[cls] = {
                "n_slices": int(keep.sum()),
                "slice_dsc": float(dsc[keep].mean()) if keep.any() else float("nan"),
                "slice_hd95": float(np.nanmean(hd[keep])) if keep.any() else float("nan"),
            }
        backbone48[name] = row

    starter = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    ender = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    def time_ms(fn) -> float:
        for _ in range(3):
            fn()
        if starter is None:
            t0 = time.perf_counter()
            fn()
            return (time.perf_counter() - t0) * 1000
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        return float(starter.elapsed_time(ender))

    # 타이밍은 실제 2.5D 입력이 필요하다. image를 3번 이어 채널을 맞춘다.
    image_b = torch.from_numpy(image[:16]).to(device)
    image_25d = image_b.repeat(1, 3, 1, 1)[:, :6]
    cost = {
        "classifier_params": count_params(pipeline.classifier),
        "small_params": count_params(pipeline.expert_small),
        "medium_params": count_params(pipeline.expert_medium),
        "large_params": count_params(pipeline.expert_large),
        "band_params": count_params(ppo),
        "trio_params": count_params(pipeline) + count_params(ppo),
    }
    if "unet" in baselines:
        cost["unet_params"] = count_params(baselines["unet"])
        cost["unet_ms_per_16"] = time_ms(lambda: baseline_probability(baselines["unet"], image_b, False))
    cost["trio_route_tta_ms_per_16"] = time_ms(lambda: tta_probability(pipeline, image_25d, *pipeline(image_25d)))
    init = routed_tta[:16]
    cost["band_5step_ms_per_16"] = time_ms(lambda: refine_masks(
        ppo, image[:16], prob[:16], init, 5, 0.35, 0.65, 2, True, 1, 16, device,
    ))

    routing_acc = float(np.mean(test["route"] == oracle_route))
    payload = {
        "protocol": {
            "excluded": "빈 슬라이스 평가, 3D Dice, HD95 mm 환산은 하지 않았다.",
            "split": "seed 42, 개발 400명(train 280/val 60/test 60), 평가 851명 종양 슬라이스 환자 평균",
            "heldout_patients": len(held),
            "overlap_with_210_pool": len(overlap),
            "independent_patients": len(independent),
            "routing_slice_accuracy": routing_acc,
            "n_slices": int(len(test["pid"])),
            "note": "띠 범위·반경·스텝·가드는 학습 설정이 아니라 추론 제약만 바꿨다. 300/700 외 구간도 재학습이 아니다.",
        },
        "threshold_grids": grids,
        "methods": {k: public_summary(v) for k, v in methods.items()},
        "paired_patient_dsc": {k: v for k, v in comparisons.items() if not k.startswith("hd95_")},
        "paired_patient_hd95_common_slices": {k: v for k, v in comparisons.items() if k.startswith("hd95_")},
        "flip_stats": flip_stats,
        "boundary_bin_sensitivity_ppo5": boundary_bins,
        "backbone_only_48_slice_mean": backbone48,
        "cost": cost,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"저장 {args.out}", flush=True)


if __name__ == "__main__":
    main()
