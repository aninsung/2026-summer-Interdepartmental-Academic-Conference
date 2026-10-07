"""단일 백본+단일 PPO 다섯 개를 크기 행, 모델 열로 그린다."""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import list_patient_ids
from src.envs.vanilla_refine_env import VanillaRefineEnv
from src.models.attention_unet import build_attention_unet
from src.models.segresnet import build_segresnet
from src.models.unet import build_unet
from src.models.unet3plus import build_unet3plus
from src.models.unetplusplus import build_unetplusplus
from src.utils.evaluation_records import measured_metrics
from src.utils.metrics import dice, gt_size_class

MODELS = [
    ("unet", "U-Net", "checkpoints/single_backbone/unet.pt", "checkpoints/single_backbone/ppo_unet"),
    ("unetplusplus", "UNet++", "checkpoints/single_backbone/unetplusplus.pt", "checkpoints/single_backbone/ppo_unetplusplus"),
    ("unet3plus", "UNet+++", "checkpoints/single_backbone/unet3plus.pt", "checkpoints/single_backbone/ppo_unet3plus"),
    ("attention_unet", "Attention U-Net", "checkpoints/single_backbone/attention_unet.pt", "checkpoints/single_backbone/ppo_attention_unet"),
    ("segresnet", "SegResNet", "checkpoints/single_backbone/segresnet.pt", "checkpoints/single_backbone/ppo_segresnet"),
]
ROW_NAMES = ["Small", "Medium", "Large"]


def build_model(model_type: str, in_channels: int, device: torch.device):
    builders = {
        "unet": build_unet,
        "unetplusplus": build_unetplusplus,
        "unet3plus": lambda **kw: build_unet3plus(DSV=False, **kw),
        "segresnet": build_segresnet,
        "attention_unet": build_attention_unet,
    }
    model = builders[model_type](in_channels=in_channels, out_channels=1).to(device)
    model.eval()
    return model


def predict_prob(model, images: np.ndarray, device: torch.device, batch_size: int = 32) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            stop = min(start + batch_size, len(images))
            batch = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
            probs.append(torch.sigmoid(model(batch).float()).squeeze(1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


def rl_mask(agent, image, rough, gt, prob, max_steps: int) -> np.ndarray:
    env = VanillaRefineEnv(
        images=image[None], gt_masks=gt[None], rough_masks=rough[None], probability_maps=prob[None],
        max_steps=max_steps,
    )
    obs, _ = env.reset(seed=0)
    for _ in range(max_steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return env._current_mask.copy()


def infer_trio(images, images_25d, device: torch.device) -> np.ndarray:
    from src.models.band_refine import build_band_refine, refine_batch
    from src.models.dynamic_router import AdaptivePipeline
    from src.utils.refinement_inputs import stage2_mask

    pipeline = AdaptivePipeline(device, in_channels=images_25d.shape[1], strict_checkpoints=True)
    pipeline.eval()
    ckpt = torch.load("checkpoints/band_ppo.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(ckpt["model"])
    actor.eval()
    thresholds, cc_sizes = [0.80, 0.80, 0.50], [0, 15, 25]
    out = np.zeros((len(images), 128, 128), dtype=np.float32)
    batch_size = 8
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        image_25d = torch.from_numpy(np.ascontiguousarray(images_25d[start:stop])).to(device)
        with torch.no_grad():
            base, route = pipeline(image_25d)
            flipped_h = torch.flip(image_25d, dims=[3])
            flipped_v = torch.flip(image_25d, dims=[2])
            horizontal = torch.flip(pipeline(flipped_h, true_class_preds=route)[0], dims=[3])
            vertical = torch.flip(pipeline(flipped_v, true_class_preds=route)[0], dims=[2])
            prob = ((base + horizontal + vertical) / 3.0).squeeze(1).detach().cpu().numpy()
        route_np = route.detach().cpu().numpy()
        masks = [
            stage2_mask(prob[offset], int(route_np[offset]), thresholds, cc_sizes)
            for offset in range(stop - start)
        ]
        center = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device)
        refined = refine_batch(
            actor, center,
            torch.from_numpy(np.ascontiguousarray(prob)).unsqueeze(1).to(device),
            torch.from_numpy(np.stack(masks).astype(np.float32)).unsqueeze(1).to(device),
            center[:, cfg["flair_index"]:cfg["flair_index"] + 1],
            int(cfg["n_steps"]), float(cfg["prob_lo"]), float(cfg["prob_hi"]), int(cfg["radius"]),
            guard_last=True,
        ).detach().cpu().numpy()
        out[start:stop] = refined[:, 0]
    return out


def class_means(scores: np.ndarray, hd95: np.ndarray, gt_cls: np.ndarray) -> tuple[list[float], list[float]]:
    dsc_means, hd_means = [], []
    for cls in (0, 1, 2):
        picked = scores[gt_cls == cls]
        dsc_means.append(float(picked.mean()) if len(picked) else 0.0)
        hd = hd95[gt_cls == cls]
        hd = hd[np.isfinite(hd)]
        hd_means.append(float(hd.mean()) if len(hd) else float("nan"))
    return dsc_means, hd_means


def crop_box(gt: np.ndarray, margin: int = 12):
    ys, xs = np.where(gt > 0.5)
    if len(ys) == 0:
        return 0, gt.shape[1] - 1, 0, gt.shape[0] - 1
    return (
        max(0, int(xs.min()) - margin), min(gt.shape[1] - 1, int(xs.max()) + margin),
        max(0, int(ys.min()) - margin), min(gt.shape[0] - 1, int(ys.max()) + margin),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", default="src/data/archive")
    parser.add_argument("--patient_split", default="checkpoints/patient_split.json")
    parser.add_argument("--sample_patients", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--out", default="results/single_backbone_ppo_grid.png")
    args = parser.parse_args()

    import json
    with open(args.patient_split, encoding="utf-8") as f:
        split = json.load(f)
    used = set(split["train"]) | set(split["val"]) | set(split["test"])
    held = sorted(set(list_patient_ids(args.train_root)) - used)
    rng = np.random.default_rng(args.seed)
    sample = sorted(rng.choice(held, size=min(args.sample_patients, len(held)), replace=False).tolist())
    dataset = BraTS2020Dataset(
        root_dir=args.train_root, modality="t1ce+flair", target_size=128,
        patient_ids=sample, slice_selection="tumor", simulate_rough=False, num_workers=4,
    )
    images, gts, _ = dataset.get_numpy_arrays()
    images_25d = dataset.get_numpy_25d_arrays()
    gt_cls = np.array([gt_size_class(g) for g in gts], dtype=np.int32)
    print(f"평가 슬라이스 {len(gts)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def pack(title: str, masks: np.ndarray) -> dict:
        scores = np.array([dice(masks[i], gts[i]) for i in range(len(gts))], dtype=np.float32)
        hd_values = []
        for i in range(len(gts)):
            value = measured_metrics(masks[i], gts[i])["hd95_surface_px"]
            hd_values.append(np.nan if value is None else value)
        hd95 = np.array(hd_values, dtype=np.float32)
        dsc_means, hd_means = class_means(scores, hd95, gt_cls)
        print(title, "DSC", [round(m, 3) for m in dsc_means], "HD95", [round(m, 3) for m in hd_means], flush=True)
        return {"title": title, "masks": masks, "scores": scores, "means": dsc_means, "hd_means": hd_means}

    trio = infer_trio(images, images_25d, device)
    stored = [pack("TRIO", trio)]
    del trio
    torch.cuda.empty_cache()

    for key, title, weight, agent_path in MODELS:
        model = build_model(key, images.shape[1], device)
        model.load_state_dict(torch.load(weight, map_location=device, weights_only=True))
        prob = predict_prob(model, images, device)
        rough = (prob > 0.5).astype(np.float32)
        agent = PPO.load(agent_path, device=str(device))
        masks = np.zeros_like(rough)
        for i in range(len(gts)):
            masks[i] = rl_mask(agent, images[i], rough[i], gts[i], prob[i], args.max_steps)
            if (i + 1) % 400 == 0:
                print(f"  {title} {i+1}/{len(gts)}", flush=True)
        stored.append(pack(title, masks))
        del model, agent
        torch.cuda.empty_cache()

    chosen = []
    for cls in (0, 1, 2):
        idx = np.where(gt_cls == cls)[0]
        dist = np.zeros(len(idx), dtype=np.float64)
        for item in stored:
            dist += np.abs(item["scores"][idx] - item["means"][cls])
        chosen.append(int(idx[int(np.argmin(dist))]))

    fig, axes = plt.subplots(3, len(stored), figsize=(4.6 * len(stored), 13.5))
    for col, item in enumerate(stored):
        for row, index in enumerate(chosen):
            ax = axes[row, col]
            center = images[index]
            img = center[1] if center.ndim == 3 and center.shape[0] > 1 else (center[0] if center.ndim == 3 else center)
            gt = gts[index]
            mask = item["masks"][index]
            x0, x1, y0, y1 = crop_box(gt)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.contour(gt, levels=[0.5], colors="#39ff14", linewidths=1.6)
            ax.contour(mask, levels=[0.5], colors="#00e5ff", linewidths=1.6)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(item["title"], fontsize=22, fontweight="bold", pad=12)
            if col == 0:
                ax.set_ylabel(ROW_NAMES[row], fontsize=22, fontweight="bold", labelpad=12)
            ax.set_xlabel(
                f"RL DSC={item['scores'][index]:.3f}\n(mean={item['means'][row]:.3f})",
                fontsize=16,
                fontweight="bold",
                labelpad=8,
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="white")
    summary = {
        "rows": ROW_NAMES,
        "aggregation": "slice mean on the held-out sample used for the figure",
        "n_slices": int(len(gts)),
        "models": [
            {
                "name": item["title"],
                "dsc_mean": {ROW_NAMES[i]: item["means"][i] for i in range(3)},
                "hd95_mean": {ROW_NAMES[i]: item["hd_means"][i] for i in range(3)},
            }
            for item in stored
        ],
    }
    metrics_path = os.path.splitext(args.out)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print(f"저장: {args.out}", flush=True)
    print(f"저장: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
