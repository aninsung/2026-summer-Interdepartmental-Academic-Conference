"""클래스 공통 경계 띠 PPO.

정책은 하나다. Small, Medium, Large를 같이 보고, 시작 가중치는 Medium 지도학습
체크포인트다. 보상은 뒤집은 픽셀이 GT와 맞는지와, 그 스텝에서 HD95가 줄어든 양이다.
가드는 마지막 스텝의 행동 공간에만 건다.

사용 예:
  BRATS_NUM_WORKERS=16 python -u scripts/train/train_band_ppo.py \
      --pretrained checkpoints/band_refine_medium.pt \
      --save_path checkpoints/band_ppo.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.contour_dataset import build_stage2_entries
from src.models.band_refine import (
    BandActorCritic,
    build_band_refine,
    constrain_logits,
    refine_batch,
)
from src.utils.band import edit_band, flair_z_map, pixel_flip_reward
from src.utils.evaluation_records import measured_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = {0: "small", 1: "medium", 2: "large"}


def _hd(mask: np.ndarray, gt: np.ndarray) -> float | None:
    return measured_metrics(mask, gt)["hd95_surface_px"]


def _tensors(entries, device):
    image = torch.stack([
        torch.from_numpy(np.asarray(e.image, dtype=np.float32)) for e in entries
    ]).to(device)
    prob = torch.stack([
        torch.from_numpy(np.asarray(e.probability, dtype=np.float32)) for e in entries
    ]).unsqueeze(1).to(device)
    mask = torch.stack([
        torch.from_numpy(np.asarray(e.stage2, dtype=np.float32)) for e in entries
    ]).unsqueeze(1).to(device)
    return image, prob, mask


@torch.no_grad()
def evaluate_actor(actor, entries, device, args) -> dict:
    buckets: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    actor.eval()
    for start in range(0, len(entries), args.eval_batch):
        chunk = entries[start:start + args.eval_batch]
        image, prob, mask = _tensors(chunk, device)
        flair = image[:, args.flair_index:args.flair_index + 1]
        pred = refine_batch(
            actor, image, prob, mask, flair, args.n_steps,
            args.prob_lo, args.prob_hi, args.radius, guard_last=True,
        )
        pred_np = pred[:, 0].detach().cpu().numpy()
        for index, entry in enumerate(chunk):
            buckets[entry.route].append((
                measured_metrics(entry.stage2, entry.gt),
                measured_metrics(pred_np[index], entry.gt),
            ))

    def mean_metric(pairs, key, source):
        if key == "dsc":
            vals = [pair[source][key] for pair in pairs]
        else:
            vals = [pair[source][key] for pair in pairs if pair[source][key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {}
    for route, pairs in sorted(buckets.items()):
        summary[CLASS_NAMES[route]] = {
            "n": len(pairs),
            "stage2_dsc": mean_metric(pairs, "dsc", 0),
            "ppo_dsc": mean_metric(pairs, "dsc", 1),
            "stage2_hd95": mean_metric(pairs, "hd95_surface_px", 0),
            "ppo_hd95": mean_metric(pairs, "hd95_surface_px", 1),
        }
    classes = [v for k, v in summary.items() if k in CLASS_NAMES.values()]
    summary["macro_stage2_dsc"] = float(np.mean([c["stage2_dsc"] for c in classes]))
    summary["macro_ppo_dsc"] = float(np.mean([c["ppo_dsc"] for c in classes]))
    summary["macro_stage2_hd95"] = float(np.mean([c["stage2_hd95"] for c in classes]))
    summary["macro_ppo_hd95"] = float(np.mean([c["ppo_hd95"] for c in classes]))
    return summary


@torch.no_grad()
def collect_rollout(model, entries, device, args):
    image, prob, mask = _tensors(entries, device)
    flair = image[:, args.flair_index:args.flair_index + 1]
    gt = np.stack([(e.gt > 0.5) for e in entries])
    hd_prev = [_hd(e.stage2, e.gt) for e in entries]
    height, width = mask.shape[-2:]
    states, old_logp, actions = [], [], []
    values, rewards, bands, zs = [], [], [], []
    flip_count = 0
    reward_abs = 0.0

    for step in range(args.n_steps):
        state = torch.cat([image, mask, prob], dim=1)
        logits, value = model.forward_with_value(state)
        prob_np = prob[:, 0].detach().cpu().numpy()
        mask_np = mask[:, 0].detach().cpu().numpy()
        flair_np = flair[:, 0].detach().cpu().numpy()
        band_list, z_list = [], []
        for index in range(len(entries)):
            band = edit_band(prob_np[index], mask_np[index], args.prob_lo, args.prob_hi, args.radius)
            band_list.append(band)
            z_list.append(flair_z_map(flair_np[index], band))
        band_t = torch.from_numpy(np.stack(band_list)).to(device)
        z_t = torch.from_numpy(np.stack(z_list)).to(device)
        guard = step == args.n_steps - 1
        constrained = constrain_logits(logits, band_t, z_t, guard=guard)
        dist = Categorical(logits=constrained.permute(0, 2, 3, 1).reshape(-1, 3))
        action = dist.sample().reshape(len(entries), height, width)
        logp = dist.log_prob(action.reshape(-1)).reshape(len(entries), height, width)
        new_mask = mask.clone()
        plane = new_mask[:, 0]
        plane[band_t & (action == 2)] = 1.0
        plane[band_t & (action == 0)] = 0.0
        new_np = new_mask[:, 0].detach().cpu().numpy()
        reward = np.zeros((len(entries), height, width), dtype=np.float32)
        hd_next = []
        for index in range(len(entries)):
            hd_new = _hd(new_np[index], gt[index])
            reward[index] = pixel_flip_reward(
                mask_np[index], new_np[index], gt[index], hd_prev[index], hd_new, args.hd_coef)
            hd_next.append(hd_new)
        flip_count += int((mask_np != (new_np > 0.5)).sum())
        reward_abs += float(np.abs(reward).mean())
        states.append(state.detach())
        old_logp.append(logp.detach())
        actions.append(action.detach())
        values.append(value.detach())
        rewards.append(torch.from_numpy(reward).to(device))
        bands.append(band_t)
        zs.append(z_t)
        hd_prev = hd_next
        mask = new_mask

    pack = {
        "states": torch.stack(states),
        "old_logp": torch.stack(old_logp),
        "actions": torch.stack(actions),
        "values": torch.stack(values),
        "rewards": torch.stack(rewards),
        "bands": torch.stack(bands),
        "zs": torch.stack(zs),
        "flip_count": flip_count,
        "reward_abs": reward_abs / max(args.n_steps, 1),
    }
    return pack


def ppo_loss(model, pack, args):
    states = pack["states"]
    steps, batch, channels, height, width = states.shape
    logits, value = model.forward_with_value(states.reshape(steps * batch, channels, height, width))
    logits = logits.view(steps, batch, 3, height, width)
    value = value.view(steps, batch, height, width)
    constrained = torch.stack([
        constrain_logits(logits[t], pack["bands"][t], pack["zs"][t], guard=(t == steps - 1))
        for t in range(steps)
    ])
    dist = Categorical(logits=constrained.permute(0, 1, 3, 4, 2).reshape(-1, 3))
    new_logp = dist.log_prob(pack["actions"].reshape(-1)).view(steps, batch, height, width)
    entropy = dist.entropy().view(steps, batch, height, width)

    adv = torch.zeros_like(pack["rewards"])
    running = torch.zeros_like(pack["values"][-1])
    for t in reversed(range(steps)):
        next_value = pack["values"][t + 1] if t + 1 < steps else torch.zeros_like(running)
        delta = pack["rewards"][t] + args.gamma * next_value - pack["values"][t]
        running = delta + args.gamma * args.gae_lambda * running
        adv[t] = running
    returns = adv + pack["values"]

    band = pack["bands"].bool()
    flat = adv.reshape(-1)[band.reshape(-1)]
    if flat.numel() < 8:
        return None
    norm = (flat - flat.mean()) / flat.std().clamp(min=1e-6)
    adv_norm = torch.zeros_like(adv.reshape(-1))
    adv_norm[band.reshape(-1)] = norm
    adv_norm = adv_norm.view_as(adv).detach()

    ratio = (new_logp - pack["old_logp"]).exp()
    clipped = ratio.clamp(1.0 - args.clip, 1.0 + args.clip)
    weight = band.float()
    denom = weight.sum().clamp(min=1.0)
    policy = -(torch.minimum(ratio * adv_norm, clipped * adv_norm) * weight).sum() / denom
    value_loss = ((value - returns).pow(2) * weight).sum() / denom
    ent = (entropy * weight).sum() / denom
    return policy + args.value_coef * value_loss - args.ent_coef * ent, policy, value_loss, ent


def main() -> None:
    parser = argparse.ArgumentParser(description="클래스 공통 경계 띠 PPO")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    parser.add_argument("--max_train_patients", type=int, default=400)
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--flair_index", type=int, default=1)
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--pretrained", type=str, default="checkpoints/band_refine_medium.pt")
    parser.add_argument("--prob_lo", type=float, default=0.35)
    parser.add_argument("--prob_hi", type=float, default=0.65)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch", type=int, default=32)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--value_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--ent_coef", type=float, default=0.001)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--hd_coef", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage2_thresholds", type=str, default="0.80,0.80,0.50")
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--save_path", type=str, default="checkpoints/band_ppo.pt")
    args = parser.parse_args()

    from src.utils.seed import set_seed
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("checkpoints/caranet_best.pt"):
        raise SystemExit("Small Expert(checkpoints/caranet_best.pt)가 없습니다.")
    if not os.path.exists(args.pretrained):
        raise SystemExit(f"초기 가중치가 없습니다: {args.pretrained}")

    from src.data.patient_split import load_split_brats_datasets
    train_ds, val_ds = load_split_brats_datasets(
        train_root=args.train_root, modality=args.modality, target_size=args.target_size,
        max_patients=args.max_train_patients, patient_split=args.patient_split,
        refinement_mode=None, simulate_rough=False,
    )
    from src.models.dynamic_router import AdaptivePipeline
    images, _, _ = train_ds.get_numpy_arrays()
    in_ch = images.shape[1] if images.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch, strict_checkpoints=True)
    pipeline.eval()
    classes = (0, 1, 2)
    thresholds = [float(t) for t in args.stage2_thresholds.split(",")]
    if len(thresholds) != 3:
        raise SystemExit("--stage2_thresholds 는 소형,중형,대형 3개 값이어야 합니다.")
    log.info("Stage 2 임계값 %s", thresholds)
    log.info("Stage 2 확률맵 계산 (train)")
    train_entries = build_stage2_entries(
        train_ds, pipeline, device, classes, thresholds=thresholds, use_tta=not args.no_tta,
    )
    log.info("Stage 2 확률맵 계산 (val)")
    val_entries = build_stage2_entries(
        val_ds, pipeline, device, classes, thresholds=thresholds, use_tta=not args.no_tta,
    )
    del pipeline
    torch.cuda.empty_cache()

    supervised = torch.load(args.pretrained, map_location=device, weights_only=False)
    cfg = supervised["config"]
    actor = build_band_refine(in_channels=cfg["in_channels"], width=cfg["width"]).to(device)
    actor.load_state_dict(supervised["model"])
    model = BandActorCritic(actor).to(device)
    log.info("초기 정책: %s | 파라미터 %.2fM", args.pretrained,
             sum(p.numel() for p in model.parameters()) / 1e6)

    before = evaluate_actor(model.actor, val_entries, device, args)
    log.info("학습 전 macro DSC %.4f→%.4f HD95 %.3f→%.3f",
             before["macro_stage2_dsc"], before["macro_ppo_dsc"],
             before["macro_stage2_hd95"], before["macro_ppo_hd95"])
    for name in ("small", "medium", "large"):
        if name in before:
            row = before[name]
            log.info("  %s n=%d DSC %.4f→%.4f HD95 %.3f→%.3f",
                     name, row["n"], row["stage2_dsc"], row["ppo_dsc"],
                     row["stage2_hd95"], row["ppo_hd95"])

    optimizer = torch.optim.AdamW([
        {"params": model.actor.parameters(), "lr": args.lr},
        {"params": model.value_head.parameters(), "lr": args.value_lr},
    ], weight_decay=0.0)
    best_hd = before["macro_stage2_hd95"]
    best_dsc = before["macro_ppo_dsc"]
    best_state = None
    config = {
        "in_channels": cfg["in_channels"], "width": cfg["width"], "n_steps": args.n_steps,
        "prob_lo": args.prob_lo, "prob_hi": args.prob_hi, "radius": args.radius,
        "flair_index": args.flair_index, "hd_coef": args.hd_coef, "classes": "small,medium,large",
    }
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.actor.eval()
        order = np.random.permutation(len(train_entries))
        total_loss = 0.0
        updates = 0
        flips = 0
        for start in range(0, len(order), args.batch_size):
            batch = [train_entries[int(i)] for i in order[start:start + args.batch_size]]
            if len(batch) < 2:
                continue
            pack = collect_rollout(model, batch, device, args)
            flips += pack["flip_count"]
            for _ in range(args.ppo_epochs):
                out = ppo_loss(model, pack, args)
                if out is None:
                    continue
                loss, policy, value_loss, ent = out
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += float(loss.detach())
                updates += 1
            if updates and updates % 200 == 0:
                log.info("  epoch %d update %d loss %.4f", epoch, updates, total_loss / updates)

        stats = evaluate_actor(model.actor, val_entries, device, args)
        log.info(
            "Epoch [%02d/%d] loss=%.4f flips=%d | macro DSC %.4f→%.4f HD95 %.3f→%.3f",
            epoch, args.epochs, total_loss / max(updates, 1), flips,
            stats["macro_stage2_dsc"], stats["macro_ppo_dsc"],
            stats["macro_stage2_hd95"], stats["macro_ppo_hd95"],
        )
        for name in ("small", "medium", "large"):
            if name in stats:
                row = stats[name]
                log.info("  %s DSC %.4f→%.4f HD95 %.3f→%.3f",
                         name, row["stage2_dsc"], row["ppo_dsc"], row["stage2_hd95"], row["ppo_hd95"])
        keeps = stats["macro_ppo_dsc"] >= stats["macro_stage2_dsc"]
        if keeps and stats["macro_ppo_hd95"] < best_hd:
            best_hd = stats["macro_ppo_hd95"]
            best_dsc = stats["macro_ppo_dsc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()}
            torch.save({
                "model": best_state,
                "config": config,
                "val_macro_dsc": best_dsc,
                "val_macro_hd95": best_hd,
                "val_stage2_dsc": stats["macro_stage2_dsc"],
                "val_stage2_hd95": stats["macro_stage2_hd95"],
                "val_classes": {k: stats[k] for k in ("small", "medium", "large") if k in stats},
            }, args.save_path)
            log.info("  ✔ 저장 (macro DSC %.4f, HD95 %.3f)", best_dsc, best_hd)

    if best_state is None:
        torch.save({
            "model": {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()},
            "config": config,
            "val_macro_dsc": stats["macro_ppo_dsc"],
            "val_macro_hd95": stats["macro_ppo_hd95"],
            "val_stage2_dsc": before["macro_stage2_dsc"],
            "val_stage2_hd95": before["macro_stage2_hd95"],
            "saved_without_gate": True,
        }, args.save_path)
        log.info("macro 지표가 Stage 2를 넘지 못해 마지막 가중치를 저장했다.")
    else:
        log.info("학습 완료. macro DSC %.4f / HD95 %.3f (Stage2 %.4f / %.3f)",
                 best_dsc, best_hd, before["macro_stage2_dsc"], before["macro_stage2_hd95"])


if __name__ == "__main__":
    main()
