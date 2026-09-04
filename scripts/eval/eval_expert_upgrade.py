"""Medium/Large Expert 교체 Stage 2 평가.

Small은 CaraNet 유지.
Medium: Attention U-Net / UNet 3+ / KAIST(nnU-Net 각색)
Large: KAIST
같은 val 42명, 분류기 라우팅, t1ce+flair.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import DEFAULT_SPLIT_PATH, load_or_create_patient_split
from src.models import build_attention_unet, build_caranet, build_segresnet, build_unetplusplus
from src.models.shape_classifier import build_shape_classifier
from src.models.unet3plus import build_unet3plus
from src.models.segresnet import region_logits_to_wt
from src.utils.metrics import dice, filter_small_components, hd95
from src.utils.zoom_crop import refine_with_zoom

CLASS_NAMES = {0: "Small", 1: "Medium", 2: "Large"}


def _first_conv_in(model: nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.weight.shape[1])
    return 1


def _match_in_channels(img: torch.Tensor, exp_in_ch: int) -> torch.Tensor:
    c = img.shape[1]
    if c == exp_in_ch:
        return img
    if c % 3 == 0:
        base = c // 3
        if exp_in_ch == base:
            return img[:, base : 2 * base]
    if c < exp_in_ch and exp_in_ch % c == 0:
        return img.repeat(1, exp_in_ch // c, 1, 1)
    if c > exp_in_ch:
        return img[:, :exp_in_ch]
    return img.repeat(1, exp_in_ch, 1, 1)[:, :exp_in_ch]


def _to_wt(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    if prob.shape[1] == 1:
        return prob
    return region_logits_to_wt(prob, from_logits=False)


def _load_state_dict_model(path: str, builder, device, default_in, default_out=1):
    try:
        st = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        st = torch.load(path, map_location=device, weights_only=False)
    if isinstance(st, dict) and "state_dict" in st and not any(
        hasattr(v, "ndim") and getattr(v, "ndim", 0) == 4 for v in st.values()
    ):
        st = st["state_dict"]
    convs = [v for v in st.values() if hasattr(v, "ndim") and v.ndim == 4]
    in_ch = convs[0].shape[1] if convs else default_in
    out_ch = convs[-1].shape[0] if convs else default_out
    model = builder(in_channels=in_ch, out_channels=out_ch).to(device)
    try:
        model.load_state_dict(st)
    except Exception as exc:
        print(f"[load] strict fail {path}: {exc}; trying strict=False")
        model = builder(in_channels=in_ch, out_channels=default_out).to(device)
        model.load_state_dict(st, strict=False)
    model.eval()
    return model, in_ch


def _load_kaist(path: str, device):
    from baselines.eval_baseline import load_baseline

    model, method, _ = load_baseline(path, device)
    return model, _first_conv_in(model)


def _load_classifier(device, in_channels):
    ckpt = "checkpoints/shape_classifier_best.pt"
    st = torch.load(ckpt, map_location=device, weights_only=True)
    cls_in = st["conv1.weight"].shape[1] if "conv1.weight" in st else in_channels
    model = build_shape_classifier(in_channels=cls_in, num_classes=3).to(device)
    model.load_state_dict(st)
    model.eval()
    return model, cls_in


def _binarize(prob, c, stage2_thr, cc_min, micro_area_floor=80.0, micro_thr_floor=0.15):
    mask = (prob > stage2_thr[c]).astype(np.float32)
    if c == 0 and np.sum(mask) < micro_area_floor:
        for thr in np.arange(stage2_thr[c] - 0.05, micro_thr_floor - 1e-9, -0.05):
            mask = (prob > thr).astype(np.float32)
            if np.sum(mask) >= micro_area_floor:
                break
    return filter_small_components(mask, cc_min[c])


@torch.no_grad()
def _predict_wt(model, images, device, batch_size=32):
    n = len(images)
    out = np.zeros((n,) + images.shape[-2:], dtype=np.float32)
    in_ch = _first_conv_in(model)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(images[start:end]).to(device)
        if batch.ndim == 3:
            batch = batch.unsqueeze(1)
        logits = model(_match_in_channels(batch, in_ch))
        out[start:end] = _to_wt(logits).squeeze(1).cpu().numpy()
    return out


def _summarize(dscs, hds, routed):
    routed = np.asarray(routed)
    dscs = np.asarray(dscs)
    hds = np.asarray(hds)
    summary = {"n": int(len(dscs)), "dsc": float(np.mean(dscs)), "hd95": float(np.mean(hds)), "by_class": {}}
    for c, name in CLASS_NAMES.items():
        idx = np.where(routed == c)[0]
        if len(idx) == 0:
            continue
        summary["by_class"][name] = {
            "n": int(len(idx)),
            "dsc": float(np.mean(dscs[idx])),
            "hd95": float(np.mean(hds[idx])),
        }
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", default="src/data/archive")
    p.add_argument("--modality", default="t1ce+flair")
    p.add_argument("--max_patients", type=int, default=210)
    p.add_argument("--patient_split", default=DEFAULT_SPLIT_PATH)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--stage2_thresholds", default="0.80,0.80,0.50")
    p.add_argument("--cc_min_sizes", default="0,15,25")
    p.add_argument("--kaist_ckpt", default="baselines/checkpoints/kaist_best.pt")
    p.add_argument("--attention_ckpt", default="checkpoints/attention_unet_best.pt")
    p.add_argument("--unet3plus_ckpt", default="checkpoints/unet3plus_best.pt")
    p.add_argument("--ppo_unet3plus_ckpt", default="ppo/checkpoints/unet3plus_best.pt")
    p.add_argument("--out", default="results/expert_upgrade_stage2.json")
    return p.parse_args()


def main():
    args = parse_args()
    stage2_thr = [float(x) for x in args.stage2_thresholds.split(",")]
    cc_min = [int(x) for x in args.cc_min_sizes.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split = load_or_create_patient_split(args.train_root, args.max_patients, args.patient_split)
    patient_ids = split["val"]
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
    print(f"Val slices: {len(images)}")

    classifier, cls_in = _load_classifier(device, images.shape[1] if images.ndim == 4 else 1)
    caranet, _ = _load_state_dict_model( "checkpoints/caranet_best.pt", build_caranet, device, default_in=6)
    unetpp, _ = _load_state_dict_model("checkpoints/unetplusplus_best.pt", build_unetplusplus, device, default_in=2)
    segres, _ = _load_state_dict_model("checkpoints/segresnet_best.pt", build_segresnet, device, default_in=2, default_out=2)

    medium_candidates = {}
    if os.path.exists(args.attention_ckpt):
        medium_candidates["attention"] = _load_state_dict_model(
            args.attention_ckpt, build_attention_unet, device, default_in=2
        )
        print(f"Attention U-Net in_ch={medium_candidates['attention'][1]}  ({args.attention_ckpt})")
    else:
        print(f"[skip] missing {args.attention_ckpt}")
    if os.path.exists(args.unet3plus_ckpt):
        u3_path = args.unet3plus_ckpt
    elif os.path.exists(getattr(args, "ppo_unet3plus_ckpt", "")):
        u3_path = args.ppo_unet3plus_ckpt
    else:
        u3_path = args.unet3plus_ckpt
    if os.path.exists(u3_path):
        medium_candidates["unet3plus"] = _load_state_dict_model(
            u3_path, build_unet3plus, device, default_in=2
        )
        print(f"UNet3+ in_ch={medium_candidates['unet3plus'][1]}  ({u3_path})")
    else:
        print(f"[skip] missing UNet3+ checkpoint")

    kaist_paths = [
        args.kaist_ckpt,
        "baselines/checkpoints/kaist_best.pt",
        "checkpoints/kaist_best.pt",
        "baselines/checkpoints/kaist_nnunet_best.pt",
    ]
    kaist = None
    kaist_in = None
    kaist_used = None
    for path in kaist_paths:
        if os.path.exists(path):
            kaist, kaist_in = _load_kaist(path, device)
            medium_candidates["kaist"] = (kaist, kaist_in)
            kaist_used = path
            print(f"KAIST in_ch={kaist_in}  ({path})")
            break
    if kaist is None:
        raise SystemExit("KAIST checkpoint 없음. 찾은 경로: " + ", ".join(kaist_paths))

    print("Classifier routing...")
    routed = []
    with torch.no_grad():
        for start in range(0, len(images), args.batch_size):
            end = min(start + args.batch_size, len(images))
            batch = torch.from_numpy(images[start:end]).to(device)
            if batch.ndim == 3:
                batch = batch.unsqueeze(1)
            logits = classifier(_match_in_channels(batch, cls_in))
            routed.extend(torch.argmax(logits, 1).cpu().tolist())
    routed = np.asarray(routed, dtype=np.int64)
    print({CLASS_NAMES[c]: int((routed == c).sum()) for c in CLASS_NAMES})

    print("CaraNet Small + zoom...")
    small_prob = np.zeros((len(images),) + images.shape[-2:], dtype=np.float32)
    small_idx = np.where(routed == 0)[0]
    caranet_in = _first_conv_in(caranet)
    with torch.no_grad():
        for i in small_idx:
            img = torch.from_numpy(images_25d[i]).unsqueeze(0).to(device)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            logits = caranet(_match_in_channels(img, caranet_in))
            prob = _to_wt(logits)
            prob = refine_with_zoom(
                lambda z: caranet(_match_in_channels(z, caranet_in)),
                img,
                prob,
                patch_size=64,
            )
            if prob.shape[1] != 1:
                prob = _to_wt(prob)
            small_prob[i] = prob.squeeze().cpu().numpy()

    print("Current Medium UNet++ / Large SegResNet ...")
    med_idx = np.where(routed == 1)[0]
    large_idx = np.where(routed == 2)[0]
    unetpp_all = _predict_wt(unetpp, images, device, args.batch_size)
    segres_all = _predict_wt(segres, images, device, args.batch_size)
    kaist_all = _predict_wt(kaist, images, device, args.batch_size)

    def combo_probs(medium_name):
        probs = np.zeros_like(small_prob)
        probs[small_idx] = small_prob[small_idx]
        if medium_name == "unetpp":
            probs[med_idx] = unetpp_all[med_idx]
        elif medium_name == "kaist":
            probs[med_idx] = kaist_all[med_idx]
        else:
            model, _ = medium_candidates[medium_name]
            med_imgs = images[med_idx]
            med_pred = _predict_wt(model, med_imgs, device, args.batch_size)
            probs[med_idx] = med_pred
        probs[large_idx] = kaist_all[large_idx] if medium_name != "current" else segres_all[large_idx]
        if medium_name == "current":
            probs[large_idx] = segres_all[large_idx]
        return probs

    configs = [("current", "unetpp", "segresnet")]
    for name in medium_candidates:
        configs.append((f"med={name}, large=kaist", name, "kaist"))

    results = {}
    print("\n=== Stage 2 Expert swap (classifier routing) ===")
    for label, med_name, large_name in configs:
        if med_name == "unetpp" and large_name == "segresnet":
            probs = np.zeros_like(small_prob)
            probs[small_idx] = small_prob[small_idx]
            probs[med_idx] = unetpp_all[med_idx]
            probs[large_idx] = segres_all[large_idx]
        else:
            probs = combo_probs(med_name)
        dscs, hds = [], []
        for i in range(len(images)):
            c = int(routed[i])
            mask = _binarize(probs[i], c, stage2_thr, cc_min)
            dscs.append(dice(mask, gt_masks[i]))
            hds.append(hd95(mask, gt_masks[i]))
        summary = _summarize(dscs, hds, routed)
        results[label] = {"medium": med_name, "large": large_name, **summary}
        print(f"\n{label}")
        print(f"  overall n={summary['n']}  DSC={summary['dsc']:.4f}  HD95={summary['hd95']:.4f}")
        for name, row in summary["by_class"].items():
            print(f"  {name:<8} n={row['n']:4d}  DSC={row['dsc']:.4f}  HD95={row['hd95']:.4f}")

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "thresholds": stage2_thr,
        "cc_min": cc_min,
        "notes": {
            "small": "CaraNet 2.5D + zoom-crop 유지",
            "attention_unet3_channel": "체크포인트 in_ch가 1이면 t1ce만 사용(_match_in_channels)",
            "kaist": "2D 각색 nnU-Net, 전 구간 학습 가중치를 Medium/Large에 재사용 (재학습 없음)",
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
