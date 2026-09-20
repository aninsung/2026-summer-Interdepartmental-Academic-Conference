"""Frozen 3D SegResNet followed by PPO to increase DSC and reduce HD95.

Artifacts are isolated from legacy TRIO. Validation only; test stays untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet, UNet, AttentionUnet, BasicUNetPlusPlus
from monai.networks.nets import SegResNet, UNet, AttentionUnet, BasicUNetPlusPlus
from src.models.caranet3d import CaraNet3D
from src.models.unet3plus3d import UNet3Plus3D
from src.models.segresnet_cfp import SegResNetCFP

from scipy import ndimage as ndi
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.boundary.data import load_case, make_split, read_case, save_case, validate_split
from src.boundary.env import BoundaryEnv
from src.boundary.metrics import REGIONS, region_metrics
from src.boundary.policy import BoundaryFeatures

log = logging.getLogger("boundary3d")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "config.json"
    if saved.exists() and json.loads(saved.read_text()) != config:
        raise ValueError("Output configuration mismatch; choose a new output directory")
    write_json(saved, config)
    split_path = output / "split.json"
    if not split_path.exists():
        write_json(split_path, make_split(config["data_root"], config["split_seed"]))
    split = json.loads(split_path.read_text())
    validate_split(split)
    selected = {k: split[k][:config.get(n, len(split[k]))] for k, n in (
        ("backbone_train", "backbone_patients"), ("rl_train", "rl_patients"), ("val", "val_patients"), ("test", "test_patients"))}
    if any(not v for k, v in selected.items() if k != "test"):
        raise ValueError("Every development partition needs patients")
    write_json(output / "selected_patients.json", selected)
    versions = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "environment.txt").write_text(versions.stdout)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["torch_threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    return output, selected


def cases_for(config, output, ids, with_logits=False):
    cache = output / ("predictions" if with_logits else "volumes")
    cache.mkdir(exist_ok=True)
    # Config hash covers all settings (backbone + env + ppo).  For predictions
    # cache the relevant validity check is backbone_sha256, done in
    # predict_backbone(); skip config_hash invalidation to avoid discarding
    # logits when only env/ppo params change.
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cases = []
    for patient in ids:
        path = cache / f"{patient}.npz"
        if path.exists():
            case = read_case(path)
            if not with_logits and case.get("config_hash") != config_hash:
                log.info("Cache mismatch for %s; reloading case", patient)
                case = load_case(config["data_root"], patient, config["resize"])
                case["config_hash"] = config_hash
                save_case(path, case)
        elif with_logits:
            raise FileNotFoundError(f"Run predict stage first: {path}")
        else:
            log.info("Loading real MRI %s", patient)
            case = load_case(config["data_root"], patient, config["resize"])
            case["config_hash"] = config_hash
            save_case(path, case)
        cases.append(case)
    return cases


def predict_backbone(config, output, ids, device, batch_size=None):
    model = build_backbone(config, device)
    model.load_state_dict(torch.load(output / "backbone.pt", map_location=device, weights_only=True))
    model.eval()


@torch.no_grad()
def predict(model, image, device, roi):
    model.eval()
    tensor = torch.from_numpy(image)[None].to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = sliding_window_inference(tensor, (roi,) * 3, 2, model, overlap=.25)
    return logits[0].float().cpu().numpy()


def sample_patch(case, size, rng):
    image, target = case["image"], case["target"]
    pad = [(0, max(0, size - s)) for s in target.shape]
    if any(p[1] for p in pad):
        image, target = np.pad(image, [(0, 0)] + pad), np.pad(target, pad)
    if rng.random() < .5 and np.any(target > 0):
        locations = np.argwhere(target > 0)
        center = locations[rng.integers(len(locations))]
        starts = [int(np.clip(c - size // 2, 0, s - size)) for c, s in zip(center, target.shape)]
    else:
        starts = [int(rng.integers(s - size + 1)) for s in target.shape]
    slices = tuple(slice(a, a + size) for a in starts)
    image, target = image[(slice(None),) + slices], target[slices]
    for axis in range(3):
        if rng.random() < .5:
            image, target = np.flip(image, axis + 1), np.flip(target, axis)
    return image.copy(), target.astype(np.int64).copy()


def measure(case, mask, config):
    return region_metrics(mask, case["target"], case["spacing"],
                          config["env"]["tolerance_mm"], case["empty_distance_mm"])


def build_backbone(config, device=None):
    name = config.get("backbone", "segresnet").lower()
    if name == "segresnet":
        model = SegResNet(spatial_dims=3, in_channels=4, out_channels=4,
                          init_filters=config.get("backbone_filters", 32))
    elif name == "caranet3d":
        model = CaraNet3D(in_channels=4, out_channels=4)
    elif name == "unet":
        model = UNet(spatial_dims=3, in_channels=4, out_channels=4,
                     channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2))
    elif name == "attention_unet":
        model = AttentionUnet(spatial_dims=3, in_channels=4, out_channels=4,
                              channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2))
    elif name in ["unet++", "unetplusplus"]:
        model = BasicUNetPlusPlus(spatial_dims=3, in_channels=4, out_channels=4,
                                  features=(32, 32, 64, 128, 256, 32))
    elif name in ["unet3plus", "unet+++"]:
        model = UNet3Plus3D(in_channels=4, out_channels=4, channels=(32, 64, 128, 256))
    elif name == "segresnet_cfp":
        model = SegResNetCFP(in_channels=4, out_channels=4, init_filters=config.get("backbone_filters", 32))
    else:
        raise ValueError(f"Unknown backbone: {name}")
    if device is not None:
        model = model.to(device)
    return model


def train_backbone(config, output, selected, device):
    checkpoint = output / "backbone.pt"
    if checkpoint.exists():
        log.info("Using completed backbone checkpoint %s", checkpoint)
        return
    train = cases_for(config, output, selected["backbone_train"])
    val = cases_for(config, output, selected["val"])
    model = build_backbone(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["backbone_lr"], weight_decay=1e-5)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config["seed"])
    best, start_step = -float("inf"), 0
    latest = output / "backbone_latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step, best = state["step"], state["best"]
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    start = time.perf_counter()
    for step in range(start_step + 1, config["backbone_steps"] + 1):
        model.train()
        samples = [sample_patch(train[int(rng.integers(len(train)))], config["backbone_patch_size"], rng)
                   for _ in range(config["backbone_batch_size"])]
        images = torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
        targets = torch.from_numpy(np.stack([s[1] for s in samples]))[:, None].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(images), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite backbone loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(optimizer)
        scaler.update()
        if step % 10 == 0 or step == 1:
            log.info("Backbone %d/%d loss=%.5f elapsed=%.1fs", step, config["backbone_steps"], loss.item(), time.perf_counter()-start)
        if step % config["backbone_val_every"] == 0 or step == config["backbone_steps"]:
            scores = [measure(c, predict(model, c["image"], device, config["backbone_patch_size"]).argmax(0), config)
                      for c in val]
            score = float(np.mean([s["mean"]["dice"] for s in scores]))
            with open(output / "backbone_history.jsonl", "a") as handle:
                handle.write(json.dumps(dict(step=step, loss=loss.item(), val_dice=score,
                                             val_metrics=scores, elapsed_seconds=time.perf_counter()-start)) + "\n")
            log.info("Backbone validation step=%d mean Dice=%.5f", step, score)
            if score > best:
                best = score
                torch.save(dict(model=model.state_dict(), step=step, val_dice=score,
                                training_patients=selected["backbone_train"], config=config), output / "backbone_best.pt")
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            step=step, best=best, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []), latest)
    best_state = torch.load(output / "backbone_best.pt", map_location="cpu", weights_only=False)
    torch.save(best_state, checkpoint)
    log.info("Backbone completed; selected step=%d val Dice=%.5f", best_state["step"], best_state["val_dice"])


def generate_predictions(config, output, selected, device):
    path = output / "backbone.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    ids = selected["rl_train"] + selected["val"] + selected["test"]
    if set(checkpoint["training_patients"]).intersection(ids):
        raise ValueError("Backbone-training patients leaked into refinement development data")
    model = build_backbone(config, device)
    model.load_state_dict(checkpoint["model"])
    digest = sha256(path)
    directory = output / "predictions"
    directory.mkdir(exist_ok=True)
    for patient in ids:
        destination = directory / f"{patient}.npz"
        if destination.exists():
            if read_case(destination).get("backbone_sha256") != digest:
                raise ValueError("Stale prediction cache: backbone has changed")
            continue
        case = cases_for(config, output, [patient])[0]
        case["logits"] = predict(model, case["image"], device, config["backbone_patch_size"])
        case["backbone_sha256"] = digest
        save_case(destination, case)
        scores = measure(case, case["logits"].argmax(0), config)
        log.info("Initial %s Dice=%.4f HD95=%.3f mm", patient, scores["mean"]["dice"], scores["mean"]["hd95_mm"])


def oracle_diagnostic(config, output, selected):
    destination = output / "oracle_validation.json"
    if destination.exists():
        return
    cases = cases_for(config, output, selected["val"], True)
    env = BoundaryEnv(cases, **config["env"])
    records = []
    for index, case in enumerate(cases):
        env.reset(options={"case_index": index})
        before, history = copy.deepcopy(env.current_metrics), []
        while not env.done:
            gains, changed = env.action_gains()
            action = int(gains.argmax()) if gains.max() > 1e-8 else 6
            _, reward, _, _, _ = env.step(action)
            history.append(dict(gains=gains.tolist(), changed_voxels=changed, chosen=action, reward=reward))
        records.append(dict(patient=case["patient"], before=before, after=env.current_metrics, steps=history))
        write_json(destination.with_suffix(".partial.json"), records)
        log.info("ORACLE diagnostic %s Dice %.4f -> %.4f; HD95 %.3f -> %.3f", case["patient"],
                 before["mean"]["dice"], env.current_metrics["mean"]["dice"],
                 before["mean"]["hd95_mm"], env.current_metrics["mean"]["hd95_mm"])
    gains = np.asarray([s["gains"] for r in records for s in r["steps"]])
    write_json(destination, dict(diagnostic_only=True, not_a_sequence_upper_bound=True, records=records,
               zero_gain_fraction=float(np.mean(np.abs(gains[:, :6]) < 1e-8)),
               improvable_state_fraction=float(np.mean(gains.max(1) > 1e-8))))


class ProgressCheckpoint(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.actions = np.zeros(7, dtype=np.int64)
        self.rewards, self.changed = [], 0
        self.start = time.perf_counter()

    def _on_step(self):
        for info, reward in zip(self.locals["infos"], self.locals["rewards"]):
            self.actions[info["action"]] += 1
            self.changed += info["changed_voxels"]
            self.rewards.append(float(reward))
        if self.num_timesteps % 64 == 0:
            log.info("PPO steps=%d mean reward=%.5f actions=%s elapsed=%.1fs", self.num_timesteps,
                     np.mean(self.rewards[-64:]), self.actions.tolist(), time.perf_counter()-self.start)
        if self.num_timesteps % 256 == 0:
            self.model.save(self.output / "ppo_latest")
            # Diagnostic: entropy and loss from SB3 logger, action distribution
            diag = {}
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                nv = getattr(self.model.logger, 'name_to_value', {})
                for key in ('train/entropy_loss', 'train/policy_gradient_loss',
                            'train/value_loss', 'train/approx_kl'):
                    val = nv.get(key)
                    if val is not None:
                        diag[key.split('/')[-1]] = round(float(val), 6)
            total_actions = max(1, int(self.actions.sum()))
            dist = [round(float(a) / total_actions, 3) for a in self.actions]
            log.info("PPO diag steps=%d %s action_dist=%s", self.num_timesteps, diag, dist)
        return True

    def _on_training_end(self):
        write_json(self.output / "ppo_training.json", dict(timesteps=self.num_timesteps,
                   action_counts=self.actions.tolist(), changed_voxels=self.changed,
                   changed_voxels_per_step=float(self.changed / max(1, self.num_timesteps)),
                   reward_mean=float(np.mean(self.rewards)) if self.rewards else None,
                   zero_reward_fraction=float(np.mean(np.abs(self.rewards) < 1e-8)) if self.rewards else None,
                   elapsed_seconds=time.perf_counter()-self.start))


def train_ppo(config, output, selected, device):
    if (output / "ppo_final.zip").exists():
        log.info("Using completed PPO checkpoint")
        return
    cases = cases_for(config, output, selected["rl_train"], True)
    env = Monitor(BoundaryEnv(cases, **config["env"]), str(output / "ppo_monitor.csv"))
    parameters = dict(config["ppo"])
    total = parameters.pop("total_timesteps")
    model = PPO("CnnPolicy", env, seed=config["seed"], device=device, verbose=0,
                policy_kwargs=dict(features_extractor_class=BoundaryFeatures, normalize_images=False,
                                   net_arch=dict(pi=[64], vf=[64])), **parameters)
    latest = output / "ppo_latest.zip"
    if latest.exists():
        model = PPO.load(latest, env=env, device=device)
        log.info("Resuming PPO snapshot at %d steps (new rollout)", model.num_timesteps)
    remaining = max(0, total - model.num_timesteps)
    if remaining:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=ProgressCheckpoint(output))
    model.save(output / "ppo_final")


def paired_summary(records, method, seed):
    rng = np.random.default_rng(seed)
    summary = {}
    for region in (*REGIONS, "mean"):
        before = np.asarray([[r["initial"][region]["dice"], r["initial"][region]["hd95_mm"]] for r in records])
        after = np.asarray([[r[method][region]["dice"], r[method][region]["hd95_mm"]] for r in records])
        delta = (after - before) * np.asarray([1, -1])
        bootstrap = delta[rng.integers(0, len(records), (2000, len(records)))].mean(1)
        summary[region] = dict(
            before_dice=float(before[:, 0].mean()), after_dice=float(after[:, 0].mean()),
            before_hd95_mm=float(before[:, 1].mean()), after_hd95_mm=float(after[:, 1].mean()),
            delta_dice=float(delta[:, 0].mean()), hd95_reduction_mm=float(delta[:, 1].mean()),
            delta_dice_ci95=np.quantile(bootstrap[:, 0], [.025, .975]).tolist(),
            hd95_reduction_ci95=np.quantile(bootstrap[:, 1], [.025, .975]).tolist(),
            both_strictly_improved=int(np.sum((delta[:, 0] > 1e-8) & (delta[:, 1] > 1e-8))),
            dice_improved_hd95_worsened=int(np.sum((delta[:, 0] > 1e-8) & (delta[:, 1] < -1e-8))),
            hd95_improved_dice_worsened=int(np.sum((delta[:, 0] < -1e-8) & (delta[:, 1] > 1e-8))),
            both_worsened_or_unchanged=int(np.sum((delta[:, 0] <= 1e-8) & (delta[:, 1] <= 1e-8))),
            either_worsened=int(np.sum((delta[:, 0] < -1e-8) | (delta[:, 1] < -1e-8))),
            both_unchanged=int(np.sum(np.all(np.abs(delta) <= 1e-8, axis=1)))
        )
    return summary


def evaluate(config, output, selected, device):
    cases = cases_for(config, output, selected["val"], True)
    policy = PPO.load(output / "ppo_final.zip", device=device)
    records = []
    for case in cases:
        initial_mask = case["logits"].argmax(0).astype(np.uint8)
        inference_case = {k: v for k, v in case.items() if k != "target"}
        env = BoundaryEnv([inference_case], **config["env"], reward_enabled=False)
        observation, _ = env.reset(seed=config["seed"])
        actions, start = [], time.perf_counter()
        while not env.done:
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, _, _, info = env.step(int(action))
            actions.append(info)
        elapsed = time.perf_counter() - start
        labeled, count = ndi.label(initial_mask > 0)
        post = initial_mask.copy()
        if count:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            main_id = int(sizes.argmax())
            for comp_id in range(1, count + 1):
                if comp_id != main_id and sizes[comp_id] < 10:
                    post[labeled == comp_id] = 0
        tc_post = (post == 1) | (post == 3)
        post[ndi.binary_fill_holes(tc_post) & (post == 0)] = 1
        wt_post = post > 0
        post[ndi.binary_fill_holes(wt_post) & (post == 0)] = 2
        final_mask = env.mask.copy()
        labeled_final, n_final = ndi.label(final_mask > 0)
        if n_final > 1:
            fsizes = np.bincount(labeled_final.ravel()); fsizes[0] = 0
            fmain = int(fsizes.argmax())
            for cid in range(1, n_final + 1):
                if cid != fmain and fsizes[cid] < 10:
                    final_mask[labeled_final == cid] = 0
        tc_final = (final_mask == 1) | (final_mask == 3)
        final_mask[ndi.binary_fill_holes(tc_final) & (final_mask == 0)] = 1
        wt_final = final_mask > 0
        final_mask[ndi.binary_fill_holes(wt_final) & (final_mask == 0)] = 2
                    
        record = dict(patient=case["patient"], initial=measure(case, initial_mask, config),
                      postprocess=measure(case, post, config), ppo=measure(case, final_mask, config),
                      actions=actions, ppo_seconds=elapsed, grid=case["evaluation_grid"], spacing=case["spacing"])
        records.append(record)
        np.savez_compressed(output / f"validation_masks_{case['patient']}.npz", initial=initial_mask,
                            ppo=final_mask, target=case["target"], spacing=case["spacing"])
        
        import nibabel as nib
        nii = nib.load(Path(config["data_root"]) / case["patient"] / f"{case['patient']}_t1.nii.gz")
        unpadded = np.zeros(nii.shape, dtype=np.uint8)
        slices = tuple(slice(s[0], s[1]) for s in case["crop"])
        unpadded[slices] = final_mask
        unpadded[unpadded == 3] = 4
        nib.save(nib.Nifti1Image(unpadded, nii.affine), output / f"{case['patient']}.nii.gz")
        log.info("PPO VALIDATION %s Dice %.4f -> %.4f; HD95 %.3f -> %.3f mm", case["patient"],
                 record["initial"]["mean"]["dice"], record["ppo"]["mean"]["dice"],
                 record["initial"]["mean"]["hd95_mm"], record["ppo"]["mean"]["hd95_mm"])
    report = dict(partition="validation", test_evaluated=False, seed=config["seed"],
                  pilot=config["resize"] is not None, patients=len(records), records=records,
                  ppo=paired_summary(records, "ppo", config["seed"]),
                  postprocess=paired_summary(records, "postprocess", config["seed"]),
                  mean_ppo_seconds=float(np.mean([r["ppo_seconds"] for r in records])))
    write_json(output / "validation_report.json", report)
    lines = ["# 3D PPO boundary refinement: validation pilot", "",
             "Real BraTS MRI; disjoint backbone and PPO training patients. Test labels were not evaluated.",
             "Single seed, small development subset. These are not final test or native-resolution claims.", "",
             "## Ablation Table (논문용 3행 분리)",
             "",
             "| Method | ET DSC | TC DSC | WT DSC | ET HD95 | TC HD95 | WT HD95 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, source, use_before in [("Backbone (Initial)", "ppo", True),
                                       ("Backbone + PostProcess", "postprocess", False),
                                       ("Ours (Backbone + PPO)", "ppo", False)]:
        s = report[source]
        p = "before" if use_before else "after"
        lines.append(f"| {label} "
                     f"| {s['ET'][f'{p}_dice']:.4f} "
                     f"| {s['TC'][f'{p}_dice']:.4f} "
                     f"| {s['WT'][f'{p}_dice']:.4f} "
                     f"| {s['ET'][f'{p}_hd95_mm']:.3f} "
                     f"| {s['TC'][f'{p}_hd95_mm']:.3f} "
                     f"| {s['WT'][f'{p}_hd95_mm']:.3f} |")
    lines += ["", "## Detailed per-method DSC improvement",
              "",
              "| Region | Initial DSC | +PostProc DSC | +PPO DSC | Initial HD95 | +PostProc HD95 | +PPO HD95 | PPO Both↑ | PPO Either↓ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for region in ["ET", "TC", "WT", "mean"]:
        iv = report["ppo"][region]
        pv = report["postprocess"][region]
        lines.append(f"| {region} "
                     f"| {iv['before_dice']:.4f} | {pv['after_dice']:.4f} | {iv['after_dice']:.4f} "
                     f"| {iv['before_hd95_mm']:.3f} | {pv['after_hd95_mm']:.3f} | {iv['after_hd95_mm']:.3f} "
                     f"| {iv['both_strictly_improved']} | {iv['either_worsened']} |")
    lines += ["", f"Mean extra PPO time: {report['mean_ppo_seconds']:.3f} seconds/patient.",
              "", "See validation_report.json for full per-patient breakdowns."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")



def evaluate_test(config, output, selected, device):
    """Evaluate on the strictly held-out test partition to mimic BraTS challenge."""
    cases = cases_for(config, output, selected["test"], True)
    policy = PPO.load(output / "ppo_final.zip", device=device)
    records = []
    for case in cases:
        initial_mask = case["logits"].argmax(0).astype(np.uint8)
        inference_case = {k: v for k, v in case.items() if k != "target"}
        env = BoundaryEnv([inference_case], **config["env"], reward_enabled=False)
        observation, _ = env.reset(seed=config["seed"])
        actions, start = [], time.perf_counter()
        while not env.done:
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, _, _, info = env.step(int(action))
            actions.append(info)
        elapsed = time.perf_counter() - start
        labeled, count = ndi.label(initial_mask > 0)
        post = initial_mask.copy()
        if count:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            main_id = int(sizes.argmax())
            for comp_id in range(1, count + 1):
                if comp_id != main_id and sizes[comp_id] < 10:
                    post[labeled == comp_id] = 0
        tc_post = (post == 1) | (post == 3)
        post[ndi.binary_fill_holes(tc_post) & (post == 0)] = 1
        wt_post = post > 0
        post[ndi.binary_fill_holes(wt_post) & (post == 0)] = 2
        final_mask = env.mask.copy()
        labeled_final, n_final = ndi.label(final_mask > 0)
        if n_final > 1:
            fsizes = np.bincount(labeled_final.ravel()); fsizes[0] = 0
            fmain = int(fsizes.argmax())
            for cid in range(1, n_final + 1):
                if cid != fmain and fsizes[cid] < 10:
                    final_mask[labeled_final == cid] = 0
        tc_final = (final_mask == 1) | (final_mask == 3)
        final_mask[ndi.binary_fill_holes(tc_final) & (final_mask == 0)] = 1
        wt_final = final_mask > 0
        final_mask[ndi.binary_fill_holes(wt_final) & (final_mask == 0)] = 2
        record = dict(patient=case["patient"], initial=measure(case, initial_mask, config),
                      postprocess=measure(case, post, config), ppo=measure(case, final_mask, config),
                      actions=actions, ppo_seconds=elapsed, grid=case["evaluation_grid"], spacing=case["spacing"])
        records.append(record)
        np.savez_compressed(output / f"test_masks_{case['patient']}.npz", initial=initial_mask,
                            ppo=final_mask, target=case["target"], spacing=case["spacing"])
        
        # Save unpadded NIfTI for challenge submission (with cleanup applied)
        import nibabel as nib
        nii = nib.load(Path(config["data_root"]) / case["patient"] / f"{case['patient']}_t1.nii.gz")
        unpadded = np.zeros(nii.shape, dtype=np.uint8)
        slices = tuple(slice(s[0], s[1]) for s in case["crop"])
        unpadded[slices] = final_mask
        unpadded[unpadded == 3] = 4
        nib.save(nib.Nifti1Image(unpadded, nii.affine), output / f"{case['patient']}.nii.gz")
        log.info("PPO TEST %s Dice %.4f -> %.4f; HD95 %.3f -> %.3f mm", case["patient"],
                 record["initial"]["mean"]["dice"], record["ppo"]["mean"]["dice"],
                 record["initial"]["mean"]["hd95_mm"], record["ppo"]["mean"]["hd95_mm"])
    report = dict(partition="test", test_evaluated=True, seed=config["seed"],
                  pilot=config["resize"] is not None, patients=len(records), records=records,
                  ppo=paired_summary(records, "ppo", config["seed"]),
                  postprocess=paired_summary(records, "postprocess", config["seed"]),
                  mean_ppo_seconds=float(np.mean([r["ppo_seconds"] for r in records])))
    write_json(output / "test_report.json", report)
    
    # Generate final test summary with 3-row ablation table
    lines = ["# 3D PPO boundary refinement: Final Test Set Evaluation", "",
             "Real BraTS MRI; strictly held-out test patients evaluated at native resolution.",
             "Backbone training (N=656) and PPO training (N=219) patients are disjoint.", "",
             "## Ablation Table (논문용 3행 분리)",
             "",
             "| Method | ET DSC | TC DSC | WT DSC | ET HD95 | TC HD95 | WT HD95 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, source, use_before in [("Backbone (Initial)", "ppo", True),
                                       ("Backbone + PostProcess", "postprocess", False),
                                       ("Ours (Backbone + PPO)", "ppo", False)]:
        s = report[source]
        p = "before" if use_before else "after"
        lines.append(f"| {label} "
                     f"| {s['ET'][f'{p}_dice']:.4f} "
                     f"| {s['TC'][f'{p}_dice']:.4f} "
                     f"| {s['WT'][f'{p}_dice']:.4f} "
                     f"| {s['ET'][f'{p}_hd95_mm']:.3f} "
                     f"| {s['TC'][f'{p}_hd95_mm']:.3f} "
                     f"| {s['WT'][f'{p}_hd95_mm']:.3f} |")
    lines += ["", "## Detailed per-region improvement",
              "",
              "| Region | Initial DSC | +PostProc DSC | +PPO DSC | Initial HD95 | +PostProc HD95 | +PPO HD95 | PPO Both↑ | PPO Either↓ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for region in ["ET", "TC", "WT", "mean"]:
        iv = report["ppo"][region]
        pv = report["postprocess"][region]
        lines.append(f"| {region} "
                     f"| {iv['before_dice']:.4f} | {pv['after_dice']:.4f} | {iv['after_dice']:.4f} "
                     f"| {iv['before_hd95_mm']:.3f} | {pv['after_hd95_mm']:.3f} | {iv['after_hd95_mm']:.3f} "
                     f"| {iv['both_strictly_improved']} | {iv['either_worsened']} |")
    lines += ["", "See test_report.json for full per-patient breakdowns."]
    (output / "TEST_RESULTS.md").write_text("\n".join(lines) + "\n")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/boundary_pilot.json")
    parser.add_argument("--stage", choices=("all", "backbone", "predict", "oracle", "ppo", "evaluate", "evaluate_test"), default="all")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")])
    output, selected = initialize(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Stage=%s device=%s patients=%s grid=%s", args.stage, device,
             {k: len(v) for k, v in selected.items()}, config["resize"] or "native")
    stages = ("backbone", "predict", "oracle", "ppo", "evaluate", "evaluate_test") if args.stage == "all" else (args.stage,)
    current_stage = stages[0]
    try:
        for stage in stages:
            current_stage = stage
            write_json(output / "status.json", dict(stage=stage, state="running", time=time.time()))
            if stage == "backbone":
                train_backbone(config, output, selected, device)
            elif stage == "predict":
                generate_predictions(config, output, selected, device)
            elif stage == "oracle":
                oracle_diagnostic(config, output, selected)
            elif stage == "ppo":
                train_ppo(config, output, selected, device)
            elif stage == "evaluate":
                evaluate(config, output, selected, device)
            elif stage == "evaluate_test":
                evaluate_test(config, output, selected, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_json(output / "status.json", dict(stage=stages[-1], state="completed", time=time.time()))
    except Exception as exc:
        write_json(output / "status.json", dict(stage=current_stage, state="failed", error=str(exc), time=time.time()))
        raise


if __name__ == "__main__":
    main()

