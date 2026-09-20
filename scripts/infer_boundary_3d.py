"""Ground-truth-free inference pipeline for 3D SegResNet + PPO refinement.

Loads 4-modality MRI NIfTI files, performs initial backbone prediction,
applies PPO boundary refinement, and exports refined NIfTI masks with
original affine, header, shape, and BraTS labels (0, 1, 2, 4).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from scipy.special import softmax
from stable_baselines3 import PPO

from src.boundary.env import BoundaryEnv
from src.boundary.policy import BoundaryFeatures

log = logging.getLogger("boundary3d_infer")


def load_mri_case(t1_path, t1ce_path, t2_path, flair_path, resize=None):
    paths = [t1_path, t1ce_path, t2_path, flair_path]
    images, reference = [], None
    for path in paths:
        nii = nib.load(path)
        if reference is None:
            reference = nii
        elif nii.shape != reference.shape or not np.allclose(nii.affine, reference.affine):
            raise ValueError(f"Misaligned modality: {path}")
        images.append(nii.get_fdata(dtype=np.float32))

    image = np.stack(images)
    foreground = np.any(image != 0, axis=0)
    coords = np.where(foreground)
    if not coords[0].size:
        raise ValueError("Empty foreground MRI input")

    original_shape = reference.shape
    slices = tuple(slice(max(0, int(c.min()) - 2), min(s, int(c.max()) + 3))
                   for c, s in zip(coords, original_shape))
    spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    cropped_image = image[(slice(None),) + slices].copy()
    cropped_shape = cropped_image.shape[1:]

    for channel in cropped_image:
        valid = channel != 0
        if valid.any():
            channel[valid] = np.clip((channel[valid] - channel[valid].mean()) /
                                     max(float(channel[valid].std()), 1e-6), -5, 5)

    processed_image = cropped_image
    current_spacing = spacing.copy()
    if resize is not None:
        shape = (int(resize),) * 3
        processed_image = F.interpolate(torch.from_numpy(cropped_image)[None], size=shape,
                                        mode="trilinear", align_corners=False)[0].numpy()
        current_spacing = spacing * np.asarray(cropped_shape) / np.asarray(shape)

    return dict(
        image=processed_image.astype(np.float32),
        reference_nii=reference,
        original_shape=list(original_shape),
        cropped_shape=list(cropped_shape),
        crop=slices,
        spacing=current_spacing.tolist(),
        original_spacing=spacing.tolist(),
    )


def build_backbone(filters=16, device=torch.device("cpu")):
    return SegResNet(spatial_dims=3, in_channels=4, out_channels=4,
                     init_filters=filters, blocks_down=(1, 1, 2, 2),
                     blocks_up=(1, 1, 1), dropout_prob=.1).to(device)


@torch.no_grad()
def predict_backbone(model, image, device, roi=96):
    model.eval()
    tensor = torch.from_numpy(image)[None].to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = sliding_window_inference(tensor, (roi,) * 3, 2, model, overlap=.25)
    return logits[0].float().cpu().numpy()


def restore_full_mask(mask_3d, case_info):
    cropped_shape = case_info["cropped_shape"]
    original_shape = case_info["original_shape"]
    slices = case_info["crop"]

    if mask_3d.shape != tuple(cropped_shape):
        mask_tensor = torch.from_numpy(mask_3d.astype(np.float32))[None, None]
        restored_cropped = F.interpolate(mask_tensor, size=tuple(cropped_shape),
                                         mode="nearest-exact")[0, 0].numpy().astype(np.uint8)
    else:
        restored_cropped = mask_3d.astype(np.uint8)

    full_mask = np.zeros(original_shape, dtype=np.uint8)
    full_mask[slices] = restored_cropped
    # Convert internal class 3 back to BraTS label 4 (Enhancing Tumor)
    full_mask[full_mask == 3] = 4
    return full_mask


def run_inference(t1_path, t1ce_path, t2_path, flair_path, backbone_path, ppo_path,
                  output_dir, patient_id="infer_patient", config_path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "resize": 96,
        "backbone_filters": 16,
        "backbone_patch_size": 96,
        "env": {
            "patch_size": 24, "max_steps": 16, "band_mm": 3.0,
            "logit_delta": 1.0, "tolerance_mm": 2.0
        }
    }
    if config_path and Path(config_path).exists():
        loaded_cfg = json.loads(Path(config_path).read_text())
        config.update({k: loaded_cfg[k] for k in ("resize", "backbone_filters", "backbone_patch_size") if k in loaded_cfg})
        if "env" in loaded_cfg:
            config["env"].update(loaded_cfg["env"])

    log.info("Loading MRI modalities for %s...", patient_id)
    case_info = load_mri_case(t1_path, t1ce_path, t2_path, flair_path, resize=config["resize"])
    case_info["patient"] = patient_id

    log.info("Loading backbone model from %s...", backbone_path)
    checkpoint = torch.load(backbone_path, map_location=device, weights_only=False)
    model = build_backbone(filters=config["backbone_filters"], device=device)
    model.load_state_dict(checkpoint["model"])

    log.info("Generating backbone initial logits...")
    logits = predict_backbone(model, case_info["image"], device=device, roi=config["backbone_patch_size"])
    case_info["logits"] = logits
    initial_mask = logits.argmax(0).astype(np.uint8)

    log.info("Loading PPO policy from %s...", ppo_path)
    env = BoundaryEnv([case_info], **config["env"], reward_enabled=False)
    policy = PPO.load(ppo_path, device=device)

    log.info("Running PPO boundary refinement...")
    obs, _ = env.reset()
    actions = []
    while not env.done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(int(action))
        actions.append(info)

    refined_mask = env.mask

    full_initial = restore_full_mask(initial_mask, case_info)
    full_ppo = restore_full_mask(refined_mask, case_info)

    ref_nii = case_info["reference_nii"]
    ppo_nii = nib.Nifti1Image(full_ppo, ref_nii.affine, ref_nii.header)
    initial_nii = nib.Nifti1Image(full_initial, ref_nii.affine, ref_nii.header)

    ppo_out_path = output_dir / f"{patient_id}_seg_ppo.nii.gz"
    initial_out_path = output_dir / f"{patient_id}_seg_initial.nii.gz"
    actions_out_path = output_dir / f"{patient_id}_actions.json"

    nib.save(ppo_nii, ppo_out_path)
    nib.save(initial_nii, initial_out_path)
    actions_out_path.write_text(json.dumps(dict(patient=patient_id, actions=actions), indent=2))

    log.info("Saved PPO segmentation to %s", ppo_out_path)
    log.info("Saved initial segmentation to %s", initial_out_path)
    log.info("Saved action logs to %s", actions_out_path)

    return dict(ppo_path=str(ppo_out_path), initial_path=str(initial_out_path), actions=actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", required=True, help="Path to t1 NIfTI")
    parser.add_argument("--t1ce", required=True, help="Path to t1ce NIfTI")
    parser.add_argument("--t2", required=True, help="Path to t2 NIfTI")
    parser.add_argument("--flair", required=True, help="Path to flair NIfTI")
    parser.add_argument("--backbone", required=True, help="Path to backbone_best.pt")
    parser.add_argument("--ppo", required=True, help="Path to ppo_final.zip")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--patient_id", default="infer_patient", help="Patient identifier")
    parser.add_argument("--config", default=None, help="Config JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_inference(args.t1, args.t1ce, args.t2, args.flair,
                  args.backbone, args.ppo, args.output_dir,
                  patient_id=args.patient_id, config_path=args.config)


if __name__ == "__main__":
    main()
