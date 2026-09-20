"""Paired four-arm evaluation; labels are used only after policy inference.

Synthetic tests exercise the measurement contract, not clinical performance.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from .env import BoundaryEnv
from .metrics import REGIONS, region_metrics
from .postprocessing import remove_small_components

log = logging.getLogger("boundary3d")
METHODS = (
    ("initial", "Backbone (initial)"),
    ("postprocess", "Backbone + postprocessing"),
    ("ppo", "Backbone + PPO (before postprocessing)"),
    ("ppo_postprocess", "Backbone + PPO + postprocessing"),
)
CHANGE_TOLERANCE = 1e-8


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=_json_value) + "\n")
    temporary.replace(path)


def _change_status(gain):
    if gain > CHANGE_TOLERANCE:
        return "improved"
    if gain < -CHANGE_TOLERANCE:
        return "worsened"
    return "unchanged"


def change_category(before, after):
    """A decrease of HD95 is an improvement; near-zero changes are ties."""
    dice = _change_status(after["dice"] - before["dice"])
    hd95 = _change_status(before["hd95_mm"] - after["hd95_mm"])
    return f"dice_{dice}_hd95_{hd95}"


def paired_summary(records, method, seed):
    """Bootstrap paired patients (not voxels, regions, steps, or seeds)."""
    if not records:
        raise ValueError("Cannot summarize an empty evaluation partition")
    rng = np.random.default_rng(seed)
    # Shared indices retain the patient pairing across all regions and metrics.
    indices = rng.integers(0, len(records), (2000, len(records)))
    summary = {}
    for region in (*REGIONS, "mean"):
        before = np.asarray([[r["initial"][region]["dice"], r["initial"][region]["hd95_mm"]]
                             for r in records], dtype=float)
        after = np.asarray([[r[method][region]["dice"], r[method][region]["hd95_mm"]]
                            for r in records], dtype=float)
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("Nonfinite metrics cannot enter a paired summary")
        delta = (after - before) * np.asarray([1, -1])
        bootstrap = delta[indices].mean(1)
        categories = {f"dice_{d}_hd95_{h}": 0
                      for d in ("improved", "unchanged", "worsened")
                      for h in ("improved", "unchanged", "worsened")}
        for record in records:
            categories[change_category(record["initial"][region], record[method][region])] += 1
        summary[region] = dict(
            before_dice=float(before[:, 0].mean()), after_dice=float(after[:, 0].mean()),
            before_hd95_mm=float(before[:, 1].mean()), after_hd95_mm=float(after[:, 1].mean()),
            delta_dice=float(delta[:, 0].mean()), hd95_reduction_mm=float(delta[:, 1].mean()),
            delta_dice_ci95=np.quantile(bootstrap[:, 0], [.025, .975]).tolist(),
            hd95_reduction_ci95=np.quantile(bootstrap[:, 1], [.025, .975]).tolist(),
            change_categories=categories,
            both_strictly_improved=categories["dice_improved_hd95_improved"],
            dice_improved_hd95_worsened=categories["dice_improved_hd95_worsened"],
            hd95_improved_dice_worsened=categories["dice_worsened_hd95_improved"],
            # Compatibility counters overlap; only change_categories is exhaustive and disjoint.
            both_worsened_or_unchanged=int(np.sum(np.all(delta <= CHANGE_TOLERANCE, axis=1))),
            either_worsened=int(np.sum(np.any(delta < -CHANGE_TOLERANCE, axis=1))),
            both_unchanged=categories["dice_unchanged_hd95_unchanged"],
        )
    return summary


def _export_nifti(masks, case, data_root, output, partition):
    import nibabel as nib
    from .nifti import restore_full_mask

    patient = case["patient"]
    reference = nib.load(Path(data_root) / patient / f"{patient}_t1.nii.gz")
    if tuple(reference.shape) != tuple(case["original_shape"]):
        raise ValueError(f"Reference NIfTI shape disagrees with cached metadata: {patient}")
    for stage, mask in masks.items():
        full = restore_full_mask(mask, case)
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(full, reference.affine, header),
                 output / f"{partition}_{patient}_{stage}.nii.gz")


def _markdown_report(report):
    lines = [f"# 3D PPO boundary refinement: {report['partition']}", "",
             f"Patients: {report['patients']}; seed: {report['seed']}.",
             f"Metric grids from case metadata: {', '.join(report['evaluation_grids'])}.",
             "Metrics are computed on the cached evaluation grid. Native-grid NIfTI export "
             "does not change the metric grid.",
             "The partition name does not establish independence from earlier experiments "
             "or equivalence to an official challenge evaluation.", "",
             "| Method | ET DSC | TC DSC | WT DSC | ET HD95 (mm) | TC HD95 (mm) | WT HD95 (mm) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, label in METHODS:
        values = report[key]
        dice = " | ".join(f"{values[r]['after_dice']:.4f}" for r in REGIONS)
        hd95 = " | ".join(f"{values[r]['after_hd95_mm']:.3f}" for r in REGIONS)
        lines.append(f"| {label} | {dice} | {hd95} |")
    lines += ["", "All changes below are paired against the same initial masks.", "",
              "| Method | Mean DSC change (95% CI) | Mean HD95 reduction, mm (95% CI) | Both improved | Either worsened |",
              "|---|---:|---:|---:|---:|"]
    for key, label in METHODS[1:]:
        s = report[key]["mean"]
        dc, hc = s["delta_dice_ci95"], s["hd95_reduction_ci95"]
        lines.append(f"| {label} | {s['delta_dice']:.5f} [{dc[0]:.5f}, {dc[1]:.5f}] "
                     f"| {s['hd95_reduction_mm']:.3f} [{hc[0]:.3f}, {hc[1]:.3f}] "
                     f"| {s['both_strictly_improved']} | {s['either_worsened']} |")
    lines += ["", f"Mean PPO inference time: {report['mean_ppo_seconds']:.3f} seconds/patient "
              "including environment construction, reset/candidate generation, observations and policy steps; "
              "excluding backbone inference, checkpoint loading, scoring, postprocessing and file export.",
              "Connected-component postprocessing is evaluated separately and contributes no policy reward.",
              f"The nine mutually exclusive change categories and complete per-patient traces "
              f"are in {report['partition']}_report.json."]
    return "\n".join(lines) + "\n"


def evaluate_cases(cases, policy, env_config, output: Path, partition: str, seed: int,
                   data_root=None, provenance=None):
    """Measure initial, cleanup-only, raw PPO, and PPO+cleanup on identical cases.

    ``target`` is used only by the scorer after label-free inference. ``data_root``
    optionally enables original-lattice NIfTI export, not a different metric grid.
    This function performs no model fitting, checkpoint selection, or oracle control.
    """
    if partition not in ("validation", "test"):
        raise ValueError("partition must be 'validation' or 'test'")
    cases = list(cases)
    if not cases:
        raise ValueError("Cannot evaluate an empty partition")
    patient_ids = [case["patient"] for case in cases]
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Evaluation patients must be unique")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parameters = dict(env_config)
    parameters["reward_enabled"] = False
    tolerance = parameters.get("tolerance_mm", 2.)
    records = []
    for case in cases:
        if "target" not in case:
            raise ValueError(f"Labels required for scoring: {case['patient']}")
        initial_mask = np.asarray(case["logits"]).argmax(0).astype(np.uint8)
        # Whitelist the actual inference inputs rather than removing one known label key.
        inference_case = {key: case[key] for key in ("patient", "image", "logits", "spacing")}
        start = time.perf_counter()
        env = BoundaryEnv([inference_case], **parameters)
        observation, reset_info = env.reset(seed=seed)
        reset_seconds = time.perf_counter() - start
        actions = []
        loop_start = time.perf_counter()
        while not env.done:
            center = np.asarray(env.candidates[env.t]).tolist()
            action, _ = policy.predict(observation, deterministic=True)
            action = int(np.asarray(action).item())
            observation, _, terminated, truncated, info = env.step(action)
            actions.append(dict(info, center=center, terminated=bool(terminated),
                                truncated=bool(truncated)))
        loop_seconds = time.perf_counter() - loop_start
        elapsed = time.perf_counter() - start
        ppo_mask = env.mask.copy()
        env.close()
        cleanup_start = time.perf_counter()
        post = remove_small_components(initial_mask, min_voxels=10)
        baseline_cleanup_seconds = time.perf_counter() - cleanup_start
        cleanup_start = time.perf_counter()
        ppo_post = remove_small_components(ppo_mask, min_voxels=10)
        ppo_cleanup_seconds = time.perf_counter() - cleanup_start
        masks = dict(initial=initial_mask, postprocess=post, ppo=ppo_mask, ppo_postprocess=ppo_post)
        scores = {key: region_metrics(mask, case["target"], case["spacing"], tolerance,
                                      case.get("empty_distance_mm"))
                  for key, mask in masks.items()}
        categories = {key: {region: change_category(scores["initial"][region], scores[key][region])
                            for region in (*REGIONS, "mean")}
                      for key, _ in METHODS[1:]}
        spacing = np.asarray(case["spacing"], dtype=float).tolist()
        record = dict(patient=case["patient"], **scores, actions=actions,
                      candidates=int(reset_info["candidates"]),
                      actual_steps=len(actions),
                      non_keep_actions=sum(a["action"] != 6 for a in actions),
                      changed_steps=sum(a["changed_voxels"] > 0 for a in actions),
                      changed_voxel_events=sum(a["changed_voxels"] for a in actions),
                      ppo_changed_voxels=int(np.count_nonzero(ppo_mask != initial_mask)),
                      postprocess_changed_voxels=int(np.count_nonzero(post != initial_mask)),
                      ppo_postprocess_changed_voxels=int(np.count_nonzero(ppo_post != ppo_mask)),
                      ppo_seconds=elapsed, ppo_reset_seconds=reset_seconds,
                      ppo_step_seconds=loop_seconds,
                      postprocess_seconds=baseline_cleanup_seconds,
                      ppo_postprocess_seconds=ppo_cleanup_seconds,
                      grid=case.get("evaluation_grid", "unspecified"),
                      metric_shape=list(initial_mask.shape), spacing=spacing,
                      empty_distance_mm=float(case.get("empty_distance_mm",
                                                       np.linalg.norm(np.asarray(initial_mask.shape) * spacing))),
                      change_categories=categories)
        records.append(record)
        np.savez_compressed(output / f"{partition}_masks_{case['patient']}.npz", **masks,
                            target=case["target"], spacing=spacing)
        if data_root is not None:
            _export_nifti(masks, case, data_root, output, partition)
        log.info("PPO %s %s Dice %.4f -> %.4f; HD95 %.3f -> %.3f mm (before cleanup)",
                 partition.upper(), case["patient"], scores["initial"]["mean"]["dice"],
                 scores["ppo"]["mean"]["dice"], scores["initial"]["mean"]["hd95_mm"],
                 scores["ppo"]["mean"]["hd95_mm"])
    grids = sorted({r["grid"] for r in records})
    report = dict(schema_version=2, partition=partition, test_evaluated=partition == "test",
                  seed=int(seed), pilot=any("resampled" in grid for grid in grids),
                  native_resolution=all(grid in ("native", "native_cropped") for grid in grids),
                  evaluation_grids=grids, patients=len(records), records=records,
                  **{key: paired_summary(records, key, seed) for key, _ in METHODS},
                  mean_ppo_seconds=float(np.mean([r["ppo_seconds"] for r in records])),
                  timing_includes="environment construction, reset, candidate generation, observations, policy steps",
                  timing_excludes="backbone inference, checkpoint loading, postprocessing, scoring, export",
                  metric_protocol=dict(region_order=list(REGIONS), distance_unit="mm",
                                       surface_dice_tolerance_mm=tolerance,
                                       hd95="maximum of the two directed surfel-area-weighted 95th percentiles",
                                       both_empty="DSC=1, HD95=0",
                                       one_empty="DSC=0, HD95=case empty_distance_mm or evaluation-grid physical diagonal",
                                       change_tolerance=CHANGE_TOLERANCE,
                                       confidence_interval="2000 paired patient bootstrap resamples; one policy seed"),
                  postprocessing=dict(min_voxels=10, connectivity=6,
                                      largest_component_preserved=True,
                                      applied_outside_policy=True),
                  provenance=provenance or {})
    _write_json(output / f"{partition}_report.json", report)
    name = "RESULTS.md" if partition == "validation" else "TEST_RESULTS.md"
    (output / name).write_text(_markdown_report(report))
    return report
