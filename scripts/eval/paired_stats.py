"""Paired Wilcoxon + cluster bootstrap CI for Init vs Final DSC/HD95.

Primary unit is the patient (BraTS convention). Slice-level tests are reported
only as a sensitivity check because slices from the same patient are dependent.
Class imbalance (e.g. Large n=385) is handled by (1) per-class patient tests and
(2) a patient-cluster bootstrap of the slice-mean that the paper actually reports.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

CLASS_NAMES = {0: "Small", 1: "Medium", 2: "Large"}


def _wilcoxon_delta(delta: np.ndarray, alternative: str) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    if delta.size < 2 or np.allclose(delta, 0.0):
        return float("nan")
    try:
        return float(wilcoxon(delta, alternative=alternative, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _cluster_bootstrap_mean_delta(
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
    n_boot: int,
    seed: int,
    higher_is_better: bool,
) -> tuple[float, float]:
    """Resample patients, pool their slices, CI for mean(final)-mean(init)."""
    pids = list(groups)
    n = len(pids)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, n, size=n)
        init_parts = []
        fin_parts = []
        for j in draw:
            init, fin = groups[pids[j]]
            init_parts.append(init)
            fin_parts.append(fin)
        init_all = np.concatenate(init_parts)
        fin_all = np.concatenate(fin_parts)
        d = float(fin_all.mean() - init_all.mean())
        stats[b] = d if higher_is_better else -d
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _patient_groups(pid, values_init, values_fin, cls=None, class_id=None):
    groups = defaultdict(lambda: ([], []))
    for i, p in enumerate(pid):
        if class_id is not None and int(cls[i]) != class_id:
            continue
        groups[str(p)][0].append(values_init[i])
        groups[str(p)][1].append(values_fin[i])
    out = {}
    for p, (a, b) in groups.items():
        if not a:
            continue
        out[p] = (np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return out


def summarize_metric(pid, cls, init, final, name: str, higher_is_better: bool, n_boot=10000, seed=42):
    init = np.asarray(init, dtype=np.float64)
    final = np.asarray(final, dtype=np.float64)
    pid = np.asarray(pid)
    cls = np.asarray(cls)
    slice_delta = final - init if higher_is_better else init - final

    groups = _patient_groups(pid, init, final)
    patient_init = np.array([v[0].mean() for v in groups.values()])
    patient_fin = np.array([v[1].mean() for v in groups.values()])
    patient_delta = (patient_fin - patient_init) if higher_is_better else (patient_init - patient_fin)

    slice_mean_init = float(init.mean())
    slice_mean_fin = float(final.mean())
    slice_delta_mean = float(slice_mean_fin - slice_mean_init) if higher_is_better else float(slice_mean_init - slice_mean_fin)
    ci = _cluster_bootstrap_mean_delta(groups, n_boot, seed, higher_is_better)

    result = {
        "metric": name,
        "n_slices": int(len(init)),
        "n_patients": int(len(groups)),
        "slice_mean_init": slice_mean_init,
        "slice_mean_final": slice_mean_fin,
        "slice_delta": float(slice_mean_fin - slice_mean_init),
        "cluster_bootstrap_95ci": {"low": ci[0], "high": ci[1], "on": "signed_improvement"},
        "patient_mean_init": float(patient_init.mean()),
        "patient_mean_final": float(patient_fin.mean()),
        "patient_delta": float(patient_delta.mean()),
        "wilcoxon_patient": {
            "n": int(patient_delta.size),
            "p": _wilcoxon_delta(patient_delta, "greater"),
            "alternative": "greater (improvement)",
        },
        "wilcoxon_slice_anticonservative": {
            "n": int(slice_delta.size),
            "p": _wilcoxon_delta(slice_delta, "greater"),
            "note": "slices are not independent; do not use as primary p",
        },
        "by_class": {},
    }

    for c, cname in CLASS_NAMES.items():
        g = _patient_groups(pid, init, final, cls, c)
        if not g:
            continue
        idx = cls == c
        c_init, c_fin = init[idx], final[idx]
        p_init = np.array([v[0].mean() for v in g.values()])
        p_fin = np.array([v[1].mean() for v in g.values()])
        p_delta = (p_fin - p_init) if higher_is_better else (p_init - p_fin)
        c_ci = _cluster_bootstrap_mean_delta(g, n_boot, seed + 1 + c, higher_is_better)
        result["by_class"][cname] = {
            "n_slices": int(idx.sum()),
            "n_patients": int(len(g)),
            "slice_mean_init": float(c_init.mean()),
            "slice_mean_final": float(c_fin.mean()),
            "slice_delta": float(c_fin.mean() - c_init.mean()),
            "cluster_bootstrap_95ci": {"low": c_ci[0], "high": c_ci[1]},
            "patient_delta": float(p_delta.mean()),
            "wilcoxon_patient_p": _wilcoxon_delta(p_delta, "greater"),
        }
    return result


def format_report(dsc_res: dict, hd_res: dict) -> str:
    lines = []
    lines.append("=== Paired statistics (primary = patient cluster) ===")
    d = dsc_res
    ci = d["cluster_bootstrap_95ci"]
    lines.append(
        f"DSC  {d['slice_mean_init']:.4f} → {d['slice_mean_final']:.4f}  "
        f"(Δ {d['slice_delta']:+.4f}, 95% CI [{ci['low']:.4f}, {ci['high']:.4f}], "
        f"patient Wilcoxon p={d['wilcoxon_patient']['p']:.4g}, n_pat={d['n_patients']})"
    )
    h = hd_res
    hci = h["cluster_bootstrap_95ci"]
    lines.append(
        f"HD95 {h['slice_mean_init']:.4f} → {h['slice_mean_final']:.4f}  "
        f"(Δ {h['slice_delta']:+.4f}, improvement {(-h['slice_delta']):+.4f}, "
        f"95% CI [{hci['low']:.4f}, {hci['high']:.4f}], "
        f"patient Wilcoxon p={h['wilcoxon_patient']['p']:.4g}, n_pat={h['n_patients']})"
    )
    lines.append(
        f"  (CI is patient-cluster bootstrap of the slice-mean; "
        f"Wilcoxon is on n={d['n_patients']} patient means. "
        f"Slice Wilcoxon p_DSC={d['wilcoxon_slice_anticonservative']['p']:.4g} is anti-conservative.)"
    )
    lines.append("\n--- By class (patient-cluster; Large has fewer slices) ---")
    for cname in ("Small", "Medium", "Large"):
        dc = d["by_class"].get(cname)
        hc = h["by_class"].get(cname)
        if not dc:
            continue
        lines.append(
            f"{cname:<6} n_slice={dc['n_slices']:4d}  n_pat={dc['n_patients']:2d}  "
            f"DSC {dc['slice_mean_init']:.4f}→{dc['slice_mean_final']:.4f} "
            f"(Δ {dc['slice_delta']:+.4f}, 95% CI [{dc['cluster_bootstrap_95ci']['low']:.4f}, "
            f"{dc['cluster_bootstrap_95ci']['high']:.4f}], p={dc['wilcoxon_patient_p']:.4g})  "
            f"HD95 Δ {hc['slice_delta']:+.4f} p={hc['wilcoxon_patient_p']:.4g}"
        )
    return "\n".join(lines)


def report_from_npz(path: str, n_boot: int = 10000, seed: int = 42, json_out: str | None = None) -> dict:
    data = np.load(path, allow_pickle=True)
    pid = data["pid"]
    cls = data["cls"]
    dsc = summarize_metric(pid, cls, data["init_dsc"], data["final_dsc"], "DSC", True, n_boot, seed)
    hd = summarize_metric(pid, cls, data["init_hd95"], data["final_hd95"], "HD95", False, n_boot, seed)
    payload = {"dsc": dsc, "hd95": hd}
    print(format_report(dsc, hd))
    out = json_out or path.replace(".npz", "_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {out}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="results/pipeline_slice_metrics.npz")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report_from_npz(args.npz, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
