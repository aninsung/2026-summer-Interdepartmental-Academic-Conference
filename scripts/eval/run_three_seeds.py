"""같은 400/851 분할에서 학습 시드 42, 7, 123을 평가한다.

시드 42는 이미 돌고 있는 파이프라인이 끝나면 그 가중치를 쓴다.
7과 123은 그 다음에 같은 명령으로 다시 학습한다. 환자 분할 파일은 바꾸지 않는다.
0.003은 제안 방법 환자 평균 DSC의 시드 범위와 비교한다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WEIGHTS = (
    "shape_classifier_best.pt",
    "caranet_best.pt",
    "unetplusplus_best.pt",
    "segresnet_best.pt",
    "band_refine_medium.pt",
    "band_ppo.pt",
)
SEEDS = (42, 7, 123)


def pipeline_running() -> bool:
    proc = subprocess.run(["pgrep", "-af", "run_pipeline.py"], capture_output=True, text=True)
    lines = [
        line for line in proc.stdout.splitlines()
        if "run_pipeline.py" in line and "pgrep" not in line and "run_three_seeds.py" not in line
    ]
    return bool(lines)


def snapshot(seed: int) -> str:
    dest = os.path.join(ROOT, "checkpoints", "seeds", str(seed))
    os.makedirs(dest, exist_ok=True)
    for name in WEIGHTS:
        src = os.path.join(ROOT, "checkpoints", name)
        if not os.path.exists(src):
            raise SystemExit(f"가중치가 없습니다: {src}")
        shutil.copy2(src, os.path.join(dest, name))
    return dest


def train(seed: int) -> None:
    env = os.environ.copy()
    env["BRATS_CACHE_DIR"] = os.path.join(ROOT, "cache")
    cmd = [
        sys.executable, "-u", "run_pipeline.py",
        "--refinement_profile", "band_ppo",
        "--max_train_patients", "400",
        "--num_workers", "24",
        "--parallel_stages",
        "--no_deterministic",
        "--batch_size", "64",
        "--skip_eval",
        "--seed", str(seed),
    ]
    print("학습", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def evaluate(seed: int, checkpoint_dir: str) -> str:
    out = os.path.join(ROOT, "results", "protocol_gaps", f"seed_{seed}.json")
    env = os.environ.copy()
    env["BRATS_CACHE_DIR"] = os.path.join(ROOT, "cache")
    env["BRATS_SKIP_CACHE_SAVE"] = "1"
    cmd = [
        sys.executable, "-u", "scripts/eval/evaluate_protocol_gaps.py",
        "--checkpoint_dir", checkpoint_dir,
        "--seed_label", str(seed),
        "--out", out,
    ]
    print("평가", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)
    return out


def summarize(paths: dict[int, str]) -> None:
    rows = []
    for seed, path in paths.items():
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        dsc = payload["tumor_slice_patient_dsc"]
        rows.append({
            "seed": seed,
            "ppo_dsc": dsc["ppo"]["mean"],
            "stage2_dsc": dsc["stage2"]["mean"],
            "large_dsc": dsc["large"]["mean"],
            "ppo_minus_large": dsc["ppo"]["mean"] - dsc["large"]["mean"],
            "empty_fp_ppo": payload["empty_slice_false_positive"]["ppo"]["false_positive_rate"],
            "dice_3d_ppo": payload["dice_3d"]["ppo"]["mean"],
            "common_hd95_mm_ppo": payload["hd95_common_slices"]["stage2_and_ppo"]["methods"]["ppo"]["patient_mean_mm"]["mean"],
            "common_hd95_mm_stage2": payload["hd95_common_slices"]["stage2_and_ppo"]["methods"]["stage2"]["patient_mean_mm"]["mean"],
        })
    ppo = [row["ppo_dsc"] for row in rows]
    spread = max(ppo) - min(ppo)
    summary = {
        "seeds": rows,
        "ppo_dsc_mean": sum(ppo) / len(ppo),
        "ppo_dsc_min": min(ppo),
        "ppo_dsc_max": max(ppo),
        "ppo_dsc_range": spread,
        "reference_gap": 0.003,
        "range_covers_0.003": bool(spread >= 0.003),
        "note": "0.003은 서로 다른 방법의 한 번 실행 차이다. 여기 범위는 제안 파이프라인을 같은 분할에서 시드만 바꿔 학습한 환자 평균 DSC다.",
    }
    dest = os.path.join(ROOT, "results", "protocol_gaps", "three_seeds.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    os.chdir(ROOT)
    while pipeline_running():
        print("시드 42 파이프라인이 끝날 때까지 대기", flush=True)
        time.sleep(30)
    paths = {}
    snapshot_42 = snapshot(42)
    paths[42] = evaluate(42, snapshot_42)
    for seed in SEEDS:
        if seed == 42:
            continue
        train(seed)
        paths[seed] = evaluate(seed, snapshot(seed))
    summarize(paths)


if __name__ == "__main__":
    main()
