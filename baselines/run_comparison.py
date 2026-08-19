"""두 베이스라인을 학습·평가하고 비교 표를 만드는 오케스트레이터.

파이프라인과 같은 환자 분할·모달리티·해상도·지표를 쓰므로, 여기서 나온 표를
적응형 파이프라인 결과와 그대로 나란히 놓을 수 있다.

사용 예시
    python baselines/run_comparison.py
    python baselines/run_comparison.py --epochs 30 --skip_train   # 이미 학습했다면
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

METHOD_LABELS = {
    "kaist": "Extending nnU-Net (KAIST, BraTS21 1위)",
    "nvauto": "SegResNet + 중복 감소 (NVAUTO, BraTS21 2위)",
}


def run(cmd: list[str], desc: str) -> None:
    log.info(f"=== 시작: {desc} ===")
    log.info("명령어: " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error(f"❌ [{desc}] 실패 (return code {result.returncode})")
        sys.exit(result.returncode)
    log.info(f"✅ === 완료: {desc} ===\n")


def build_markdown(rows: dict[str, dict]) -> str:
    lines = [
        "| 방법 | DSC | HD95 | Precision | Recall |",
        "|---|---|---|---|---|",
    ]
    for label, res in rows.items():
        o = res["Overall"]
        lines.append(
            f"| {label} | {o['dsc']:.4f} | {o['hd95']:.3f} | "
            f"{o['precision']:.4f} | {o['recall']:.4f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="BraTS 2021 상위 입상 방법 베이스라인 비교")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_train_patients", type=int, default=210)
    p.add_argument("--patient_split", type=str, default="checkpoints/patient_split.json")
    p.add_argument("--modality", type=str, default="t1ce+flair")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", type=str, nargs="+", default=["kaist", "nvauto"])
    p.add_argument("--skip_train", action="store_true", help="학습을 건너뛰고 평가만 한다.")
    p.add_argument("--sweep", action="store_true", help="평가 시 임계값 스윕을 함께 수행한다.")
    args = p.parse_args()

    python_exec = sys.executable
    common = [
        "--max_train_patients", str(args.max_train_patients),
        "--patient_split", args.patient_split,
        "--modality", args.modality,
        "--seed", str(args.seed),
    ]

    # ── 학습 ──
    if not args.skip_train:
        for method in args.methods:
            cmd = [
                python_exec, "baselines/train_baseline.py",
                "--method", method,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
            ] + common
            run(cmd, f"학습: {METHOD_LABELS[method]}")

    # ── 평가 ──
    rows = {}
    for method in args.methods:
        ckpt = f"baselines/checkpoints/{method}_best.pt"
        if not os.path.exists(ckpt):
            log.warning(f"체크포인트가 없어 건너뜁니다: {ckpt}")
            continue
        out_json = f"baselines/results/{method}_metrics.json"
        cmd = [
            python_exec, "baselines/eval_baseline.py",
            "--checkpoint", ckpt,
            "--max_patients", str(args.max_train_patients),
            "--patient_split", args.patient_split,
            "--modality", args.modality,
            "--out_json", out_json,
        ]
        if args.sweep:
            cmd.append("--sweep")
        run(cmd, f"평가: {METHOD_LABELS[method]}")

        with open(out_json, encoding="utf-8") as f:
            rows[METHOD_LABELS[method]] = json.load(f)["results"]

    if not rows:
        log.error("평가 결과가 없습니다.")
        sys.exit(1)

    # ── 비교 표 ──
    table = build_markdown(rows)
    log.info("\n" + table)

    os.makedirs("baselines/results", exist_ok=True)
    out_md = "baselines/results/comparison.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# BraTS 2021 상위 입상 방법 베이스라인 비교\n\n")
        f.write(
            "모든 결과는 checkpoints/patient_split.json 의 동일한 42명 검증 환자에서\n"
            "동일한 지표 구현(src/utils/metrics.py)으로 측정했다.\n"
            "3D/4모달리티/앙상블을 제외한 2D 각색 구현이므로 원 논문의 리더보드\n"
            "점수와 직접 비교할 수 없다. 자세한 차이는 baselines/README.md 참고.\n\n"
        )
        f.write(table + "\n")
    log.info(f"\n비교 표 저장: {out_md}")


if __name__ == "__main__":
    main()
