"""
subregion_pipeline/run_subregion_pipeline.py
---------------------------------------------
Subregion-Guided (ET / TC / WT) 3-Stage Dynamic Routing 파이프라인 마스터 실행 스크립트

실행 순서:
  Stage 1: Subregion Classifier (ET / TC / WT 분류기) 학습
  Stage 2: Subregion Expert 백본 모델 (ET: Attention U-Net, TC: UNet++, WT: SegResNet) 학습
  Stage 3: Subregion 맞춤형 PPO 에이전트 (ET, TC, WT Refiner) 학습
  Stage 4: Subregion Dynamic Routing 파이프라인 최종 벤치마크 평가 및 시각화 저장
"""

import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_command_cmd(cmd_list, description):
    log.info(f"=== 시작: {description} ===")
    log.info(f"명령어: {' '.join(cmd_list)}")
    ret = subprocess.run(cmd_list)
    if ret.returncode != 0:
        log.error(f"❌ 실패: {description} (exit code: {ret.returncode})")
        sys.exit(ret.returncode)
    log.info(f"✅ 완료: {description}\n")


def main():
    parser = argparse.ArgumentParser(description="Subregion-Guided Dynamic Routing Master Pipeline Runner")
    parser.add_argument("--train_root", type=str, default="src/data/archive")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_train_patients", type=int, default=210)
    parser.add_argument("--max_eval_patients", type=int, default=20)
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--skip_stage3", action="store_true")
    args = parser.parse_args()

    python_exec = sys.executable

    log.info("🚀 Subregion-Guided (ET / TC / WT) 3-Stage Dynamic Routing 파이프라인 전체 실행을 시작합니다.\n")

    # ── Stage 1: Subregion Classifier ──────────────────────
    if not args.skip_stage1:
        cmd_stage1 = [
            python_exec, "subregion_pipeline/scripts/train_subregion_classifier.py",
            "--train_root", args.train_root,
            "--batch_size", str(args.batch_size),
            "--max_train_patients", str(args.max_train_patients),
        ]
        run_command_cmd(cmd_stage1, "Stage 1: Subregion Classifier (ET / TC / WT 분류기) 학습")

    # ── Stage 2: Subregion Expert Backbones ────────────────
    if not args.skip_stage2:
        for sub_mode in ["et", "tc", "wt"]:
            cmd_stage2 = [
                python_exec, "subregion_pipeline/scripts/train_subregion_experts.py",
                "--subregion", sub_mode,
                "--train_root", args.train_root,
                "--batch_size", str(args.batch_size),
                "--max_train_patients", str(args.max_train_patients),
            ]
            run_command_cmd(cmd_stage2, f"Stage 2 ({sub_mode.upper()}): Subregion Expert 백본 모델 학습")

    # ── Stage 3: Subregion PPO Refiners ───────────────────
    if not args.skip_stage3:
        for sub_mode in ["et", "tc", "wt"]:
            cmd_stage3 = [
                python_exec, "subregion_pipeline/scripts/train_subregion_ppo.py",
                "--subregion", sub_mode,
                "--train_root", args.train_root,
                "--max_train_patients", str(args.max_train_patients),
                "--total_timesteps", "40000",
            ]
            run_command_cmd(cmd_stage3, f"Stage 3 ({sub_mode.upper()}): Subregion 맞춤형 PPO 에이전트 학습")

    # ── Stage 4: Evaluation & Visualization ────────────────
    cmd_stage4 = [
        python_exec, "subregion_pipeline/scripts/evaluate_subregion_pipeline.py",
        "--train_root", args.train_root,
        "--max_patients", str(args.max_eval_patients),
    ]
    run_command_cmd(cmd_stage4, "Stage 4: Subregion-Guided Dynamic Routing 파이프라인 최종 성능 검증 및 시각화")

    log.info("🎉 Subregion-Guided Dynamic Routing 파이프라인 전 과정이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()
