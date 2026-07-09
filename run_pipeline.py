import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def run_command(cmd, desc):
    log.info(f"=== 시작: {desc} ===")
    log.info(f"명령어: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error(f"❌ [{desc}] 실행 중 오류가 발생했습니다. (Return code: {result.returncode})")
        sys.exit(result.returncode)
    log.info(f"✅ === 완료: {desc} ===\n")

def main():
    parser = argparse.ArgumentParser(description="RL-Refiner End-to-End 파이프라인 실행 스크립트")
    parser.add_argument("--skip_unet", action="store_true", help="U-Net 학습 단계(Step 1)를 건너뜁니다.")
    parser.add_argument("--skip_agent", action="store_true", help="RL 에이전트 학습 단계(Step 3)를 건너뜁니다.")
    parser.add_argument("--skip_eval", action="store_true", help="성능 검증 단계(Step 4)를 건너뜁니다.")
    
    # 추가 인자들 (필요시 각 스크립트에 전달할 수도 있습니다)
    parser.add_argument("--config", type=str, default="configs/ppo_brats.yaml", help="Agent 학습용 YAML 설정 파일 경로")
    
    args = parser.parse_args()

    python_exec = sys.executable

    log.info("🚀 RL-Refiner 전체 파이프라인 실행을 시작합니다.")

    # 1. Step 1: U-Net 학습
    if not args.skip_unet:
        cmd_unet = [python_exec, "train_unet.py"]
        run_command(cmd_unet, "Step 1: U-Net (초기 모델) 학습")
    else:
        log.info("⏭️  Step 1: U-Net 학습 단계를 건너뜁니다.\n")

    # 2. Step 2 & 3: 환경 및 PPO 에이전트 학습
    if not args.skip_agent:
        cmd_agent = [python_exec, "train_agent.py", "--config", args.config]
        run_command(cmd_agent, "Step 2 & 3: RL 환경 기반 PPO 에이전트 학습")
    else:
        log.info("⏭️  Step 3: PPO 에이전트 학습 단계를 건너뜁니다.\n")

    # 3. Step 4: 성능 평가 및 검증
    if not args.skip_eval:
        cmd_eval = [python_exec, "evaluate.py"]
        run_command(cmd_eval, "Step 4: 성능 검증 및 벤치마크 (Evaluation)")
    else:
        log.info("⏭️  Step 4: 성능 검증 단계를 건너뜁니다.\n")

    log.info("🎉 모든 파이프라인 실행이 완료되었습니다!")

if __name__ == "__main__":
    main()
