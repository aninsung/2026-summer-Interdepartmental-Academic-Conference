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
    # CLI 인자의 '=' 형태 자동 유연 보정 (예: batch_size=64 -> --batch_size 64)
    import sys
    new_argv = []
    for arg in sys.argv:
        if "=" in arg and not arg.startswith("-"):
            key, val = arg.split("=", 1)
            new_argv += [f"--{key}", val]
        else:
            new_argv.append(arg)
    sys.argv = new_argv

    parser = argparse.ArgumentParser(description="RL-Refiner Dynamic Routing 파이프라인 실행 스크립트")
    parser.add_argument("--skip_classifier", action="store_true", help="Stage 1: Shape Classifier 학습 단계를 건너뜁니다.")
    parser.add_argument("--skip_experts", action="store_true", help="Stage 2: Expert 백본 모델(3종) 학습 단계를 건너뜁니다.")
    parser.add_argument("--skip_agents", action="store_true", help="Stage 3: 맞춤형 PPO 에이전트(3종) 학습 단계를 건너뜁니다.")
    parser.add_argument("--skip_eval", action="store_true", help="Stage 4: 전체 파이프라인 성능 검증 단계를 건너뜁니다.")
    
    # 공통 설정
    parser.add_argument("--config", type=str, default="configs/ppo_brats.yaml", help="Agent 학습용 YAML 설정 파일 경로")
    parser.add_argument("--batch_size", type=int, default=None, help="배치 크기 (기본값: 각 스크립트 기본값 사용)")
    parser.add_argument("--max_train_patients", type=int, default=None, help="학습 환자 수 제한")
    parser.add_argument("--epochs", type=int, default=None, help="학습 에폭 수")

    args = parser.parse_args()

    python_exec = sys.executable

    log.info("🚀 RL-Refiner 3-Stage Dynamic Routing 파이프라인 전체 실행을 시작합니다.")

    extra_args = ["--train_root", "src/data/archive/BraTS2021_Training_Data"]
    if args.batch_size is not None:
        extra_args += ["--batch_size", str(args.batch_size)]
    if args.max_train_patients is not None:
        extra_args += ["--max_train_patients", str(args.max_train_patients)]
    if args.epochs is not None:
        extra_args += ["--epochs", str(args.epochs)]

    # 1. Stage 1: Shape Classifier 학습
    if not args.skip_classifier:
        cmd_cls = [python_exec, "train_shape_classifier.py"] + extra_args
        run_command(cmd_cls, "Stage 1: Shape Classifier (종양 크기 판별기) 학습")
    else:
        log.info("⏭️  Stage 1: Shape Classifier 학습 단계를 건너뜁니다.\n")

    # 2. Stage 2: Expert 백본 모델 3종 학습 (Pre-training -> Fine-tuning 기법 적용)
    if not args.skip_experts:
        # ── Small Expert (Attention U-Net) ──
        gen_path_att = "checkpoints/attention_unet_general.pt"
        if not os.path.exists(gen_path_att):
            cmd_att_gen = [python_exec, "train_attention_unet.py"] + extra_args + ["--save_path", gen_path_att]
            run_command(cmd_att_gen, "Stage 2 (Small): Attention U-Net 일반 사전 학습 (General Pre-training)")
        else:
            log.info(f"ℹ️  기존 Attention U-Net 일반 사전 학습 가중치 발견 ({gen_path_att}) -> 사전 학습 생략")
        
        cmd_att_ft = [python_exec, "train_attention_unet.py"] + extra_args + ["--refinement_mode", "small", "--pretrained_path", gen_path_att]
        run_command(cmd_att_ft, "Stage 2 (Small): Attention U-Net 소형 종양 특화 파인튜닝 (Fine-tuning)")
        
        # ── Medium Expert (UNet++) ──
        gen_path_unetpp = "checkpoints/unetplusplus_general.pt"
        if not os.path.exists(gen_path_unetpp):
            cmd_unetpp_gen = [python_exec, "train_unetplusplus.py"] + extra_args + ["--save_path", gen_path_unetpp]
            run_command(cmd_unetpp_gen, "Stage 2 (Medium): UNet++ 일반 사전 학습 (General Pre-training)")
        else:
            log.info(f"ℹ️  기존 UNet++ 일반 사전 학습 가중치 발견 ({gen_path_unetpp}) -> 사전 학습 생략")
            
        cmd_unetpp_ft = [python_exec, "train_unetplusplus.py"] + extra_args + ["--refinement_mode", "medium", "--pretrained_path", gen_path_unetpp]
        run_command(cmd_unetpp_ft, "Stage 2 (Medium): UNet++ 중형 종양 특화 파인튜닝 (Fine-tuning)")
        
        # ── Large Expert (SegResNet) ──
        gen_path_seg = "checkpoints/segresnet_general.pt"
        if not os.path.exists(gen_path_seg):
            cmd_seg_gen = [python_exec, "train_segresnet.py"] + extra_args + ["--save_path", gen_path_seg]
            run_command(cmd_seg_gen, "Stage 2 (Large): SegResNet 일반 사전 학습 (General Pre-training)")
        else:
            log.info(f"ℹ️  기존 SegResNet 일반 사전 학습 가중치 발견 ({gen_path_seg}) -> 사전 학습 생략")
            
        cmd_seg_ft = [python_exec, "train_segresnet.py"] + extra_args + ["--refinement_mode", "large", "--pretrained_path", gen_path_seg]
        run_command(cmd_seg_ft, "Stage 2 (Large): SegResNet 대형 종양 특화 파인튜닝 (Fine-tuning)")
    else:
        log.info("⏭️  Stage 2: Expert 백본 모델 3종 학습 단계를 건너뜁니다.\n")

    # 3. Stage 3: 맞춤형 PPO 에이전트 3종 학습
    if not args.skip_agents:
        agent_base_cmd = [python_exec, "train_agent.py", "--config", args.config, "--train_root", "src/data/archive/BraTS2021_Training_Data"]
        
        # Small Agent
        run_command(agent_base_cmd + ["--model_type", "attention_unet", "--refinement_mode", "small"], 
                    "Stage 3 (Small): Class 0 맞춤형 PPO 에이전트 학습")
        
        # Medium Agent
        run_command(agent_base_cmd + ["--model_type", "unetplusplus", "--refinement_mode", "medium"], 
                    "Stage 3 (Medium): Class 1 맞춤형 PPO 에이전트 학습")
        
        # Large Agent
        run_command(agent_base_cmd + ["--model_type", "segresnet", "--refinement_mode", "large"], 
                    "Stage 3 (Large): Class 2 맞춤형 PPO 에이전트 학습")
    else:
        log.info("⏭️  Stage 3: 맞춤형 PPO 에이전트 3종 학습 단계를 건너뜁니다.\n")

    # 4. Stage 4: 전체 동적 라우팅 파이프라인 성능 평가 (`evaluate_pipeline.py`)
    if not args.skip_eval:
        cmd_eval = [python_exec, "evaluate_pipeline.py"]
        run_command(cmd_eval, "Stage 4: 3-Stage Adaptive Pipeline 최종 성능 검증")
    else:
        log.info("⏭️  Stage 4: 성능 검증 단계를 건너뜁니다.\n")

    log.info("🎉 Dynamic Routing 파이프라인의 모든 과정이 완료되었습니다!")

if __name__ == "__main__":
    main()

