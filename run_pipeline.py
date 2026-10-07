import os
import sys
import time
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")

def run_command(cmd, desc, wait=True):
    log.info(f"=== 시작: {desc} ===")
    log.info(f"명령어: {' '.join(cmd)}")
    if wait:
        ret = subprocess.call(cmd)
        if ret != 0:
            log.error(f"❌ [{desc}] 실행 중 오류가 발생했습니다. (Return code: {ret})")
            sys.exit(ret)
        log.info(f"✅ === 완료: {desc} ===\n")
        return None
    else:
        return subprocess.Popen(cmd)

def wait_for_external_seed_train():
    """오케스트레이터가 시드 123을 또 학습하지 않게, 밖에서 겹쳐 돌린 학습이 끝나길 기다린다."""
    if os.environ.get("EXTERNAL_SEED_TRAIN") == "1":
        return
    if "--seed" not in sys.argv:
        return
    seed_arg = sys.argv[sys.argv.index("--seed") + 1]
    if seed_arg != "123":
        return
    root = os.path.dirname(os.path.abspath(__file__))
    gap = os.path.join(root, "results", "protocol_gaps")
    lock = os.path.join(gap, "seed123_external.lock")
    done = os.path.join(gap, "seed123_external.done")
    fail = os.path.join(gap, "seed123_external.fail")
    if not os.path.exists(lock):
        return
    log.info("시드 123 학습은 평가와 겹쳐 이미 실행 중이다. 끝날 때까지 대기한다.")
    while not (os.path.exists(done) or os.path.exists(fail)):
        time.sleep(5)
    if os.path.exists(fail):
        os.remove(fail)
        os.remove(lock)
        log.info("겹친 시드 123 학습이 실패해서 이 프로세스에서 다시 학습한다.")
        return
    sys.exit(0)


def main():
    wait_for_external_seed_train()
    parser = argparse.ArgumentParser(description="RL-Refiner Dynamic Routing 파이프라인 실행 스크립트")
    parser.add_argument("--skip_classifier", action="store_true")
    parser.add_argument("--skip_experts", action="store_true")
    parser.add_argument("--skip_agents", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    
    parser.add_argument("--config", type=str, default="configs/ppo_brats.yaml")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_patients", type=int, default=400)
    parser.add_argument("--refinement_profile", choices=["legacy", "ppo_v2", "ppo_v3", "ppo_v4", "ppo_v5", "band_ppo"], default=None)
    parser.add_argument("--slice_selection", choices=["tumor", "all"], default="all")
    parser.add_argument("--agent_dir", default=None)
    parser.add_argument("--eval_mode", choices=["stage2", "augmentation", "ppo_raw", "heuristic", "quality", "oracle", "compare"], default=None)
    parser.add_argument("--split_role", choices=["val", "test", "all"], default="val")
    parser.add_argument("--output_dir")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--allow_oracle_gate", action="store_true")
    parser.add_argument("--disable_edge_gate", action="store_true")
    parser.add_argument("--stage2_thresholds", default="0.80,0.80,0.50")
    parser.add_argument("--cc_min_sizes", default="0,15,25")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--modality", type=str, default="t1ce+flair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_deterministic", action="store_true")
    parser.add_argument("--parallel_stages", action="store_true")
    parser.add_argument("--auto_tegda", action="store_true")

    args = parser.parse_args()
    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    # 기본 config의 ppo_v2는 예전 크기별 에이전트다. 프로필을 지정하지 않으면
    # 경계 띠 PPO 하나를 학습하고 851명 평가를 한다.
    if args.refinement_profile is None:
        from_config = config.get("refinement_profile", "ppo_v2")
        args.refinement_profile = "band_ppo" if from_config == "ppo_v2" else from_config
    if args.refinement_profile not in {"legacy", "ppo_v2", "ppo_v3", "ppo_v4", "ppo_v5", "band_ppo"}:
        parser.error("Unknown refinement_profile")
    if args.refinement_profile == "band_ppo" and args.auto_tegda:
        parser.error("band_ppo는 TEGDA·품질 gate를 쓰지 않습니다")
    args.agent_dir = args.agent_dir or ("checkpoints" if args.refinement_profile == "legacy"
                                       else f"checkpoints/{args.refinement_profile}")
    args.eval_mode = args.eval_mode or ("compare" if args.refinement_profile == "ppo_v3" else "heuristic")
    if args.refinement_profile == "ppo_v3":
        if args.auto_tegda or args.eval_mode in {"quality", "heuristic"}:
            parser.error("ppo_v3 is an ungated ablation; use --eval_mode compare or ppo_raw")
        if config.get("max_steps", 30) != 15:
            parser.error("ppo_v3 requires max_steps: 15 in the config")
        if args.slice_selection != "all":
            parser.error("ppo_v3 requires --slice_selection all")
    os.environ["BRATS_NUM_WORKERS"] = str(max(1, args.num_workers))
    deterministic = not args.no_deterministic
    python_exec = sys.executable
    n_patients = args.max_train_patients
    log.info("🚀 파이프라인 시작")

    from src.data.patient_split import load_or_create_patient_split
    split_path = "checkpoints/patient_split.json"
    training = not (args.skip_classifier and args.skip_experts and args.skip_agents)
    if training:
        load_or_create_patient_split("src/data/archive", n_patients, split_path)
    
    seed_args = ["--seed", str(args.seed)]
    if deterministic:
        seed_args.append("--deterministic")

    extra_args = ["--train_root", "src/data/archive", "--max_train_patients", str(n_patients), "--patient_split", split_path] + seed_args
    agent_data_args = extra_args.copy() + ["--modality", args.modality]
    if args.batch_size is not None:
        extra_args += ["--batch_size", str(args.batch_size)]
    if args.epochs is not None:
        extra_args += ["--epochs", str(args.epochs)]
    if args.modality is not None:
        extra_args += ["--modality", str(args.modality)]
    
    worker_arg = ["--num_workers", str(args.num_workers)]

    # 1. Stage 1
    if not args.skip_classifier:
        cmd_cls = [python_exec, "scripts/train/train_shape_classifier.py"] + extra_args + worker_arg
        run_command(cmd_cls, "Stage 1: Shape Classifier")

    # 2. Stage 2
    if not args.skip_experts:
        cmds = []
        cmds.append(([python_exec, "scripts/train/train_caranet.py"] + extra_args + worker_arg + ["--refinement_mode", "small", "--save_path", "checkpoints/caranet_best.pt"], "Stage 2 (Small): CaraNet"))
        cmds.append(([python_exec, "scripts/train/train_unetplusplus.py"] + extra_args + worker_arg + ["--refinement_mode", "medium", "--save_path", "checkpoints/unetplusplus_best.pt"], "Stage 2 (Medium): UNet++"))
        cmds.append(([python_exec, "scripts/train/train_segresnet.py"] + extra_args + worker_arg + ["--save_path", "checkpoints/segresnet_best.pt"], "Stage 2 (Large): SegResNet"))
        
        if args.parallel_stages:
            procs = [(run_command(c, d, wait=False), d) for c, d in cmds]
            for p, d in procs:
                if p.wait() != 0:
                    log.error(f"❌ {d} 실패")
                    sys.exit(1)
        else:
            for c, d in cmds:
                run_command(c, d)

    # 3. Stage 3
    if not args.skip_agents and args.refinement_profile == "band_ppo":
        band_data = ["--train_root", "src/data/archive", "--max_train_patients", str(n_patients),
                     "--patient_split", split_path, "--modality", args.modality, "--seed", str(args.seed)]
        run_command(
            [python_exec, "scripts/train/train_band_refine.py", "--classes", "medium", "--epochs", "20",
             "--num_workers", str(args.num_workers), "--save_path", "checkpoints/band_refine_medium.pt"] + band_data,
            "Stage 3: Medium 경계 띠 지도학습",
        )
        run_command(
            [python_exec, "scripts/train/train_band_ppo.py", "--epochs", "6",
             "--pretrained", "checkpoints/band_refine_medium.pt",
             "--save_path", "checkpoints/band_ppo.pt"] + band_data,
            "Stage 3: 경계 띠 PPO",
        )
    elif not args.skip_agents:
        agent_base = [python_exec, "scripts/train/train_agent.py", "--config", args.config, "--refinement_profile", args.refinement_profile, "--stage2_thresholds", args.stage2_thresholds, "--cc_min_sizes", args.cc_min_sizes] + agent_data_args
        cmds = [
            (agent_base + ["--model_type", "caranet", "--refinement_mode", "small", "--save_path", os.path.join(args.agent_dir, "ppo_small.zip")], "Stage 3 (Small)"),
            (agent_base + ["--model_type", "unetplusplus", "--refinement_mode", "medium", "--save_path", os.path.join(args.agent_dir, "ppo_medium.zip")], "Stage 3 (Medium)"),
            (agent_base + ["--model_type", "segresnet", "--refinement_mode", "large", "--save_path", os.path.join(args.agent_dir, "ppo_large.zip")], "Stage 3 (Large)"),
        ]
        
        if args.parallel_stages:
            procs = [(run_command(c, d, wait=False), d) for c, d in cmds]
            for p, d in procs:
                if p.wait() != 0:
                    log.error(f"❌ {d} 실패")
                    sys.exit(1)
        else:
            for c, d in cmds:
                run_command(c, d)

    # 3.5 TEGDA
    if args.auto_tegda:
        log.info("Stage 3.5 TEGDA 학습 시작")
        cmd_eval_train = [python_exec, "scripts/eval/evaluate_pipeline.py", "--max_patients", str(n_patients), "--patient_split", split_path, "--split_role", "train", "--eval_mode", "compare", "--output_dir", "results/gate_train", "--modality", str(args.modality), "--refinement_profile", args.refinement_profile, "--agent_dir", args.agent_dir, "--slice_selection", args.slice_selection] + seed_args
        run_command(cmd_eval_train, "TEGDA 데이터 생성 (Train split)")
        run_command([python_exec, "scripts/train/train_energy_model.py", "--train_root", "src/data/archive", "--patient_split", split_path, "--max_patients", str(n_patients), "--output", "checkpoints/energy_model.pt"], "Energy Model 학습")
        run_command([python_exec, "scripts/train/train_quality_gate.py", "--records", "results/gate_train", "--output", "checkpoints/quality_gate.json"], "Quality Gate 학습")

    # 4. Stage 4
    if not args.skip_eval and args.refinement_profile == "band_ppo":
        cmd_eval = [python_exec, "scripts/eval/evaluate_band_ppo_locked.py",
                    "--train_root", "src/data/archive", "--patient_split", split_path,
                    "--checkpoint", "checkpoints/band_ppo.pt", "--seed", str(args.seed)]
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            cmd_eval += ["--out", os.path.join(args.output_dir, "band_ppo_locked.json")]
        run_command(cmd_eval, "Stage 4: 851명 환자 평균 평가")
    elif not args.skip_eval:
        cmd_eval = [python_exec, "scripts/eval/evaluate_pipeline.py", "--max_patients", str(n_patients), "--patient_split", split_path, "--split_role", args.split_role, "--slice_selection", args.slice_selection, "--refinement_profile", args.refinement_profile, "--agent_dir", args.agent_dir]
        cmd_eval += ["--eval_mode", args.eval_mode, "--stage2_thresholds", args.stage2_thresholds,
                     "--cc_min_sizes", args.cc_min_sizes, "--seed", str(args.seed),
                     "--deterministic" if deterministic else "--no_deterministic"]
        if args.output_dir:
            cmd_eval += ["--output_dir", args.output_dir]
        for flag in ("no_plots", "allow_oracle_gate", "disable_edge_gate"):
            if getattr(args, flag):
                cmd_eval.append(f"--{flag}")
        if args.modality is not None:
            cmd_eval += ["--modality", str(args.modality)]
        if args.auto_tegda:
            cmd_eval += ["--eval_mode", "quality", "--quality_gate", "checkpoints/quality_gate.json", "--energy_model", "checkpoints/energy_model.pt"]
        run_command(cmd_eval, "Stage 4: 최종 성능 검증")

if __name__ == "__main__":
    main()
