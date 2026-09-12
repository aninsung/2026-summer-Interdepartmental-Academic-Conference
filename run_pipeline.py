import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def run_command(cmd, desc):
    log.info(f"=== 시작: {desc} ===")
    log.info(f"명령어: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        log.error(f"❌ [{desc}] 실행 중 오류가 발생했습니다. (Return code: {result.returncode})")
        sys.exit(result.returncode)
    log.info(f"✅ === 완료: {desc} ===\n")

def archive_existing_checkpoints(paths, reason):
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join("checkpoints", "archived_before_fresh_train", stamp)
    os.makedirs(archive_dir, exist_ok=True)
    for path in existing:
        dst = os.path.join(archive_dir, os.path.basename(path))
        os.replace(path, dst)
        log.info("기존 checkpoint 미사용(%s): %s → %s", reason, path, dst)

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
    parser.add_argument("--skip_agents", action="store_true", help="Stage 3: PPO mask refiner 학습을 건너뜁니다.")
    parser.add_argument("--skip_eval", action="store_true", help="Stage 4: 전체 파이프라인 성능 검증 단계를 건너뜁니다.")
    parser.add_argument(
        "--alt_ppo_timesteps",
        type=int,
        default=40000,
        help="PPO mask refiner 학습 step 수 (0이면 Stage3 학습만 스킵하려면 --skip_agents 권장)",
    )
    parser.add_argument("--skip_plot", action="store_true", help="Stage4 시각화 PNG 생략")
    
    # 공통 설정
    parser.add_argument("--config", type=str, default="configs/ppo_brats.yaml", help="Agent 학습용 YAML 설정 파일 경로")
    parser.add_argument("--batch_size", type=int, default=None, help="배치 크기 (기본값: 각 스크립트 기본값 사용)")
    parser.add_argument("--max_train_patients", type=int, default=1251, help="Stage 1–4 공통 환자 수")
    parser.add_argument("--epochs", type=int, default=None, help="학습 에폭 수")
    parser.add_argument("--modality", type=str, default="t1ce+flair", help="MRI 모달리티 ('t1ce', 't1ce+flair', 't1ce+t2' 등)")
    parser.add_argument("--seed", type=int, default=42, help="전역 시드 (Stage 1–4 공통)")
    parser.add_argument("--deterministic", action="store_true",
                        help="cuDNN 결정적 모드 (기본 활성화, 이 플래그는 하위 호환용)")
    parser.add_argument("--no_deterministic", action="store_true",
                        help="결정적 모드 해제 (재현성을 포기하고 속도를 높임)")
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="tqdm 진행바 활성화 (기본 OFF — tee/에이전트 로그 비대화 방지)",
    )

    parser.add_argument("--fast", action="store_true", help="Speed profile: fewer rounds/epochs, larger batches, nondeterministic CUDA")
    parser.add_argument("--under_1h", action="store_true", help="Sub-hour profile for the full patient pool")
    args = parser.parse_args()
    if args.fast:
        args.no_deterministic = True
        args.batch_size = args.batch_size or 128
        args.epochs = args.epochs or 3
        args.skip_plot = True
    if args.under_1h:
        args.no_deterministic = True
        args.batch_size = args.batch_size or 128
        args.epochs = min(args.epochs or 3, 3)
        args.skip_plot = True
        args.alt_ppo_timesteps = min(args.alt_ppo_timesteps, 10000)

    # 재현성을 기본값으로 둔다. 해제는 --no_deterministic 으로만 가능하다.
    deterministic = not args.no_deterministic

    from src.utils.progress import configure_quiet_logs
    configure_quiet_logs(verbose_progress=bool(args.verbose_progress))

    python_exec = sys.executable

    n_patients = args.max_train_patients if args.max_train_patients is not None else 1251
    log.info("🚀 RL-Refiner 4-Stage Dynamic Routing 파이프라인 전체 실행을 시작합니다.")
    if not args.verbose_progress:
        log.info("진행바 OFF (TQDM_DISABLE=1). 켜려면 --verbose-progress")

    # GPU 전용: CUDA 없으면 즉시 종료 (CPU fallback 금지)
    from src.utils.device import require_cuda_device
    require_cuda_device()

    from src.data.patient_split import load_or_create_patient_split
    split_path = "checkpoints/patient_split.json"
    split = load_or_create_patient_split("src/data/archive", n_patients, split_path)
    log.info(
        f"환자 풀 {n_patients}명 → train {len(split['train'])} / val {len(split['val'])} "
        f"(seed={split.get('seed')}, 파일={split_path})"
    )

    seed_args = ["--seed", str(args.seed)]
    if deterministic:
        seed_args.append("--deterministic")
    log.info(f"시드 {args.seed} / 결정적 모드 {'ON' if deterministic else 'OFF'}")

    extra_args = [
        "--train_root", "src/data/archive",
        "--max_train_patients", str(n_patients),
        "--patient_split", split_path,
    ] + seed_args
    if args.batch_size is not None:
        extra_args += ["--batch_size", str(args.batch_size)]
    if args.epochs is not None:
        extra_args += ["--epochs", str(args.epochs)]
    if args.modality is not None:
        extra_args += ["--modality", str(args.modality)]

    # 1. Stage 1: Shape Classifier 학습 (P2: P1 + boundary soft + ordinal)
    if not args.skip_classifier:
        archive_existing_checkpoints(
            ["checkpoints/shape_classifier_best.pt"],
            "Stage1 fresh train",
        )
        cmd_cls = (
            [python_exec, "scripts/train/train_shape_classifier.py"]
            + extra_args
            + ["--recipe", "p2", "--backbone", "resnet18"]
        )
        run_command(cmd_cls, "Stage 1: Shape Classifier (종양 크기 판별기) 학습")
    else:
        log.info("⏭️  Stage 1: Shape Classifier 학습 단계를 건너뜁니다.\n")

    # 2. Stage 2: Expert 3종 — BraTS 1회 로드 후 크기별 필터/학습
    if not args.skip_experts:
        archive_existing_checkpoints(
            [
                "checkpoints/caranet_best.pt",
                "checkpoints/unetplusplus_best.pt",
                "checkpoints/segresnet_best.pt",
            ],
            "Stage2 fresh train",
        )
        cmd_stage2 = [python_exec, "scripts/train/train_stage2_all.py"] + extra_args
        run_command(cmd_stage2, "Stage 2: CaraNet/UNet++/SegResNet (BraTS 1회 로드)")
    else:
        log.info("⏭️  Stage 2: Expert 백본 모델 3종 학습 단계를 건너뜁니다.\n")

    # 3. Stage 3: single PPO mask refiner
    if not args.skip_agents:
        if int(args.alt_ppo_timesteps) <= 0:
            log.info("⏭️  Stage 3: alt_ppo_timesteps<=0 이므로 PPO mask refiner 학습을 건너뜁니다.\n")
        else:
            archive_existing_checkpoints(
                [
                    "checkpoints/ppo_mask_refiner.zip",
                    "checkpoints/ppo_mask_refiner.json",
                    "checkpoints/ppo_unified.zip",
                    "checkpoints/ppo_unified.json",
                ],
                "Stage3 fresh train",
            )
            cmd_alt = [
                python_exec, "scripts/train/train_ppo_mask_refiner.py",
                "--train_root", "src/data/archive",
                "--max_train_patients", str(n_patients),
                "--patient_split", split_path,
                "--modality", str(args.modality),
                "--seed", str(args.seed),
                "--ppo_timesteps", str(args.alt_ppo_timesteps),
            ]
            run_command(cmd_alt, "Stage 3: PPO mask refiner")
    else:
        log.info("⏭️  Stage 3: PPO mask refiner 학습 단계를 건너뜁니다.\n")

    # 4. Stage 4: 배포형 평가 (PPO mask refiner)
    if not args.skip_eval:
        cmd_deploy = [
            python_exec, "scripts/eval/evaluate_pipeline.py",
            "--max_patients", str(n_patients),
            "--patient_split", split_path,
            "--split_role", "val",
            "--deploy_mode",
            "--stage3_mode", "ppo",
            "--stage3_skip_classes", "",
            "--boundary_band_px", "2",
            "--boundary_band_mode", "expand",
            "--boundary_band_classes", "1,2",
            "--medium_active_max_area", "0",
            "--medium_skip_min_area", "0",
            "--sl_zoom_patches", "0",
            "--stage2_thresholds", "0.70,0.75,0.50",
            "--cc_min_sizes", "0,15,25",
            "--stage2_erode_classes", "",
            "--stage2_erode_px", "0",
            "--area_gate_lo", "0.85",
            "--area_gate_hi", "1.2",
            "--area_gate_hi_medium", "1.5",
            "--area_gate_hi_large", "1.35",
            "--metrics_out", "results/pipeline_slice_metrics_deploy.npz",
            "--eval_batch_size", str(args.batch_size or 64),
        ]
        if args.skip_plot:
            cmd_deploy += ["--skip_plot"]
        if args.modality is not None:
            cmd_deploy += ["--modality", str(args.modality)]
        run_command(
            cmd_deploy,
            "Stage 4: Deploy eval (PPO mask refiner)",
        )
    else:
        log.info("⏭️  Stage 4: 성능 검증 단계를 건너뜁니다.\n")

    log.info("🎉 Dynamic Routing 파이프라인의 모든 과정이 완료되었습니다!")

if __name__ == "__main__":
    main()
