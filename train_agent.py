"""
Step 3: PPO 에이전트 학습 스크립트
Stable-Baselines3의 PPO 알고리즘을 활용합니다.

사용 예시:
  # YAML 설정 파일로 실행
  python train_agent.py --config configs/ppo_brats.yaml

  # 인자 직접 지정
  python train_agent.py --use_real_data --total_timesteps 200000

  # 합성 데이터
  python train_agent.py --num_samples 300
"""

import os
import sys
import signal
import argparse
import logging
from typing import Optional

import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.envs.mask_refinement_env import MaskRefinementEnv


# ──────────────────────────────────────────────
# 진행 상황 콜백
# ──────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    """
    N 스텝마다 학습 진행 현황을 한 줄로 출력합니다.
    예) [Step  10000/200000  5%] reward=-0.021 | ep_len=18.3 | fps=248 | eta=12m30s
    """
    def __init__(self, total_timesteps: int, print_freq: int = 2048):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self._last_print = 0
        self._start_time = None

    def _on_training_start(self) -> None:
        import time
        self._start_time = time.time()
        print("")
        print("="*70)
        print(f"  PPO 학습 시작  |  총 스텐: {self.total_timesteps:,}")
        print("="*70)
        print(f"  {'Step':>18}  {'Progress':>8}  {'MeanReward':>12}  {'EpLen':>7}  {'FPS':>6}  {'ETA':>8}")
        print("-"*70)

    def _on_step(self) -> bool:
        import time
        if self.num_timesteps - self._last_print < self.print_freq:
            return True
        self._last_print = self.num_timesteps

        elapsed = time.time() - self._start_time
        progress = self.num_timesteps / self.total_timesteps
        remaining = (elapsed / progress - elapsed) if progress > 0 else 0
        eta_m, eta_s = divmod(int(remaining), 60)
        eta_str = f"{eta_m}m{eta_s:02d}s" if eta_m else f"{eta_s}s"

        # locals()['infos'] 또는 로그에서 환경 통계 추출
        ep_rew  = self.locals.get("mean_reward", float("nan"))
        ep_len  = float("nan")
        fps     = int(self.num_timesteps / elapsed) if elapsed > 0 else 0

        # rollout buffer’s infos
        infos = self.locals.get("infos", [])
        if infos:
            lens = [i.get("episode", {}).get("l", None) for i in infos if "episode" in i]
            rews = [i.get("episode", {}).get("r", None) for i in infos if "episode" in i]
            if lens:  ep_len = sum(lens) / len(lens)
            if rews:  ep_rew = sum(rews) / len(rews)

        print(
            f"  {self.num_timesteps:>8,}/{self.total_timesteps:,}"
            f"  {progress*100:>7.1f}%"
            f"  {ep_rew:>12.4f}"
            f"  {ep_len:>7.1f}"
            f"  {fps:>6}"
            f"  {eta_str:>8}"
        )
        return True

    def _on_training_end(self) -> None:
        import time
        elapsed = time.time() - self._start_time
        m, s = divmod(int(elapsed), 60)
        print("-"*70)
        print(f"  학습 완료!  소요 시간: {m}m{s:02d}s  |  총 스텐: {self.num_timesteps:,}")
        print("="*70)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 데이터 로드 헬퍼
# ──────────────────────────────────────────────

def load_synthetic_data(num_samples: int, image_size: int, seed: int = 42):
    raise ValueError("합성 데이터 생성기(synthetic_brats.py)가 삭제되어 더 이상 합성 데이터를 사용할 수 없습니다. --use_real_data 옵션을 사용해 주세요.")


def load_real_data(
    train_root: str,
    modality:   str,
    target_size: int,
    max_patients: Optional[int],
    noise_seed: int = 42,
):
    """실제 BraTS2020 데이터를 NumPy 배열로 반환."""
    from src.data.brats2020_dataset import BraTS2020Dataset
    log.info(f"실제 BraTS2020 데이터 로드 중: {train_root}")
    ds = BraTS2020Dataset(
        root_dir=train_root,
        modality=modality,
        target_size=target_size,
        max_patients=max_patients,
        simulate_rough=True,
        noise_seed=noise_seed,
    )
    return ds.get_numpy_arrays()   # (N,H,W), (N,H,W), (N,H,W)


# ──────────────────────────────────────────────
# VecEnv 빌더
# ──────────────────────────────────────────────

def make_env_fn(images, gt_masks, rough_masks, max_steps, target_dsc):
    """MaskRefinementEnv 팩토리 함수 반환."""
    def _init():
        return MaskRefinementEnv(
            images=images,
            gt_masks=gt_masks,
            rough_masks=rough_masks,
            max_steps=max_steps,
            target_dsc=target_dsc,
        )
    return _init


# ──────────────────────────────────────────────
# 메인 학습 함수
# ──────────────────────────────────────────────

def train_agent(
    # 데이터
    use_real_data:       bool  = True,
    train_root:          str   = r"src\data\archive (1)\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    modality:            str   = "t1ce",
    image_size:          int   = 128,
    max_train_patients:  Optional[int] = None,
    num_samples:         int   = 300,         # 합성 데이터용
    # RL 환경
    max_steps:           int   = 20,
    target_dsc:          float = 0.90,
    # PPO 하이퍼파라미터
    total_timesteps:     int   = 200_000,
    n_envs:              int   = 4,
    n_steps:             int   = 512,
    batch_size:          int   = 64,
    n_epochs:            int   = 4,
    gamma:               float = 0.99,
    gae_lambda:          float = 0.95,
    clip_range:          float = 0.2,
    ent_coef:            float = 0.01,
    learning_rate:       float = 3e-4,
    net_arch:            list  = None,
    # 저장
    save_path:           str   = "checkpoints/ppo_refiner",
    log_path:            str   = "logs/ppo",
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    if net_arch is None:
        net_arch = [256, 256]

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # ── 데이터 로드 ──────────────────────────────────────────
    if use_real_data:
        images, gt_masks, rough_masks = load_real_data(
            train_root=train_root,
            modality=modality,
            target_size=image_size,
            max_patients=max_train_patients,
        )
    else:
        images, gt_masks, rough_masks = load_synthetic_data(
            num_samples=num_samples,
            image_size=image_size,
        )

    N = len(images)
    log.info(f"RL 환경 데이터: {N}개 슬라이스 | 이미지 크기: {image_size}x{image_size}")

    # Train / Val 분할 (80 / 20)
    split = int(N * 0.8)
    tr_img,  tr_gt,  tr_rough  = images[:split],  gt_masks[:split],  rough_masks[:split]
    val_img, val_gt, val_rough = images[split:],  gt_masks[split:],  rough_masks[split:]

    # ── VecEnv 생성 ─────────────────────────────────────────
    train_env_fn = make_env_fn(tr_img,  tr_gt,  tr_rough,  max_steps, target_dsc)
    val_env_fn   = make_env_fn(val_img, val_gt, val_rough, max_steps, target_dsc)

    train_env = make_vec_env(train_env_fn, n_envs=n_envs)
    eval_env  = DummyVecEnv([val_env_fn])

    # ── PPO 에이전트 ─────────────────────────────────────────
    # TensorBoard 설치 여부 확인
    try:
        import tensorboard  # noqa: F401
        tb_log = log_path
        log.info(f"TensorBoard 로그 활성화: {log_path}")
    except ImportError:
        tb_log = None
        log.warning("TensorBoard 미설치 — 로그 비활성화 (pip install tensorboard 로 활성화 가능)")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        verbose=0,                          # 기본 SB3 테이블 출력 비활성화
        tensorboard_log=tb_log,
        policy_kwargs=dict(net_arch=net_arch),
    )

    # ── 콜백 ────────────────────────────────────────────────
    ckpt_dir = os.path.dirname(save_path) or "checkpoints"
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=ckpt_dir,
        log_path=log_path if tb_log else None,
        eval_freq=max(1000, total_timesteps // 20),
        n_eval_episodes=10,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(5000, total_timesteps // 10),
        save_path=ckpt_dir,
        name_prefix="ppo_refiner",
    )

    progress_cb = ProgressCallback(
        total_timesteps=total_timesteps,
        print_freq=max(2048, n_steps * n_envs),  # 1 iteration = n_steps * n_envs
    )

    log.info(f"PPO 학습 시작: total_timesteps={total_timesteps:,}  n_envs={n_envs}")
    log.info("  >> Ctrl+C 로 언제든 중단 가능 (현재까지 학습된 모델 자동 저장)")

    # ── Ctrl+C 핸들러: 모델 저장 후 종료 ────────────────────
    interrupted = False
    def _graceful_exit(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[중단 요청] 학습을 중지하고 모델을 저장합니다...")

    signal.signal(signal.SIGINT,  _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_cb, ckpt_cb, progress_cb],
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[중단 요청] KeyboardInterrupt 감지.")

    interrupt_path = save_path + "_interrupted"
    save_target = interrupt_path if interrupted else save_path
    model.save(save_target)
    log.info(f"모델 저장 완료: {save_target}")
    if interrupted:
        log.info(f"  (중단 시점까지 {model.num_timesteps:,} 스텝 학습됨)")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Step 3: Train PPO Refiner Agent")

    # YAML 설정 파일 (다른 인자보다 먼저 파싱)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 설정 파일 경로 (configs/ppo_brats.yaml)")

    # 데이터
    parser.add_argument("--use_real_data",  action="store_true", default=True)
    parser.add_argument("--train_root",     type=str,
                        default=r"src\data\archive (1)\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
    parser.add_argument("--modality",       type=str, default="t1ce")
    parser.add_argument("--image_size",     type=int, default=128)
    parser.add_argument("--max_train_patients", type=int, default=None)
    parser.add_argument("--num_samples",    type=int, default=300)
    # RL 환경
    parser.add_argument("--max_steps",      type=int,   default=20)
    parser.add_argument("--target_dsc",     type=float, default=0.90)
    # PPO
    parser.add_argument("--total_timesteps",type=int,   default=200_000)
    parser.add_argument("--n_envs",         type=int,   default=4)
    parser.add_argument("--n_steps",        type=int,   default=512)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--n_epochs",       type=int,   default=4)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--gae_lambda",     type=float, default=0.95)
    parser.add_argument("--clip_range",     type=float, default=0.2)
    parser.add_argument("--ent_coef",       type=float, default=0.01)
    parser.add_argument("--learning_rate",  type=float, default=3e-4)
    # 저장
    parser.add_argument("--save_path",  type=str, default="checkpoints/ppo_refiner")
    parser.add_argument("--log_path",   type=str, default="logs/ppo")

    args = parser.parse_args()

    # YAML 설정을 기본값으로 먼저 로드, CLI 인자로 덮어씀
    cfg = {}
    if args.config:
        cfg = _load_yaml(args.config)
        log.info(f"설정 파일 로드: {args.config}")

    # YAML 키를 train_agent 인자에 매핑 (snake_case 변환)
    yaml_key_map = {
        "use_real_data":       "use_real_data",
        "train_root":          "train_root",
        "modality":            "modality",
        "image_size":          "image_size",
        "max_train_patients":  "max_train_patients",
        "num_samples":         "num_samples",
        "max_steps":           "max_steps",
        "target_dsc":          "target_dsc",
        "total_timesteps":     "total_timesteps",
        "n_envs":              "n_envs",
        "n_steps":             "n_steps",
        "batch_size":          "batch_size",
        "n_epochs":            "n_epochs",
        "gamma":               "gamma",
        "gae_lambda":          "gae_lambda",
        "clip_range":          "clip_range",
        "ent_coef":            "ent_coef",
        "learning_rate":       "learning_rate",
        "net_arch":            "net_arch",
        "save_path":           "save_path",
        "log_path":            "log_path",
    }

    # 최종 파라미터: YAML 기본값 → CLI 인자로 override
    final_params = {}
    defaults = vars(parser.parse_args([]))  # 기본값만 추출
    cli_args  = vars(args)

    for fn_key, yaml_key in yaml_key_map.items():
        if fn_key == "use_real_data":
            # bool 플래그: YAML이 true이거나 CLI에서 --use_real_data 사용 시
            yaml_val = cfg.get(yaml_key, False)
            cli_val  = cli_args.get("use_real_data", False)
            final_params[fn_key] = yaml_val or cli_val
        else:
            yaml_val = cfg.get(yaml_key, None)
            cli_val  = cli_args.get(fn_key, None)
            default_val = defaults.get(fn_key, None)
            # CLI가 기본값과 다르면(사용자가 직접 지정) 우선, 아니면 YAML, 그 외 기본값
            if cli_val != default_val and cli_val is not None:
                final_params[fn_key] = cli_val
            elif yaml_val is not None:
                final_params[fn_key] = yaml_val
            else:
                final_params[fn_key] = default_val

    log.info("최종 파라미터:")
    for k, v in final_params.items():
        log.info(f"  {k}: {v}")

    train_agent(**final_params)


if __name__ == "__main__":
    main()
