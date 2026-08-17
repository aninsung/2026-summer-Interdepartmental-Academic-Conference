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
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
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


# ──────────────────────────────────────────────
# Stop Motion 콜백 모음
# ──────────────────────────────────────────────

class DSCTargetStopCallback(BaseCallback):
    """
    [Stop #1] DSC 목표 달성 Stop
    eval 환경에서 측정한 평균 DSC 가 dsc_target 이상이면 학습을 즉시 중단합니다.
    EvalCallback 이 기록하는 'eval/mean_reward' 대신, 직접 infos 로부터 dsc 를 추적합니다.

    Parameters
    ----------
    dsc_target : float
        이 값 이상의 평균 DSC 가 달성되면 중단 (기본 0.92).
    check_freq  : int
        몇 스텝마다 DSC 를 확인할지 (기본 2048).
    min_episodes : int
        최소 이 에피소드 수의 데이터가 쌓여야 판단합니다 (기본 10).
    """

    def __init__(self, dsc_target: float = 0.92, check_freq: int = 2048, min_episodes: int = 10):
        super().__init__(verbose=0)
        self.dsc_target   = dsc_target
        self.check_freq   = check_freq
        self.min_episodes = min_episodes
        self._dsc_buffer: deque = deque(maxlen=100)
        self._last_check  = 0

    def _on_step(self) -> bool:
        # infos 에서 에피소드 종료 시점의 dsc 수집
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode", {})
            dsc = info.get("dsc", None)           # MaskRefinementEnv info 키
            if dsc is not None:
                self._dsc_buffer.append(dsc)
            elif "dsc" in ep:                      # Monitor wrapper 경유 시
                self._dsc_buffer.append(ep["dsc"])

        if self.num_timesteps - self._last_check < self.check_freq:
            return True
        self._last_check = self.num_timesteps

        if len(self._dsc_buffer) < self.min_episodes:
            return True

        mean_dsc = float(np.mean(self._dsc_buffer))
        if mean_dsc >= self.dsc_target:
            print()
            print("★" * 70)
            print(f"  [Stop #1 · DSC 목표 달성]  평균 DSC={mean_dsc:.4f} ≥ {self.dsc_target}")
            print(f"  학습을 {self.num_timesteps:,} 스텝에서 중단합니다.")
            print("★" * 70)
            return False   # False 반환 → SB3 학습 루프 종료
        return True


class RewardPlateauStopCallback(BaseCallback):
    """
    [Stop #2] Reward Plateau Stop
    최근 window_size 개 에피소드의 평균 리워드가
    patience 회 연속 개선되지 않으면 학습을 중단합니다.

    Parameters
    ----------
    window_size : int
        이동 평균에 사용할 에피소드 수 (기본 50).
    min_delta   : float
        개선으로 인정할 최소 리워드 증분 (기본 0.005).
    patience    : int
        허용할 비개선 횟수 (기본 5).  check_freq 마다 1 카운트.
    check_freq  : int
        몇 스텝마다 plateau 를 확인할지 (기본 4096).
    """

    def __init__(
        self,
        window_size: int = 50,
        min_delta:   float = 0.005,
        patience:    int   = 5,
        check_freq:  int   = 4096,
    ):
        super().__init__(verbose=0)
        self.window_size  = window_size
        self.min_delta    = min_delta
        self.patience     = patience
        self.check_freq   = check_freq
        self._rew_buffer: deque = deque(maxlen=window_size)
        self._best_mean   = -np.inf
        self._no_improve  = 0
        self._last_check  = 0

    def _on_step(self) -> bool:
        # 에피소드 종료 시 리워드 수집
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode", {})
            if "r" in ep:
                self._rew_buffer.append(float(ep["r"]))

        if self.num_timesteps - self._last_check < self.check_freq:
            return True
        self._last_check = self.num_timesteps

        if len(self._rew_buffer) < self.window_size:
            return True

        mean_rew = float(np.mean(self._rew_buffer))
        if mean_rew > self._best_mean + self.min_delta:
            self._best_mean  = mean_rew
            self._no_improve = 0
        else:
            self._no_improve += 1
            print(
                f"  [Stop #2 · Plateau] 개선 없음 {self._no_improve}/{self.patience}"
                f"  (mean_rew={mean_rew:.4f}, best={self._best_mean:.4f})"
            )
            if self._no_improve >= self.patience:
                print()
                print("▲" * 70)
                print(f"  [Stop #2 · Reward Plateau]  {self.patience} 회 연속 개선 없음")
                print(f"  학습을 {self.num_timesteps:,} 스텝에서 중단합니다.")
                print("▲" * 70)
                return False
        return True


class MilestoneSnapshotCallback(BaseCallback):
    """
    [Stop #3] Milestone Snapshot (멈추지 않고 스냅샷 저장)
    milestones 에 지정한 타임스텝 비율(0~1)에 도달할 때마다
    모델을 자동으로 저장합니다.
    예) milestones=[0.25, 0.5, 0.75] → 25 / 50 / 75 % 지점에서 저장.

    Parameters
    ----------
    total_timesteps : int
        전체 학습 스텝.
    save_dir        : str
        스냅샷을 저장할 디렉터리.
    milestones      : list[float]
        저장 비율 목록 (기본 [0.25, 0.50, 0.75]).
    """

    def __init__(
        self,
        total_timesteps: int,
        save_dir:        str,
        milestones:      list = None,
    ):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.save_dir        = save_dir
        self.milestones      = milestones if milestones is not None else [0.25, 0.50, 0.75]
        # 저장 완료된 마일스톤 추적 (인덱스)
        self._triggered: set = set()

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        for i, ms in enumerate(self.milestones):
            if i in self._triggered:
                continue
            if progress >= ms:
                self._triggered.add(i)
                os.makedirs(self.save_dir, exist_ok=True)
                snap_name = f"snapshot_{int(ms*100):03d}pct_{self.num_timesteps}"
                snap_path = os.path.join(self.save_dir, snap_name)
                self.model.save(snap_path)
                print()
                print("◆" * 70)
                print(f"  [Milestone {int(ms*100)}%]  스냅샷 저장 → {snap_path}.zip")
                print("◆" * 70)
        return True


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
    unet_path: Optional[str] = None,
    model_type: str = "unet",
    refinement_mode: str = "small",
):
    """실제 BraTS2021 데이터를 NumPy 배열로 반환."""
    from src.data.brats2020_dataset import BraTS2020Dataset
    import torch
    
    log.info(f"실제 BraTS2021 데이터 로드 중: {train_root}")
    ds = BraTS2020Dataset(
        root_dir=train_root,
        modality=modality,
        target_size=target_size,
        max_patients=max_patients,
        simulate_rough=False,  # 실제/합성 믹스업을 위해 일단 False로 로드
        noise_seed=noise_seed,
    )
    imgs, gts, _ = ds.get_numpy_arrays()

    rng = np.random.default_rng(noise_seed)

    # 1. 합성 노이즈 마스크 및 시뮬레이션된 확률 맵 생성
    from src.data.brats2020_dataset import make_noisy_mask
    from scipy.ndimage import gaussian_filter
    morph_px = 5  # train/eval 동일: max_morph_px=5 (분포 일치)
    log.info(f"학습 데이터에 대한 합성 노이즈 마스크 및 시뮬레이션 확률 맵 생성 중... (max_morph_px={morph_px})")
    synthetic_roughs = np.stack(
        [make_noisy_mask(gt, rng, max_morph_px=morph_px) for gt in gts], axis=0
    )
    synthetic_probs = np.zeros_like(synthetic_roughs)
    for i in range(len(synthetic_roughs)):
        synthetic_probs[i] = gaussian_filter(synthetic_roughs[i].astype(float), sigma=2.0)

    # 2. 실제 모델 예측 마스크 및 Sigmoid 확률 맵 생성 (AdaptivePipeline 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.models.dynamic_router import AdaptivePipeline
    log.info(f"AdaptivePipeline 3-Stage 라우터 로드 중... (Device: {device})")
    in_ch = imgs.shape[1] if imgs.ndim == 4 else 1
    pipeline = AdaptivePipeline(device, in_channels=in_ch)
    
    batch_size = 64
    num_slices = len(imgs)
    preds = []
    probs = []
    class_preds_all = []
    log.info("학습 데이터에 대한 AdaptivePipeline 초안 마스크 및 확률 맵 생성 중...")
    with torch.no_grad():
        for start_idx in range(0, num_slices, batch_size):
            end_idx = min(start_idx + batch_size, num_slices)
            batch_imgs = imgs[start_idx:end_idx]
            if batch_imgs.ndim == 3:
                batch_t = torch.from_numpy(batch_imgs).unsqueeze(1).to(device)
            else:
                batch_t = torch.from_numpy(batch_imgs).to(device)
            
            rough_masks_t, class_preds = pipeline(batch_t)
            batch_preds_bin = (rough_masks_t > 0.5).float().squeeze(1).cpu().numpy()
            batch_preds_prob = rough_masks_t.squeeze(1).cpu().numpy()
            
            preds.append(batch_preds_bin)
            probs.append(batch_preds_prob)
            class_preds_all.extend(class_preds.cpu().numpy().tolist())
            
    actual_roughs = np.concatenate(preds, axis=0)
    actual_probs = np.concatenate(probs, axis=0)
    if actual_roughs.ndim == 4 and actual_roughs.shape[1] == 1:
        actual_roughs = np.squeeze(actual_roughs, axis=1)
    if actual_probs.ndim == 4 and actual_probs.shape[1] == 1:
        actual_probs = np.squeeze(actual_probs, axis=1)
    class_preds_all = np.array(class_preds_all)
    log.info(f"AdaptivePipeline 초안 마스크 생성 완료 (개수: {len(actual_roughs)})")

    # 3. Shape Class (refinement_mode) 에 따른 필터링
    target_class = {"small": 0, "medium": 1, "large": 2}[refinement_mode.lower()]
    mask_indices = (class_preds_all == target_class)
    
    imgs = imgs[mask_indices]
    gts = gts[mask_indices]
    synthetic_roughs = synthetic_roughs[mask_indices]
    synthetic_probs = synthetic_probs[mask_indices]
    actual_roughs = actual_roughs[mask_indices]
    actual_probs = actual_probs[mask_indices]
    
    if len(imgs) == 0:
        raise ValueError(f"해당 클래스({refinement_mode})로 분류된 데이터가 하나도 없습니다!")
        
    log.info(f"'{refinement_mode}' (Class {target_class}) 필터링 완료: {len(imgs)}개 슬라이스 사용")

    # 데이터 믹스업: 실제 예측값 50% + 합성 노이즈 50%
    # → 실제 예측 1회 + 합성 1회 = 1:1 비율 (일반화 향상을 위해 합성 비중 증가)
    imgs = np.concatenate([imgs, imgs], axis=0)
    gts = np.concatenate([gts, gts], axis=0)
    roughs = np.concatenate([actual_roughs, synthetic_roughs], axis=0)
    probs_all = np.concatenate([actual_probs, synthetic_probs], axis=0)
    
    # 순열(permutation) 믹스
    perm = rng.permutation(len(imgs))
    imgs = imgs[perm]
    gts = gts[perm]
    roughs = roughs[perm]
    probs_all = probs_all[perm]
    log.info(f"실제 예측 50% + 합성 노이즈 50% 믹스업 완료 (최종 데이터 슬라이스 수: {len(imgs)})")
    
    return imgs, gts, roughs, probs_all


# ──────────────────────────────────────────────
# VecEnv 빌더
# ──────────────────────────────────────────────

def make_env_fn(images, gt_masks, rough_masks, uncertainty_maps, max_steps, target_dsc, step_penalty=0.01, model_type="unet"):
    """MaskRefinementEnv 팩토리 함수 반환."""
    def _init():
        return MaskRefinementEnv(
            images=images,
            gt_masks=gt_masks,
            rough_masks=rough_masks,
            uncertainty_maps=uncertainty_maps,
            max_steps=max_steps,
            target_dsc=target_dsc,
            step_penalty=0.01,
            model_type="adaptive_pipeline",
        )
    return _init


# ──────────────────────────────────────────────
# 메인 학습 함수
# ──────────────────────────────────────────────

def train_agent(
    # 데이터
    use_real_data:       bool  = True,
    train_root:          str   = "src/data/archive",
    modality:            str   = "t1ce",
    image_size:          int   = 128,
    max_train_patients:  Optional[int] = None,
    num_samples:         int   = 300,         # 합성 데이터용
    unet_path:           Optional[str] = "checkpoints/segresnet_best.pt",
    model_type:          str   = "segresnet",
    # RL 환경
    max_steps:           int   = 30,          # 20 → 30
    target_dsc:          float = 0.88,        # 0.95 → 0.88: SegResNet rough DSC≈0.85 기준 현실적 목표
    step_penalty:        float = 0.005,       # 0.01 → 0.005: 탐색 억제 완화
    # PPO 하이퍼파라미터
    total_timesteps:     int   = 300_000,      # 200K → 300K (5-class 행동 공간 확장 대응)
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
    # ── Stop Motion 설정 ──────────────────────────────────────
    # Stop #1: DSC 목표 달성 시 조기 종료
    stop_dsc_target:     float = 0.92,        # 이 DSC 달성 시 즉시 중단 (0이면 비활성)
    stop_dsc_check_freq: int   = 2048,        # DSC 확인 주기 (스텝)
    # Stop #2: Reward Plateau 감지 시 조기 종료
    stop_plateau:        bool  = True,        # Plateau Stop 활성화 여부
    plateau_window:      int   = 50,          # 이동 평균 에피소드 수
    plateau_min_delta:   float = 0.005,       # 최소 개선 임계값
    plateau_patience:    int   = 5,           # 비개선 허용 횟수
    plateau_check_freq:  int   = 4096,        # Plateau 확인 주기 (스텝)
    # Stop #3: Milestone Snapshot (중단 없이 자동 저장)
    milestone_ratios:    list  = None,        # 저장 비율 목록 (기본 [0.25, 0.5, 0.75])
    # 모드
    refinement_mode:     str   = "small",     # "small", "medium", "large"
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    if net_arch is None:
        net_arch = [256, 256]

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # ── 데이터 로드 ──────────────────────────────────────────
    if use_real_data:
        images, gt_masks, rough_masks, uncertainty_maps = load_real_data(
            train_root=train_root,
            modality=modality,
            target_size=image_size,
            max_patients=max_train_patients,
            unet_path=unet_path,
            model_type=model_type,
            refinement_mode=refinement_mode,
        )
    else:
        images, gt_masks, rough_masks, uncertainty_maps = load_synthetic_data(
            num_samples=num_samples,
            image_size=image_size,
        )

    N = len(images)
    log.info(f"RL 환경 데이터: {N}개 슬라이스 | 이미지 크기: {image_size}x{image_size}")

    # Train / Val 분할 (80 / 20)
    split = int(N * 0.8)
    tr_img,  tr_gt,  tr_rough, tr_uncert  = images[:split],  gt_masks[:split],  rough_masks[:split], uncertainty_maps[:split]
    val_img, val_gt, val_rough, val_uncert = images[split:],  gt_masks[split:],  rough_masks[split:], uncertainty_maps[split:]

    # ── VecEnv 생성 ─────────────────────────────────────────
    # (더 이상 사용되지 않는 make_env_fn은 무시하고 아래의 make_monitored_env_fn을 사용합니다)

    # Monitor wrapper 적용 팩토리
    def make_monitored_env_fn(images, gt_masks, rough_masks, uncertainty_maps, max_steps, target_dsc, step_penalty, model_type, refinement_mode):
        def _init():
            env = MaskRefinementEnv(
                images=images,
                gt_masks=gt_masks,
                rough_masks=rough_masks,
                uncertainty_maps=uncertainty_maps,
                max_steps=max_steps,
                target_dsc=target_dsc,
                step_penalty=step_penalty,
                model_type=model_type,
                refinement_mode=refinement_mode,
            )
            return Monitor(env)
        return _init

    train_env = make_vec_env(make_monitored_env_fn(tr_img, tr_gt, tr_rough, tr_uncert, max_steps, target_dsc, step_penalty, model_type, refinement_mode), n_envs=n_envs)
    eval_env  = DummyVecEnv([make_monitored_env_fn(val_img, val_gt, val_rough, val_uncert, max_steps, target_dsc, step_penalty, model_type, refinement_mode)])

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
        policy="CnnPolicy",      # GPU CNN으로 이미지 (2,H,W) 처리
        env=train_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        verbose=0,
        device="cuda",           # GPU 명시
        tensorboard_log=tb_log,
        policy_kwargs=dict(
            net_arch=net_arch,
            # 관측값이 float32 [0,1] 이므로 SB3 내부 정규화 비활성화
            normalize_images=False,
        ),
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

    # ── Stop Motion 콜백 ────────────────────────────────────
    stop_callbacks = []

    # Stop #1: DSC 목표 달성 Stop
    if stop_dsc_target > 0:
        dsc_stop_cb = DSCTargetStopCallback(
            dsc_target=stop_dsc_target,
            check_freq=stop_dsc_check_freq,
        )
        stop_callbacks.append(dsc_stop_cb)
        log.info(f"  [Stop #1] DSC 목표 달성 Stop 활성화: dsc_target={stop_dsc_target}")
    else:
        log.info("  [Stop #1] DSC 목표 달성 Stop 비활성화 (stop_dsc_target=0)")

    # Stop #2: Reward Plateau Stop
    if stop_plateau:
        plateau_cb = RewardPlateauStopCallback(
            window_size=plateau_window,
            min_delta=plateau_min_delta,
            patience=plateau_patience,
            check_freq=plateau_check_freq,
        )
        stop_callbacks.append(plateau_cb)
        log.info(
            f"  [Stop #2] Plateau Stop 활성화: "
            f"window={plateau_window}, patience={plateau_patience}, "
            f"min_delta={plateau_min_delta}"
        )
    else:
        log.info("  [Stop #2] Plateau Stop 비활성화")

    # Stop #3: Milestone Snapshot
    snap_dir = os.path.join(ckpt_dir, "snapshots")
    milestone_cb = MilestoneSnapshotCallback(
        total_timesteps=total_timesteps,
        save_dir=snap_dir,
        milestones=milestone_ratios,
    )
    stop_callbacks.append(milestone_cb)
    ratios = milestone_ratios if milestone_ratios is not None else [0.25, 0.50, 0.75]
    log.info(f"  [Stop #3] Milestone Snapshot 활성화: {[f'{int(r*100)}%' for r in ratios]}")

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
            callback=[eval_cb, ckpt_cb, progress_cb] + stop_callbacks,
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
                        default="src/data/archive")
    parser.add_argument("--modality",       type=str, default="t1ce+flair")
    parser.add_argument("--image_size",     type=int, default=128)
    parser.add_argument("--max_train_patients", type=int, default=None)
    parser.add_argument("--num_samples",    type=int, default=300)
    parser.add_argument("--unet_path",      type=str, default="checkpoints/segresnet_best.pt",
                        help="가중치 파일 경로 (checkpoints/unet_best.pt 또는 checkpoints/segresnet_best.pt)")
    parser.add_argument("--model_type",     type=str, default="segresnet", choices=["unet", "segresnet", "unetplusplus", "unet++", "unet3plus", "unet3+", "attention_unet", "attunet", "caranet"],
                        help="세그멘테이션 모델 종류 (기본값: segresnet)")
    # RL 환경
    parser.add_argument("--max_steps",      type=int,   default=30)
    parser.add_argument("--target_dsc",     type=float, default=0.95)
    parser.add_argument("--step_penalty",   type=float, default=0.01,
                        help="유지(Keep) 이외의 액션을 취할 때 보상에서 차감하는 미세 패널티")
    # PPO
    parser.add_argument("--total_timesteps",type=int,   default=300_000)
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
    # ── Stop Motion ───────────────────────────────────────────────
    # Stop #1: DSC 목표 달성 시 조기 종료
    parser.add_argument("--stop_dsc_target",     type=float, default=0.92,
                        help="평균 DSC 이 이 값 이상이면 학습 없다. 0으로 설정 시 비활성")
    parser.add_argument("--stop_dsc_check_freq", type=int,   default=2048,
                        help="DSC 확인 주기 (스텝)")
    # Stop #2: Reward Plateau 감지 시 조기 종료
    parser.add_argument("--no_plateau_stop",     action="store_true", default=False,
                        help="Plateau Stop 비활성화 플래그")
    parser.add_argument("--plateau_window",      type=int,   default=50)
    parser.add_argument("--plateau_min_delta",   type=float, default=0.005)
    parser.add_argument("--plateau_patience",    type=int,   default=5)
    parser.add_argument("--plateau_check_freq",  type=int,   default=4096)
    # Stop #3: Milestone Snapshot
    parser.add_argument("--milestone_ratios",    type=float, nargs="+",
                        default=None,
                        help="마일스톤 저장 비율 (0~1). 예: 0.25 0.5 0.75")
    # 모드
    parser.add_argument("--refinement_mode",     type=str, default="small", choices=["small", "medium", "large"],
                        help="학습할 PPO 에이전트의 타겟 Shape Class (small, medium, large)")

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
        "unet_path":           "unet_path",
        "model_type":          "model_type",
        "max_steps":           "max_steps",
        "target_dsc":          "target_dsc",
        "step_penalty":        "step_penalty",
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
        # Stop Motion
        "stop_dsc_target":     "stop_dsc_target",
        "stop_dsc_check_freq": "stop_dsc_check_freq",
        "stop_plateau":        "stop_plateau",
        "plateau_window":      "plateau_window",
        "plateau_min_delta":   "plateau_min_delta",
        "plateau_patience":    "plateau_patience",
        "plateau_check_freq":  "plateau_check_freq",
        "milestone_ratios":    "milestone_ratios",
        "refinement_mode":     "refinement_mode",
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
            if fn_key == "unet_path":
                yaml_val = cfg.get("unet_path", cfg.get("unet_checkpoint", None))
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

    # unet3+ 와 unet3plus 표준화 및 unet++ 지원
    if final_params.get("model_type") == "unet3+":
        final_params["model_type"] = "unet3plus"
    elif final_params.get("model_type") == "unet++":
        final_params["model_type"] = "unetplusplus"
    elif final_params.get("model_type") == "attunet":
        final_params["model_type"] = "attention_unet"

    # model_type에 따라 기본 unet_path 및 save_path 자동 분기 매핑
    m_type = final_params.get("model_type", "segresnet")
    
    # 1. unet_path 자동 설정
    if final_params.get("unet_path") in [None, "checkpoints/segresnet_best.pt", "checkpoints/unet_best.pt", "checkpoints/unetplusplus_best.pt", "checkpoints/unet3plus_best.pt", "checkpoints/attention_unet_best.pt", "checkpoints/caranet_best.pt"]:
        if m_type == "segresnet":
            final_params["unet_path"] = "checkpoints/segresnet_best.pt"
        elif m_type == "unetplusplus":
            final_params["unet_path"] = "checkpoints/unetplusplus_best.pt"
        elif m_type == "unet3plus":
            final_params["unet_path"] = "checkpoints/unet3plus_best.pt"
        elif m_type == "attention_unet":
            final_params["unet_path"] = "checkpoints/attention_unet_best.pt"
        elif m_type == "caranet":
            final_params["unet_path"] = "checkpoints/caranet_best.pt"
        else:
            final_params["unet_path"] = "checkpoints/unet_best.pt"

    # 2. save_path 자동 설정 (refinement_mode 또는 m_type에 맞춰 분기)
    ref_mode = final_params.get("refinement_mode", "small").lower()
    if cli_args.get("save_path") is None:
        if ref_mode == "small":
            final_params["save_path"] = "checkpoints/ppo_refiner_attention_unet"
        elif ref_mode == "medium":
            final_params["save_path"] = "checkpoints/ppo_refiner_unetplusplus"
        elif ref_mode == "large":
            final_params["save_path"] = "checkpoints/ppo_refiner_segresnet"
        elif final_params.get("save_path") == "checkpoints/ppo_refiner":
            final_params["save_path"] = f"checkpoints/ppo_refiner_{m_type}"

    log.info("최종 파라미터:")
    for k, v in final_params.items():
        log.info(f"  {k}: {v}")

    train_agent(**final_params)


if __name__ == "__main__":
    main()
