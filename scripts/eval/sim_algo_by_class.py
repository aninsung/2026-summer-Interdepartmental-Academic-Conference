"""
클래스별 알고리즘 조합 짧은 시뮬레이션 (메인 파이프라인/체크포인트 미변경).

Small : PPO vs SAC (연속)
Medium: PPO only   (이산 — SAC 불가)
Large : PPO only   (이산 — SAC 불가)

결과: results/sim_algo_by_class.md
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.mask_refinement_env import MaskRefinementEnv, _dice as dice


def make_blob(h, w, cy, cx, ry, rx):
    yy, xx = np.ogrid[:h, :w]
    return (((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0).astype(np.float32)


def synth_batch(n: int, mode: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    h = w = 128
    images, gts, roughs = [], [], []
    for i in range(n):
        img = rng.normal(0.35, 0.12, (h, w)).astype(np.float32)
        img = np.clip(img, 0, 1)
        if mode == "small":
            cy, cx = rng.integers(40, 90), rng.integers(40, 90)
            gt = make_blob(h, w, cy, cx, rng.integers(4, 8), rng.integers(4, 8))
        elif mode == "medium":
            cy, cx = rng.integers(35, 95), rng.integers(35, 95)
            gt = make_blob(h, w, cy, cx, rng.integers(12, 18), rng.integers(12, 18))
        else:
            cy, cx = rng.integers(30, 100), rng.integers(30, 100)
            gt = make_blob(h, w, cy, cx, rng.integers(22, 32), rng.integers(22, 32))
        # rough = noisy GT
        r = gt.copy()
        if rng.random() < 0.5:
            r = binary_dilation(r, iterations=rng.integers(1, 3)).astype(np.float32)
        else:
            r = binary_erosion(r, iterations=1).astype(np.float32)
        if r.sum() < 5:
            r = gt.copy()
        img = np.clip(img + 0.25 * gaussian_filter(gt, 1.5), 0, 1).astype(np.float32)
        images.append(img)
        gts.append(gt)
        roughs.append(r)
    return (
        np.stack(images),
        np.stack(gts),
        np.stack(roughs),
    )


def make_env(mode: str, images, gts, roughs):
    def _f():
        return MaskRefinementEnv(
            images=images,
            gt_masks=gts,
            rough_masks=roughs,
            max_steps=15,
            target_dsc=1.0,
            step_penalty=0.001,
            refinement_mode=mode,
            enable_stop=False,
        )
    return _f


def eval_policy(model, images, gts, roughs, mode: str, n_eval: int = 40, max_steps: int = 15):
    env = MaskRefinementEnv(
        images=images,
        gt_masks=gts,
        rough_masks=roughs,
        max_steps=max_steps,
        target_dsc=1.0,
        step_penalty=0.001,
        refinement_mode=mode,
        enable_stop=False,
    )
    init_dsc, final_dsc, ep_rews = [], [], []
    for i in range(min(n_eval, len(images))):
        obs, _ = env.reset(seed=i)
        init = float(dice(env.rough_masks[env._idx], env.gt_masks[env._idx]))
        total_r = 0.0
        fin = init
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += float(r)
            fin = float(info.get("dsc", dice(env._current_mask, env.gt_masks[env._idx])))
            if term or trunc:
                break
        init_dsc.append(init)
        final_dsc.append(fin)
        ep_rews.append(total_r)
    return {
        "init_dsc": float(np.mean(init_dsc)),
        "final_dsc": float(np.mean(final_dsc)),
        "delta": float(np.mean(final_dsc) - np.mean(init_dsc)),
        "mean_reward": float(np.mean(ep_rews)),
        "n": len(init_dsc),
    }


def train_and_eval(mode: str, algo: str, timesteps: int, seed: int = 0):
    print(f"\n=== {mode.upper()} | {algo} | steps={timesteps} ===", flush=True)
    train_img, train_gt, train_r = synth_batch(48, mode, seed=seed)
    eval_img, eval_gt, eval_r = synth_batch(24, mode, seed=seed + 99)

    venv = DummyVecEnv([make_env(mode, train_img, train_gt, train_r)])
    policy = "CnnPolicy"
    policy_kwargs = dict(normalize_images=False, net_arch=[64, 64])
    device = "cpu"
    t0 = time.time()

    if algo == "PPO":
        model = PPO(
            policy,
            venv,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=3,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=device,
        )
    elif algo == "SAC":
        if mode != "small":
            raise ValueError("SAC only for continuous (small)")
        model = SAC(
            policy,
            venv,
            learning_rate=3e-4,
            buffer_size=10_000,
            batch_size=64,
            learning_starts=200,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=device,
        )
    elif algo == "A2C":
        model = A2C(
            policy,
            venv,
            learning_rate=3e-4,
            n_steps=16,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=device,
        )
    else:
        raise ValueError(algo)

    model.learn(total_timesteps=timesteps, progress_bar=False)
    train_s = time.time() - t0
    metrics = eval_policy(model, eval_img, eval_gt, eval_r, mode, n_eval=24)
    metrics["train_sec"] = train_s
    metrics["algo"] = algo
    metrics["mode"] = mode
    metrics["timesteps"] = timesteps
    print(
        f"  init={metrics['init_dsc']:.4f}  final={metrics['final_dsc']:.4f}  "
        f"Δ={metrics['delta']:+.4f}  rew={metrics['mean_reward']:.3f}  "
        f"time={train_s:.1f}s",
        flush=True,
    )
    venv.close()
    del model
    return metrics


def main():
    plans = [
        ("small", "PPO", 3_000),
        ("small", "SAC", 3_000),
        ("medium", "PPO", 3_000),
        ("medium", "A2C", 3_000),
        ("large", "PPO", 3_000),
    ]
    # SAC on medium should fail — record as N/A
    rows = []
    for mode, algo, steps in plans:
        try:
            rows.append(train_and_eval(mode, algo, steps, seed=42))
        except Exception as e:
            print(f"  FAIL {mode}/{algo}: {e}")
            rows.append(
                dict(
                    mode=mode,
                    algo=algo,
                    timesteps=steps,
                    init_dsc=float("nan"),
                    final_dsc=float("nan"),
                    delta=float("nan"),
                    mean_reward=float("nan"),
                    train_sec=0.0,
                    n=0,
                    note=str(e),
                )
            )

    # document SAC incompatibility for medium/large
    for mode in ("medium", "large"):
        rows.append(
            dict(
                mode=mode,
                algo="SAC",
                timesteps=0,
                init_dsc=float("nan"),
                final_dsc=float("nan"),
                delta=float("nan"),
                mean_reward=float("nan"),
                train_sec=0.0,
                n=0,
                note="이산 MultiDiscrete → SAC 불가 (시뮬 스킵)",
            )
        )

    out = ROOT / "results" / "sim_algo_by_class.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 클래스별 알고리즘 시뮬레이션 (합성 데이터, 단기 학습)",
        "",
        "- 메인 파이프라인/체크포인트 **미변경**",
        "- GT 단조 게이트 **없음** (raw RL DSC)",
        "- 합성 blob + 노이즈 rough, 학습 8k step / 평가 40 에피소드",
        "",
        "| Class | Algo | Init DSC | Final DSC | ΔDSC | MeanRew | Train(s) | Note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        note = r.get("note", "")
        def fmt(x):
            return "—" if x != x else f"{x:.4f}"  # NaN check

        lines.append(
            f"| {r['mode']} | {r['algo']} | {fmt(r['init_dsc'])} | {fmt(r['final_dsc'])} | "
            f"{fmt(r['delta'])} | {fmt(r['mean_reward'])} | {r['train_sec']:.1f} | {note} |"
        )
    lines += [
        "",
        "## 해석 (이 시뮬 한정)",
        "- Small에서 SAC vs PPO는 짧은 스텝이라 승패가 뒤집힐 수 있음 → 경향만 참고.",
        "- Medium/Large는 SAC 사용 불가; A2C는 참고용이며 보통 PPO가 더 안정적.",
        "- 실제 BraTS·300k step 결과와 수치를 직접 비교하면 안 됨.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
