#!/usr/bin/env python3
"""
Train a policy (PPO by default) on the Rocket League dummy env and save model + optional ONNX.

Usage:
    cd python && pip install -e . && python scripts/train.py
    python scripts/train.py --algo SAC --steps 50_000 --save-dir runs/
    python scripts/train.py --load runs/ppo_50k.zip --export-onnx model.onnx

This script is the entry point for the ML training path. The env is a dummy by default;
replace with an RLBot-backed or simulated env for real training.
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from boostrap_bot.env import RocketLeagueEnv
from boostrap_bot.env.spaces import OBS_SIZE, ACTION_SIZE


def parse_args():
    p = argparse.ArgumentParser(description="Train RL policy for boostrap-bot")
    p.add_argument("--algo", choices=["PPO", "SAC", "DDPG"], default="PPO")
    p.add_argument("--steps", type=int, default=100_000, help="Training timesteps")
    p.add_argument("--save-dir", type=str, default="runs", help="Directory for checkpoints and final model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4, help="Number of parallel envs (PPO)")
    p.add_argument("--load", type=str, default="", help="Resume from this checkpoint")
    p.add_argument("--export-onnx", type=str, default="", help="Export loaded/saved model to this ONNX path after training")
    return p.parse_args()


def make_env(n_envs: int = 1, seed: int = 0):
    def _fn():
        return RocketLeagueEnv(max_episode_steps=5000, seed=seed)

    if n_envs <= 1:
        return DummyVecEnv([_fn])
    return make_vec_env(_fn, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(n_envs=args.n_envs if args.algo == "PPO" else 1, seed=args.seed)

    algo_kw = {
        "PPO": dict(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        ),
        "SAC": dict(
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.99,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
        ),
        "DDPG": dict(
            learning_rate=1e-3,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.99,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
        ),
    }

    if args.load:
        path = Path(args.load)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        loaders = {"PPO": PPO.load, "SAC": SAC.load, "DDPG": DDPG.load}
        model = loaders[args.algo](str(path), env=env)
        print(f"Resumed from {path}")
    else:
        constructors = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG}
        model = constructors[args.algo](
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            **algo_kw[args.algo],
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10000 // args.n_envs, 1),
        save_path=str(save_dir / "checkpoints"),
        name_prefix=args.algo.lower(),
    )
    eval_env = make_env(n_envs=1, seed=args.seed + 1000)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir),
        eval_freq=5000,
        deterministic=True,
        n_eval_episodes=5,
    )

    model.learn(total_timesteps=args.steps, callback=[checkpoint_cb, eval_cb])
    final_path = save_dir / f"{args.algo.lower()}_final"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}.zip")

    if args.export_onnx:
        onnx_path = Path(args.export_onnx)
        if not onnx_path.suffix:
            onnx_path = onnx_path.with_suffix(".onnx")
        from boostrap_bot.export_onnx import export_policy_to_onnx
        export_policy_to_onnx(str(final_path) + ".zip", onnx_path)
        print(f"Exported ONNX to {onnx_path}")


if __name__ == "__main__":
    main()
