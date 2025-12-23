"""
Train SAC on GHM 1D equity model with impulse controls.

This script trains a Soft Actor-Critic (SAC) agent on the GHM equity
management environment. The agent learns to approximate the optimal
barrier policy for dividend payments and equity issuance.

Action space: 2D continuous
  - action[0]: Dividend amount to pay out (≥ 0)
  - action[1]: Gross equity amount to raise (≥ 0)

The optimal policy should learn a barrier structure:
  - High cash (c > c*): Pay dividends
  - Low cash (c ≈ 0): Issue equity or liquidate
  - Middle range: Do nothing (both actions ≈ 0)

Usage:
    python scripts/train_ghm.py
    python scripts/train_ghm.py --timesteps 1000000 --output models/ghm_equity
    python scripts/train_ghm.py --timesteps 500000 --n-envs 8 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from macro_rl.envs import GHMEquityEnv


def make_env(seed: int = 0):
    """Create a single monitored environment."""
    def _init():
        env = GHMEquityEnv(
            dt=0.01,
            max_steps=1000,
            dividend_max=2.0,
            equity_max=2.0,
        )
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train SAC on GHM equity model")
    parser.add_argument("--timesteps", type=int, default=1000000)  # Increased default
    parser.add_argument("--output", type=str, default="models/ghm_equity")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=10000)  # Adjusted for longer training
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create vectorized environments
    print(f"Creating {args.n_envs} parallel environments...")
    env = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(args.seed + 100)])

    # Compute correct discount factor from environment
    # For continuous-time discounting: γ = exp(-ρ * dt) where ρ = r - μ
    temp_env = GHMEquityEnv(dt=0.01, max_steps=1000, dividend_max=2.0, equity_max=2.0)
    gamma = temp_env.get_expected_discount_factor()
    print(f"\nUsing discount factor γ = {gamma:.6f} (from continuous-time rate ρ = r - μ)")
    print(f"Action space: {temp_env.action_space} (2D: [dividend_amount, equity_gross_amount])")
    print(f"Liquidation value: {temp_env._dynamics.liquidation_value():.4f}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "logs"),
        eval_freq=args.eval_freq // args.n_envs,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // args.n_envs,  # Save every 50k steps
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_ghm",
    )

    # SAC hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200000,  # Increased for longer training
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=gamma,  # FIX: Use correct discount factor from environment
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    # Train
    print(f"\nTraining SAC on GHM equity for {args.timesteps} timesteps...")
    print(f"Output directory: {output_dir}")
    print(f"Monitor with: tensorboard --logdir {output_dir / 'tensorboard'}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(output_dir / "final_model")
    print(f"\nTraining complete. Model saved to {output_dir / 'final_model'}")


if __name__ == "__main__":
    main()
