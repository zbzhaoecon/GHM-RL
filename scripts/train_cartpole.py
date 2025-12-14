"""
Debug script: Train SAC on Pendulum to verify infrastructure.

Expected: Reward improves from ~-1500 to ~-200 within 20k timesteps.

Usage:
    python scripts/train_cartpole.py
    python scripts/train_cartpole.py --timesteps 50000 --output models/pendulum
"""

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


def main():
    parser = argparse.ArgumentParser(description="Debug: Train SAC on Pendulum")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--output", type=str, default="models/pendulum_debug")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment (Pendulum for continuous actions)
    env = make_vec_env("Pendulum-v1", n_envs=4, seed=args.seed)
    eval_env = make_vec_env("Pendulum-v1", n_envs=1, seed=args.seed + 100)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "logs"),
        eval_freq=2000,
        deterministic=True,
        render=False,
    )

    # Create SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    # Train
    print(f"Training SAC on Pendulum-v1 for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save final model
    model.save(output_dir / "final_model")
    print(f"Model saved to {output_dir / 'final_model'}")

    # Quick evaluation
    print("\nFinal evaluation:")
    obs = eval_env.reset()
    total_reward = 0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward[0]
        if done[0]:
            break
    print(f"Episode reward: {total_reward:.1f}")


if __name__ == "__main__":
    main()
