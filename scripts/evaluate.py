"""
Evaluate trained SAC model on GHM equity environment.

Outputs:
1. Policy plot: a(c) vs c
2. Value function estimate: V(c) vs c
3. Estimated threshold c*

Usage:
    python scripts/evaluate.py --model models/ghm_equity/final_model
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from macro_rl.envs import GHMEquityEnv


def extract_policy(model, c_grid: np.ndarray) -> np.ndarray:
    """Extract learned policy a(c) at grid points."""
    actions = []
    for c in c_grid:
        obs = np.array([[c]], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0, 0])
    return np.array(actions)


def estimate_value_function(
    model, env, c_grid: np.ndarray, n_episodes: int = 50
) -> np.ndarray:
    """Estimate V(c) via Monte Carlo rollouts."""
    values = []

    for c in c_grid:
        episode_returns = []

        for _ in range(n_episodes):
            obs, _ = env.reset(options={"initial_state": np.array([c])})
            total_return = 0
            gamma = 0.99
            discount = 1.0

            for _ in range(env.max_steps):
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action[0])
                total_return += discount * reward
                discount *= gamma

                if terminated or truncated:
                    break

            episode_returns.append(total_return)

        values.append(np.mean(episode_returns))

    return np.array(values)


def find_threshold(c_grid: np.ndarray, actions: np.ndarray, threshold: float = 0.5) -> float:
    """Find c* where policy switches from low to high action."""
    above = actions > threshold
    if above.any() and (~above).any():
        idx = np.where(above)[0][0]
        return c_grid[idx]
    return None


def plot_results(c_grid, actions, values, c_star, output_path: Path):
    """Create and save evaluation plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Policy plot
    ax1 = axes[0]
    ax1.plot(c_grid, actions, 'b-', linewidth=2)
    if c_star:
        ax1.axvline(c_star, color='r', linestyle='--', label=f'c* ≈ {c_star:.3f}')
        ax1.legend()
    ax1.set_xlabel('Cash ratio c')
    ax1.set_ylabel('Dividend rate a(c)')
    ax1.set_title('Learned Policy')
    ax1.grid(True, alpha=0.3)

    # Value function plot
    ax2 = axes[1]
    ax2.plot(c_grid, values, 'g-', linewidth=2)
    if c_star:
        ax2.axvline(c_star, color='r', linestyle='--', label=f'c* ≈ {c_star:.3f}')
        ax2.legend()
    ax2.set_xlabel('Cash ratio c')
    ax2.set_ylabel('Value V(c)')
    ax2.set_title('Estimated Value Function')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "evaluation.png", dpi=150)
    plt.close()
    print(f"Plots saved to {output_path / 'evaluation.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GHM model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output) if args.output else model_path.parent

    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path)

    env = GHMEquityEnv()
    c_max = env.observation_space.high[0]
    c_grid = np.linspace(0.01, c_max - 0.01, args.n_points)

    print("Extracting policy...")
    actions = extract_policy(model, c_grid)

    print(f"Estimating value function...")
    values = estimate_value_function(model, env, c_grid, args.n_episodes)

    c_star = find_threshold(c_grid, actions)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"Value range: [{values.min():.3f}, {values.max():.3f}]")
    if c_star:
        print(f"Estimated c*: {c_star:.3f}")

    plot_results(c_grid, actions, values, c_star, output_path)
    np.savez(output_path / "results.npz", c=c_grid, actions=actions, values=values)


if __name__ == "__main__":
    main()
