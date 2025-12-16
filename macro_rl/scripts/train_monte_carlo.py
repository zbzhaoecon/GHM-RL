"""
Training script for Monte Carlo Policy Gradient solver.

Example usage:
    python macro_rl/scripts/train_monte_carlo.py --n_iterations 10000
"""

import torch
import argparse

# TODO: Implement Monte Carlo training pipeline


def main():
    parser = argparse.ArgumentParser(description="Train GHM policy with Monte Carlo PG")
    parser.add_argument("--n_iterations", type=int, default=10000)
    parser.add_argument("--n_trajectories", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print(f"Training GHM with Monte Carlo PG")
    print("TODO: Implement training pipeline")
    raise NotImplementedError


if __name__ == "__main__":
    main()
