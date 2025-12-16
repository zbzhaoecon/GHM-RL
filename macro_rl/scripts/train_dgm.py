"""
Training script for Deep Galerkin Method.

Example usage:
    python macro_rl/scripts/train_dgm.py --n_iterations 10000
"""

import torch
import argparse

# TODO: Implement Deep Galerkin training pipeline


def main():
    parser = argparse.ArgumentParser(description="Train GHM value function with Deep Galerkin")
    parser.add_argument("--n_iterations", type=int, default=10000)
    parser.add_argument("--n_interior", type=int, default=1000)
    parser.add_argument("--n_boundary", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print(f"Training GHM with Deep Galerkin Method")
    print("TODO: Implement training pipeline")
    raise NotImplementedError


if __name__ == "__main__":
    main()
