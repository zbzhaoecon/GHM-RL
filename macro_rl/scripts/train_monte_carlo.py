"""
Training script for Monte Carlo Policy Gradient solver.

Example usage:
    python macro_rl/scripts/train_monte_carlo.py --n_iterations 5000
"""

import torch
import argparse
import os
from pathlib import Path

from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.policies.neural import GaussianPolicy
from macro_rl.values.neural import ValueNetwork
from macro_rl.simulation.trajectory import TrajectorySimulator
from macro_rl.solvers.monte_carlo import MonteCarloPolicyGradient


def main():
    parser = argparse.ArgumentParser(description="Train GHM policy with Monte Carlo PG")

    # Training
    parser.add_argument("--n_iterations", type=int, default=5000)
    parser.add_argument("--n_trajectories", type=int, default=500)
    parser.add_argument("--lr_policy", type=float, default=3e-4)
    parser.add_argument("--lr_baseline", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=100)

    # Network architecture
    parser.add_argument("--policy_hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--value_hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--no_baseline", action="store_true")

    # Simulation
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--T", type=float, default=5.0)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="results/monte_carlo")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("Monte Carlo Policy Gradient Training")
    print("=" * 80)
    print(f"Iterations: {args.n_iterations}, Trajectories: {args.n_trajectories}")
    print(f"Policy LR: {args.lr_policy}, Baseline LR: {args.lr_baseline}")
    print(f"Use baseline: {not args.no_baseline}")
    print("=" * 80)

    # Initialize GHM components
    params = GHMEquityParams()
    dynamics = GHMEquityDynamics(params)
    control_spec = GHMControlSpec(
        a_L_max=10.0,
        a_E_max=0.5,
        issuance_cost=params.p - 1.0,
    )
    reward_fn = GHMRewardFunction(
        discount_rate=params.r - params.mu,
        issuance_cost=params.p - 1.0,
        liquidation_rate=params.omega,
        liquidation_flow=params.alpha,
    )

    # Create policy and baseline
    policy = GaussianPolicy(
        state_dim=1,
        action_dim=2,
        hidden_dims=args.policy_hidden,
    ).to(device)

    baseline = None if args.no_baseline else ValueNetwork(
        state_dim=1,
        hidden_dims=args.value_hidden,
    ).to(device)

    # Create simulator and solver
    simulator = TrajectorySimulator(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        dt=args.dt,
        T=args.T,
    )

    solver = MonteCarloPolicyGradient(
        policy=policy,
        simulator=simulator,
        baseline=baseline,
        n_trajectories=args.n_trajectories,
        lr_policy=args.lr_policy,
        lr_baseline=args.lr_baseline,
    )

    print("Starting training...")
    result = solver.solve(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        n_iterations=args.n_iterations,
        log_interval=args.log_interval,
    )

    print("=" * 80)
    print(f"Training complete! Final return: {result.diagnostics['returns'][-1]:.4f}")

    # Save checkpoint
    save_path = Path(args.save_dir) / "checkpoint_final.pt"
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'args': vars(args),
        'diagnostics': result.diagnostics,
    }
    if baseline is not None:
        checkpoint['baseline_state_dict'] = baseline.state_dict()
    torch.save(checkpoint, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
