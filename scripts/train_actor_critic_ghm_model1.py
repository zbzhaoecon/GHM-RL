"""
Training script for Actor-Critic on GHM Model 1 (Equity Management).

This script trains the ModelBasedActorCritic solver on the GHM equity dynamics
with TensorBoard logging for monitoring learning progress, policy behavior,
and value function diagnostics.

Usage:
    python scripts/train_actor_critic_ghm_model1.py
    python scripts/train_actor_critic_ghm_model1.py --n_iterations 100000 --lr 1e-4
    python scripts/train_actor_critic_ghm_model1.py --seed 456 --device cuda
"""

import argparse
import os
from dataclasses import asdict, replace
from typing import Dict, Any

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from macro_rl.dynamics import GHMEquityDynamics, GHMEquityParams
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.networks.actor_critic import ActorCritic
from macro_rl.solvers.actor_critic import ModelBasedActorCritic

from utils_training import TrainConfig, create_writer, save_checkpoint


def parse_args() -> TrainConfig:
    """Parse command-line arguments and return TrainConfig."""
    parser = argparse.ArgumentParser(
        description="Train Actor-Critic on GHM Model 1 (Equity Management)"
    )

    # Core parameters
    parser.add_argument("--dt", type=float, default=0.01, help="Time discretization")
    parser.add_argument("--T", type=float, default=10.0, help="Episode horizon")

    # Training parameters
    parser.add_argument("--n_iterations", type=int, default=50000, help="Number of training iterations")
    parser.add_argument("--n_trajectories", type=int, default=256, help="Number of trajectories per iteration")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--critic_loss", type=str, default="mc+hjb",
                       choices=["mc", "td", "mc+td", "mc+hjb", "td+hjb"],
                       help="Critic loss type")
    parser.add_argument("--actor_loss", type=str, default="pathwise",
                       choices=["pathwise", "reinforce"],
                       help="Actor loss type")
    parser.add_argument("--hjb_weight", type=float, default=0.1, help="HJB loss weight")
    parser.add_argument("--entropy_weight", type=float, default=0.01, help="Entropy regularization weight")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")

    # Network architecture
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256],
                       help="Hidden layer dimensions")
    parser.add_argument("--shared_layers", type=int, default=1,
                       help="Number of shared layers between actor and critic")

    # Logging and checkpoints
    parser.add_argument("--log_dir", type=str, default="runs/ghm_model1", help="TensorBoard log directory")
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency (iterations)")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Evaluation frequency (iterations)")
    parser.add_argument("--ckpt_freq", type=int, default=5000, help="Checkpoint frequency (iterations)")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/ghm_model1", help="Checkpoint directory")

    # Misc
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, default: auto-detect)")

    args = parser.parse_args()

    # Create config from args
    config = TrainConfig(
        dt=args.dt,
        T=args.T,
        n_iterations=args.n_iterations,
        n_trajectories=args.n_trajectories,
        lr=args.lr,
        critic_loss=args.critic_loss,
        actor_loss=args.actor_loss,
        hjb_weight=args.hjb_weight,
        entropy_weight=args.entropy_weight,
        max_grad_norm=args.max_grad_norm,
        hidden_dims=tuple(args.hidden_dims),
        shared_layers=args.shared_layers,
        log_dir=args.log_dir,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        ckpt_freq=args.ckpt_freq,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return config


def log_actor_critic_details(
    writer: SummaryWriter,
    solver: ModelBasedActorCritic,
    step: int,
    n_samples: int = 128,
):
    """
    Log detailed actor-critic diagnostics to TensorBoard.

    This includes:
    - Policy statistics (log_std, actions, entropy)
    - Value function statistics
    - HJB residuals (if applicable)
    """
    solver.ac.eval()

    with torch.no_grad():
        # Sample random states from a reasonable range
        # For GHM equity model, cash reserves typically in [0, 10]
        device = next(solver.ac.parameters()).device
        state_dim = solver.dynamics.state_space.dim
        states = torch.rand(n_samples, state_dim, device=device) * 10.0

        # Pass states through shared trunk if present
        features = solver.ac._features(states)

        # Get policy distribution and sample actions
        dist = solver.ac.actor.get_distribution(features)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # Get value predictions
        values = solver.ac.critic(features).squeeze(-1)

        # Log policy statistics
        if hasattr(dist, 'scale'):
            log_stds = torch.log(dist.scale)
            writer.add_scalar("policy/log_std_mean", log_stds.mean().item(), step)
            writer.add_scalar("policy/log_std_max", log_stds.max().item(), step)
            writer.add_scalar("policy/log_std_min", log_stds.min().item(), step)

        writer.add_histogram("policy/actions", actions, step)
        writer.add_scalar("policy/actions_mean", actions.mean().item(), step)
        writer.add_scalar("policy/actions_std", actions.std().item(), step)

        writer.add_scalar("policy/entropy_sampled", entropy.mean().item(), step)
        writer.add_scalar("policy/logp_sampled_mean", log_probs.mean().item(), step)

        # Log value statistics
        writer.add_histogram("value/V_sampled", values, step)
        writer.add_scalar("value/V_mean", values.mean().item(), step)
        writer.add_scalar("value/V_std", values.std().item(), step)
        writer.add_scalar("value/V_min", values.min().item(), step)
        writer.add_scalar("value/V_max", values.max().item(), step)

        # Log HJB residuals if using HJB loss
        if "hjb" in solver.critic_loss_type:
            # Sample trajectories for HJB residual computation
            n_traj = min(32, n_samples)
            initial_states = solver._sample_initial_states(n_traj)
            traj = solver.simulator.rollout(solver.ac, initial_states)

            # Compute HJB residuals: dV/dt + H(x, dV/dx)
            # Approximate using finite differences
            states_t = traj.states[:, :-1].reshape(-1, state_dim)
            states_tp1 = traj.states[:, 1:].reshape(-1, state_dim)
            rewards = traj.rewards.reshape(-1)

            # Pass through shared trunk if present
            feat_t = solver.ac._features(states_t)
            feat_tp1 = solver.ac._features(states_tp1)

            V_t = solver.ac.critic(feat_t).squeeze(-1)
            V_tp1 = solver.ac.critic(feat_tp1).squeeze(-1)

            # HJB residual: r + V_{t+1} - V_t (simplified, without discount for diagnostic)
            hjb_residual = rewards + V_tp1 - V_t

            writer.add_histogram("diagnostics/hjb_residual_sampled", hjb_residual, step)
            writer.add_scalar("diagnostics/hjb_residual_mean", hjb_residual.mean().item(), step)
            writer.add_scalar("diagnostics/hjb_residual_std", hjb_residual.std().item(), step)

    solver.ac.train()


def log_training_metrics(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    step: int,
):
    """Log training metrics from solver.train_step() to TensorBoard."""
    # Log returns
    if "return/mean" in metrics:
        writer.add_scalar("return/mean", metrics["return/mean"], step)
    if "return/std" in metrics:
        writer.add_scalar("return/std", metrics["return/std"], step)

    # Log actor losses
    if "actor/loss" in metrics:
        writer.add_scalar("actor/loss", metrics["actor/loss"], step)
    if "actor/entropy" in metrics:
        writer.add_scalar("actor/entropy", metrics["actor/entropy"], step)
    if "actor/advantage" in metrics:
        writer.add_scalar("actor/advantage", metrics["actor/advantage"], step)

    # Log critic losses
    if "critic/mc" in metrics:
        writer.add_scalar("critic/mc", metrics["critic/mc"], step)
    if "critic/td" in metrics:
        writer.add_scalar("critic/td", metrics["critic/td"], step)
    if "critic/hjb" in metrics:
        writer.add_scalar("critic/hjb", metrics["critic/hjb"], step)

    # Log total loss
    if "loss/total" in metrics:
        writer.add_scalar("loss/total", metrics["loss/total"], step)

    # Log episode length
    if "episode_length/mean" in metrics:
        writer.add_scalar("episode_length/mean", metrics["episode_length/mean"], step)


def log_evaluation_metrics(
    writer: SummaryWriter,
    eval_metrics: Dict[str, float],
    step: int,
):
    """Log evaluation metrics to TensorBoard under eval/ namespace."""
    if "return_mean" in eval_metrics:
        writer.add_scalar("eval/return_mean", eval_metrics["return_mean"], step)
    if "return_std" in eval_metrics:
        writer.add_scalar("eval/return_std", eval_metrics["return_std"], step)
    if "episode_length" in eval_metrics:
        writer.add_scalar("eval/episode_length", eval_metrics["episode_length"], step)


def main():
    """Main training loop."""
    # Parse configuration
    config = parse_args()

    print("=" * 80)
    print("Training Actor-Critic on GHM Model 1 (Equity Management)")
    print("=" * 80)
    print(f"Configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print("=" * 80)

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Set device
    device = torch.device(config.device)
    print(f"\nUsing device: {device}")

    # =========================================================================
    # 1. Setup dynamics
    # =========================================================================
    print("\n[1/5] Setting up GHM Equity dynamics...")
    params = GHMEquityParams()
    dynamics = GHMEquityDynamics(params)
    print(f"  State dimension: {dynamics.state_space.dim}")
    print(f"  Discount rate (ρ = r - μ): {params.r - params.mu:.4f}")

    # =========================================================================
    # 2. Setup control specification
    # =========================================================================
    print("\n[2/5] Setting up control specification...")
    control_spec = GHMControlSpec()
    print(f"  Action dimension: 2 (dividend, equity issuance)")
    print(f"  Action bounds: [{control_spec.lower}, {control_spec.upper}]")

    # =========================================================================
    # 3. Setup reward function
    # =========================================================================
    print("\n[3/5] Setting up reward function...")
    reward_fn = GHMRewardFunction(
        discount_rate=params.r - params.mu,
        issuance_cost=params.lambda_,
        liquidation_rate=params.omega,
        liquidation_flow=params.alpha,
    )
    print(f"  Discount rate: {params.r - params.mu:.4f}")
    print(f"  Issuance cost: {params.lambda_:.4f}")

    # =========================================================================
    # 4. Setup actor-critic network
    # =========================================================================
    print("\n[4/5] Setting up actor-critic network...")
    state_dim = dynamics.state_space.dim
    action_dim = 2  # dividend, equity issuance

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=config.shared_layers,
        hidden_dims=list(config.hidden_dims),
        action_bounds=(control_spec.lower, control_spec.upper),
    ).to(device)

    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Shared layers: {config.shared_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in ac.parameters())}")

    # =========================================================================
    # 5. Setup solver
    # =========================================================================
    print("\n[5/5] Setting up ModelBasedActorCritic solver...")
    solver = ModelBasedActorCritic(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        actor_critic=ac,
        dt=config.dt,
        T=config.T,
        critic_loss=config.critic_loss,
        actor_loss=config.actor_loss,
        hjb_weight=config.hjb_weight,
        entropy_weight=config.entropy_weight,
        n_trajectories=config.n_trajectories,
        lr=config.lr,
        max_grad_norm=config.max_grad_norm,
    )

    print(f"  Critic loss: {config.critic_loss}")
    print(f"  Actor loss: {config.actor_loss}")
    print(f"  HJB weight: {config.hjb_weight}")
    print(f"  Entropy weight: {config.entropy_weight}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Max grad norm: {config.max_grad_norm}")

    # =========================================================================
    # Setup TensorBoard
    # =========================================================================
    print("\nSetting up TensorBoard...")
    writer = create_writer(config)
    print(f"  Log directory: {writer.log_dir}")
    print(f"  Monitor with: tensorboard --logdir {config.log_dir}")

    # =========================================================================
    # Training loop
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"Starting training for {config.n_iterations} iterations...")
    print("=" * 80)

    best_return = -float('inf')

    for step in range(1, config.n_iterations + 1):
        # Training step
        metrics = solver.train_step(n_samples=config.n_trajectories)

        # Log training metrics
        if step % config.log_freq == 0:
            log_training_metrics(writer, metrics, step)

            print(f"[Step {step}/{config.n_iterations}]")
            print(f"  Return: {metrics['return/mean']:.4f} ± {metrics['return/std']:.4f}")
            print(f"  Loss: {metrics['loss/total']:.4f}")
            if "actor/entropy" in metrics:
                print(f"  Entropy: {metrics['actor/entropy']:.4f}")
            if "critic/mc" in metrics:
                print(f"  Critic MC: {metrics['critic/mc']:.4f}")
            if "critic/hjb" in metrics:
                print(f"  Critic HJB: {metrics['critic/hjb']:.4f}")

        # Detailed logging (less frequent)
        if step % (config.log_freq * 5) == 0:
            log_actor_critic_details(writer, solver, step, n_samples=128)

        # Evaluation
        if step % config.eval_freq == 0:
            print(f"\n[Evaluation at step {step}]")
            eval_metrics = solver.evaluate(n_episodes=50)
            log_evaluation_metrics(writer, eval_metrics, step)

            print(f"  Eval Return: {eval_metrics['return_mean']:.4f} ± {eval_metrics['return_std']:.4f}")
            print(f"  Eval Episode Length: {eval_metrics['episode_length']:.2f}")

            # Save best model
            if eval_metrics['return_mean'] > best_return:
                best_return = eval_metrics['return_mean']
                save_checkpoint(solver, config, step, ckpt_name="best_model.pt")
                print(f"  New best model! Return: {best_return:.4f}")
            print()

        # Save periodic checkpoint
        if step % config.ckpt_freq == 0:
            save_checkpoint(solver, config, step)

    # =========================================================================
    # Save final model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    save_checkpoint(solver, config, config.n_iterations, ckpt_name="final_model.pt")

    # Final evaluation
    print("\nFinal evaluation...")
    final_eval = solver.evaluate(n_episodes=100)
    print(f"  Final Return: {final_eval['return_mean']:.4f} ± {final_eval['return_std']:.4f}")
    print(f"  Best Return: {best_return:.4f}")

    writer.close()
    print(f"\nTensorBoard logs saved to: {writer.log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
