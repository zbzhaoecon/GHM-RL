"""
Unified training script using configuration files.

This script can train both Monte Carlo and Actor-Critic solvers using
a YAML or JSON configuration file, making it easy to test different
configurations without modifying code.

Usage:
    # Train with default configuration
    python scripts/train_with_config.py

    # Train with custom configuration
    python scripts/train_with_config.py --config configs/my_config.yaml

    # Override specific parameters
    python scripts/train_with_config.py --config configs/default_config.yaml --lr 1e-3 --seed 456

    # Resume from checkpoint
    python scripts/train_with_config.py --config configs/my_config.yaml --resume checkpoints/ghm_rl/step_5000.pt
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from macro_rl.config import ConfigManager, load_config
from macro_rl.config.setup_utils import setup_from_config, print_config_summary
from macro_rl.solvers.monte_carlo import MonteCarloPolicyGradient
from macro_rl.solvers.actor_critic import ModelBasedActorCritic

# Import utilities from time-augmented training script and new modules
from scripts.train_monte_carlo_ghm_time_augmented import PolicyAdapter
from macro_rl.visualization import (
    compute_policy_value_time_augmented,
    compute_policy_value_standard,
    create_training_visualization,
)
from macro_rl.evaluation import evaluate_policy as evaluate_policy_unified


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GHM-RL with configuration file"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file (YAML or JSON)"
    )

    # Override parameters
    parser.add_argument("--lr_policy", type=float, help="Override policy learning rate")
    parser.add_argument("--lr_baseline", type=float, help="Override baseline learning rate")
    parser.add_argument("--lr", type=float, help="Override combined learning rate (actor-critic)")
    parser.add_argument("--n_iterations", type=int, help="Override number of iterations")
    parser.add_argument("--n_trajectories", type=int, help="Override trajectories per iteration")
    parser.add_argument("--entropy_weight", type=float, help="Override entropy weight")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--experiment_name", type=str, help="Experiment name for logging")

    return parser.parse_args()


def create_solver_from_config(
    config_manager: ConfigManager,
    policy,
    baseline,
    simulator,
    dynamics,
    control_spec,
    reward_fn,
):
    """Create solver instance based on configuration.

    Args:
        config_manager: ConfigManager instance
        policy: Policy network
        baseline: Baseline/value network
        simulator: Trajectory simulator
        dynamics: Dynamics model
        control_spec: Control specification
        reward_fn: Reward function

    Returns:
        Solver instance (MonteCarloPolicyGradient or ModelBasedActorCritic)
    """
    config = config_manager.config

    if config.solver.solver_type == "monte_carlo":
        # Wrap policy with adapter for Monte Carlo
        policy_adapted = PolicyAdapter(policy)

        solver = MonteCarloPolicyGradient(
            policy=policy_adapted,
            simulator=simulator,
            baseline=baseline,
            n_trajectories=config.training.n_trajectories,
            lr_policy=config.training.lr_policy,
            lr_baseline=config.training.lr_baseline,
            advantage_normalization=config.training.advantage_normalization,
            max_grad_norm=config.training.max_grad_norm,
            entropy_weight=config.training.entropy_weight,
        )
    elif config.solver.solver_type == "actor_critic":
        solver = ModelBasedActorCritic(
            dynamics=dynamics,
            control_spec=control_spec,
            reward_fn=reward_fn,
            actor_critic=policy,  # ActorCritic network
            dt=config.training.dt,
            T=config.training.T,
            critic_loss=config.solver.critic_loss,
            actor_loss=config.solver.actor_loss,
            hjb_weight=config.solver.hjb_weight,
            entropy_weight=config.training.entropy_weight,
            n_trajectories=config.training.n_trajectories,
            lr=config.training.lr,
            max_grad_norm=config.training.max_grad_norm,
            use_parallel=config.solver.use_parallel,
            n_workers=config.solver.n_workers,
        )
    else:
        raise ValueError(f"Unknown solver type: {config.solver.solver_type}")

    return solver


def save_checkpoint(solver, config_manager: ConfigManager, step: int, ckpt_name: str = None):
    """Save training checkpoint."""
    config = config_manager.config
    os.makedirs(config.logging.ckpt_dir, exist_ok=True)

    if ckpt_name is None:
        ckpt_name = f"step_{step:06d}.pt"

    checkpoint = {
        'step': step,
        'config': config_manager.config.to_dict(),
    }

    # Handle different solver types
    if config.solver.solver_type == "monte_carlo":
        # Unwrap policy if adapted
        policy_to_save = solver.policy.policy if hasattr(solver.policy, 'policy') else solver.policy
        checkpoint['policy_state_dict'] = policy_to_save.state_dict()
        checkpoint['policy_optimizer_state_dict'] = solver.policy_optimizer.state_dict()

        if solver.baseline is not None:
            checkpoint['baseline_state_dict'] = solver.baseline.state_dict()
            checkpoint['baseline_optimizer_state_dict'] = solver.baseline_optimizer.state_dict()

    elif config.solver.solver_type == "actor_critic":
        checkpoint['actor_critic_state_dict'] = solver.ac.state_dict()
        checkpoint['optimizer_state_dict'] = solver.optimizer.state_dict()

    ckpt_path = os.path.join(config.logging.ckpt_dir, ckpt_name)
    torch.save(checkpoint, ckpt_path)
    print(f"[Checkpoint] Saved to {ckpt_path}")


def load_checkpoint(checkpoint_path: str, solver, config_manager: ConfigManager) -> int:
    """Load training checkpoint."""
    config = config_manager.config
    print(f"[Checkpoint] Loading from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=config.misc.device)

    if config.solver.solver_type == "monte_carlo":
        policy_to_load = solver.policy.policy if hasattr(solver.policy, 'policy') else solver.policy
        policy_to_load.load_state_dict(checkpoint['policy_state_dict'])
        solver.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

        if solver.baseline is not None and 'baseline_state_dict' in checkpoint:
            solver.baseline.load_state_dict(checkpoint['baseline_state_dict'])
            solver.baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state_dict'])

    elif config.solver.solver_type == "actor_critic":
        solver.ac.load_state_dict(checkpoint['actor_critic_state_dict'])
        solver.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    step = checkpoint.get('step', 0)
    print(f"[Checkpoint] Resumed from step {step}")
    return step


def log_training_metrics(writer: SummaryWriter, metrics: dict, step: int):
    """Log training metrics to TensorBoard."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, step)


def log_visualizations(writer: SummaryWriter, solver, dynamics, step: int, config_manager):
    """Generate and log policy/value visualizations."""
    config = config_manager.config

    # Get policy and baseline based on solver type
    if config.solver.solver_type == "monte_carlo":
        policy = solver.policy.policy if hasattr(solver.policy, 'policy') else solver.policy
        baseline = solver.baseline
    elif config.solver.solver_type == "actor_critic":
        policy = solver.ac  # ActorCritic has same interface
        baseline = solver.ac

    # Compute visualizations (automatically detects time-augmented vs standard)
    if dynamics.state_space.dim == 2:
        # Time-augmented dynamics
        results = compute_policy_value_time_augmented(policy, baseline, dynamics, n_points=100)
    else:
        # Standard dynamics
        results = compute_policy_value_standard(policy, baseline, dynamics, n_points=100)

    fig = create_training_visualization(results, step)

    # Log to TensorBoard
    writer.add_figure('policy_value/visualization', fig, step)

    # Save to file
    viz_dir = os.path.join(config.logging.ckpt_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    viz_path = os.path.join(viz_dir, f'step_{step:06d}.png')
    fig.savefig(viz_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved visualization to {viz_path}")


def evaluate_policy_local(solver, config_manager: ConfigManager, dynamics, n_episodes: int = 50) -> dict:
    """Evaluate policy deterministically using unified evaluation module."""
    config = config_manager.config

    # Use the unified evaluation function from macro_rl.evaluation
    return evaluate_policy_unified(
        solver=solver,
        solver_type=config.solver.solver_type,
        dynamics=dynamics,
        n_episodes=n_episodes,
        deterministic=True,
    )


def main():
    """Main training loop."""
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config_manager = load_config(args.config)

    # Apply command-line overrides
    overrides = {}
    if args.lr_policy is not None:
        overrides['training.lr_policy'] = args.lr_policy
    if args.lr_baseline is not None:
        overrides['training.lr_baseline'] = args.lr_baseline
    if args.lr is not None:
        overrides['training.lr'] = args.lr
    if args.n_iterations is not None:
        overrides['training.n_iterations'] = args.n_iterations
    if args.n_trajectories is not None:
        overrides['training.n_trajectories'] = args.n_trajectories
    if args.entropy_weight is not None:
        overrides['training.entropy_weight'] = args.entropy_weight
    if args.seed is not None:
        overrides['misc.seed'] = args.seed
    if args.device is not None:
        overrides['misc.device'] = args.device
    if args.resume is not None:
        overrides['misc.resume'] = args.resume
    if args.experiment_name is not None:
        overrides['misc.experiment_name'] = args.experiment_name

    if overrides:
        print("\nApplying command-line overrides:")
        for key, value in overrides.items():
            print(f"  {key}: {value}")
            config_manager.update({key: value})

    # Print configuration summary
    print_config_summary(config_manager)

    # Setup components from configuration
    print("\nSetting up training components...")
    dynamics, control_spec, reward_fn, policy, baseline, simulator, device = setup_from_config(
        config_manager
    )

    print(f"\nUsing device: {device}")
    print(f"State dimension: {dynamics.state_space.dim}")
    print(f"Action dimension: 2")
    print(f"Max steps per episode: {simulator.max_steps}")

    # Create solver
    print(f"\nCreating {config_manager.config.solver.solver_type} solver...")
    solver = create_solver_from_config(
        config_manager, policy, baseline, simulator, dynamics, control_spec, reward_fn
    )

    # Setup TensorBoard
    config = config_manager.config
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if config.misc.experiment_name:
        log_dir = os.path.join(config.logging.log_dir, config.misc.experiment_name, timestamp)
    else:
        log_dir = os.path.join(config.logging.log_dir, timestamp)

    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard log directory: {log_dir}")
    print(f"Monitor with: tensorboard --logdir {config.logging.log_dir}")

    # Save configuration
    config_path = os.path.join(config.logging.ckpt_dir, "config.yaml")
    os.makedirs(config.logging.ckpt_dir, exist_ok=True)
    config_manager.save(config_path)
    print(f"Saved configuration to: {config_path}")

    # Resume from checkpoint if specified
    start_step = 1
    if config.misc.resume:
        start_step = load_checkpoint(config.misc.resume, solver, config_manager) + 1

    # Training loop
    print("\n" + "=" * 80)
    print(f"Starting training from step {start_step} to {config.training.n_iterations}")
    print("=" * 80)

    # Store dynamics for state sampling (Monte Carlo solver needs this)
    if config.solver.solver_type == "monte_carlo":
        solver.dynamics = dynamics

    best_return = -float('inf')

    for step in range(start_step, config.training.n_iterations + 1):
        # Training step
        metrics = solver.train_step()

        # Log metrics
        if step % config.logging.log_freq == 0:
            log_training_metrics(writer, metrics, step)

            print(f"\n[Step {step}/{config.training.n_iterations}]")
            if "return/mean" in metrics:
                print(f"  Return: {metrics['return/mean']:7.3f} ± {metrics.get('return/std', 0):6.3f}")
            if "loss/policy" in metrics:
                print(f"  Policy Loss: {metrics['loss/policy']:8.4f}")
            if "loss/baseline" in metrics or "loss/critic" in metrics:
                baseline_loss = metrics.get("loss/baseline", metrics.get("loss/critic", 0))
                print(f"  Critic Loss: {baseline_loss:8.4f}")
            if "policy/entropy" in metrics:
                print(f"  Entropy: {metrics['policy/entropy']:6.4f}")

        # Evaluation
        if step % config.logging.eval_freq == 0:
            print(f"\n[Evaluation at step {step}]")
            eval_metrics = evaluate_policy_local(solver, config_manager, dynamics, n_episodes=1024)

            writer.add_scalar("eval/return_mean", eval_metrics['return_mean'], step)
            writer.add_scalar("eval/return_std", eval_metrics['return_std'], step)
            writer.add_scalar("eval/episode_length", eval_metrics['episode_length'], step)

            print(f"  Eval Return: {eval_metrics['return_mean']:.4f} ± {eval_metrics['return_std']:.4f}")
            print(f"  Episode Length: {eval_metrics['episode_length']:.2f}")

            # Generate visualizations
            log_visualizations(writer, solver, dynamics, step, config_manager)

            # Save best model
            if eval_metrics['return_mean'] > best_return:
                best_return = eval_metrics['return_mean']
                save_checkpoint(solver, config_manager, step, ckpt_name="best_model.pt")
                print(f"  ✓ New best model! Return: {best_return:.4f}")

        # Save periodic checkpoint
        if step % config.logging.ckpt_freq == 0:
            save_checkpoint(solver, config_manager, step)

    # Final checkpoint
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    save_checkpoint(solver, config_manager, config.training.n_iterations, ckpt_name="final_model.pt")

    # Final evaluation
    print("\nFinal evaluation...")
    final_eval = evaluate_policy_local(solver, config_manager, dynamics, n_episodes=100)
    print(f"  Final Return: {final_eval['return_mean']:.4f} ± {final_eval['return_std']:.4f}")
    print(f"  Best Return: {best_return:.4f}")

    writer.close()
    print(f"\nLogs saved to: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
