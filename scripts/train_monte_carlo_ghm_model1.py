"""
Training script for Monte Carlo Policy Gradient on GHM Model 1 (Equity Management).

This script trains the MonteCarloPolicyGradient solver on the GHM equity dynamics
with TensorBoard logging for monitoring learning progress, policy behavior,
and value function diagnostics.

Usage:
    # Start new training
    python scripts/train_monte_carlo_ghm_model1.py
    python scripts/train_monte_carlo_ghm_model1.py --n_iterations 10000 --lr_policy 3e-4
    python scripts/train_monte_carlo_ghm_model1.py --seed 456 --device cuda

    # Resume from checkpoint
    python scripts/train_monte_carlo_ghm_model1.py --resume checkpoints/monte_carlo_model1/step_5000.pt
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.tensorboard import SummaryWriter

from macro_rl.dynamics import GHMEquityDynamics, GHMEquityParams
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.networks.policy import GaussianPolicy  # FIXED: Use actor-critic policy
from macro_rl.networks.value import ValueNetwork    # FIXED: Use actor-critic value network
from macro_rl.simulation.trajectory import TrajectorySimulator
from macro_rl.solvers.monte_carlo import MonteCarloPolicyGradient


class PolicyAdapter(torch.nn.Module):
    """
    Adapter to make actor-critic GaussianPolicy compatible with Monte Carlo solver.

    The Monte Carlo solver expects:
    - policy.log_prob(state, action) -> log_prob
    - policy.entropy(state) -> entropy

    But actor-critic GaussianPolicy has:
    - policy.log_prob_and_entropy(state, action) -> (log_prob, entropy)
    - policy.get_distribution(state) -> distribution
    """
    def __init__(self, policy: GaussianPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, state):
        """Forward pass for distribution."""
        return self.policy.get_distribution(state)

    def act(self, state):
        """Sample action (used by TrajectorySimulator)."""
        action, _ = self.policy.sample(state, deterministic=False)
        return action

    def log_prob(self, state, action):
        """Compute log probability."""
        log_prob, _ = self.policy.log_prob_and_entropy(state, action)
        return log_prob

    def entropy(self, state):
        """Compute entropy."""
        dist = self.policy.get_distribution(state)
        return dist.entropy().sum(dim=-1)

    def parameters(self):
        """Pass through to underlying policy."""
        return self.policy.parameters()

    def train(self, mode=True):
        """Set training mode."""
        self.policy.train(mode)
        return super().train(mode)

    def eval(self):
        """Set evaluation mode."""
        self.policy.eval()
        return super().eval()


@dataclass
class TrainConfig:
    """Training configuration for Monte Carlo Policy Gradient."""
    # Core parameters
    dt: float = 0.01
    T: float = 10.0

    # Training parameters
    n_iterations: int = 10000
    n_trajectories: int = 500
    lr_policy: float = 3e-4
    lr_baseline: float = 1e-3
    max_grad_norm: float = 0.5
    advantage_normalization: bool = True
    entropy_weight: float = 0.05  # INCREASED from 0.01
    action_reg_weight: float = 0.01  # NEW: action magnitude regularization (INCREASED)

    # Network architecture
    policy_hidden: tuple = (64, 64)
    value_hidden: tuple = (64, 64)
    use_baseline: bool = True

    # Logging and checkpoints
    log_dir: str = "runs/monte_carlo_model1"
    log_freq: int = 100
    eval_freq: int = 1000
    ckpt_freq: int = 5000
    ckpt_dir: str = "checkpoints/monte_carlo_model1"

    # Misc
    seed: int = 123
    device: str = "cpu"
    resume: str = None


def parse_args() -> TrainConfig:
    """Parse command-line arguments and return TrainConfig."""
    parser = argparse.ArgumentParser(
        description="Train Monte Carlo PG on GHM Model 1 (Equity Management)"
    )

    # Core parameters
    parser.add_argument("--dt", type=float, default=0.01, help="Time discretization")
    parser.add_argument("--T", type=float, default=10.0, help="Episode horizon")

    # Training parameters
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--n_trajectories", type=int, default=500, help="Number of trajectories per iteration")
    parser.add_argument("--lr_policy", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--lr_baseline", type=float, default=1e-3, help="Baseline learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
    parser.add_argument("--entropy_weight", type=float, default=0.05, help="Entropy regularization weight")
    parser.add_argument("--action_reg_weight", type=float, default=0.01, help="Action magnitude regularization weight")
    parser.add_argument("--no_baseline", action="store_true", help="Disable baseline (pure REINFORCE)")

    # Network architecture
    parser.add_argument("--policy_hidden", type=int, nargs="+", default=[64, 64],
                       help="Policy hidden layer dimensions")
    parser.add_argument("--value_hidden", type=int, nargs="+", default=[64, 64],
                       help="Value hidden layer dimensions")

    # Logging and checkpoints
    parser.add_argument("--log_dir", type=str, default="runs/monte_carlo_model1", help="TensorBoard log directory")
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency (iterations)")
    parser.add_argument("--eval_freq", type=int, default=500, help="Evaluation frequency (iterations)")
    parser.add_argument("--ckpt_freq", type=int, default=5000, help="Checkpoint frequency (iterations)")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/monte_carlo_model1", help="Checkpoint directory")

    # Misc
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, default: auto-detect)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    # Create config from args
    config = TrainConfig(
        dt=args.dt,
        T=args.T,
        n_iterations=args.n_iterations,
        n_trajectories=args.n_trajectories,
        lr_policy=args.lr_policy,
        lr_baseline=args.lr_baseline,
        max_grad_norm=args.max_grad_norm,
        entropy_weight=args.entropy_weight,
        action_reg_weight=args.action_reg_weight,
        policy_hidden=tuple(args.policy_hidden),
        value_hidden=tuple(args.value_hidden),
        use_baseline=not args.no_baseline,
        log_dir=args.log_dir,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        ckpt_freq=args.ckpt_freq,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
        resume=args.resume,
    )

    return config


def save_config_to_file(config: TrainConfig, output_dir: str):
    """Save training configuration to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "training_config.json")

    config_dict = asdict(config)
    # Convert tuples to lists for JSON serialization
    if isinstance(config_dict.get('policy_hidden'), tuple):
        config_dict['policy_hidden'] = list(config_dict['policy_hidden'])
    if isinstance(config_dict.get('value_hidden'), tuple):
        config_dict['value_hidden'] = list(config_dict['value_hidden'])

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"[Config] Saved configuration to {config_path}")


def create_writer(config: TrainConfig) -> SummaryWriter:
    """Create TensorBoard writer with timestamped directory."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config.log_dir, timestamp)
    return SummaryWriter(log_dir)


def save_checkpoint(
    solver: MonteCarloPolicyGradient,
    config: TrainConfig,
    step: int,
    ckpt_name: str = None,
    best_return: float = None,
):
    """Save training checkpoint."""
    os.makedirs(config.ckpt_dir, exist_ok=True)

    if ckpt_name is None:
        ckpt_name = f"step_{step:06d}.pt"

    # Unwrap policy if it's adapted
    policy_to_save = solver.policy.policy if hasattr(solver.policy, 'policy') else solver.policy

    checkpoint = {
        'step': step,
        'policy_state_dict': policy_to_save.state_dict(),
        'policy_optimizer_state_dict': solver.policy_optimizer.state_dict(),
        'config': asdict(config),
    }

    if solver.baseline is not None:
        checkpoint['baseline_state_dict'] = solver.baseline.state_dict()
        checkpoint['baseline_optimizer_state_dict'] = solver.baseline_optimizer.state_dict()

    if best_return is not None:
        checkpoint['best_return'] = best_return

    ckpt_path = os.path.join(config.ckpt_dir, ckpt_name)
    torch.save(checkpoint, ckpt_path)
    print(f"[Checkpoint] Saved to {ckpt_path}")


def load_checkpoint(
    checkpoint_path: str,
    solver: MonteCarloPolicyGradient,
    config: TrainConfig,
) -> int:
    """Load training checkpoint and return the step number."""
    print(f"[Checkpoint] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Unwrap policy if it's adapted
    policy_to_load = solver.policy.policy if hasattr(solver.policy, 'policy') else solver.policy
    policy_to_load.load_state_dict(checkpoint['policy_state_dict'])
    solver.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

    if solver.baseline is not None and 'baseline_state_dict' in checkpoint:
        solver.baseline.load_state_dict(checkpoint['baseline_state_dict'])
        solver.baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state_dict'])

    step = checkpoint.get('step', 0)
    print(f"[Checkpoint] Resumed from step {step}")

    return step


def compute_policy_value_for_visualization(
    policy: GaussianPolicy,
    baseline: ValueNetwork,
    dynamics: GHMEquityDynamics,
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Compute policy and value function on a grid for visualization.

    Args:
        policy: Trained policy
        baseline: Trained value function (baseline)
        dynamics: GHM dynamics model
        n_points: Number of points to sample in state space

    Returns:
        Dictionary containing c_values, actions_mean, actions_std, values, V_s
    """
    policy.eval()
    if baseline is not None:
        baseline.eval()

    state_space = dynamics.state_space
    c_values = np.linspace(state_space.lower[0].item(), state_space.upper[0].item(), n_points)
    states = torch.tensor(c_values, dtype=torch.float32).unsqueeze(1)

    device = next(policy.parameters()).device
    states = states.to(device)

    with torch.no_grad():
        # Get policy distribution parameters
        dist = policy.get_distribution(states)
        # Get mean actions (with bounds applied via sigmoid)
        actions_mean, _ = policy.sample(states, deterministic=True)
        actions_mean = actions_mean.cpu()
        actions_std = dist.stddev.cpu()

        # Get value estimates
        if baseline is not None:
            values = baseline(states).squeeze().cpu()
        else:
            values = torch.zeros(n_points)

    # Get value gradients (requires gradients enabled)
    if baseline is not None:
        states_grad = states.clone().requires_grad_(True)
        values_grad = baseline(states_grad)
        V_s = torch.autograd.grad(
            values_grad.sum(),
            states_grad,
            create_graph=False
        )[0].detach().cpu().squeeze()
    else:
        V_s = torch.zeros(n_points)

    policy.train()
    if baseline is not None:
        baseline.train()

    return {
        'c_values': c_values,
        'actions_mean': actions_mean.numpy(),
        'actions_std': actions_std.numpy(),
        'values': values.numpy(),
        'V_s': V_s.numpy(),
    }


def create_training_visualization(
    results: Dict[str, np.ndarray],
    step: int
) -> plt.Figure:
    """Create a comprehensive visualization figure for logging during training."""
    c_values = results['c_values']
    actions_mean = results['actions_mean']
    actions_std = results['actions_std']
    values = results['values']
    V_s = results['V_s']

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Plot 1: Value Function
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(c_values, values, 'b-', linewidth=2)
    ax1.set_xlabel('Cash Reserves Ratio (c)', fontsize=10)
    ax1.set_ylabel('Value Function V(c)', fontsize=10)
    ax1.set_title('Value Function (Baseline)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Value Function Gradient
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(c_values, V_s, 'g-', linewidth=2)
    ax2.set_xlabel('Cash Reserves Ratio (c)', fontsize=10)
    ax2.set_ylabel("Value Gradient V'(c)", fontsize=10)
    ax2.set_title('Value Function Gradient', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Dividend Payout Policy (a_L)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(c_values, actions_mean[:, 0], 'b-', linewidth=2, label='Mean')
    ax3.fill_between(c_values,
                     actions_mean[:, 0] - actions_std[:, 0],
                     actions_mean[:, 0] + actions_std[:, 0],
                     alpha=0.3, label='±1 std')
    ax3.set_xlabel('Cash Reserves Ratio (c)', fontsize=10)
    ax3.set_ylabel('Dividend Payout Rate (a_L)', fontsize=10)
    ax3.set_title('Policy: Dividend Payout', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Equity Issuance Policy (a_E)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(c_values, actions_mean[:, 1], 'r-', linewidth=2, label='Mean')
    ax4.fill_between(c_values,
                     actions_mean[:, 1] - actions_std[:, 1],
                     actions_mean[:, 1] + actions_std[:, 1],
                     alpha=0.3, label='±1 std')
    ax4.set_xlabel('Cash Reserves Ratio (c)', fontsize=10)
    ax4.set_ylabel('Equity Issuance (a_E)', fontsize=10)
    ax4.set_title('Policy: Equity Issuance', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Combined Policy View
    ax5 = fig.add_subplot(gs[2, :])
    ax5_twin = ax5.twinx()

    line1 = ax5.plot(c_values, actions_mean[:, 0], 'b-', linewidth=2, label='Dividend Payout (a_L)')
    line2 = ax5_twin.plot(c_values, actions_mean[:, 1], 'r-', linewidth=2, label='Equity Issuance (a_E)')

    ax5.set_xlabel('Cash Reserves Ratio (c)', fontsize=10)
    ax5.set_ylabel('Dividend Payout Rate (a_L)', fontsize=10, color='b')
    ax5_twin.set_ylabel('Equity Issuance (a_E)', fontsize=10, color='r')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5_twin.tick_params(axis='y', labelcolor='r')
    ax5.set_title('Combined Policy View', fontsize=11, fontweight='bold')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Add main title
    fig.suptitle(f'Policy and Value Function Visualization (Step {step})',
                 fontsize=13, fontweight='bold', y=0.995)

    return fig


def log_policy_value_visualization(
    writer: SummaryWriter,
    policy: PolicyAdapter,
    baseline: ValueNetwork,
    dynamics: GHMEquityDynamics,
    step: int,
    config: TrainConfig,
):
    """Generate and log policy/value visualizations to TensorBoard."""
    print(f"  Generating policy/value visualizations...")

    # Compute policy and value on grid (unwrap the policy)
    results = compute_policy_value_for_visualization(policy.policy, baseline, dynamics, n_points=100)

    # Create visualization figure
    fig = create_training_visualization(results, step)

    # Log to TensorBoard
    writer.add_figure('policy_value/visualization', fig, step)

    # Also save to checkpoint directory for archival
    viz_dir = os.path.join(config.ckpt_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    viz_path = os.path.join(viz_dir, f'step_{step:06d}.png')
    fig.savefig(viz_path, dpi=100, bbox_inches='tight')

    plt.close(fig)

    # Log some key statistics as scalars
    writer.add_scalar('policy_value/mean_value', results['values'].mean(), step)
    writer.add_scalar('policy_value/mean_dividend', results['actions_mean'][:, 0].mean(), step)
    writer.add_scalar('policy_value/mean_equity_issuance', results['actions_mean'][:, 1].mean(), step)

    print(f"  Saved visualization to {viz_path}")


def log_training_metrics(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    step: int,
):
    """Log training metrics to TensorBoard."""
    # Returns
    if "return/mean" in metrics:
        writer.add_scalar("return/mean", metrics["return/mean"], step)
    if "return/std" in metrics:
        writer.add_scalar("return/std", metrics["return/std"], step)
    if "return/min" in metrics:
        writer.add_scalar("return/min", metrics["return/min"], step)
    if "return/max" in metrics:
        writer.add_scalar("return/max", metrics["return/max"], step)

    # Losses
    if "loss/policy" in metrics:
        writer.add_scalar("loss/policy", metrics["loss/policy"], step)
    if "loss/baseline" in metrics:
        writer.add_scalar("loss/baseline", metrics["loss/baseline"], step)
    if "loss/action_reg" in metrics:
        writer.add_scalar("loss/action_reg", metrics["loss/action_reg"], step)

    # Advantages
    if "advantage/mean" in metrics:
        writer.add_scalar("advantage/mean", metrics["advantage/mean"], step)
    if "advantage/std" in metrics:
        writer.add_scalar("advantage/std", metrics["advantage/std"], step)

    # Episode statistics
    if "episode_length/mean" in metrics:
        writer.add_scalar("episode_length/mean", metrics["episode_length/mean"], step)
    if "termination_rate" in metrics:
        writer.add_scalar("episode/termination_rate", metrics["termination_rate"], step)

    # Policy statistics
    if "policy/mean_action_0" in metrics:
        writer.add_scalar("policy/mean_action_0", metrics["policy/mean_action_0"], step)
    if "policy/std_action_0" in metrics:
        writer.add_scalar("policy/std_action_0", metrics["policy/std_action_0"], step)
    if "policy/entropy" in metrics:
        writer.add_scalar("policy/entropy", metrics["policy/entropy"], step)
    if "policy/action_magnitude" in metrics:
        writer.add_scalar("policy/action_magnitude", metrics["policy/action_magnitude"], step)

    # Gradients
    if "grad_norm/policy" in metrics:
        writer.add_scalar("grad_norm/policy", metrics["grad_norm/policy"], step)
    if "grad_norm/baseline" in metrics:
        writer.add_scalar("grad_norm/baseline", metrics["grad_norm/baseline"], step)


def evaluate_policy(
    solver: MonteCarloPolicyGradient,
    n_episodes: int = 50,
) -> Dict[str, float]:
    """Evaluate policy on deterministic rollouts."""
    solver.policy.eval()

    # Sample initial states
    initial_states = solver._sample_initial_states(n_episodes)

    # Rollout with deterministic actions
    # Create a wrapper that samples deterministically
    class DeterministicPolicy:
        def __init__(self, policy):
            # Unwrap the adapted policy to get the underlying policy
            self.policy = policy.policy if hasattr(policy, 'policy') else policy

        def act(self, state):
            action, _ = self.policy.sample(state, deterministic=True)
            return action

    with torch.no_grad():
        det_policy = DeterministicPolicy(solver.policy)
        trajectories = solver.simulator.rollout(det_policy, initial_states)

    solver.policy.train()

    return {
        'return_mean': trajectories.returns.mean().item(),
        'return_std': trajectories.returns.std().item(),
        'episode_length': trajectories.masks.sum(dim=-1).mean().item(),
    }


def main():
    """Main training loop."""
    # Parse configuration
    config = parse_args()

    print("=" * 80)
    print("Training Monte Carlo Policy Gradient on GHM Model 1 (Equity Management)")
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
    print("\n[1/6] Setting up GHM Equity dynamics...")
    params = GHMEquityParams()
    dynamics = GHMEquityDynamics(params)
    print(f"  State dimension: {dynamics.state_space.dim}")
    print(f"  Discount rate (ρ = r - μ): {params.r - params.mu:.4f}")

    # =========================================================================
    # 2. Setup control specification
    # =========================================================================
    print("\n[2/6] Setting up control specification...")
    control_spec = GHMControlSpec()
    print(f"  Action dimension: 2 (dividend, equity issuance)")
    print(f"  Action bounds: [{control_spec.lower}, {control_spec.upper}]")

    # =========================================================================
    # 3. Setup reward function
    # =========================================================================
    print("\n[3/6] Setting up reward function...")
    reward_fn = GHMRewardFunction(
        discount_rate=params.r - params.mu,
        issuance_cost=params.lambda_,  # FIXED: Use lambda_ instead of p-1
        liquidation_rate=params.omega,
        liquidation_flow=params.alpha,
    )
    print(f"  Discount rate: {params.r - params.mu:.4f}")
    print(f"  Issuance cost: {params.lambda_:.4f}")

    # =========================================================================
    # 4. Setup policy and baseline networks
    # =========================================================================
    print("\n[4/6] Setting up policy and baseline networks...")
    state_dim = dynamics.state_space.dim
    action_dim = 2  # dividend, equity issuance

    # FIXED: Use actor-critic policy with sigmoid squashing for proper action bounds
    policy = GaussianPolicy(
        input_dim=state_dim,
        output_dim=action_dim,
        hidden_dims=list(config.policy_hidden),
        action_bounds=(control_spec.lower, control_spec.upper),
    ).to(device)

    baseline = None
    if config.use_baseline:
        baseline = ValueNetwork(
            input_dim=state_dim,
            hidden_dims=list(config.value_hidden),
        ).to(device)

    print(f"  Policy hidden dims: {config.policy_hidden}")
    print(f"  Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    if baseline is not None:
        print(f"  Baseline hidden dims: {config.value_hidden}")
        print(f"  Baseline parameters: {sum(p.numel() for p in baseline.parameters())}")
    else:
        print(f"  Baseline: None (pure REINFORCE)")

    # =========================================================================
    # 5. Setup simulator
    # =========================================================================
    print("\n[5/6] Setting up trajectory simulator...")
    simulator = TrajectorySimulator(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        dt=config.dt,
        T=config.T,
    )
    print(f"  dt: {config.dt}, T: {config.T}")
    print(f"  Max steps per episode: {simulator.max_steps}")

    # =========================================================================
    # 6. Setup solver
    # =========================================================================
    print("\n[6/6] Setting up MonteCarloPolicyGradient solver...")

    # Wrap policy with adapter to make it compatible with Monte Carlo solver
    policy_adapted = PolicyAdapter(policy)

    solver = MonteCarloPolicyGradient(
        policy=policy_adapted,
        simulator=simulator,
        baseline=baseline,
        n_trajectories=config.n_trajectories,
        lr_policy=config.lr_policy,
        lr_baseline=config.lr_baseline,
        advantage_normalization=config.advantage_normalization,
        max_grad_norm=config.max_grad_norm,
        entropy_weight=config.entropy_weight,
        action_reg_weight=config.action_reg_weight,
    )

    print(f"  Number of trajectories: {config.n_trajectories}")
    print(f"  Policy learning rate: {config.lr_policy}")
    print(f"  Baseline learning rate: {config.lr_baseline}")
    print(f"  Max grad norm: {config.max_grad_norm}")
    print(f"  Advantage normalization: {config.advantage_normalization}")
    print(f"  Entropy weight: {config.entropy_weight}")
    print(f"  Action reg weight: {config.action_reg_weight}")

    # =========================================================================
    # Setup TensorBoard and save configuration
    # =========================================================================
    print("\nSetting up TensorBoard...")
    writer = create_writer(config)
    print(f"  Log directory: {writer.log_dir}")
    print(f"  Monitor with: tensorboard --logdir {config.log_dir}")

    # Save configuration to checkpoint directory
    print("\nSaving training configuration...")
    save_config_to_file(config, config.ckpt_dir)

    # =========================================================================
    # Resume from checkpoint if provided
    # =========================================================================
    start_step = 1
    best_return = -float('inf')

    if config.resume:
        start_step = load_checkpoint(config.resume, solver, config) + 1
        print(f"\n[Checkpoint] Resuming from step {start_step}")

        # Try to load best_return from checkpoint metadata if available
        try:
            checkpoint = torch.load(config.resume, map_location=config.device)
            if 'best_return' in checkpoint:
                best_return = checkpoint['best_return']
                print(f"[Checkpoint] Restored best_return: {best_return:.4f}")
        except:
            print("[Checkpoint] Could not restore best_return, starting fresh")

    # =========================================================================
    # Training loop
    # =========================================================================
    print("\n" + "=" * 80)
    if start_step == 1:
        print(f"Starting training for {config.n_iterations} iterations...")
    else:
        print(f"Resuming training from step {start_step} to {config.n_iterations}...")
    print("=" * 80)

    # Store dynamics for state sampling
    solver.dynamics = dynamics

    for step in range(start_step, config.n_iterations + 1):
        # Training step
        metrics = solver.train_step()

        # Log training metrics
        if step % config.log_freq == 0:
            log_training_metrics(writer, metrics, step)

            print(f"\n[Step {step}/{config.n_iterations}]")
            print(f"  Return: {metrics['return/mean']:7.3f} ± {metrics['return/std']:6.3f}")
            print(f"  Policy Loss: {metrics['loss/policy']:8.4f}")
            if config.use_baseline:
                print(f"  Baseline Loss: {metrics['loss/baseline']:8.4f}")
            print(f"  Advantage: {metrics['advantage/mean']:7.3f} ± {metrics['advantage/std']:6.3f}")
            print(f"  Entropy: {metrics['policy/entropy']:6.4f}")
            print(f"  Action Magnitude: {metrics['policy/action_magnitude']:6.4f}")

            # DIAGNOSTIC: Check if policy has collapsed to zero
            if metrics['policy/action_magnitude'] < 0.01:
                print(f"  ⚠️  WARNING: Policy may have collapsed! Action magnitude very low.")

            # DIAGNOSTIC: Sample test policy to see what it's doing
            with torch.no_grad():
                test_states = torch.linspace(0.1, 2.0, 10).unsqueeze(1).to(device)
                test_actions, _ = policy.sample(test_states, deterministic=True)
                print(f"  Test policy outputs (c=0.5-2.0):")
                print(f"    Dividend (a_L): {test_actions[:, 0].cpu().numpy()}")
                print(f"    Equity (a_E):   {test_actions[:, 1].cpu().numpy()}")

        # Evaluation
        if step % config.eval_freq == 0:
            print(f"\n[Evaluation at step {step}]")
            eval_metrics = evaluate_policy(solver, n_episodes=50)

            writer.add_scalar("eval/return_mean", eval_metrics['return_mean'], step)
            writer.add_scalar("eval/return_std", eval_metrics['return_std'], step)
            writer.add_scalar("eval/episode_length", eval_metrics['episode_length'], step)

            print(f"  Eval Return: {eval_metrics['return_mean']:.4f} ± {eval_metrics['return_std']:.4f}")
            print(f"  Eval Episode Length: {eval_metrics['episode_length']:.2f}")

            # Generate and log policy/value visualizations
            log_policy_value_visualization(writer, solver.policy, baseline, dynamics, step, config)

            # Save best model
            if eval_metrics['return_mean'] > best_return:
                best_return = eval_metrics['return_mean']
                save_checkpoint(solver, config, step, ckpt_name="best_model.pt", best_return=best_return)
                print(f"  New best model! Return: {best_return:.4f}")
            print()

        # Save periodic checkpoint
        if step % config.ckpt_freq == 0:
            save_checkpoint(solver, config, step, best_return=best_return)

    # =========================================================================
    # Save final model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    save_checkpoint(solver, config, config.n_iterations, ckpt_name="final_model.pt", best_return=best_return)

    # Final evaluation
    print("\nFinal evaluation...")
    final_eval = evaluate_policy(solver, n_episodes=100)
    print(f"  Final Return: {final_eval['return_mean']:.4f} ± {final_eval['return_std']:.4f}")
    print(f"  Best Return: {best_return:.4f}")

    writer.close()
    print(f"\nTensorBoard logs saved to: {writer.log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
