"""
Compare RL and numerical solutions for GHM equity model.

This script loads both a trained RL model and a numerical VFI solution,
then creates side-by-side visualizations and computes comparison metrics.

Usage:
    python scripts/compare_rl_numerical.py \
        --rl-checkpoint checkpoints/ghm_time_augmented_sparse/policy_step_5000.pt \
        --numerical-solution numerical_benchmark_results/vfi_solution.npz \
        --config configs/time_augmented_sparse_config.yaml
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from macro_rl.dynamics.ghm_equity import GHMEquityParams, GHMEquityTimeAugmentedDynamics
from macro_rl.networks.policy import GaussianPolicy
from macro_rl.networks.actor_critic import ActorCritic
from macro_rl.solvers.monte_carlo_evaluator import MonteCarloEvaluator, MonteCarloConfig, RLPolicyWrapper


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_rl_policy(checkpoint_path: str, state_dim: int, action_dim: int, config: dict):
    """Load trained RL policy from checkpoint."""
    # Try to load as ActorCritic first
    try:
        network_config = config['network']
        model = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=network_config.get('hidden_dims', [64, 64]),
            shared_layers=network_config.get('shared_layers', 0),
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'actor_critic_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['actor_critic_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model.actor  # Return actor part
    except:
        # Fall back to GaussianPolicy
        network_config = config['network']
        policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=network_config.get('policy_hidden', [64, 64]),
            activation=network_config.get('policy_activation', 'relu'),
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'policy_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            policy.load_state_dict(checkpoint)

        policy.eval()
        return policy


def compute_rl_policy_on_grid(policy, c_grid, tau_grid, device='cpu'):
    """Evaluate RL policy on a grid."""
    n_c = len(c_grid)
    n_tau = len(tau_grid)

    policy_dividend = np.zeros((n_c, n_tau))
    policy_equity = np.zeros((n_c, n_tau))

    for i, c in enumerate(c_grid):
        for j, tau in enumerate(tau_grid):
            state = torch.tensor([[c, tau]], dtype=torch.float32).to(device)
            with torch.no_grad():
                action, _ = policy.sample(state, deterministic=True)
                action = action.cpu().numpy()[0]

            policy_dividend[i, j] = action[0]
            policy_equity[i, j] = action[1]

    return policy_dividend, policy_equity


def create_comparison_visualization(
    c_grid,
    tau_grid,
    numerical_dividend,
    numerical_equity,
    numerical_value,
    rl_dividend,
    rl_equity,
    rl_value=None,
    save_path=None
):
    """Create side-by-side comparison visualization."""
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.4)

    tau_slices = [0.5, 2.5, 5.0, 7.5, 10.0]

    # Row 1: Numerical Solution
    # Dividend heatmap
    ax1 = fig.add_subplot(gs[0, 0:2])
    im1 = ax1.contourf(c_grid, tau_grid, numerical_dividend.T, levels=20, cmap='viridis')
    ax1.set_xlabel('Cash Reserves (c)')
    ax1.set_ylabel('Time-to-Horizon (τ)')
    ax1.set_title('Numerical: Dividend Policy π(c, τ)', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Dividend')

    # Equity heatmap
    ax2 = fig.add_subplot(gs[0, 2:4])
    im2 = ax2.contourf(c_grid, tau_grid, numerical_equity.T, levels=20, cmap='plasma')
    ax2.set_xlabel('Cash Reserves (c)')
    ax2.set_ylabel('Time-to-Horizon (τ)')
    ax2.set_title('Numerical: Equity Issuance π(c, τ)', fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Equity')

    # Value heatmap
    ax3 = fig.add_subplot(gs[0, 4:6])
    im3 = ax3.contourf(c_grid, tau_grid, numerical_value.T, levels=20, cmap='coolwarm')
    ax3.set_xlabel('Cash Reserves (c)')
    ax3.set_ylabel('Time-to-Horizon (τ)')
    ax3.set_title('Numerical: Value Function V(c, τ)', fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Value')

    # Row 2: RL Solution
    # Dividend heatmap
    ax4 = fig.add_subplot(gs[1, 0:2])
    im4 = ax4.contourf(c_grid, tau_grid, rl_dividend.T, levels=20, cmap='viridis')
    ax4.set_xlabel('Cash Reserves (c)')
    ax4.set_ylabel('Time-to-Horizon (τ)')
    ax4.set_title('RL: Dividend Policy π(c, τ)', fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Dividend')

    # Equity heatmap
    ax5 = fig.add_subplot(gs[1, 2:4])
    im5 = ax5.contourf(c_grid, tau_grid, rl_equity.T, levels=20, cmap='plasma')
    ax5.set_xlabel('Cash Reserves (c)')
    ax5.set_ylabel('Time-to-Horizon (τ)')
    ax5.set_title('RL: Equity Issuance π(c, τ)', fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='Equity')

    # Value heatmap (if available)
    ax6 = fig.add_subplot(gs[1, 4:6])
    if rl_value is not None:
        im6 = ax6.contourf(c_grid, tau_grid, rl_value.T, levels=20, cmap='coolwarm')
        ax6.set_xlabel('Cash Reserves (c)')
        ax6.set_ylabel('Time-to-Horizon (τ)')
        ax6.set_title('RL: Value Function V(c, τ)', fontweight='bold')
        plt.colorbar(im6, ax=ax6, label='Value')
    else:
        ax6.text(0.5, 0.5, 'RL Value\nNot Available',
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)
        ax6.set_xticks([])
        ax6.set_yticks([])

    fig.suptitle('Comparison: Numerical VFI vs RL Policy', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")

    return fig


def create_difference_plots(
    c_grid,
    tau_grid,
    numerical_dividend,
    numerical_equity,
    numerical_value,
    rl_dividend,
    rl_equity,
    rl_value=None,
    save_path=None
):
    """Create difference plots showing (RL - Numerical)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Dividend difference
    diff_dividend = rl_dividend - numerical_dividend
    im1 = axes[0].contourf(c_grid, tau_grid, diff_dividend.T, levels=20, cmap='RdBu_r',
                           vmin=-np.abs(diff_dividend).max(), vmax=np.abs(diff_dividend).max())
    axes[0].set_xlabel('Cash Reserves (c)')
    axes[0].set_ylabel('Time-to-Horizon (τ)')
    axes[0].set_title('Dividend Difference (RL - Numerical)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Difference')

    # Equity difference
    diff_equity = rl_equity - numerical_equity
    im2 = axes[1].contourf(c_grid, tau_grid, diff_equity.T, levels=20, cmap='RdBu_r',
                           vmin=-np.abs(diff_equity).max(), vmax=np.abs(diff_equity).max())
    axes[1].set_xlabel('Cash Reserves (c)')
    axes[1].set_ylabel('Time-to-Horizon (τ)')
    axes[1].set_title('Equity Difference (RL - Numerical)', fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Difference')

    # Value difference
    if rl_value is not None:
        diff_value = rl_value - numerical_value
        im3 = axes[2].contourf(c_grid, tau_grid, diff_value.T, levels=20, cmap='RdBu_r',
                              vmin=-np.abs(diff_value).max(), vmax=np.abs(diff_value).max())
        axes[2].set_xlabel('Cash Reserves (c)')
        axes[2].set_ylabel('Time-to-Horizon (τ)')
        axes[2].set_title('Value Difference (RL - Numerical)', fontweight='bold')
        plt.colorbar(im3, ax=axes[2], label='Difference')
    else:
        axes[2].text(0.5, 0.5, 'Value Difference\nNot Available',
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved difference plots to {save_path}")

    return fig


def compute_metrics(numerical_dividend, numerical_equity, numerical_value,
                   rl_dividend, rl_equity, rl_value=None):
    """Compute comparison metrics."""
    metrics = {}

    # Policy differences
    metrics['dividend_mae'] = np.mean(np.abs(rl_dividend - numerical_dividend))
    metrics['dividend_rmse'] = np.sqrt(np.mean((rl_dividend - numerical_dividend)**2))
    metrics['dividend_max_diff'] = np.max(np.abs(rl_dividend - numerical_dividend))

    metrics['equity_mae'] = np.mean(np.abs(rl_equity - numerical_equity))
    metrics['equity_rmse'] = np.sqrt(np.mean((rl_equity - numerical_equity)**2))
    metrics['equity_max_diff'] = np.max(np.abs(rl_equity - numerical_equity))

    if rl_value is not None:
        metrics['value_mae'] = np.mean(np.abs(rl_value - numerical_value))
        metrics['value_rmse'] = np.sqrt(np.mean((rl_value - numerical_value)**2))
        metrics['value_max_diff'] = np.max(np.abs(rl_value - numerical_value))
        metrics['value_relative_error'] = np.mean(np.abs((rl_value - numerical_value) / (numerical_value + 1e-8)))

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compare RL and numerical solutions')
    parser.add_argument('--rl-checkpoint', type=str, required=True,
                       help='Path to RL model checkpoint')
    parser.add_argument('--numerical-solution', type=str, required=True,
                       help='Path to numerical VFI solution (.npz file)')
    parser.add_argument('--config', type=str, default='configs/time_augmented_sparse_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for comparison plots')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for RL policy evaluation')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Load numerical solution
    print(f"Loading numerical solution from {args.numerical_solution}")
    numerical_data = np.load(args.numerical_solution)

    c_grid = numerical_data['c_grid']
    tau_grid = numerical_data['tau_grid']
    numerical_dividend = numerical_data['policy_dividend']
    numerical_equity = numerical_data['policy_equity']
    numerical_value = numerical_data['V']

    print(f"Numerical grid: {len(c_grid)} × {len(tau_grid)}")

    # Load RL policy
    print(f"Loading RL policy from {args.rl_checkpoint}")
    state_dim = 2  # (c, tau)
    action_dim = 2  # (dividend, equity)
    rl_policy = load_rl_policy(args.rl_checkpoint, state_dim, action_dim, config)

    # Evaluate RL policy on grid
    print("Evaluating RL policy on grid...")
    rl_dividend, rl_equity = compute_rl_policy_on_grid(
        rl_policy, c_grid, tau_grid, device=args.device
    )

    # TODO: Load RL value function if available
    rl_value = None

    # Compute metrics
    print("\nComputing comparison metrics...")
    metrics = compute_metrics(
        numerical_dividend, numerical_equity, numerical_value,
        rl_dividend, rl_equity, rl_value
    )

    print("\n" + "="*60)
    print("COMPARISON METRICS")
    print("="*60)
    print(f"\nDividend Policy:")
    print(f"  MAE:      {metrics['dividend_mae']:.6f}")
    print(f"  RMSE:     {metrics['dividend_rmse']:.6f}")
    print(f"  Max Diff: {metrics['dividend_max_diff']:.6f}")

    print(f"\nEquity Policy:")
    print(f"  MAE:      {metrics['equity_mae']:.6f}")
    print(f"  RMSE:     {metrics['equity_rmse']:.6f}")
    print(f"  Max Diff: {metrics['equity_max_diff']:.6f}")

    if rl_value is not None:
        print(f"\nValue Function:")
        print(f"  MAE:            {metrics['value_mae']:.6f}")
        print(f"  RMSE:           {metrics['value_rmse']:.6f}")
        print(f"  Max Diff:       {metrics['value_max_diff']:.6f}")
        print(f"  Relative Error: {metrics['value_relative_error']:.2%}")

    print("="*60)

    # Save metrics
    metrics_path = output_dir / 'comparison_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("COMPARISON METRICS\n")
        f.write("="*60 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    print(f"\nMetrics saved to {metrics_path}")

    # Create visualizations
    print("\nCreating comparison visualizations...")

    fig1 = create_comparison_visualization(
        c_grid, tau_grid,
        numerical_dividend, numerical_equity, numerical_value,
        rl_dividend, rl_equity, rl_value,
        save_path=output_dir / 'comparison_heatmaps.png'
    )

    fig2 = create_difference_plots(
        c_grid, tau_grid,
        numerical_dividend, numerical_equity, numerical_value,
        rl_dividend, rl_equity, rl_value,
        save_path=output_dir / 'difference_plots.png'
    )

    print(f"\nAll results saved to {output_dir}")
    print("Done!")

    plt.show()


if __name__ == '__main__':
    main()
