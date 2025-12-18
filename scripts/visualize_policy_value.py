#!/usr/bin/env python3
"""
Visualization script for GHM Actor-Critic Policy and Value Functions.

This script loads a trained model checkpoint and visualizes:
1. Policy function: Mean actions (dividend payout a_L, equity issuance a_E) across state space
2. Value function: V(c) across the cash reserves ratio state space
3. Policy standard deviations
4. HJB residuals for model validation
5. Action distribution samples
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from macro_rl.core.state_space import StateSpace
from macro_rl.dynamics.ghm_equity import GHMEquityDynamics
from macro_rl.networks.actor_critic import ActorCritic


def load_checkpoint(checkpoint_path: str):
    """Load model checkpoint and reconstruct actor-critic network."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    actor_critic_state = checkpoint['actor_critic_state']

    # Reconstruct actor-critic network
    state_dim = config.get('state_dim', 1)
    action_dim = config.get('action_dim', 2)
    hidden_dims = config.get('hidden_dims', [256, 256])
    shared_layers = config.get('shared_layers', 1)

    # Check if action bounds are in the checkpoint
    action_bounds = None
    if 'actor.action_low' in actor_critic_state and 'actor.action_high' in actor_critic_state:
        action_low = actor_critic_state['actor.action_low']
        action_high = actor_critic_state['actor.action_high']
        action_bounds = (action_low, action_high)

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        shared_layers=shared_layers,
        action_bounds=action_bounds
    )
    ac.load_state_dict(actor_critic_state)
    ac.eval()

    print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    print(f"  Step: {checkpoint['step']}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    print(f"  Hidden dims: {hidden_dims}, Shared layers: {shared_layers}")

    return ac, config


def compute_policy_value_grid(ac: ActorCritic, state_space: StateSpace, n_points: int = 100):
    """Compute policy and value function on a grid over the state space."""
    # Create state grid
    c_values = np.linspace(state_space.lower[0].item(), state_space.upper[0].item(), n_points)
    states = torch.tensor(c_values, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        # Get deterministic actions (mean of policy)
        actions_mean = ac.act(states, deterministic=True)

        # Get value estimates
        values = ac.evaluate(states).squeeze()

        # Get policy distribution parameters (need to pass through shared features)
        feat = ac._features(states)
        mean, log_std = ac.actor._get_mean_log_std(feat)
        # log_std is state-independent, so expand it to match batch size
        actions_std = log_std.exp().expand_as(mean)

    # Get value gradients and Hessian diagonal (for HJB residuals)
    # Note: This requires gradients enabled, so it's outside the no_grad context
    values_grad, V_s, V_ss_diag = ac.evaluate_with_grad(states)

    return {
        'c_values': c_values,
        'actions_mean': actions_mean.numpy(),
        'actions_std': actions_std.numpy(),
        'values': values.numpy(),
        'V_s': V_s.detach().numpy().squeeze(),
        'V_ss_diag': V_ss_diag.detach().numpy().squeeze(),
    }


def compute_hjb_residuals(results: dict, dynamics: GHMEquityDynamics, dt: float = 0.01):
    """Compute HJB residuals for validation of the value function."""
    c_values = results['c_values']
    actions = results['actions_mean']
    V = results['values']
    V_s = results['V_s']
    V_ss = results['V_ss_diag']

    # Convert to tensors
    states = torch.tensor(c_values, dtype=torch.float32).unsqueeze(1)
    actions_t = torch.tensor(actions, dtype=torch.float32)
    V_t = torch.tensor(V, dtype=torch.float32).unsqueeze(1)
    V_s_t = torch.tensor(V_s, dtype=torch.float32).unsqueeze(1)
    V_ss_t = torch.tensor(V_ss, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        # Compute drift and diffusion
        drift = dynamics.drift(states, actions_t)  # [N, 1]
        diffusion = dynamics.diffusion(states)  # [N, 1]

        # Reward
        a_L = actions_t[:, 0:1]
        a_E = actions_t[:, 1:2]
        reward = a_L * dt - (1 + dynamics.p.lambda_) * a_E  # [N, 1]

        # HJB residual: r(s,a) + V_s·f(s,a) + 0.5·V_ss·σ²(s,a) - ρ·V
        discount_rate = dynamics.p.r - dynamics.p.mu
        hjb_residual = (
            reward +
            V_s_t * drift +
            0.5 * V_ss_t * diffusion.pow(2) -
            discount_rate * V_t
        )

    return hjb_residual.squeeze().numpy()


def sample_action_distributions(ac: ActorCritic, states_to_sample: np.ndarray, n_samples: int = 1000):
    """Sample actions from the policy at specific states."""
    states = torch.tensor(states_to_sample, dtype=torch.float32)

    with torch.no_grad():
        # Sample actions multiple times by repeating states
        states_repeated = states.repeat(n_samples, 1)  # [n_samples * n_states, state_dim]
        action_samples = ac.act(states_repeated, deterministic=False)  # [n_samples * n_states, action_dim]
        # Reshape to [n_samples, n_states, action_dim]
        action_samples = action_samples.reshape(n_samples, len(states_to_sample), -1)

    return action_samples.numpy()


def create_visualizations(results: dict, hjb_residuals: np.ndarray,
                          checkpoint_path: str, output_dir: str):
    """Create comprehensive visualization plots."""
    c_values = results['c_values']
    actions_mean = results['actions_mean']
    actions_std = results['actions_std']
    values = results['values']
    V_s = results['V_s']

    # Setup seaborn style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # === Plot 1: Value Function ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(c_values, values, 'b-', linewidth=2)
    ax1.set_xlabel('Cash Reserves Ratio (c)', fontsize=11)
    ax1.set_ylabel('Value Function V(c)', fontsize=11)
    ax1.set_title('Value Function', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Value Function Gradient ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(c_values, V_s, 'g-', linewidth=2)
    ax2.set_xlabel('Cash Reserves Ratio (c)', fontsize=11)
    ax2.set_ylabel('Value Gradient V\'(c)', fontsize=11)
    ax2.set_title('Value Function Gradient', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # === Plot 3: HJB Residuals ===
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(c_values, hjb_residuals, 'r-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Cash Reserves Ratio (c)', fontsize=11)
    ax3.set_ylabel('HJB Residual', fontsize=11)
    ax3.set_title('HJB Residuals (Validation)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # === Plot 4: Dividend Payout Policy (a_L) ===
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(c_values, actions_mean[:, 0], 'b-', linewidth=2, label='Mean')
    ax4.fill_between(c_values,
                     actions_mean[:, 0] - actions_std[:, 0],
                     actions_mean[:, 0] + actions_std[:, 0],
                     alpha=0.3, label='±1 std')
    ax4.set_xlabel('Cash Reserves Ratio (c)', fontsize=11)
    ax4.set_ylabel('Dividend Payout Rate (a_L)', fontsize=11)
    ax4.set_title('Policy: Dividend Payout', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # === Plot 5: Equity Issuance Policy (a_E) ===
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(c_values, actions_mean[:, 1], 'r-', linewidth=2, label='Mean')
    ax5.fill_between(c_values,
                     actions_mean[:, 1] - actions_std[:, 1],
                     actions_mean[:, 1] + actions_std[:, 1],
                     alpha=0.3, label='±1 std')
    ax5.set_xlabel('Cash Reserves Ratio (c)', fontsize=11)
    ax5.set_ylabel('Equity Issuance (a_E)', fontsize=11)
    ax5.set_title('Policy: Equity Issuance', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # === Plot 6: Policy Standard Deviations ===
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(c_values, actions_std[:, 0], 'b-', linewidth=2, label='σ(a_L)')
    ax6.plot(c_values, actions_std[:, 1], 'r-', linewidth=2, label='σ(a_E)')
    ax6.set_xlabel('Cash Reserves Ratio (c)', fontsize=11)
    ax6.set_ylabel('Policy Std Dev', fontsize=11)
    ax6.set_title('Policy Uncertainty', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # === Plot 7: Combined Policy View ===
    ax7 = fig.add_subplot(gs[2, :2])
    ax7_twin = ax7.twinx()

    line1 = ax7.plot(c_values, actions_mean[:, 0], 'b-', linewidth=2, label='Dividend Payout (a_L)')
    line2 = ax7_twin.plot(c_values, actions_mean[:, 1], 'r-', linewidth=2, label='Equity Issuance (a_E)')

    ax7.set_xlabel('Cash Reserves Ratio (c)', fontsize=11)
    ax7.set_ylabel('Dividend Payout Rate (a_L)', fontsize=11, color='b')
    ax7_twin.set_ylabel('Equity Issuance (a_E)', fontsize=11, color='r')
    ax7.tick_params(axis='y', labelcolor='b')
    ax7_twin.tick_params(axis='y', labelcolor='r')
    ax7.set_title('Combined Policy View', fontsize=12, fontweight='bold')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left')
    ax7.grid(True, alpha=0.3)

    # === Plot 8: State-Action Value Surface ===
    ax8 = fig.add_subplot(gs[2, 2])
    # Create a small heatmap showing which actions are taken at which states
    n_sample_states = 20
    sample_indices = np.linspace(0, len(c_values)-1, n_sample_states, dtype=int)
    sample_c = c_values[sample_indices]
    sample_actions = actions_mean[sample_indices]

    # Create color-coded scatter
    scatter = ax8.scatter(sample_actions[:, 0], sample_actions[:, 1],
                          c=sample_c, cmap='viridis', s=100, alpha=0.7)
    ax8.set_xlabel('Dividend Payout (a_L)', fontsize=11)
    ax8.set_ylabel('Equity Issuance (a_E)', fontsize=11)
    ax8.set_title('Action Space Coverage', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax8)
    cbar.set_label('Cash Ratio (c)', fontsize=10)
    ax8.grid(True, alpha=0.3)

    # Add main title
    fig.suptitle(f'GHM Actor-Critic: Policy and Value Function Visualization\n{Path(checkpoint_path).name}',
                 fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    output_path = os.path.join(output_dir, 'policy_value_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    return fig


def create_action_distribution_plots(ac: ActorCritic, state_space: StateSpace, output_dir: str):
    """Create detailed action distribution visualizations at key states."""
    # Sample at 3 key states: low, medium, high cash
    key_states = np.array([
        [0.2],   # Low cash
        [1.0],   # Medium cash
        [1.8],   # High cash
    ])

    state_labels = ['Low Cash (c=0.2)', 'Medium Cash (c=1.0)', 'High Cash (c=1.8)']

    # Sample actions
    action_samples = sample_action_distributions(ac, key_states, n_samples=5000)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Action Distribution Samples at Key States', fontsize=14, fontweight='bold')

    for i, (state, label) in enumerate(zip(key_states, state_labels)):
        # Dividend payout distribution
        ax_L = axes[0, i]
        ax_L.hist(action_samples[:, i, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax_L.set_xlabel('Dividend Payout (a_L)', fontsize=10)
        ax_L.set_ylabel('Frequency', fontsize=10)
        ax_L.set_title(f'{label}\nDividend Distribution', fontsize=11)
        ax_L.grid(True, alpha=0.3)

        # Equity issuance distribution
        ax_E = axes[1, i]
        ax_E.hist(action_samples[:, i, 1], bins=50, alpha=0.7, color='red', edgecolor='black')
        ax_E.set_xlabel('Equity Issuance (a_E)', fontsize=10)
        ax_E.set_ylabel('Frequency', fontsize=10)
        ax_E.set_title(f'{label}\nEquity Issuance Distribution', fontsize=11)
        ax_E.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'action_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved action distributions to: {output_path}")

    return fig


def print_summary_statistics(results: dict, hjb_residuals: np.ndarray):
    """Print summary statistics of the policy and value function."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print("\nValue Function:")
    print(f"  Mean V(c):      {results['values'].mean():.4f}")
    print(f"  Min V(c):       {results['values'].min():.4f} at c={results['c_values'][results['values'].argmin()]:.3f}")
    print(f"  Max V(c):       {results['values'].max():.4f} at c={results['c_values'][results['values'].argmax()]:.3f}")
    print(f"  Mean V'(c):     {results['V_s'].mean():.4f}")

    print("\nPolicy - Dividend Payout (a_L):")
    print(f"  Mean:           {results['actions_mean'][:, 0].mean():.4f}")
    print(f"  Std:            {results['actions_mean'][:, 0].std():.4f}")
    print(f"  Min:            {results['actions_mean'][:, 0].min():.4f}")
    print(f"  Max:            {results['actions_mean'][:, 0].max():.4f}")
    print(f"  Mean σ(a_L):    {results['actions_std'][:, 0].mean():.4f}")

    print("\nPolicy - Equity Issuance (a_E):")
    print(f"  Mean:           {results['actions_mean'][:, 1].mean():.4f}")
    print(f"  Std:            {results['actions_mean'][:, 1].std():.4f}")
    print(f"  Min:            {results['actions_mean'][:, 1].min():.4f}")
    print(f"  Max:            {results['actions_mean'][:, 1].max():.4f}")
    print(f"  Mean σ(a_E):    {results['actions_std'][:, 1].mean():.4f}")

    print("\nHJB Residuals (Validation):")
    print(f"  Mean:           {hjb_residuals.mean():.6f}")
    print(f"  Std:            {hjb_residuals.std():.6f}")
    print(f"  Mean Abs:       {np.abs(hjb_residuals).mean():.6f}")
    print(f"  Max Abs:        {np.abs(hjb_residuals).max():.6f}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GHM Actor-Critic Policy and Value Functions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/ghm_model1/final_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Directory to save visualization plots'
    )
    parser.add_argument(
        '--n-points',
        type=int,
        default=200,
        help='Number of points to sample in state space'
    )
    parser.add_argument(
        '--c-max',
        type=float,
        default=2.0,
        help='Maximum cash reserves ratio'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("GHM ACTOR-CRITIC VISUALIZATION")
    print("="*70)

    # Load checkpoint
    ac, config = load_checkpoint(args.checkpoint)

    # Setup environment components
    from macro_rl.dynamics.ghm_equity import GHMEquityParams
    params = GHMEquityParams(c_max=args.c_max)
    dynamics = GHMEquityDynamics(params)
    state_space = dynamics.state_space

    print(f"\n✓ State space: c ∈ [0, {args.c_max}]")
    print(f"✓ Sampling {args.n_points} points")

    # Compute policy and value function on grid
    print(f"\nComputing policy and value function...")
    results = compute_policy_value_grid(ac, state_space, n_points=args.n_points)

    # Compute HJB residuals
    print("Computing HJB residuals...")
    hjb_residuals = compute_hjb_residuals(results, dynamics)

    # Print summary statistics
    print_summary_statistics(results, hjb_residuals)

    # Create visualizations
    print(f"\nGenerating visualizations...")
    create_visualizations(results, hjb_residuals, args.checkpoint, args.output_dir)
    create_action_distribution_plots(ac, state_space, args.output_dir)

    print("\n✓ Visualization complete!")
    print(f"✓ All plots saved to: {args.output_dir}/")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
