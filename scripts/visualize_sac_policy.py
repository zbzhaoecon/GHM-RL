#!/usr/bin/env python3
"""
Visualization script for SAC-trained GHM Policy.

This script loads a trained SAC model and visualizes:
1. Learned policy: Mean actions (dividend, equity) vs cash state
2. Episode trajectories showing cash evolution and actions taken
3. Barrier policy approximation analysis
4. Action distributions at different cash levels
5. Performance metrics and statistics

Designed for models trained with Stable-Baselines3 SAC.

Usage:
    python scripts/visualize_sac_policy.py --model models/ghm_equity/final_model
    python scripts/visualize_sac_policy.py --model models/best_model --n-episodes 10
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from stable_baselines3 import SAC
from macro_rl.envs import GHMEquityEnv


def load_sac_model(model_path: str):
    """Load trained SAC model."""
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    model = SAC.load(model_path)
    print(f"✓ Loaded SAC model from: {model_path}.zip")

    return model


def compute_policy_over_states(model, env, n_points=100, deterministic=True):
    """Compute policy actions over a grid of cash states."""
    c_min = env.observation_space.low[0]
    c_max = env.observation_space.high[0]

    c_values = np.linspace(c_min, c_max, n_points)

    dividends_mean = []
    equity_mean = []
    dividends_samples = []
    equity_samples = []

    for c in c_values:
        obs = np.array([c], dtype=np.float32)

        # Get deterministic action (mean)
        action_det, _ = model.predict(obs, deterministic=True)
        dividends_mean.append(action_det[0])
        equity_mean.append(action_det[1])

        # Get stochastic samples
        samples = []
        for _ in range(100):
            action_stoch, _ = model.predict(obs, deterministic=False)
            samples.append(action_stoch)
        samples = np.array(samples)

        dividends_samples.append(samples[:, 0])
        equity_samples.append(samples[:, 1])

    return {
        'c_values': c_values,
        'dividends_mean': np.array(dividends_mean),
        'equity_mean': np.array(equity_mean),
        'dividends_std': np.array([np.std(s) for s in dividends_samples]),
        'equity_std': np.array([np.std(s) for s in equity_samples]),
        'dividends_samples': dividends_samples,
        'equity_samples': equity_samples,
    }


def run_episodes(model, env, n_episodes=10, deterministic=True):
    """Run multiple episodes and collect trajectories."""
    trajectories = []

    for ep in range(n_episodes):
        obs, _ = env.reset()

        trajectory = {
            'cash': [obs[0]],
            'dividends': [],
            'equity': [],
            'rewards': [],
            'terminated': False,
            'steps': 0,
        }

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            trajectory['cash'].append(obs[0])
            trajectory['dividends'].append(action[0])
            trajectory['equity'].append(action[1])
            trajectory['rewards'].append(reward)
            trajectory['steps'] += 1

            if terminated or truncated:
                trajectory['terminated'] = terminated
                break

        trajectories.append(trajectory)

    return trajectories


def analyze_barrier_structure(policy_results, threshold_percentile=10):
    """Analyze if the policy approximates a barrier structure."""
    c_values = policy_results['c_values']
    dividends = policy_results['dividends_mean']
    equity = policy_results['equity_mean']

    # Find potential dividend barrier (where dividends become significant)
    div_threshold = np.percentile(dividends[dividends > 0], threshold_percentile) if any(dividends > 0) else 0.01
    dividend_barrier_idx = np.where(dividends > div_threshold)[0]
    dividend_barrier = c_values[dividend_barrier_idx[0]] if len(dividend_barrier_idx) > 0 else None

    # Find equity issuance region (low cash)
    equity_threshold = np.percentile(equity[equity > 0], 90) if any(equity > 0) else 0.01
    equity_region_idx = np.where(equity > equity_threshold)[0]
    equity_region = (c_values[equity_region_idx[0]], c_values[equity_region_idx[-1]]) if len(equity_region_idx) > 0 else None

    # Inaction region (both actions near zero)
    inaction_threshold = 0.05
    inaction_idx = np.where((dividends < inaction_threshold) & (equity < inaction_threshold))[0]
    inaction_region = (c_values[inaction_idx[0]], c_values[inaction_idx[-1]]) if len(inaction_idx) > 0 else None

    return {
        'dividend_barrier': dividend_barrier,
        'equity_region': equity_region,
        'inaction_region': inaction_region,
    }


def create_policy_visualization(policy_results, barrier_analysis, output_dir):
    """Create comprehensive policy visualization."""
    c_values = policy_results['c_values']
    dividends_mean = policy_results['dividends_mean']
    equity_mean = policy_results['equity_mean']
    dividends_std = policy_results['dividends_std']
    equity_std = policy_results['equity_std']

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # === Plot 1: Dividend Policy ===
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(c_values, dividends_mean, 'b-', linewidth=2.5, label='Mean Policy')
    ax1.fill_between(c_values,
                     dividends_mean - dividends_std,
                     dividends_mean + dividends_std,
                     alpha=0.25, color='blue', label='±1 Std')

    # Mark barrier if found
    if barrier_analysis['dividend_barrier'] is not None:
        ax1.axvline(barrier_analysis['dividend_barrier'], color='red',
                   linestyle='--', linewidth=2, label=f'Barrier c* ≈ {barrier_analysis["dividend_barrier"]:.3f}')

    ax1.set_xlabel('Cash Reserves Ratio (c)', fontsize=12)
    ax1.set_ylabel('Dividend Amount', fontsize=12)
    ax1.set_title('Learned Dividend Policy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Equity Issuance Policy ===
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(c_values, equity_mean, 'r-', linewidth=2.5, label='Mean Policy')
    ax2.fill_between(c_values,
                     equity_mean - equity_std,
                     equity_mean + equity_std,
                     alpha=0.25, color='red', label='±1 Std')

    # Mark equity region if found
    if barrier_analysis['equity_region'] is not None:
        ax2.axvspan(barrier_analysis['equity_region'][0],
                   barrier_analysis['equity_region'][1],
                   alpha=0.2, color='yellow', label='Equity Region')

    ax2.set_xlabel('Cash Reserves Ratio (c)', fontsize=12)
    ax2.set_ylabel('Equity Issuance Amount', fontsize=12)
    ax2.set_title('Learned Equity Policy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # === Plot 3: Combined Policy (Dual Y-axis) ===
    ax3 = fig.add_subplot(gs[1, :])
    ax3_twin = ax3.twinx()

    line1 = ax3.plot(c_values, dividends_mean, 'b-', linewidth=2.5, label='Dividend')
    line2 = ax3_twin.plot(c_values, equity_mean, 'r-', linewidth=2.5, label='Equity')

    # Shade regions
    if barrier_analysis['dividend_barrier'] is not None:
        ax3.axvline(barrier_analysis['dividend_barrier'], color='green',
                   linestyle='--', linewidth=1.5, alpha=0.7)
    if barrier_analysis['inaction_region'] is not None:
        ax3.axvspan(barrier_analysis['inaction_region'][0],
                   barrier_analysis['inaction_region'][1],
                   alpha=0.15, color='gray', label='Inaction Region')

    ax3.set_xlabel('Cash Reserves Ratio (c)', fontsize=12)
    ax3.set_ylabel('Dividend Amount', fontsize=12, color='b')
    ax3_twin.set_ylabel('Equity Issuance Amount', fontsize=12, color='r')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3.set_title('Combined Policy View (Barrier Structure)', fontsize=13, fontweight='bold')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # === Plot 4: Policy Uncertainty (Std Dev) ===
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(c_values, dividends_std, 'b-', linewidth=2, label='σ(dividend)')
    ax4.plot(c_values, equity_std, 'r-', linewidth=2, label='σ(equity)')
    ax4.set_xlabel('Cash Reserves Ratio (c)', fontsize=12)
    ax4.set_ylabel('Policy Std Dev', fontsize=12)
    ax4.set_title('Policy Uncertainty', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # === Plot 5: Action Space Coverage ===
    ax5 = fig.add_subplot(gs[2, 1])
    n_sample = 30
    sample_idx = np.linspace(0, len(c_values)-1, n_sample, dtype=int)
    scatter = ax5.scatter(dividends_mean[sample_idx], equity_mean[sample_idx],
                         c=c_values[sample_idx], cmap='viridis', s=100, alpha=0.8,
                         edgecolors='black', linewidths=0.5)
    ax5.set_xlabel('Dividend Amount', fontsize=12)
    ax5.set_ylabel('Equity Issuance Amount', fontsize=12)
    ax5.set_title('Action Space Coverage', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Cash (c)', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # === Plot 6: Net Payout to Shareholders ===
    ax6 = fig.add_subplot(gs[2, 2])
    net_payout = dividends_mean - equity_mean
    ax6.plot(c_values, net_payout, 'g-', linewidth=2.5)
    ax6.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax6.fill_between(c_values, 0, net_payout, where=(net_payout > 0),
                     alpha=0.3, color='green', label='Net Positive')
    ax6.fill_between(c_values, 0, net_payout, where=(net_payout < 0),
                     alpha=0.3, color='red', label='Net Negative')
    ax6.set_xlabel('Cash Reserves Ratio (c)', fontsize=12)
    ax6.set_ylabel('Net Payout (Dividend - Equity)', fontsize=12)
    ax6.set_title('Net Shareholder Value', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    fig.suptitle('SAC-Learned Policy for GHM Equity Management',
                 fontsize=15, fontweight='bold', y=0.995)

    output_path = os.path.join(output_dir, 'sac_policy_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved policy visualization to: {output_path}")

    return fig


def create_trajectory_visualization(trajectories, env, output_dir):
    """Visualize episode trajectories."""
    n_trajectories = min(len(trajectories), 5)  # Show at most 5

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Episode Trajectories (n={n_trajectories})', fontsize=14, fontweight='bold')

    # Plot 1: Cash evolution
    ax1 = axes[0, 0]
    for i, traj in enumerate(trajectories[:n_trajectories]):
        label = f"Ep {i+1} ({'Liq.' if traj['terminated'] else 'OK'})"
        ax1.plot(traj['cash'], linewidth=2, label=label, alpha=0.7)
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Cash Reserves (c)', fontsize=11)
    ax1.set_title('Cash Evolution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Plot 2: Dividend actions
    ax2 = axes[0, 1]
    for i, traj in enumerate(trajectories[:n_trajectories]):
        ax2.plot(traj['dividends'], linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Dividend Amount', fontsize=11)
    ax2.set_title('Dividend Actions', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Equity issuance
    ax3 = axes[1, 0]
    for i, traj in enumerate(trajectories[:n_trajectories]):
        ax3.plot(traj['equity'], linewidth=2, alpha=0.7)
    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('Equity Issuance Amount', fontsize=11)
    ax3.set_title('Equity Issuance Actions', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative rewards
    ax4 = axes[1, 1]
    for i, traj in enumerate(trajectories[:n_trajectories]):
        cumsum = np.cumsum(traj['rewards'])
        ax4.plot(cumsum, linewidth=2, alpha=0.7)
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_ylabel('Cumulative Reward', fontsize=11)
    ax4.set_title('Cumulative Rewards', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'sac_trajectories.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved trajectory visualization to: {output_path}")

    return fig


def create_action_distributions(policy_results, output_dir):
    """Create detailed action distribution plots at key states."""
    key_indices = [
        0,  # Very low cash
        len(policy_results['c_values']) // 4,  # Low-medium
        len(policy_results['c_values']) // 2,  # Medium
        3 * len(policy_results['c_values']) // 4,  # Medium-high
        -1,  # High cash
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle('Action Distributions at Different Cash Levels', fontsize=14, fontweight='bold')

    for i, idx in enumerate(key_indices):
        c_val = policy_results['c_values'][idx]

        # Dividend distribution
        ax_div = axes[0, i]
        div_samples = policy_results['dividends_samples'][idx]
        ax_div.hist(div_samples, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax_div.axvline(policy_results['dividends_mean'][idx], color='red',
                      linestyle='--', linewidth=2, label='Mean')
        ax_div.set_xlabel('Dividend', fontsize=10)
        ax_div.set_ylabel('Frequency', fontsize=10)
        ax_div.set_title(f'c = {c_val:.2f}', fontsize=11)
        ax_div.grid(True, alpha=0.3)
        if i == 0:
            ax_div.legend(fontsize=9)

        # Equity distribution
        ax_eq = axes[1, i]
        eq_samples = policy_results['equity_samples'][idx]
        ax_eq.hist(eq_samples, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax_eq.axvline(policy_results['equity_mean'][idx], color='blue',
                     linestyle='--', linewidth=2, label='Mean')
        ax_eq.set_xlabel('Equity', fontsize=10)
        ax_eq.set_ylabel('Frequency', fontsize=10)
        ax_eq.grid(True, alpha=0.3)
        if i == 0:
            ax_eq.legend(fontsize=9)

    axes[0, 0].set_ylabel('Dividend\nFrequency', fontsize=11)
    axes[1, 0].set_ylabel('Equity\nFrequency', fontsize=11)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'sac_action_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved action distributions to: {output_path}")

    return fig


def print_summary_statistics(policy_results, barrier_analysis, trajectories):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("LEARNED POLICY SUMMARY")
    print("="*80)

    print("\nBarrier Structure Analysis:")
    if barrier_analysis['dividend_barrier'] is not None:
        print(f"  Dividend barrier c*:     {barrier_analysis['dividend_barrier']:.4f}")
    else:
        print(f"  Dividend barrier c*:     Not clearly identified")

    if barrier_analysis['equity_region'] is not None:
        print(f"  Equity issuance region:  [{barrier_analysis['equity_region'][0]:.4f}, {barrier_analysis['equity_region'][1]:.4f}]")
    else:
        print(f"  Equity issuance region:  Not clearly identified")

    if barrier_analysis['inaction_region'] is not None:
        print(f"  Inaction region:         [{barrier_analysis['inaction_region'][0]:.4f}, {barrier_analysis['inaction_region'][1]:.4f}]")
    else:
        print(f"  Inaction region:         Not clearly identified")

    print("\nDividend Policy Statistics:")
    div_mean = policy_results['dividends_mean']
    print(f"  Mean dividend:           {div_mean.mean():.4f}")
    print(f"  Max dividend:            {div_mean.max():.4f} at c={policy_results['c_values'][div_mean.argmax()]:.3f}")
    print(f"  Avg std dev:             {policy_results['dividends_std'].mean():.4f}")

    print("\nEquity Issuance Policy Statistics:")
    eq_mean = policy_results['equity_mean']
    print(f"  Mean equity issuance:    {eq_mean.mean():.4f}")
    print(f"  Max equity issuance:     {eq_mean.max():.4f} at c={policy_results['c_values'][eq_mean.argmax()]:.3f}")
    print(f"  Avg std dev:             {policy_results['equity_std'].mean():.4f}")

    print("\nEpisode Performance (n={}):")
    avg_steps = np.mean([t['steps'] for t in trajectories])
    avg_reward = np.mean([sum(t['rewards']) for t in trajectories])
    liquidation_rate = np.mean([t['terminated'] for t in trajectories])

    print(f"  Average episode length:  {avg_steps:.1f} steps")
    print(f"  Average total reward:    {avg_reward:.4f}")
    print(f"  Liquidation rate:        {liquidation_rate*100:.1f}%")

    if liquidation_rate < 1.0:
        survived = [sum(t['rewards']) for t in trajectories if not t['terminated']]
        if survived:
            print(f"  Avg reward (survived):   {np.mean(survived):.4f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize SAC-trained GHM Policy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/ghm_equity/final_model',
        help='Path to SAC model (without .zip extension)'
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
        default=100,
        help='Number of points to sample in state space'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=10,
        help='Number of episodes to run for trajectory visualization'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("SAC POLICY VISUALIZATION FOR GHM EQUITY MANAGEMENT")
    print("="*80)

    # Load model
    model = load_sac_model(args.model)

    # Create environment
    env = GHMEquityEnv(dt=0.01, max_steps=1000, dividend_max=2.0, equity_max=2.0)
    print(f"✓ Created environment")
    print(f"  State space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Liquidation value: {env._dynamics.liquidation_value():.4f}")

    # Compute policy over states
    print(f"\nComputing policy over {args.n_points} state points...")
    policy_results = compute_policy_over_states(model, env, n_points=args.n_points)

    # Analyze barrier structure
    print("Analyzing barrier structure...")
    barrier_analysis = analyze_barrier_structure(policy_results)

    # Run episodes
    print(f"Running {args.n_episodes} evaluation episodes...")
    trajectories = run_episodes(model, env, n_episodes=args.n_episodes, deterministic=True)

    # Print statistics
    print_summary_statistics(policy_results, barrier_analysis, trajectories)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_policy_visualization(policy_results, barrier_analysis, args.output_dir)
    create_trajectory_visualization(trajectories, env, args.output_dir)
    create_action_distributions(policy_results, args.output_dir)

    print("\n✓ Visualization complete!")
    print(f"✓ All plots saved to: {args.output_dir}/")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
