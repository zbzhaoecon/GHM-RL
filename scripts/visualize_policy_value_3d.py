#!/usr/bin/env python3
"""
3D Visualization script for GHM Actor-Critic Policy and Value Functions.

This script loads a trained model checkpoint and visualizes the full
time-dependent policy and value functions using 3D surface plots.

python scripts/visualize_policy_value_time_augmented_3d.py \
    --checkpoint checkpoints/ghm_time_augmented_sparse_mc/final_model.pt \
    --output-dir visualizations/monte_carlo_3d

python scripts/visualize_policy_value_time_augmented_3d.py \
    --checkpoint checkpoints/ghm_time_augmented_sparse_mc/final_model.pt \
    --output-dir visualizations/monte_carlo_3d
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from macro_rl.core.state_space import StateSpace
from macro_rl.dynamics.ghm_equity import GHMEquityTimeAugmentedDynamics, GHMEquityParams
from macro_rl.networks.actor_critic import ActorCritic


def load_checkpoint(checkpoint_path: str):
    """Load model checkpoint and reconstruct actor-critic network."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain a 'config' dictionary.")

    config_data = checkpoint['config']
    if isinstance(config_data, argparse.Namespace):
        config = vars(config_data)
    else:
        config = config_data

    state_dim = config.get('state_dim', 2)
    action_dim = config.get('action_dim', 2)
    
    hidden_dims = None
    shared_layers = 0

    if 'network' in config and isinstance(config['network'], dict):
        hidden_dims = config['network'].get('hidden_dims')
        shared_layers = config['network'].get('shared_layers', 0)
    elif 'policy_hidden' in config:
        hidden_dims = config.get('policy_hidden')
        shared_layers = 0
    
    if hidden_dims is None:
        hidden_dims = [256, 256]

    if isinstance(hidden_dims, tuple):
        hidden_dims = list(hidden_dims)

    action_bounds = None
    state_dict_for_bounds = checkpoint.get('actor_critic_state_dict') or checkpoint.get('policy_state_dict')
    
    if state_dict_for_bounds:
        low_key = 'actor.action_low' if 'actor.action_low' in state_dict_for_bounds else 'action_low'
        high_key = 'actor.action_high' if 'actor.action_high' in state_dict_for_bounds else 'action_high'
        
        if low_key in state_dict_for_bounds and high_key in state_dict_for_bounds:
            action_bounds = (state_dict_for_bounds[low_key], state_dict_for_bounds[high_key])

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        shared_layers=shared_layers,
        action_bounds=action_bounds
    )

    has_critic = False
    if 'actor_critic_state_dict' in checkpoint:
        ac.load_state_dict(checkpoint['actor_critic_state_dict'], strict=False)
        has_critic = True
    elif 'policy_state_dict' in checkpoint:
        ac.actor.load_state_dict(checkpoint['policy_state_dict'], strict=False)
        if 'baseline_state_dict' in checkpoint and hasattr(ac, 'critic'):
            ac.critic.load_state_dict(checkpoint['baseline_state_dict'], strict=False)
            has_critic = True
    else:
        raise ValueError("Could not find a valid model state_dict key in checkpoint.")

    ac.eval()
    return ac, config, has_critic


def compute_policy_value_full_grid(ac, state_space, n_points=50):
    """Computes policy and value over a full 2D grid of (c, τ)."""
    print(f"\nComputing policy and value over a {n_points}x{n_points} grid...")
    c_values = np.linspace(state_space.lower[0].item(), state_space.upper[0].item(), n_points)
    tau_values = np.linspace(0.01, state_space.upper[1].item(), n_points)
    c_grid, tau_grid = np.meshgrid(c_values, tau_values)

    states_flat = torch.tensor(np.stack([c_grid.flatten(), tau_grid.flatten()], axis=-1), dtype=torch.float32)

    with torch.no_grad():
        actions_mean = ac.act(states_flat, deterministic=True).numpy()
        values = ac.evaluate(states_flat).squeeze().numpy()

    return {
        'c_grid': c_grid,
        'tau_grid': tau_grid,
        'dividend_grid': actions_mean[:, 0].reshape(n_points, n_points),
        'equity_grid': actions_mean[:, 1].reshape(n_points, n_points),
        'value_grid': values.reshape(n_points, n_points),
    }


def estimate_value_full_grid_mc(policy, dynamics, state_space, n_points=50, n_episodes=8):
    """Estimate value function over a full 2D grid via Monte Carlo rollouts."""
    print(f"\nEstimating value function via {n_episodes} rollouts over a {n_points}x{n_points} grid...")
    from macro_rl.simulation.trajectory import TrajectorySimulator
    from macro_rl.control.ghm_control import GHMControlSpec
    from macro_rl.rewards.ghm_rewards import GHMRewardFunction

    reward_fn = GHMRewardFunction(
        discount_rate=dynamics.p.r - dynamics.p.mu,
        issuance_cost=dynamics.p.lambda_,
        liquidation_rate=dynamics.p.omega,
        liquidation_flow=dynamics.p.alpha,
    )
    simulator = TrajectorySimulator(
        dynamics=dynamics,
        control_spec=GHMControlSpec(),
        reward_fn=reward_fn,
        dt=dynamics.dt,
        T=dynamics.T,
        use_sparse_rewards=True
    )

    c_values = np.linspace(state_space.lower[0].item(), state_space.upper[0].item(), n_points)
    tau_values = np.linspace(0.01, state_space.upper[1].item(), n_points)
    c_grid, tau_grid = np.meshgrid(c_values, tau_values)
    
    states_flat = torch.tensor(np.stack([c_grid.flatten(), tau_grid.flatten()], axis=-1), dtype=torch.float32)
    value_grid_flat = np.zeros(states_flat.shape[0])

    # Batch the states to avoid memory issues
    batch_size = 1024 
    for i in tqdm(range(0, len(states_flat), batch_size), desc="MC Value Estimation"):
        batch_states = states_flat[i:i+batch_size]
        batch_returns = []
        for _ in range(n_episodes):
            trajectories = simulator.rollout(policy, batch_states.clone())
            batch_returns.append(trajectories.returns.numpy())
        
        # Average returns over episodes
        value_grid_flat[i:i+batch_size] = np.mean(batch_returns, axis=0)
        
    return value_grid_flat.reshape(n_points, n_points)


def create_3d_visualizations(results: dict, checkpoint_path: str, output_dir: str):
    """Create 3D surface plots for policy and value function."""
    print("\nGenerating 3D visualizations...")
    c_grid = results['c_grid']
    tau_grid = results['tau_grid']

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f'3D Policy and Value Functions\n{Path(checkpoint_path).name}', fontsize=16, y=1.02)

    # Plot 1: Value Function
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(c_grid, tau_grid, results['value_grid'], cmap='viridis', edgecolor='none')
    ax1.set_xlabel('Cash Ratio (c)')
    ax1.set_ylabel('Time-to-Horizon (τ)')
    ax1.set_zlabel('Value V(c, τ)')
    ax1.set_title('Value Function')
    ax1.view_init(elev=20, azim=-120)

    # Plot 2: Dividend Policy
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(c_grid, tau_grid, results['dividend_grid'], cmap='plasma', edgecolor='none')
    ax2.set_xlabel('Cash Ratio (c)')
    ax2.set_ylabel('Time-to-Horizon (τ)')
    ax2.set_zlabel('Dividend Rate a_L(c, τ)')
    ax2.set_title('Policy: Dividend Payout')
    ax2.view_init(elev=20, azim=-120)

    # Plot 3: Equity Issuance Policy
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(c_grid, tau_grid, results['equity_grid'], cmap='magma', edgecolor='none')
    ax3.set_xlabel('Cash Ratio (c)')
    ax3.set_ylabel('Time-to-Horizon (τ)')
    ax3.set_zlabel('Equity Issuance a_E(c, τ)')
    ax3.set_title('Policy: Equity Issuance')
    ax3.view_init(elev=20, azim=-120)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'policy_value_3d.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved 3D visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D visualizations for GHM policies.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save plots')
    parser.add_argument('--n-points', type=int, default=50, help='Grid points per dimension')
    parser.add_argument('--c-max', type=float, default=2.0, help='Maximum cash reserves ratio')
    parser.add_argument('--T', type=float, default=10.0, help='Time horizon T')
    parser.add_argument('--n-episodes', type=int, default=32, help='Episodes for MC value estimation')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("GHM 3D VISUALIZATION")
    print("="*70)

    ac, config, has_critic = load_checkpoint(args.checkpoint)

    if 'dynamics' in config and isinstance(config['dynamics'], dict):
        d_config = config['dynamics']
        params = GHMEquityParams(c_max=args.c_max, alpha=d_config.get('alpha'), mu=d_config.get('mu'), r=d_config.get('r'))
    else:
        params = GHMEquityParams(c_max=args.c_max)

    dynamics = GHMEquityTimeAugmentedDynamics(params, T=args.T)
    dynamics.dt = config.get('dt', 0.01)
    state_space = dynamics.state_space

    results = compute_policy_value_full_grid(ac, state_space, n_points=args.n_points)
    
    if not has_critic:
        # Re-estimate value function via rollouts for MC models without a critic
        value_grid = estimate_value_full_grid_mc(
            ac.actor, dynamics, state_space, 
            n_points=args.n_points,
            n_episodes=args.n_episodes
        )
        results['value_grid'] = value_grid

    create_3d_visualizations(results, args.checkpoint, args.output_dir)

    print("\n✓ 3D Visualization complete!")

if __name__ == '__main__':
    main()
