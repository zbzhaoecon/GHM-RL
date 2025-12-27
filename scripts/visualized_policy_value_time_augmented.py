#!/usr/bin/env python3
"""
Visualization script for GHM Actor-Critic Policy and Value Functions.
This script is adapted to handle time-augmented state spaces (c, τ).

This script loads a trained model checkpoint and visualizes:
1. Policy function: Mean actions (dividend payout a_L, equity issuance a_E) across state space for a fixed τ
2. Value function: V(c, τ) across the cash reserves ratio state space for a fixed τ
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
from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityTimeAugmentedDynamics
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

    # --- Robustly extract network and action bound parameters ---
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
        print("Warning: Could not determine hidden_dims from config. Using default [256, 256].")
        hidden_dims = [256, 256]

    if isinstance(hidden_dims, tuple):
        hidden_dims = list(hidden_dims)

    # --- Extract Action Bounds from State Dict ---
    action_bounds = None
    state_dict_for_bounds = checkpoint.get('actor_critic_state_dict') or checkpoint.get('policy_state_dict')
    
    if state_dict_for_bounds:
        low_key = 'actor.action_low' if 'actor.action_low' in state_dict_for_bounds else 'action_low'
        high_key = 'actor.action_high' if 'actor.action_high' in state_dict_for_bounds else 'action_high'
        
        if low_key in state_dict_for_bounds and high_key in state_dict_for_bounds:
            print(f"Found action bounds in checkpoint.")
            action_bounds = (state_dict_for_bounds[low_key], state_dict_for_bounds[high_key])

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        shared_layers=shared_layers,
        action_bounds=action_bounds
    )

    # --- Flexible loading of state dict ---
    has_critic = False
    if 'actor_critic_state_dict' in checkpoint:
        print("Found 'actor_critic_state_dict', loading combined state dict.")
        ac.load_state_dict(checkpoint['actor_critic_state_dict'], strict=False)
        has_critic = True
    elif 'policy_state_dict' in checkpoint:
        print("Found separate 'policy_state_dict', loading components.")
        ac.actor.load_state_dict(checkpoint['policy_state_dict'], strict=False)
        if 'baseline_state_dict' in checkpoint and hasattr(ac, 'critic'):
            ac.critic.load_state_dict(checkpoint['baseline_state_dict'], strict=False)
            has_critic = True
        else:
            print("Warning: No 'baseline_state_dict' found for critic. Value function will be estimated via rollouts.")
    else:
        raise ValueError("Could not find a valid model state_dict key in checkpoint.")

    ac.eval()

    print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    print(f"  Hidden dims: {hidden_dims}, Shared layers: {shared_layers}")

    return ac, config, has_critic


def estimate_value_function_mc(policy, dynamics, state_space, tau_value, n_points=100, n_episodes=32):
    """Estimate value function via Monte Carlo rollouts."""
    print(f"\nEstimating value function via {n_episodes} Monte Carlo rollouts per point...")
    from macro_rl.simulation.trajectory import TrajectorySimulator
    from macro_rl.control.ghm_control import GHMControlSpec
    from macro_rl.rewards.ghm_rewards import GHMRewardFunction

    # Setup components needed for simulation
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
        use_sparse_rewards=True # Use trajectory return for value
    )

    c_values = np.linspace(state_space.lower[0].item(), state_space.upper[0].item(), n_points)
    values = []
    
    from tqdm import tqdm
    for c in tqdm(c_values, desc="Value Estimation"):
        tau_values = np.full_like(c, tau_value)
        initial_states = torch.tensor(np.stack([c, tau_values], axis=-1), dtype=torch.float32).reshape(1, -1)
        
        episode_returns = []
        for _ in range(n_episodes):
            # The simulator expects a batch of initial states
            trajectories = simulator.rollout(policy, initial_states.clone())
            episode_returns.append(trajectories.returns.item())
        
        values.append(np.mean(episode_returns))

    return np.array(values)


def compute_policy_value_grid(ac: ActorCritic, state_space: StateSpace, tau_value: float, n_points: int = 100):
    """Compute policy and value function on a grid over the state space for a fixed tau."""
    # Create state grid
    c_values = np.linspace(state_space.lower[0].item(), state_space.upper[0].item(), n_points)
    tau_values = np.full_like(c_values, tau_value)
    
    states = torch.tensor(np.stack([c_values, tau_values], axis=-1), dtype=torch.float32)

    with torch.no_grad():
        # Get deterministic actions (mean of policy)
        actions_mean = ac.act(states, deterministic=True)

        # Get value estimates from critic
        values = ac.evaluate(states).squeeze()

        # Get policy distribution parameters
        feat = ac._features(states)
        mean, std = ac.actor._get_distribution_params(feat)
        actions_std = std.expand_as(mean)

    # Get value gradients and Hessian diagonal
    values_grad, V_s, V_ss_diag = ac.evaluate_with_grad(states)

    return {
        'c_values': c_values,
        'tau_value': tau_value,
        'actions_mean': actions_mean.numpy(),
        'actions_std': actions_std.numpy(),
        'values': values.numpy(),
        'V_s': V_s.detach().numpy(),
        'V_ss_diag': V_ss_diag.detach().numpy(),
    }


def compute_hjb_residuals(results: dict, dynamics: GHMEquityTimeAugmentedDynamics, dt: float = 0.01):
    """Compute HJB residuals for validation of the value function."""
    c_values = results['c_values']
    tau_value = results['tau_value']
    actions = results['actions_mean']
    V = results['values']
    V_s = results['V_s']
    V_ss_diag = results['V_ss_diag']

    # Convert to tensors
    tau_values = np.full_like(c_values, tau_value)
    states = torch.tensor(np.stack([c_values, tau_values], axis=-1), dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.float32)
    V_t = torch.tensor(V, dtype=torch.float32).unsqueeze(1)
    V_s_t = torch.tensor(V_s, dtype=torch.float32)
    V_ss_t = torch.tensor(V_ss_diag, dtype=torch.float32)

    with torch.no_grad():
        drift = dynamics.drift(states, actions_t)
        diffusion = dynamics.diffusion(states)

        a_L = actions_t[:, 0:1]
        a_E = actions_t[:, 1:2]
        reward = a_L * dt - a_E

        discount_rate = dynamics.p.r - dynamics.p.mu
        
        # HJB residual for time-augmented state (c, τ)
        # V_t + V_c * f(c,a) + 0.5 * V_cc * σ^2(c) + r(c,a) - ρV = 0
        # V_s has two components: [V_c, V_τ]
        V_c = V_s_t[:, 0:1]
        V_tau = V_s_t[:, 1:2]

        hjb_residual = (
            V_tau + 
            V_c * drift[:, 0:1] + 
            0.5 * V_ss_t[:, 0:1] * diffusion[:, 0:1].pow(2) +
            reward -
            discount_rate * V_t
        )

    return hjb_residual.squeeze().numpy()

def create_visualizations(results: dict, hjb_residuals: np.ndarray,
                          checkpoint_path: str, output_dir: str, has_critic: bool):
    """Create comprehensive visualization plots."""
    c_values = results['c_values']
    tau_value = results['tau_value']
    actions_mean = results['actions_mean']
    actions_std = results['actions_std']
    values = results['values']
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    fig.suptitle(f'Policy & Value Visualization (τ = {tau_value:.2f})\n{Path(checkpoint_path).name}',
                 fontsize=16, fontweight='bold', y=0.99)

    # === Plot 1: Value Function ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(c_values, values, 'b-', linewidth=2)
    ax1.set_xlabel('Cash Reserves Ratio (c)', fontsize=11)
    ax1.set_ylabel(f'Value Function V(c, τ={tau_value:.2f})', fontsize=11)
    value_title = 'Value Function' if has_critic else 'Value Function (MC Estimate)'
    ax1.set_title(value_title, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # === Plots 2 & 3 are for critic gradients, only show if critic exists ===
    if has_critic:
        V_s_c = results['V_s'][:, 0]
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(c_values, V_s_c, 'g-', linewidth=2)
        ax2.set_xlabel('Cash Reserves Ratio (c)', fontsize=11)
        ax2.set_ylabel(f'Value Gradient V_c(c, τ={tau_value:.2f})', fontsize=11)
        ax2.set_title('Value Function Gradient (w.r.t c)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

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

    output_path = os.path.join(output_dir, f'policy_value_tau_{tau_value:.2f}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GHM Actor-Critic Policy and Value Functions (Time-Augmented)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save visualization plots')
    parser.add_argument('--n-points', type=int, default=100, help='Number of points to sample in state space')
    parser.add_argument('--c-max', type=float, default=2.0, help='Maximum cash reserves ratio')
    parser.add_argument('--T', type=float, default=5.0, help='Time horizon T')
    parser.add_argument('--tau-eval', type=float, default=None, help='Specific tau to evaluate at. Defaults to T/2.')
    parser.add_argument('--n-episodes', type=int, default=32, help='Number of episodes for MC value estimation.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("GHM ACTOR-CRITIC VISUALIZATION (TIME-AUGMENTED)")
    print("="*70)

    ac, config, has_critic = load_checkpoint(args.checkpoint)

    # Use time-augmented dynamics
    from macro_rl.dynamics.ghm_equity import GHMEquityParams
    # Use params from config if available, otherwise defaults
    if 'dynamics' in config and isinstance(config['dynamics'], dict):
        d_config = config['dynamics']
        params = GHMEquityParams(c_max=args.c_max, alpha=d_config.get('alpha'), mu=d_config.get('mu'), r=d_config.get('r'))
    else:
        params = GHMEquityParams(c_max=args.c_max)

    dynamics = GHMEquityTimeAugmentedDynamics(params, T=args.T)
    dynamics.dt = config.get('dt', 0.01) # Manually set dt, as it's used by the simulator
    state_space = dynamics.state_space
    
    tau_to_eval = args.tau_eval if args.tau_eval is not None else args.T / 2.0

    print(f"\n✓ State space: c ∈ [0, {args.c_max}], τ ∈ [0, {args.T}]")
    print(f"✓ Evaluating at fixed τ = {tau_to_eval:.2f}")
    print(f"✓ Sampling {args.n_points} points for c")

    print(f"\nComputing policy and value function...")
    
    results = compute_policy_value_grid(ac, state_space, tau_value=tau_to_eval, n_points=args.n_points)
    hjb_residuals = None

    if has_critic:
        print("Computing HJB residuals...")
        hjb_residuals = compute_hjb_residuals(results, dynamics)
    else:
        # If no critic, estimate value function via rollouts
        mc_values = estimate_value_function_mc(
            ac.actor, dynamics, state_space, 
            tau_value=tau_to_eval, 
            n_points=args.n_points,
            n_episodes=args.n_episodes
        )
        results['values'] = mc_values

    print(f"\nGenerating visualizations...")
    create_visualizations(results, hjb_residuals, args.checkpoint, args.output_dir, has_critic)

    print("\n✓ Visualization complete!")
    print(f"✓ All plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
