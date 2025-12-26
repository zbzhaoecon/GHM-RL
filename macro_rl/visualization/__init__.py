"""
Visualization utilities for GHM-RL policies.

This module provides visualization functions for different types of policies:
- Time-augmented policies: π(c, τ)
- Standard policies: π(c)

Separated from training scripts for better modularity and reusability.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional


def compute_policy_value_time_augmented(
    policy,
    baseline,
    dynamics,
    n_points: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute policy and value function on 2D grid for time-augmented dynamics.

    Args:
        policy: Trained policy network
        baseline: Trained value function (baseline)
        dynamics: Time-augmented GHM dynamics model  (state_dim=2: c, τ)
        n_points: Number of points to sample in each dimension

    Returns:
        Dictionary containing visualization data with keys:
        - c_values: 1D array of cash values
        - tau_values: 1D array of time-to-horizon values
        - c_grid, tau_grid: 2D meshgrids
        - dividend_grid, equity_grid: 2D policy outputs
        - value_grid: 2D value function
        - tau_slices: Selected τ values for line plots
        - dividend_slices, equity_slices, value_slices: 1D slices at each τ
    """
    policy.eval()
    if baseline is not None:
        baseline.eval()

    state_space = dynamics.state_space
    device = next(policy.parameters()).device

    # Create 2D grid for (c, τ)
    c_values = np.linspace(state_space.lower[0].item(), state_space.upper[0].item(), n_points)
    tau_values = np.linspace(0.1, state_space.upper[1].item(), n_points)  # Avoid τ=0

    c_grid, tau_grid = np.meshgrid(c_values, tau_values)
    c_flat = torch.tensor(c_grid.flatten(), dtype=torch.float32).unsqueeze(1)
    tau_flat = torch.tensor(tau_grid.flatten(), dtype=torch.float32).unsqueeze(1)
    states = torch.cat([c_flat, tau_flat], dim=1).to(device)

    with torch.no_grad():
        # Get policy actions
        actions_mean, _ = policy.sample(states, deterministic=True)
        actions_mean = actions_mean.cpu().numpy()

        # Get value estimates
        if baseline is not None:
            # ActorCritic has evaluate() method, ValueNetwork can be called directly
            if hasattr(baseline, 'evaluate'):
                values = baseline.evaluate(states).squeeze().cpu().numpy()
            else:
                values = baseline(states).squeeze().cpu().numpy()
        else:
            values = np.zeros(len(states))

    # Reshape to 2D grids
    dividend_grid = actions_mean[:, 0].reshape(n_points, n_points)
    equity_grid = actions_mean[:, 1].reshape(n_points, n_points)
    value_grid = values.reshape(n_points, n_points)

    # Extract slices at different time horizons for line plots
    tau_slices = [0.5, 2.5, 5.0, 7.5, 10.0]
    dividend_slices = []
    equity_slices = []
    value_slices = []

    for tau_val in tau_slices:
        idx = np.argmin(np.abs(tau_values - tau_val))
        dividend_slices.append(dividend_grid[idx, :])
        equity_slices.append(equity_grid[idx, :])
        value_slices.append(value_grid[idx, :])

    return {
        'c_values': c_values,
        'tau_values': tau_values,
        'c_grid': c_grid,
        'tau_grid': tau_grid,
        'dividend_grid': dividend_grid,
        'equity_grid': equity_grid,
        'value_grid': value_grid,
        'tau_slices': tau_slices,
        'dividend_slices': dividend_slices,
        'equity_slices': equity_slices,
        'value_slices': value_slices,
        'is_time_augmented': True,
    }


def compute_policy_value_standard(
    policy,
    baseline,
    dynamics,
    n_points: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute policy and value function on 1D grid for standard dynamics.

    Args:
        policy: Trained policy network
        baseline: Trained value function (baseline)
        dynamics: Standard GHM dynamics model (state_dim=1: c only)
        n_points: Number of points to sample

    Returns:
        Dictionary containing visualization data with keys:
        - c_values: 1D array of cash values
        - actions_mean: (n_points, 2) array of policy outputs
        - values: (n_points,) array of value function
    """
    policy.eval()
    if baseline is not None:
        baseline.eval()

    state_space = dynamics.state_space
    device = next(policy.parameters()).device

    # Create 1D grid for c
    c_values = np.linspace(state_space.lower[0].item(), state_space.upper[0].item(), n_points)
    states = torch.tensor(c_values, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        # Get policy actions
        actions_mean, _ = policy.sample(states, deterministic=True)
        actions_mean = actions_mean.cpu().numpy()

        # Get value estimates
        if baseline is not None:
            # ActorCritic has evaluate() method, ValueNetwork can be called directly
            if hasattr(baseline, 'evaluate'):
                values = baseline.evaluate(states).squeeze().cpu().numpy()
            else:
                values = baseline(states).squeeze().cpu().numpy()
        else:
            values = np.zeros(len(states))

    return {
        'c_values': c_values,
        'actions_mean': actions_mean,
        'values': values,
        'is_time_augmented': False,
    }


def create_time_augmented_visualization(
    results: Dict[str, np.ndarray],
    step: int
) -> plt.Figure:
    """Create visualization for time-augmented dynamics showing policy(c, τ)."""
    c_values = results['c_values']
    tau_values = results['tau_values']
    dividend_grid = results['dividend_grid']
    equity_grid = results['equity_grid']
    value_grid = results['value_grid']

    tau_slices = results['tau_slices']
    dividend_slices = results['dividend_slices']
    equity_slices = results['equity_slices']
    value_slices = results['value_slices']

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Row 1: Heatmaps
    # Plot 1: Dividend Policy Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(c_values, tau_values, dividend_grid, levels=20, cmap='viridis')
    ax1.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax1.set_ylabel('Time-to-Horizon (τ)', fontsize=10)
    ax1.set_title('Dividend Policy π(c, τ)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Dividend Rate')

    # Plot 2: Equity Policy Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(c_values, tau_values, equity_grid, levels=20, cmap='plasma')
    ax2.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax2.set_ylabel('Time-to-Horizon (τ)', fontsize=10)
    ax2.set_title('Equity Issuance π(c, τ)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Equity Issuance')

    # Plot 3: Value Function Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(c_values, tau_values, value_grid, levels=20, cmap='coolwarm')
    ax3.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax3.set_ylabel('Time-to-Horizon (τ)', fontsize=10)
    ax3.set_title('Value Function V(c, τ)', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Value')

    # Row 2: Dividend slices at different τ
    ax4 = fig.add_subplot(gs[1, :])
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(tau_slices)))
    for i, (tau_val, div_slice) in enumerate(zip(tau_slices, dividend_slices)):
        ax4.plot(c_values, div_slice, color=colors[i], linewidth=2,
                 label=f'τ = {tau_val:.1f}', alpha=0.8)
    ax4.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax4.set_ylabel('Dividend Payout Rate', fontsize=10)
    ax4.set_title('Dividend Policy at Different Time Horizons', fontsize=11, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)

    # Row 3: Equity and Value slices
    ax5 = fig.add_subplot(gs[2, 0:2])
    colors_eq = plt.cm.Reds(np.linspace(0.3, 0.9, len(tau_slices)))
    for i, (tau_val, eq_slice) in enumerate(zip(tau_slices, equity_slices)):
        ax5.plot(c_values, eq_slice, color=colors_eq[i], linewidth=2,
                 label=f'τ = {tau_val:.1f}', alpha=0.8)
    ax5.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax5.set_ylabel('Equity Issuance', fontsize=10)
    ax5.set_title('Equity Issuance at Different Time Horizons', fontsize=11, fontweight='bold')
    ax5.legend(loc='best', fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 2])
    colors_val = plt.cm.Greens(np.linspace(0.3, 0.9, len(tau_slices)))
    for i, (tau_val, val_slice) in enumerate(zip(tau_slices, value_slices)):
        ax6.plot(c_values, val_slice, color=colors_val[i], linewidth=2,
                 label=f'τ = {tau_val:.1f}', alpha=0.8)
    ax6.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax6.set_ylabel('Value V(c, τ)', fontsize=10)
    ax6.set_title('Value Function at Different τ', fontsize=11, fontweight='bold')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Add main title
    fig.suptitle(f'Time-Augmented Policy: π(c, τ) - Step {step}',
                 fontsize=14, fontweight='bold', y=0.995)

    return fig


def create_standard_visualization(
    results: Dict[str, np.ndarray],
    step: int
) -> plt.Figure:
    """Create visualization for standard (non-time-augmented) dynamics."""
    c_values = results['c_values']
    actions_mean = results['actions_mean']
    values = results['values']

    dividends = actions_mean[:, 0]
    equity = actions_mean[:, 1]

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Dividend policy
    ax1.plot(c_values, dividends, 'b-', linewidth=2)
    ax1.set_xlabel('Cash Reserves (c)')
    ax1.set_ylabel('Dividend Rate')
    ax1.set_title('Dividend Policy π(c)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Equity issuance policy
    ax2.plot(c_values, equity, 'r-', linewidth=2)
    ax2.set_xlabel('Cash Reserves (c)')
    ax2.set_ylabel('Equity Issuance')
    ax2.set_title('Equity Issuance Policy π(c)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Value function
    ax3.plot(c_values, values, 'g-', linewidth=2)
    ax3.set_xlabel('Cash Reserves (c)')
    ax3.set_ylabel('Value')
    ax3.set_title('Value Function V(c)')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'Policy and Value Function - Step {step}', fontsize=14, fontweight='bold')
    fig.tight_layout()

    return fig


def create_training_visualization(
    results: Dict[str, np.ndarray],
    step: int
) -> plt.Figure:
    """
    Create appropriate visualization based on dynamics type.

    Args:
        results: Dictionary from compute_policy_value_*
        step: Training step number

    Returns:
        Matplotlib figure
    """
    if results.get('is_time_augmented', False):
        return create_time_augmented_visualization(results, step)
    else:
        return create_standard_visualization(results, step)
