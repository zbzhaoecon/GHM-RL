"""
Run numerical benchmark for GHM equity model.

This script:
1. Solves the HJB equation using Value Function Iteration (VFI)
2. Evaluates the numerical solution using Monte Carlo
3. Visualizes the results (policy and value functions)
4. Optionally compares with trained RL policies

Usage:
    python scripts/run_numerical_benchmark.py --config configs/time_augmented_sparse_config.yaml
    python scripts/run_numerical_benchmark.py --compare-rl checkpoints/model.pt
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from macro_rl.dynamics.ghm_equity import GHMEquityParams, GHMEquityTimeAugmentedDynamics
from macro_rl.solvers.numerical_vfi import NumericalVFISolver, VFIConfig
from macro_rl.solvers.monte_carlo_evaluator import (
    MonteCarloEvaluator,
    MonteCarloConfig,
    NumericalPolicyWrapper,
    RLPolicyWrapper
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dynamics_from_config(config: dict):
    """Create dynamics object from configuration."""
    dyn_config = config['dynamics']
    params = GHMEquityParams(
        alpha=dyn_config['alpha'],
        mu=dyn_config['mu'],
        r=dyn_config['r'],
        lambda_=dyn_config['lambda_'],
        sigma_A=dyn_config['sigma_A'],
        sigma_X=dyn_config['sigma_X'],
        rho=dyn_config['rho'],
        c_max=dyn_config['c_max'],
        p=dyn_config['p'],
        phi=dyn_config['phi'],
        omega=dyn_config['omega'],
    )

    T = config['training']['T']
    dynamics = GHMEquityTimeAugmentedDynamics(params, T=T)

    return dynamics, params


def visualize_numerical_solution(
    vfi_solver: NumericalVFISolver,
    mc_results: dict = None,
    save_path: str = None,
    title_suffix: str = ""
):
    """
    Create comprehensive visualization of numerical solution.

    Args:
        vfi_solver: Solved VFI solver
        mc_results: Optional Monte Carlo realized value function
        save_path: Path to save figure
        title_suffix: Additional text for title
    """
    c_grid = vfi_solver.c_grid
    tau_grid = vfi_solver.tau_grid
    V = vfi_solver.V
    policy_dividend = vfi_solver.policy_dividend
    policy_equity = vfi_solver.policy_equity

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Row 1: Heatmaps
    # Plot 1: Dividend Policy Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(c_grid, tau_grid, policy_dividend.T, levels=20, cmap='viridis')
    ax1.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax1.set_ylabel('Time-to-Horizon (τ)', fontsize=10)
    ax1.set_title('Dividend Policy π(c, τ)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Dividend Rate')

    # Plot 2: Equity Policy Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(c_grid, tau_grid, policy_equity.T, levels=20, cmap='plasma')
    ax2.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax2.set_ylabel('Time-to-Horizon (τ)', fontsize=10)
    ax2.set_title('Equity Issuance π(c, τ)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Equity Issuance')

    # Plot 3: Value Function Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    V_plot = mc_results['V_realized'].T if mc_results else V.T
    im3 = ax3.contourf(c_grid, tau_grid, V_plot, levels=20, cmap='coolwarm')
    ax3.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax3.set_ylabel('Time-to-Horizon (τ)', fontsize=10)
    title_v = 'Value Function V(c, τ)' + (' (MC Realized)' if mc_results else '')
    ax3.set_title(title_v, fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Value')

    # Row 2: Dividend slices at different τ
    ax4 = fig.add_subplot(gs[1, :])
    tau_slices = [0.5, 2.5, 5.0, 7.5, 10.0]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(tau_slices)))

    for tau_val, color in zip(tau_slices, colors):
        idx = np.argmin(np.abs(tau_grid - tau_val))
        ax4.plot(c_grid, policy_dividend[:, idx], color=color, linewidth=2,
                 label=f'τ = {tau_val:.1f}', alpha=0.8)

    ax4.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax4.set_ylabel('Dividend Payout Rate', fontsize=10)
    ax4.set_title('Dividend Policy at Different Time Horizons', fontsize=11, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)

    # Row 3: Equity and Value slices
    ax5 = fig.add_subplot(gs[2, 0:2])
    colors_eq = plt.cm.Reds(np.linspace(0.3, 0.9, len(tau_slices)))

    for tau_val, color in zip(tau_slices, colors_eq):
        idx = np.argmin(np.abs(tau_grid - tau_val))
        ax5.plot(c_grid, policy_equity[:, idx], color=color, linewidth=2,
                 label=f'τ = {tau_val:.1f}', alpha=0.8)

    ax5.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax5.set_ylabel('Equity Issuance', fontsize=10)
    ax5.set_title('Equity Issuance at Different Time Horizons', fontsize=11, fontweight='bold')
    ax5.legend(loc='best', fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 2])
    colors_val = plt.cm.Greens(np.linspace(0.3, 0.9, len(tau_slices)))

    for tau_val, color in zip(tau_slices, colors_val):
        idx = np.argmin(np.abs(tau_grid - tau_val))
        V_slice = mc_results['V_realized'][:, idx] if mc_results else V[:, idx]
        ax6.plot(c_grid, V_slice, color=color, linewidth=2,
                 label=f'τ = {tau_val:.1f}', alpha=0.8)

    ax6.set_xlabel('Cash Reserves (c)', fontsize=10)
    ax6.set_ylabel('Value V(c, τ)', fontsize=10)
    ax6.set_title('Value Function at Different τ', fontsize=11, fontweight='bold')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Add main title
    main_title = f'Numerical Benchmark: Value Function Iteration{title_suffix}'
    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Run numerical benchmark for GHM equity model')
    parser.add_argument('--config', type=str, default='configs/time_augmented_sparse_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--n-c', type=int, default=100,
                        help='Number of cash grid points')
    parser.add_argument('--n-tau', type=int, default=100,
                        help='Number of time grid points')
    parser.add_argument('--n-dividend', type=int, default=50,
                        help='Number of dividend action grid points')
    parser.add_argument('--n-equity', type=int, default=30,
                        help='Number of equity action grid points')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Convergence tolerance for VFI')
    parser.add_argument('--compute-mc', action='store_true',
                        help='Compute Monte Carlo realized value function')
    parser.add_argument('--mc-samples', type=int, default=100,
                        help='Number of MC samples per state')
    parser.add_argument('--output-dir', type=str, default='numerical_benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--compare-rl', type=str, default=None,
                        help='Path to trained RL model checkpoint for comparison')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Create dynamics
    print("Creating dynamics...")
    dynamics, params = create_dynamics_from_config(config)

    print(f"\nModel Parameters:")
    print(f"  α (mean cash flow): {params.alpha}")
    print(f"  r (interest rate): {params.r}")
    print(f"  μ (growth rate): {params.mu}")
    print(f"  λ (carry cost): {params.lambda_}")
    print(f"  σ_A (permanent vol): {params.sigma_A}")
    print(f"  σ_X (transitory vol): {params.sigma_X}")
    print(f"  ρ (correlation): {params.rho}")
    print(f"  p (equity cost): {params.p}")
    print(f"  φ (fixed cost): {params.phi}")
    print(f"  T (horizon): {dynamics.T}")

    # Setup VFI solver
    print("\nSetting up VFI solver...")
    vfi_config = VFIConfig(
        n_c=args.n_c,
        n_tau=args.n_tau,
        c_max=params.c_max,
        n_dividend=args.n_dividend,
        n_equity=args.n_equity,
        dt=config['training']['dt'],
        T=config['training']['T'],
        tolerance=args.tolerance,
        dividend_max=config['action_space']['dividend_max'],
        equity_max=config['action_space']['equity_max'],
    )

    vfi_solver = NumericalVFISolver(dynamics, vfi_config)

    # Solve HJB equation
    print("\nSolving HJB equation using Value Function Iteration...")
    print(f"Grid: {args.n_c} × {args.n_tau} states")
    print(f"Action grid: {args.n_dividend} × {args.n_equity} actions")
    print(f"Total evaluations: ~{args.n_c * args.n_tau * args.n_dividend * args.n_equity:,}")

    results = vfi_solver.solve(verbose=True)

    print("\nVFI Solution complete!")
    print(f"Value function range: [{vfi_solver.V.min():.3f}, {vfi_solver.V.max():.3f}]")
    print(f"Dividend policy range: [{vfi_solver.policy_dividend.min():.3f}, {vfi_solver.policy_dividend.max():.3f}]")
    print(f"Equity policy range: [{vfi_solver.policy_equity.min():.3f}, {vfi_solver.policy_equity.max():.3f}]")

    # Compute Monte Carlo realized value function if requested
    mc_results = None
    if args.compute_mc:
        print("\nComputing Monte Carlo realized value function...")
        mc_config = MonteCarloConfig(
            n_trajectories=10000,
            dt=config['training']['dt'],
            T=config['training']['T'],
        )
        mc_evaluator = MonteCarloEvaluator(dynamics, mc_config)
        policy_wrapper = NumericalPolicyWrapper(vfi_solver)

        mc_results = mc_evaluator.compute_realized_value_function(
            policy_wrapper,
            n_c=args.n_c,
            n_tau=args.n_tau,
            n_samples_per_state=args.mc_samples,
            verbose=True
        )

        print(f"MC Value function range: [{mc_results['V_realized'].min():.3f}, {mc_results['V_realized'].max():.3f}]")

    # Visualize results
    print("\nGenerating visualizations...")
    fig = visualize_numerical_solution(
        vfi_solver,
        mc_results=mc_results,
        save_path=output_dir / 'numerical_benchmark.png'
    )

    # Save numerical results
    print("\nSaving numerical results...")
    np.savez(
        output_dir / 'vfi_solution.npz',
        V=vfi_solver.V,
        policy_dividend=vfi_solver.policy_dividend,
        policy_equity=vfi_solver.policy_equity,
        c_grid=vfi_solver.c_grid,
        tau_grid=vfi_solver.tau_grid,
    )

    if mc_results:
        np.savez(
            output_dir / 'mc_realized_values.npz',
            **mc_results
        )

    print(f"\nResults saved to {output_dir}")
    print("Done!")

    # Show plot
    plt.show()


if __name__ == '__main__':
    main()
