"""Quick test of numerical VFI solver."""

import numpy as np
import torch
from macro_rl.dynamics.ghm_equity import GHMEquityParams, GHMEquityTimeAugmentedDynamics
from macro_rl.solvers.numerical_vfi import NumericalVFISolver, VFIConfig

# Create dynamics with default parameters
params = GHMEquityParams()
dynamics = GHMEquityTimeAugmentedDynamics(params, T=10.0)

print("Testing Numerical VFI Solver")
print("="*60)
print(f"Model parameters:")
print(f"  alpha: {params.alpha}")
print(f"  r: {params.r}, mu: {params.mu}, discount: {params.r - params.mu}")
print(f"  sigma_A: {params.sigma_A}, sigma_X: {params.sigma_X}, rho: {params.rho}")
print(f"  p: {params.p}, phi: {params.phi}")
print()

# Create small grid for testing
config = VFIConfig(
    n_c=20,              # Small grid for quick test
    n_tau=20,
    c_max=2.0,
    n_dividend=10,       # Small action grid
    n_equity=10,
    dt=0.5,              # Larger time step for speed
    T=10.0,
    tolerance=1e-4,      # Relaxed tolerance
)

print(f"VFI Configuration:")
print(f"  State grid: {config.n_c} x {config.n_tau}")
print(f"  Action grid: {config.n_dividend} x {config.n_equity}")
print(f"  Time step: {config.dt}")
print()

# Create solver
solver = NumericalVFISolver(dynamics, config)

print("Running VFI solver...")
print("This may take a minute...")
print()

# Solve
results = solver.solve(verbose=True)

print()
print("="*60)
print("VFI Solution Summary")
print("="*60)
print(f"Value function:")
print(f"  Min: {results['V'].min():.4f}")
print(f"  Max: {results['V'].max():.4f}")
print(f"  Mean: {results['V'].mean():.4f}")
print()

print(f"Dividend policy:")
print(f"  Min: {results['policy_dividend'].min():.4f}")
print(f"  Max: {results['policy_dividend'].max():.4f}")
print(f"  Mean: {results['policy_dividend'].mean():.4f}")
print()

print(f"Equity policy:")
print(f"  Min: {results['policy_equity'].min():.4f}")
print(f"  Max: {results['policy_equity'].max():.4f}")
print(f"  Mean: {results['policy_equity'].mean():.4f}")
print()

# Test policy evaluation at a few states
print("Testing policy at sample states:")
test_states = [
    (0.5, 5.0),   # Low cash, mid horizon
    (1.0, 5.0),   # Mid cash, mid horizon
    (1.5, 5.0),   # High cash, mid horizon
    (1.0, 1.0),   # Mid cash, near horizon
]

for c, tau in test_states:
    div, eq = solver.get_policy_at_state(c, tau)
    v = solver.get_value_at_state(c, tau)
    print(f"  c={c:.1f}, τ={tau:.1f}: dividend={div:.3f}, equity={eq:.3f}, value={v:.3f}")

print()
print("Test completed successfully!")
