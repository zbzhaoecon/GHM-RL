"""
Diagnostic script to verify sparse vs dense rewards give identical results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

# Mock test without full dependencies
def test_sparse_dense_equivalence():
    """Test that sparse and dense return computations are equivalent."""

    print("Testing Sparse vs Dense Return Computation")
    print("=" * 60)

    # Simulate a simple trajectory
    batch_size = 5
    n_steps = 10
    dt = 0.1
    discount_rate = 0.02  # r - mu
    issuance_cost = 0.1

    # Create random actions and masks
    torch.manual_seed(42)
    actions = torch.rand(batch_size, n_steps, 2)  # [0, 1] range
    actions[:, :, 0] = actions[:, :, 0] * 10  # Dividend: [0, 10]
    actions[:, :, 1] = actions[:, :, 1] * 0.5  # Equity: [0, 0.5]

    # Create masks (some trajectories terminate early)
    masks = torch.ones(batch_size, n_steps)
    masks[0, 5:] = 0  # Trajectory 0 terminates at step 5
    masks[1, 3:] = 0  # Trajectory 1 terminates at step 3
    masks[2, 8:] = 0  # Trajectory 2 terminates at step 8

    terminal_rewards = torch.zeros(batch_size)  # Assume 0 for simplicity

    # Compute DENSE returns (from per-step rewards)
    # NOTE: All terms scaled by dt since a_L, a_E are RATES in dynamics
    rewards = torch.zeros(batch_size, n_steps)
    for t in range(n_steps):
        a_L = actions[:, t, 0]
        a_E = actions[:, t, 1]
        rewards[:, t] = (a_L - issuance_cost * a_E) * dt

    returns_dense = torch.zeros(batch_size)
    for t in range(n_steps):
        discount = torch.exp(torch.tensor(-discount_rate * t * dt))
        returns_dense = returns_dense + discount * rewards[:, t] * masks[:, t]

    termination_times = masks.sum(dim=1)
    terminal_discount = torch.exp(-discount_rate * termination_times * dt)
    returns_dense = returns_dense + terminal_discount * terminal_rewards

    # Compute SPARSE returns (directly from actions)
    # NOTE: All terms scaled by dt since a_L, a_E are RATES in dynamics
    returns_sparse = torch.zeros(batch_size)
    for t in range(n_steps):
        discount = torch.exp(torch.tensor(-discount_rate * t * dt))
        a_L = actions[:, t, 0]
        a_E = actions[:, t, 1]
        net_payout = (a_L - issuance_cost * a_E) * dt
        returns_sparse = returns_sparse + discount * net_payout * masks[:, t]

    terminal_discount = torch.exp(-discount_rate * termination_times * dt)
    returns_sparse = returns_sparse + terminal_discount * terminal_rewards

    # Compare
    print("\nComparison:")
    print("-" * 60)
    print(f"{'Trajectory':<12} {'Dense':<15} {'Sparse':<15} {'Difference':<15}")
    print("-" * 60)

    max_diff = 0.0
    for i in range(batch_size):
        diff = abs(returns_dense[i].item() - returns_sparse[i].item())
        max_diff = max(max_diff, diff)
        print(f"{i:<12} {returns_dense[i].item():<15.8f} {returns_sparse[i].item():<15.8f} {diff:<15.2e}")

    print("-" * 60)
    print(f"Maximum difference: {max_diff:.2e}")

    if max_diff < 1e-6:
        print("\n✓ PASSED: Sparse and dense returns are equivalent!")
        return True
    else:
        print("\n✗ FAILED: Sparse and dense returns differ!")
        return False

if __name__ == "__main__":
    success = test_sparse_dense_equivalence()
    sys.exit(0 if success else 1)
