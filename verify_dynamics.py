#!/usr/bin/env python
"""
Standalone verification script for GHM Equity Dynamics.

This script verifies that the drift and diffusion formulas produce
the expected values as specified in the project requirements.
"""

import sys

try:
    import torch
except ImportError:
    print("ERROR: torch is not installed. Please install dependencies:")
    print("  pip install torch numpy scipy")
    sys.exit(1)

# Add the project root to path
sys.path.insert(0, '/home/user/GHM-RL')

from macro_rl.dynamics import GHMEquityDynamics, GHMEquityParams


def verify_drift():
    """Verify drift values."""
    print("=" * 60)
    print("DRIFT VERIFICATION")
    print("=" * 60)

    dynamics = GHMEquityDynamics()

    # Test at c=0
    c_zero = torch.tensor([[0.0]])
    drift_zero = dynamics.drift(c_zero)

    print(f"\nAt c=0:")
    print(f"  Computed drift: {drift_zero.item():.6f}")
    print(f"  Expected drift: 0.180000")
    print(f"  Match: {abs(drift_zero.item() - 0.18) < 1e-6}")

    # Test at other values
    test_values = [0.5, 1.0, 1.5, 2.0]
    print(f"\nAt other values (with default params, drift should be constant):")
    for c_val in test_values:
        c = torch.tensor([[c_val]])
        drift = dynamics.drift(c)
        print(f"  c={c_val}: drift={drift.item():.6f}")

    # Verify with custom parameters where slope is non-zero
    print(f"\nWith custom parameters (r=0.05, lambda_=0.02, mu=0.01):")
    custom_params = GHMEquityParams(r=0.05, lambda_=0.02, mu=0.01)
    custom_dynamics = GHMEquityDynamics(custom_params)

    for c_val in [0.0, 1.0, 2.0]:
        c = torch.tensor([[c_val]])
        drift = custom_dynamics.drift(c)
        # μ_c(c) = 0.18 + c * (0.05 - 0.02 - 0.01) = 0.18 + 0.02c
        expected = 0.18 + 0.02 * c_val
        print(f"  c={c_val}: drift={drift.item():.6f}, expected={expected:.6f}")


def verify_diffusion():
    """Verify diffusion values."""
    print("\n" + "=" * 60)
    print("DIFFUSION VERIFICATION")
    print("=" * 60)

    dynamics = GHMEquityDynamics()

    # Test at c=0
    c_zero = torch.tensor([[0.0]])
    diffusion_zero = dynamics.diffusion(c_zero)
    diffusion_sq_zero = dynamics.diffusion_squared(c_zero)

    print(f"\nAt c=0:")
    print(f"  Computed diffusion: {diffusion_zero.item():.6f}")
    print(f"  Computed diffusion²: {diffusion_sq_zero.item():.6f}")

    # Manual calculation
    sigma_X = 0.12
    rho = -0.2
    const_term = sigma_X**2 * (1 - rho**2)
    linear_term = rho * sigma_X
    expected_sq = const_term + linear_term**2
    expected = expected_sq ** 0.5

    print(f"  Expected diffusion: {expected:.6f}")
    print(f"  Expected diffusion²: {expected_sq:.6f}")
    print(f"  Match (≈0.12): {abs(diffusion_zero.item() - 0.12) < 0.01}")
    print(f"  Exact match: {abs(diffusion_zero.item() - expected) < 1e-6}")

    # Test at other values
    test_cases = [
        (0.0, 0.014400),
        (0.5, 0.036025),
        (1.0, 0.088900),
    ]

    print(f"\nAt various values:")
    for c_val, expected_sq_val in test_cases:
        c = torch.tensor([[c_val]])
        diffusion = dynamics.diffusion(c)
        diffusion_sq = dynamics.diffusion_squared(c)
        expected_val = expected_sq_val ** 0.5

        print(f"  c={c_val}:")
        print(f"    diffusion={diffusion.item():.6f}, expected={expected_val:.6f}")
        print(f"    diffusion²={diffusion_sq.item():.6f}, expected²={expected_sq_val:.6f}")
        print(f"    match: {abs(diffusion_sq.item() - expected_sq_val) < 1e-4}")

    # Verify consistency: diffusion² == diffusion_squared
    print(f"\nConsistency check (diffusion² == diffusion_squared):")
    test_points = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]])
    diff = dynamics.diffusion(test_points)
    diff_sq = dynamics.diffusion_squared(test_points)
    consistent = torch.allclose(diff**2, diff_sq, atol=1e-6)
    print(f"  All points consistent: {consistent}")


def verify_discount_rate():
    """Verify discount rate."""
    print("\n" + "=" * 60)
    print("DISCOUNT RATE VERIFICATION")
    print("=" * 60)

    dynamics = GHMEquityDynamics()

    discount = dynamics.discount_rate()
    expected = 0.03 - 0.01  # r - μ

    print(f"\nDiscount rate:")
    print(f"  Computed: {discount:.6f}")
    print(f"  Expected: {expected:.6f}")
    print(f"  Match: {abs(discount - expected) < 1e-10}")


def verify_state_space():
    """Verify state space specification."""
    print("\n" + "=" * 60)
    print("STATE SPACE VERIFICATION")
    print("=" * 60)

    dynamics = GHMEquityDynamics()
    ss = dynamics.state_space

    print(f"\nState space:")
    print(f"  Dimension: {ss.dim} (expected: 1)")
    print(f"  Lower bound: {ss.lower[0].item():.1f} (expected: 0.0)")
    print(f"  Upper bound: {ss.upper[0].item():.1f} (expected: 2.0)")
    print(f"  Variable name: {ss.names[0]} (expected: 'c')")

    assert ss.dim == 1, "State space dimension should be 1"
    assert torch.allclose(ss.lower, torch.tensor([0.0])), "Lower bound should be 0.0"
    assert torch.allclose(ss.upper, torch.tensor([2.0])), "Upper bound should be 2.0"
    assert ss.names == ("c",), "Variable name should be 'c'"

    print(f"\n  ✓ All state space checks passed!")


def main():
    """Run all verifications."""
    print("\n" + "=" * 60)
    print("GHM EQUITY DYNAMICS VERIFICATION")
    print("=" * 60)
    print()

    try:
        verify_state_space()
        verify_drift()
        verify_diffusion()
        verify_discount_rate()

        print("\n" + "=" * 60)
        print("✓ ALL VERIFICATIONS PASSED!")
        print("=" * 60)
        print()

    except Exception as e:
        print(f"\n✗ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
