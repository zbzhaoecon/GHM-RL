"""
Verification script for Phase 1: Core Abstractions

This script tests the StateSpace class and parameter utilities.
"""

import torch
from dataclasses import dataclass
from macro_rl.core import StateSpace, validate_params, params_to_dict


def test_state_space():
    """Test StateSpace implementation."""
    print("=" * 60)
    print("Testing StateSpace Implementation")
    print("=" * 60)

    # Create state space: c ∈ [0, 2], τ ∈ [0, 50]
    ss = StateSpace(dim=2, lower=[0, 0], upper=[2, 50], names=('c', 'τ'))

    print(f"✓ Created StateSpace: dim={ss.dim}, names={ss.names}")
    print(f"  Lower bounds: {ss.lower}")
    print(f"  Upper bounds: {ss.upper}")

    # Test 1: Sample uniformly
    print("\n[Test 1] Uniform sampling...")
    samples = ss.sample_uniform(1000)
    assert samples.shape == (1000, 2), f"Expected shape (1000, 2), got {samples.shape}"
    print(f"✓ Sample shape: {samples.shape}")

    # Test 2: Check bounds
    print("\n[Test 2] Checking sample bounds...")
    assert (samples >= 0).all(), "Some samples below lower bound"
    assert (samples[:, 0] <= 2).all(), "Some c samples above upper bound"
    assert (samples[:, 1] <= 50).all(), "Some τ samples above upper bound"
    print(f"✓ All samples within bounds")
    print(f"  c range: [{samples[:, 0].min():.4f}, {samples[:, 0].max():.4f}]")
    print(f"  τ range: [{samples[:, 1].min():.4f}, {samples[:, 1].max():.4f}]")

    # Test 3: Normalize/denormalize
    print("\n[Test 3] Testing normalize/denormalize...")
    test_state = torch.tensor([[1.0, 25.0]])
    normalized = ss.normalize(test_state)
    denormalized = ss.denormalize(normalized)
    assert torch.allclose(test_state, denormalized), "Normalize/denormalize roundtrip failed"
    print(f"✓ Normalize/denormalize roundtrip successful")
    print(f"  Original: {test_state}")
    print(f"  Normalized: {normalized}")
    print(f"  Denormalized: {denormalized}")

    # Test 4: Contains check
    print("\n[Test 4] Testing contains method...")
    valid_states = torch.tensor([[1.0, 25.0], [0.0, 0.0], [2.0, 50.0]])
    invalid_states = torch.tensor([[-1.0, 25.0], [3.0, 25.0], [1.0, 51.0]])

    valid_check = ss.contains(valid_states)
    invalid_check = ss.contains(invalid_states)

    assert valid_check.all(), "Valid states marked as invalid"
    assert not invalid_check.any(), "Invalid states marked as valid"
    print(f"✓ Contains check working correctly")
    print(f"  Valid states: {valid_check}")
    print(f"  Invalid states: {invalid_check}")

    # Test 5: Clip
    print("\n[Test 5] Testing clip method...")
    out_of_bounds = torch.tensor([[-1.0, 60.0], [3.0, -5.0]])
    clipped = ss.clip(out_of_bounds)
    assert ss.contains(clipped).all(), "Clipped states still out of bounds"
    print(f"✓ Clip working correctly")
    print(f"  Original: {out_of_bounds}")
    print(f"  Clipped: {clipped}")

    print("\n" + "=" * 60)
    print("✓ All StateSpace tests passed!")
    print("=" * 60)


def test_params():
    """Test parameter utilities."""
    print("\n" + "=" * 60)
    print("Testing Parameter Utilities")
    print("=" * 60)

    # Define a test parameter dataclass
    @dataclass
    class TestParams:
        r: float
        sigma: float
        bounds: torch.Tensor = torch.tensor([0.0, 1.0])

    # Test 1: validate_params
    print("\n[Test 1] Testing validate_params...")
    params = TestParams(r=0.05, sigma=0.2)
    validate_params(params)
    print(f"✓ Valid params accepted")

    # Test invalid params
    try:
        validate_params("not a dataclass")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"✓ Invalid params rejected: {e}")

    # Test 2: params_to_dict
    print("\n[Test 2] Testing params_to_dict...")
    params_dict = params_to_dict(params)
    assert params_dict['r'] == 0.05
    assert params_dict['sigma'] == 0.2
    assert isinstance(params_dict['bounds'], list)
    print(f"✓ Conversion to dict successful:")
    print(f"  {params_dict}")

    # Test tensor conversion
    assert params_dict['bounds'] == [0.0, 1.0], "Tensor not converted to list correctly"
    print(f"✓ Tensor conversion working correctly")

    print("\n" + "=" * 60)
    print("✓ All parameter utility tests passed!")
    print("=" * 60)


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION: Core Abstractions")
    print("=" * 60 + "\n")

    try:
        test_state_space()
        test_params()

        print("\n" + "=" * 60)
        print("🎉 ALL PHASE 1 TESTS PASSED! 🎉")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
