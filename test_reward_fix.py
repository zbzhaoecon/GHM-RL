"""
Test to verify the equity issuance cost correction.

This test verifies that the reward function correctly computes the
cost to existing shareholders as (p-1)/p * a_E, not (p-1) * a_E.
"""

import torch
from macro_rl.rewards.ghm_rewards import GHMRewardFunction

def test_issuance_cost_calculation():
    """Test that issuance cost is calculated correctly."""

    # Parameters
    p = 1.06  # Proportional cost from model
    phi = 0.002  # Fixed cost
    dt = 0.1

    # Create reward function with OLD formulation (should be auto-converted)
    reward_fn = GHMRewardFunction(
        discount_rate=0.02,
        issuance_cost=0.06,  # This is (p-1), should be converted to (p-1)/p
        fixed_cost=phi,
        proportional_cost=p,
    )

    # Test state and action
    state = torch.tensor([[1.0, 5.0]])  # c=1.0, τ=5.0
    action = torch.tensor([[0.5, 0.1]])  # a_L=0.5, a_E=0.1
    next_state = torch.tensor([[1.0, 5.0]])

    # Compute reward
    reward = reward_fn.step_reward(state, action, next_state, dt)

    # Expected reward with CORRECT formulation:
    # r = a_L * dt - (p-1)/p * a_E - φ
    # r = 0.5 * 0.1 - (0.06/1.06) * 0.1 - 0.002
    # r = 0.05 - 0.00566 - 0.002
    # r = 0.04234
    expected_correct = 0.5 * dt - (0.06 / 1.06) * 0.1 - phi

    # OLD (incorrect) formulation would give:
    # r = 0.5 * 0.1 - 0.06 * 0.1 - 0.002 = 0.04200
    expected_old = 0.5 * dt - 0.06 * 0.1 - phi

    print("=" * 60)
    print("Equity Issuance Cost Correction Test")
    print("=" * 60)
    print(f"Parameters: p={p}, φ={phi}, dt={dt}")
    print(f"Action: a_L={action[0,0]}, a_E={action[0,1]}")
    print()
    print(f"Stored issuance_cost: {reward_fn.issuance_cost:.6f}")
    print(f"Expected (p-1)/p:     {(p-1)/p:.6f}")
    print()
    print(f"Actual reward:        {reward.item():.6f}")
    print(f"Expected (correct):   {expected_correct:.6f}")
    print(f"Expected (old/wrong): {expected_old:.6f}")
    print()

    # Verify the conversion happened
    assert abs(reward_fn.issuance_cost - (p-1)/p) < 1e-6, \
        f"issuance_cost not converted correctly: {reward_fn.issuance_cost} != {(p-1)/p}"

    # Verify reward is correct
    assert abs(reward.item() - expected_correct) < 1e-6, \
        f"Reward calculation incorrect: {reward.item()} != {expected_correct}"

    # Verify it's different from old formulation
    assert abs(reward.item() - expected_old) > 1e-4, \
        f"Reward matches old (incorrect) formulation!"

    print("✅ All checks passed!")
    print(f"✅ Cost correctly computed as (p-1)/p = {(p-1)/p:.6f}")
    print(f"✅ Difference from old formulation: {abs(reward.item() - expected_old):.6f}")
    print("=" * 60)

if __name__ == "__main__":
    test_issuance_cost_calculation()
