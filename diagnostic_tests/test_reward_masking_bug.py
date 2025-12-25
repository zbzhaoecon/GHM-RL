"""
Test to demonstrate the critical reward masking bug at bankruptcy.

This test shows that when bankruptcy occurs, the reward earned at the
bankruptcy step is NOT zeroed out, causing the agent to receive both
the dividend payout AND the liquidation value.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
from macro_rl.rewards.ghm_rewards import GHMEquityReward
from macro_rl.simulation.trajectory import TrajectorySimulator
from macro_rl.control.ghm_control import GHMEquityControlSpec


def test_bankruptcy_reward_bug():
    """
    Demonstrate the reward masking bug.

    Setup:
    - Start with cash c = 0.1
    - Pay huge dividend (e.g., 10/dt) which exceeds available cash
    - This causes bankruptcy in the next state

    Expected (CORRECT behavior):
    - Reward at bankruptcy step should be ZEROED out
    - Agent should only receive liquidation value

    Actual (BUG):
    - Reward at bankruptcy step is NOT zeroed
    - Agent receives BOTH dividend reward AND liquidation value
    """

    print("=" * 80)
    print("TEST: Bankruptcy Reward Masking Bug")
    print("=" * 80)

    # Setup
    params = GHMEquityParams()
    print(f"\nParameters:")
    print(f"  Liquidation value: {params.liquidation_value:.4f}")
    print(f"  Issuance cost: {params.issuance_cost:.4f}")

    dynamics = GHMEquityDynamics(params)
    reward_fn = GHMEquityReward(issuance_cost=params.issuance_cost)
    control_spec = GHMEquityControlSpec()

    dt = 0.1
    max_steps = 3

    simulator = TrajectorySimulator(
        dynamics=dynamics,
        reward_fn=reward_fn,
        control_spec=control_spec,
        dt=dt,
        max_steps=max_steps
    )

    # Create a policy that pays huge dividends (forcing bankruptcy)
    class BankruptcyPolicy:
        def act(self, state):
            batch_size = state.shape[0]
            # Pay dividend rate of 100 (way more than available cash)
            # No equity issuance
            return torch.tensor([[100.0, 0.0]] * batch_size, dtype=state.dtype)

    policy = BankruptcyPolicy()

    # Initial state: small cash reserves
    initial_state = torch.tensor([[0.1]], dtype=torch.float32)

    print(f"\nInitial state: c = {initial_state[0, 0]:.4f}")
    print(f"Action: dividend_rate = 100.0 (will cause bankruptcy)")

    # Simulate
    batch = simulator.simulate(
        initial_state=initial_state,
        policy=policy,
        num_trajectories=1
    )

    print("\n" + "=" * 80)
    print("TRAJECTORY RESULTS")
    print("=" * 80)

    for t in range(max_steps):
        state_t = batch.states[0, t, 0].item()
        action_t = batch.actions[0, t, :].numpy()
        reward_t = batch.rewards[0, t].item()
        mask_t = batch.masks[0, t].item()

        print(f"\nStep {t}:")
        print(f"  State: c = {state_t:.6f}")
        print(f"  Action: dividend_rate = {action_t[0]:.4f}, equity = {action_t[1]:.4f}")
        print(f"  Reward: {reward_t:.6f}")
        print(f"  Mask: {mask_t:.1f}")

        if t < max_steps - 1:
            next_state = batch.states[0, t + 1, 0].item()
            print(f"  Next state: c = {next_state:.6f}")

            if next_state <= 0:
                print(f"  ⚠️  BANKRUPTCY at step {t}!")

    # Compute total return
    total_return = batch.returns[0].item()
    print(f"\nTotal Return: {total_return:.6f}")

    # Expected vs Actual
    liquidation_value = params.liquidation_value

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find bankruptcy step
    bankruptcy_step = None
    for t in range(max_steps - 1):
        if batch.states[0, t + 1, 0] <= 0:
            bankruptcy_step = t
            break

    if bankruptcy_step is not None:
        reward_at_bankruptcy = batch.rewards[0, bankruptcy_step].item()
        mask_at_bankruptcy = batch.masks[0, bankruptcy_step].item()

        print(f"\nBankruptcy occurred at step {bankruptcy_step}")
        print(f"Reward at bankruptcy step: {reward_at_bankruptcy:.6f}")
        print(f"Mask at bankruptcy step: {mask_at_bankruptcy:.1f}")

        print(f"\n🔍 BUG CHECK:")
        if mask_at_bankruptcy > 0.5 and abs(reward_at_bankruptcy) > 1e-6:
            print(f"  ❌ BUG CONFIRMED!")
            print(f"  The reward at bankruptcy step is NOT zeroed out.")
            print(f"  Agent receives BOTH dividend ({reward_at_bankruptcy:.6f}) AND liquidation ({liquidation_value:.6f})")
            print(f"  This makes bankruptcy highly rewarding!")

            expected_return_bug = reward_at_bankruptcy + liquidation_value
            print(f"\n  Expected return (with bug): {expected_return_bug:.6f}")
            print(f"  Actual return: {total_return:.6f}")
            print(f"  Difference: {abs(total_return - expected_return_bug):.6f}")

        else:
            print(f"  ✅ NO BUG: Reward is properly zeroed out")
            print(f"  Expected return (correct): {liquidation_value:.6f}")
            print(f"  Actual return: {total_return:.6f}")

    print("\n" + "=" * 80)

    return bankruptcy_step is not None and mask_at_bankruptcy > 0.5 and abs(reward_at_bankruptcy) > 1e-6


if __name__ == "__main__":
    has_bug = test_bankruptcy_reward_bug()

    if has_bug:
        print("\n⚠️  CRITICAL BUG DETECTED!")
        print("The reward masking logic is incorrect.")
        print("See DIAGNOSTIC_PLAN.md for fix.")
        sys.exit(1)
    else:
        print("\n✅ Test passed: Reward masking is correct")
        sys.exit(0)
