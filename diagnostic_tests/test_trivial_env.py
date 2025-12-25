"""
Test with a trivial environment to verify basic RL works.

This tests whether the RL algorithm can learn a simple policy
in a trivial environment with no economic complexity.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from macro_rl.policies.mlp_policy import MLPPolicy
from macro_rl.optimizers.policy_gradient import PolicyGradientOptimizer


class TrivialReward:
    """Reward = action (higher action = higher reward)."""

    def step_reward(self, state, action, next_state, dt):
        """Reward is just the action value."""
        return action.squeeze()

    def terminal_reward(self, state, terminated, value_function=None):
        """No terminal reward."""
        return torch.zeros(state.shape[0])


class TrivialDynamics:
    """State doesn't change."""

    def drift(self, state):
        return torch.zeros_like(state)

    def diffusion(self, state):
        return torch.zeros_like(state)

    def discount_rate(self):
        return 0.05


class TrivialControlSpec:
    """Action bounded in [0, 10]."""

    def apply_mask(self, action, state, dt):
        return torch.clamp(action, 0.0, 10.0)


def test_trivial_learning():
    """
    Test if RL can learn to maximize action in trivial environment.

    Optimal policy: action = 10.0 (max allowed)
    """

    print("=" * 80)
    print("TEST: Trivial Environment Learning")
    print("=" * 80)
    print("\nObjective: Learn to output action = 10.0")
    print("Reward: r = action (higher is better)")
    print("Constraint: action ∈ [0, 10]")

    # Setup
    policy = MLPPolicy(
        state_dim=1,
        action_dim=1,
        hidden_dims=[32, 32],
        activation=nn.ReLU
    )

    optimizer = PolicyGradientOptimizer(
        policy=policy,
        learning_rate=0.01
    )

    # Training
    num_iterations = 100
    batch_size = 32

    print(f"\nTraining for {num_iterations} iterations...")

    for i in range(num_iterations):
        # Random states (doesn't matter since dynamics don't use it)
        states = torch.randn(batch_size, 1)

        # Get actions from policy
        actions = policy.act(states)

        # Apply constraints
        actions = torch.clamp(actions, 0.0, 10.0)

        # Compute rewards
        rewards = actions.squeeze()

        # Simple policy gradient update
        log_probs = policy.log_prob(states, actions)
        loss = -(log_probs * rewards).mean()

        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.optimizer.step()

        if (i + 1) % 20 == 0:
            mean_action = actions.mean().item()
            mean_reward = rewards.mean().item()
            print(f"  Iter {i+1:3d}: Mean action = {mean_action:.4f}, Mean reward = {mean_reward:.4f}")

    # Test
    print("\n" + "=" * 80)
    print("TESTING")
    print("=" * 80)

    test_states = torch.randn(100, 1)
    test_actions = policy.act(test_states)
    test_actions = torch.clamp(test_actions, 0.0, 10.0)

    mean_action = test_actions.mean().item()
    std_action = test_actions.std().item()

    print(f"\nFinal policy:")
    print(f"  Mean action: {mean_action:.4f}")
    print(f"  Std action: {std_action:.4f}")

    print("\n🔍 TEST CHECK:")
    if mean_action > 9.0:
        print("  ✅ SUCCESS: Policy learned to maximize action!")
        print("  RL algorithm is working correctly.")
        return True
    else:
        print("  ❌ FAILURE: Policy did not converge to optimal action.")
        print("  Expected: ~10.0")
        print(f"  Got: {mean_action:.4f}")
        print("  This suggests fundamental issues with RL algorithm.")
        return False


if __name__ == "__main__":
    success = test_trivial_learning()
    sys.exit(0 if success else 1)
