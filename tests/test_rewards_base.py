"""
Unit tests for rewards/base.py - RewardFunction class.
"""

import torch
from macro_rl.rewards.base import RewardFunction


class DummyReward(RewardFunction):
    """Dummy reward function for testing."""

    def step_reward(self, state, action, next_state, dt):
        return torch.ones(state.shape[0])

    def terminal_reward(self, state, terminated):
        return 2.0 * terminated.to(state.dtype)


def test_trajectory_return_simple():
    """Test trajectory return computation with simple inputs."""
    dummy = DummyReward()

    rewards = torch.ones(2, 3)         # batch=2, T=3, all ones
    terminal = torch.tensor([0.0, 2.0])
    masks = torch.ones_like(rewards)
    discount_rate = 0.0  # No discounting
    dt = 0.1

    returns = dummy.trajectory_return(rewards, terminal, masks, discount_rate, dt)

    # With no discounting: sum of rewards + terminal
    # Sample 0: 1 + 1 + 1 + 0 = 3
    # Sample 1: 1 + 1 + 1 + 2 = 5
    expected = torch.tensor([3.0, 5.0])
    assert torch.allclose(returns, expected, atol=1e-5)


def test_trajectory_return_with_discounting():
    """Test trajectory return with continuous-time discounting."""
    dummy = DummyReward()

    rewards = torch.ones(1, 3)         # batch=1, T=3
    terminal = torch.tensor([0.0])
    masks = torch.ones_like(rewards)
    discount_rate = 0.1
    dt = 1.0

    returns = dummy.trajectory_return(rewards, terminal, masks, discount_rate, dt)

    # R = e^(-0.1*0*1)*1 + e^(-0.1*1*1)*1 + e^(-0.1*2*1)*1 + e^(-0.1*3*1)*0
    #   = 1.0 + exp(-0.1) + exp(-0.2) + 0
    import math
    expected = 1.0 + math.exp(-0.1) + math.exp(-0.2)
    assert torch.allclose(returns, torch.tensor([expected]), atol=1e-5)


def test_trajectory_return_with_masking():
    """Test trajectory return with early termination."""
    dummy = DummyReward()

    # Sample terminates at t=1
    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    terminal = torch.tensor([2.0])
    masks = torch.tensor([[1.0, 0.0, 0.0]])  # Active only at t=0
    discount_rate = 0.0
    dt = 0.1

    returns = dummy.trajectory_return(rewards, terminal, masks, discount_rate, dt)

    # Only first reward counts: 1.0 + terminal
    expected = torch.tensor([1.0 + 2.0])
    assert torch.allclose(returns, expected, atol=1e-5)


def test_trajectory_return_batched():
    """Test trajectory return with batched inputs."""
    dummy = DummyReward()

    rewards = torch.tensor([
        [1.0, 2.0, 3.0],
        [0.5, 0.5, 0.5],
    ])
    terminal = torch.tensor([1.0, 0.0])
    masks = torch.ones_like(rewards)
    discount_rate = 0.0
    dt = 0.1

    returns = dummy.trajectory_return(rewards, terminal, masks, discount_rate, dt)

    # Sample 0: 1 + 2 + 3 + 1 = 7
    # Sample 1: 0.5 + 0.5 + 0.5 + 0 = 1.5
    expected = torch.tensor([7.0, 1.5])
    assert torch.allclose(returns, expected, atol=1e-5)


def test_cumulative_reward():
    """Test cumulative reward computation."""
    dummy = DummyReward()

    rewards = torch.tensor([
        [1.0, 2.0, 3.0],
        [0.5, 0.5, 0.5],
    ])
    masks = torch.ones_like(rewards)

    cumulative = dummy.cumulative_reward(rewards, masks)

    expected = torch.tensor([6.0, 1.5])
    assert torch.allclose(cumulative, expected)


def test_cumulative_reward_with_masking():
    """Test cumulative reward with masking."""
    dummy = DummyReward()

    rewards = torch.tensor([
        [1.0, 2.0, 3.0],
        [0.5, 0.5, 0.5],
    ])
    masks = torch.tensor([
        [1.0, 1.0, 0.0],  # Last step masked
        [1.0, 0.0, 0.0],  # Last two steps masked
    ])

    cumulative = dummy.cumulative_reward(rewards, masks)

    # Sample 0: 1 + 2 = 3
    # Sample 1: 0.5 = 0.5
    expected = torch.tensor([3.0, 0.5])
    assert torch.allclose(cumulative, expected)
