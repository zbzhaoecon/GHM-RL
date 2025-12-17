"""
Unit tests for rewards/ghm_rewards.py - GHMRewardFunction class.
"""

import torch
from macro_rl.rewards.ghm_rewards import GHMRewardFunction


def test_ghm_reward_initialization():
    """Test GHMRewardFunction initialization."""
    reward_fn = GHMRewardFunction(
        discount_rate=0.03,
        issuance_cost=0.1,
        liquidation_rate=0.8,
        liquidation_flow=0.5,
    )

    assert reward_fn.discount_rate_value == 0.03
    assert reward_fn.issuance_cost == 0.1
    assert reward_fn.liquidation_rate == 0.8
    assert reward_fn.liquidation_flow == 0.5

    # Liquidation value should be omega * alpha / rho
    expected_liquidation = 0.8 * 0.5 / 0.03
    assert abs(reward_fn.liquidation_value - expected_liquidation) < 1e-6


def test_ghm_step_reward():
    """Test per-step reward computation."""
    reward_fn = GHMRewardFunction(
        discount_rate=0.03,
        issuance_cost=0.1,
    )

    state = torch.zeros(3, 2)
    next_state = torch.zeros_like(state)
    action = torch.tensor([
        [1.0, 0.1],   # a_L=1.0, a_E=0.1
        [0.5, 0.0],   # a_L=0.5, a_E=0.0
        [0.0, 0.2],   # a_L=0.0, a_E=0.2
    ])
    dt = 0.1

    r = reward_fn.step_reward(state, action, next_state, dt)

    # r_i = a_L * dt - (1 + λ) * a_E
    expected = torch.tensor([
        1.0 * 0.1 - 1.1 * 0.1,   # 0.1 - 0.11 = -0.01
        0.5 * 0.1 - 1.1 * 0.0,   # 0.05 - 0.0 = 0.05
        0.0 * 0.1 - 1.1 * 0.2,   # 0.0 - 0.22 = -0.22
    ])
    assert torch.allclose(r, expected, atol=1e-6)


def test_ghm_step_reward_no_issuance_cost():
    """Test step reward with no issuance cost."""
    reward_fn = GHMRewardFunction(
        discount_rate=0.03,
        issuance_cost=0.0,
    )

    state = torch.zeros(2, 2)
    next_state = torch.zeros_like(state)
    action = torch.tensor([
        [1.0, 0.1],
        [0.5, 0.2],
    ])
    dt = 0.1

    r = reward_fn.step_reward(state, action, next_state, dt)

    # r_i = a_L * dt - a_E (no cost multiplier)
    expected = torch.tensor([
        1.0 * 0.1 - 0.1,  # 0.1 - 0.1 = 0.0
        0.5 * 0.1 - 0.2,  # 0.05 - 0.2 = -0.15
    ])
    assert torch.allclose(r, expected, atol=1e-6)


def test_ghm_terminal_reward():
    """Test terminal reward computation."""
    reward_fn = GHMRewardFunction(
        discount_rate=0.02,
        liquidation_rate=0.8,
        liquidation_flow=0.5,
    )

    # Liquidation value = 0.8 * 0.5 / 0.02 = 20.0

    state = torch.zeros(3, 2)
    terminated = torch.tensor([0, 1, 1], dtype=torch.int32)

    term_r = reward_fn.terminal_reward(state, terminated)

    expected = torch.tensor([0.0, 20.0, 20.0])
    assert torch.allclose(term_r, expected, atol=1e-5)


def test_ghm_terminal_reward_boolean_mask():
    """Test terminal reward with boolean mask."""
    reward_fn = GHMRewardFunction(
        discount_rate=0.04,
        liquidation_rate=0.5,
        liquidation_flow=0.2,
    )

    # Liquidation value = 0.5 * 0.2 / 0.04 = 2.5

    state = torch.zeros(2, 2)
    terminated = torch.tensor([False, True])

    term_r = reward_fn.terminal_reward(state, terminated)

    expected = torch.tensor([0.0, 2.5])
    assert torch.allclose(term_r, expected, atol=1e-5)


def test_ghm_net_payout():
    """Test net payout computation."""
    reward_fn = GHMRewardFunction(
        discount_rate=0.03,
        issuance_cost=0.1,
    )

    action = torch.tensor([
        [1.0, 0.1],
        [0.5, 0.0],
        [0.0, 0.2],
    ])
    dt = 0.1

    net = reward_fn.net_payout(action, dt)

    # Should match step_reward
    state = torch.zeros(3, 2)
    expected = reward_fn.step_reward(state, action, state, dt)
    assert torch.allclose(net, expected, atol=1e-6)


def test_ghm_total_issuance_cost():
    """Test total issuance cost computation."""
    reward_fn = GHMRewardFunction(
        discount_rate=0.03,
        issuance_cost=0.2,
    )

    action = torch.tensor([
        [1.0, 0.1],
        [0.5, 0.0],
        [0.0, 0.5],
    ])

    cost = reward_fn.total_issuance_cost(action)

    # cost = (1 + λ) * a_E
    expected = torch.tensor([
        1.2 * 0.1,  # 0.12
        1.2 * 0.0,  # 0.0
        1.2 * 0.5,  # 0.6
    ])
    assert torch.allclose(cost, expected, atol=1e-6)


def test_ghm_reward_integration():
    """Test integrated reward computation over trajectory."""
    reward_fn = GHMRewardFunction(
        discount_rate=0.02,
        issuance_cost=0.1,
        liquidation_rate=0.8,
        liquidation_flow=0.1,
    )

    # Simple trajectory: 3 time steps, 1 sample
    states = torch.zeros(1, 3, 2)
    next_states = torch.zeros_like(states)
    actions = torch.tensor([
        [[1.0, 0.0], [0.5, 0.0], [0.2, 0.1]]
    ])  # shape: (1, 3, 2)

    dt = 0.1

    # Compute per-step rewards
    rewards = torch.zeros(1, 3)
    for t in range(3):
        rewards[:, t] = reward_fn.step_reward(
            states[:, t],
            actions[:, t],
            next_states[:, t],
            dt
        )

    # Terminal reward
    terminated = torch.tensor([1])
    terminal_r = reward_fn.terminal_reward(states[:, -1], terminated)

    # Compute trajectory return
    masks = torch.ones_like(rewards)
    returns = reward_fn.trajectory_return(
        rewards,
        terminal_r,
        masks,
        reward_fn.discount_rate_value,
        dt
    )

    # Just check it computes without error and has correct shape
    assert returns.shape == (1,)
    assert not torch.isnan(returns).any()
