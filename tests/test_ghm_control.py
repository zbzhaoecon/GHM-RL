"""
Unit tests for control/ghm_control.py - GHMControlSpec class.
"""

import torch
from macro_rl.control.ghm_control import GHMControlSpec


def test_ghm_control_initialization():
    """Test GHMControlSpec initialization."""
    spec = GHMControlSpec(
        a_L_max=10.0,
        a_E_max=0.5,
        issuance_threshold=0.05,
        issuance_cost=0.1,
    )

    assert spec.dim == 2
    assert len(spec.names) == 2
    assert spec.names[0] == "dividend"
    assert spec.names[1] == "equity_issuance"
    assert spec.a_L_max == 10.0
    assert spec.a_E_max == 0.5
    assert spec.issuance_threshold == 0.05
    assert spec.issuance_cost == 0.1


def test_ghm_control_basic_masking():
    """Test basic action masking."""
    spec = GHMControlSpec(a_L_max=10.0, a_E_max=0.5, issuance_threshold=0.05)

    action = torch.tensor([[0.5, 0.3]])        # (batch=1, 2)
    state = torch.tensor([[0.1, 0.5]])         # cash=0.1 (state[:, 0])

    dt = 0.01
    masked = spec.apply_mask(action, state, dt=dt)

    # Dividend cannot exceed available cash / dt
    # cash / dt = 0.1 / 0.01 = 10.0
    assert masked.shape == action.shape
    assert masked[0, 0] <= 0.1 / dt + 1e-6

    # Equity issuance is within [0, a_E_max]
    assert 0.0 <= masked[0, 1] <= spec.upper[1] + 1e-6


def test_ghm_control_dividend_constraint():
    """Test dividend constraint: a_L * dt <= c."""
    spec = GHMControlSpec(a_L_max=10.0, a_E_max=0.5, issuance_threshold=0.05)

    # Low cash scenario
    action = torch.tensor([[5.0, 0.0]])  # High dividend
    state = torch.tensor([[0.02, 0.0]])  # Low cash: c=0.02
    dt = 0.01

    masked = spec.apply_mask(action, state, dt=dt)

    # Maximum dividend should be c/dt = 0.02/0.01 = 2.0
    max_allowed_dividend = state[0, 0] / dt
    assert masked[0, 0] <= max_allowed_dividend + 1e-6

    # High cash scenario
    action = torch.tensor([[5.0, 0.0]])
    state = torch.tensor([[1.0, 0.0]])  # High cash: c=1.0
    dt = 0.01

    masked = spec.apply_mask(action, state, dt=dt)

    # With high cash, dividend should be limited by a_L_max
    # c/dt = 1.0/0.01 = 100, but a_L_max = 10.0
    assert torch.allclose(masked[0, 0], torch.tensor(5.0))  # Original action within bounds


def test_ghm_control_suppresses_small_issuance():
    """Test that small issuance below threshold is suppressed."""
    spec = GHMControlSpec(a_L_max=10.0, a_E_max=0.5, issuance_threshold=0.1)

    # Below threshold: should be zeroed out
    # Threshold = 0.1 * 0.5 = 0.05
    low_issuance = torch.tensor([[0.2, 0.01]])     # a_E = 0.01 < 0.05
    state = torch.tensor([[1.0, 0.5]])
    masked_low = spec.apply_mask(low_issuance, state, dt=0.01)
    assert torch.allclose(masked_low[0, 1], torch.tensor(0.0))

    # Above threshold: should pass through (up to clipping)
    high_issuance = torch.tensor([[0.2, 0.4]])    # a_E = 0.4 >= 0.05
    masked_high = spec.apply_mask(high_issuance, state, dt=0.01)
    assert masked_high[0, 1] > 0.0
    assert torch.allclose(masked_high[0, 1], torch.tensor(0.4))


def test_ghm_control_batched_masking():
    """Test masking with batched inputs."""
    spec = GHMControlSpec(a_L_max=10.0, a_E_max=0.5, issuance_threshold=0.05)

    # Batch of 3 samples
    actions = torch.tensor([
        [5.0, 0.3],  # High dividend, high issuance
        [1.0, 0.01], # Low dividend, low issuance (below threshold)
        [2.0, 0.0],  # Medium dividend, no issuance
    ])
    states = torch.tensor([
        [0.05, 0.0],  # Low cash
        [1.0, 0.0],   # High cash
        [0.5, 0.0],   # Medium cash
    ])
    dt = 0.01

    masked = spec.apply_mask(actions, states, dt=dt)

    # Sample 0: dividend should be constrained by cash
    assert masked[0, 0] <= states[0, 0] / dt + 1e-6

    # Sample 1: issuance should be zeroed (below threshold)
    assert torch.allclose(masked[1, 1], torch.tensor(0.0))

    # Sample 2: no issuance, should remain zero
    assert torch.allclose(masked[2, 1], torch.tensor(0.0))


def test_ghm_control_compute_net_payout():
    """Test net payout computation."""
    spec = GHMControlSpec()

    action = torch.tensor([
        [1.0, 0.0],
        [0.5, 0.2],
        [0.0, 0.1],
    ])

    net = spec.compute_net_payout(action)

    # net_payout = a_L - a_E
    expected = torch.tensor([1.0, 0.3, -0.1])
    assert torch.allclose(net, expected)


def test_ghm_control_issuance_indicator():
    """Test issuance indicator computation."""
    spec = GHMControlSpec()

    action = torch.tensor([
        [1.0, 0.0],   # No issuance
        [0.5, 0.2],   # Issuance
        [0.0, 0.001], # Tiny issuance (still counts)
    ])

    indicator = spec.issuance_indicator(action)

    expected = torch.tensor([0.0, 1.0, 1.0])
    assert torch.allclose(indicator, expected)


def test_ghm_control_total_issuance_cost():
    """Test total issuance cost computation."""
    spec = GHMControlSpec(issuance_cost=0.1)

    action = torch.tensor([
        [1.0, 0.0],   # No issuance
        [0.5, 0.2],   # Issuance
        [0.0, 0.0],   # No issuance
    ])

    cost = spec.total_issuance_cost(action)

    # cost = λ * 𝟙(a_E > 0)
    expected = torch.tensor([0.0, 0.1, 0.0])
    assert torch.allclose(cost, expected)


def test_ghm_control_verification_example():
    """Test the verification example from the Phase 2 guide."""
    control = GHMControlSpec()
    action = torch.tensor([[0.5, 0.3]])
    state = torch.tensor([[0.1, 0.5]])  # Low cash
    masked = control.apply_mask(action, state, dt=0.01)

    # Can't overdraw
    assert masked[0, 0] <= 0.1 / 0.01 + 1e-6
