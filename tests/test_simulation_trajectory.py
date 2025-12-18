"""
Unit tests for TrajectorySimulator.
"""

import torch
from macro_rl.simulation.trajectory import TrajectorySimulator, TrajectoryBatch
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
import pytest


class RandomPolicy:
    """Simple random policy for testing."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def act(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        return torch.rand(batch_size, self.action_dim, device=state.device)


def make_ghm_simulator(dt=0.01, T=0.1):
    """Create GHM simulator for testing."""
    params = GHMEquityParams()
    dynamics = GHMEquityDynamics(params)
    control_spec = GHMControlSpec()
    reward_fn = GHMRewardFunction(
        discount_rate=params.r - params.mu,
        issuance_cost=params.lambda_,
        liquidation_rate=params.omega,
        liquidation_flow=params.alpha,
    )

    simulator = TrajectorySimulator(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        dt=dt,
        T=T,
    )

    return simulator, control_spec, reward_fn, dynamics, params


def test_trajectory_simulator_shapes():
    """Test that TrajectorySimulator produces correct shapes."""
    dt = 0.01
    T = 1.0
    simulator, control_spec, reward_fn, dynamics, params = make_ghm_simulator(dt=dt, T=T)

    batch = 8
    state_dim = dynamics.state_space.dim
    initial_states = torch.ones(batch, state_dim) * 1.0  # Start with c=1.0

    random_policy = RandomPolicy(action_dim=control_spec.dim)
    trajectories = simulator.rollout(random_policy, initial_states)

    num_steps = int(round(T / dt))

    assert trajectories.states.shape == (batch, num_steps + 1, state_dim)
    assert trajectories.actions.shape == (batch, num_steps, control_spec.dim)
    assert trajectories.rewards.shape == (batch, num_steps)
    assert trajectories.masks.shape == (batch, num_steps)
    assert trajectories.returns.shape == (batch,)
    assert trajectories.terminal_rewards.shape == (batch,)


def test_trajectory_batch_validation():
    """Test TrajectoryBatch validation."""
    batch = 10
    n_steps = 50
    state_dim = 1
    action_dim = 2

    states = torch.randn(batch, n_steps + 1, state_dim)
    actions = torch.randn(batch, n_steps, action_dim)
    rewards = torch.randn(batch, n_steps)
    masks = torch.ones(batch, n_steps)
    returns = torch.randn(batch)
    terminal_rewards = torch.randn(batch)

    # Should not raise
    traj = TrajectoryBatch(states, actions, rewards, masks, returns, terminal_rewards)

    assert traj.batch_size == batch
    assert traj.n_steps == n_steps


def test_trajectory_batch_device_transfer():
    """Test TrajectoryBatch.to() method."""
    batch = 5
    n_steps = 10
    state_dim = 1
    action_dim = 2

    traj = TrajectoryBatch(
        states=torch.randn(batch, n_steps + 1, state_dim),
        actions=torch.randn(batch, n_steps, action_dim),
        rewards=torch.randn(batch, n_steps),
        masks=torch.ones(batch, n_steps),
        returns=torch.randn(batch),
        terminal_rewards=torch.randn(batch),
    )

    # Test CPU to CPU transfer
    traj_cpu = traj.to(torch.device('cpu'))
    assert traj_cpu.states.device.type == 'cpu'


def test_trajectory_simulator_reproducibility():
    """Test that simulator is reproducible with fixed noise."""
    dt = 0.01
    T = 0.5
    simulator, control_spec, _, _, _ = make_ghm_simulator(dt=dt, T=T)

    batch = 10
    state_dim = 1
    initial_states = torch.ones(batch, state_dim) * 1.0

    num_steps = int(round(T / dt))
    noise = torch.randn(batch, num_steps, state_dim)

    random_policy = RandomPolicy(action_dim=control_spec.dim)

    # Run twice with same noise
    torch.manual_seed(42)
    traj1 = simulator.rollout(random_policy, initial_states, noise=noise)

    torch.manual_seed(42)
    traj2 = simulator.rollout(random_policy, initial_states, noise=noise)

    # States should be identical (actions are still random from policy)
    # But with same seed, policy actions should also be identical
    assert torch.allclose(traj1.states, traj2.states, atol=1e-6)


def test_trajectory_simulator_termination():
    """Test that simulator handles termination correctly."""
    dt = 0.01
    T = 1.0
    simulator, control_spec, _, _, _ = make_ghm_simulator(dt=dt, T=T)

    # Start with very low cash to trigger termination
    batch = 5
    state_dim = 1
    initial_states = torch.ones(batch, state_dim) * 0.01  # Very low cash

    # Policy that always pays out cash (should trigger termination)
    class DrainCashPolicy:
        def __init__(self, action_dim):
            self.action_dim = action_dim

        def act(self, state):
            batch_size = state.shape[0]
            # Large dividend, no issuance
            actions = torch.zeros(batch_size, self.action_dim)
            actions[:, 0] = 10.0  # Large dividend rate
            return actions

    drain_policy = DrainCashPolicy(action_dim=control_spec.dim)
    trajectories = simulator.rollout(drain_policy, initial_states)

    # Check that some trajectories terminated early
    # masks should have some zeros
    assert (trajectories.masks == 0).any(), "Expected some trajectories to terminate"

    # Terminal rewards should be non-zero for terminated trajectories
    # (depends on reward function implementation)


def test_compute_returns_discounting():
    """Test that returns computation uses correct discounting."""
    dt = 0.01
    T = 0.1
    simulator, _, _, _, _ = make_ghm_simulator(dt=dt, T=T)

    batch = 3
    n_steps = int(round(T / dt))

    # Create simple reward trajectory
    rewards = torch.ones(batch, n_steps)
    terminal_rewards = torch.ones(batch)
    masks = torch.ones(batch, n_steps)
    discount_rate = 0.02

    returns = simulator._compute_returns(rewards, terminal_rewards, masks, discount_rate)

    # Manual calculation
    expected = 0.0
    for t in range(n_steps):
        expected += torch.exp(torch.tensor(-discount_rate * t * dt)) * rewards[0, t]
    expected += torch.exp(torch.tensor(-discount_rate * n_steps * dt)) * terminal_rewards[0]

    assert torch.allclose(returns[0], expected, rtol=1e-5)


def test_check_termination():
    """Test termination condition."""
    dt = 0.01
    T = 0.1
    simulator, _, _, _, _ = make_ghm_simulator(dt=dt, T=T)

    # Test with various cash levels
    states_positive = torch.tensor([[1.0], [0.5], [0.1]])
    states_negative = torch.tensor([[0.0], [-0.1], [-1.0]])

    terminated_pos = simulator._check_termination(states_positive)
    terminated_neg = simulator._check_termination(states_negative)

    assert not terminated_pos.any(), "Positive cash should not terminate"
    assert terminated_neg.all(), "Non-positive cash should terminate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
