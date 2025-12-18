"""
Unit tests for DifferentiableSimulator.
"""

import torch
from torch import nn
from macro_rl.simulation.differentiable import DifferentiableSimulator
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
import pytest

class TinyReparamPolicy(nn.Module):
    """Simple reparameterized Gaussian policy for testing."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Linear(state_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def sample_with_noise(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Reparameterized Gaussian policy: a = mu + sigma * noise.
        Assumes `noise` has shape (batch, action_dim).
        """
        mu = self.net(state)
        std = torch.exp(self.log_std)
        return mu + std * noise[:, :mu.shape[-1]]


def make_ghm_diff_simulator(dt=0.1, T=0.3):
    """Create GHM differentiable simulator for testing."""
    params = GHMEquityParams()
    dynamics = GHMEquityDynamics(params)
    control_spec = GHMControlSpec()
    reward_fn = GHMRewardFunction(
        discount_rate=params.r - params.mu,
        issuance_cost=params.lambda_,
        liquidation_rate=params.omega,
        liquidation_flow=params.alpha,
    )

    simulator = DifferentiableSimulator(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        dt=dt,
        T=T,
    )

    return simulator, control_spec, reward_fn, dynamics, params


def test_differentiable_simulator_gradient_flow():
    """Test that gradients flow through the simulator."""
    dt = 0.1
    T = 0.3
    simulator, control_spec, reward_fn, dynamics, params = make_ghm_diff_simulator(dt=dt, T=T)

    batch = 4
    state_dim = dynamics.state_space.dim
    action_dim = control_spec.dim

    initial_states = torch.ones(batch, state_dim, requires_grad=False)
    num_steps = int(round(T / dt))

    # Noise for actions/state
    noise = torch.randn(batch, num_steps, state_dim)

    policy = TinyReparamPolicy(state_dim=state_dim, action_dim=action_dim)

    returns = simulator.simulate(policy, initial_states, noise)
    loss = -returns.mean()

    loss.backward()

    # Verify that policy parameters received gradients
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0, "Policy parameters should have gradients"

    # Verify gradients are non-zero
    for grad in grads:
        assert not torch.allclose(grad, torch.zeros_like(grad)), "Gradients should be non-zero"


def test_differentiable_simulator_shapes():
    """Test output shapes of differentiable simulator."""
    dt = 0.1
    T = 0.5
    simulator, control_spec, _, dynamics, _ = make_ghm_diff_simulator(dt=dt, T=T)

    batch = 10
    state_dim = dynamics.state_space.dim
    action_dim = control_spec.dim
    num_steps = int(round(T / dt))

    initial_states = torch.ones(batch, state_dim)
    noise = torch.randn(batch, num_steps, state_dim)

    policy = TinyReparamPolicy(state_dim=state_dim, action_dim=action_dim)

    # Test without trajectory return
    returns = simulator.simulate(policy, initial_states, noise, return_trajectory=False)
    assert returns.shape == (batch,)

    # Test with trajectory return
    returns, states, actions = simulator.simulate(policy, initial_states, noise, return_trajectory=True)
    assert returns.shape == (batch,)
    assert states.shape == (batch, num_steps + 1, state_dim)
    assert actions.shape == (batch, num_steps, action_dim)


def test_differentiable_simulator_reproducibility():
    """Test that simulator is deterministic with fixed noise."""
    dt = 0.1
    T = 0.5
    simulator, control_spec, _, dynamics, _ = make_ghm_diff_simulator(dt=dt, T=T)

    batch = 5
    state_dim = dynamics.state_space.dim
    action_dim = control_spec.dim
    num_steps = int(round(T / dt))

    initial_states = torch.ones(batch, state_dim)
    noise = torch.randn(batch, num_steps, state_dim)

    policy = TinyReparamPolicy(state_dim=state_dim, action_dim=action_dim)

    # Run twice with same noise and policy (no stochasticity)
    with torch.no_grad():
        returns1 = simulator.simulate(policy, initial_states, noise)
        returns2 = simulator.simulate(policy, initial_states, noise)

    assert torch.allclose(returns1, returns2, atol=1e-6)


def test_soft_termination_mask():
    """Test soft termination mask function."""
    dt = 0.1
    T = 0.3
    simulator, _, _, _, _ = make_ghm_diff_simulator(dt=dt, T=T)

    # Test with various cash levels
    states = torch.tensor([
        [2.0],   # High cash
        [1.0],   # Medium cash
        [0.1],   # Low cash
        [0.0],   # Zero cash
        [-0.5],  # Negative cash
    ])

    masks = simulator._soft_termination_mask(states)

    # Masks should be in [0, 1]
    assert (masks >= 0).all()
    assert (masks <= 1).all()

    # High cash should have mask close to 1
    assert masks[0] > 0.9

    # Negative cash should have mask close to 0
    assert masks[-1] < 0.1

    # Masks should be differentiable
    assert masks.requires_grad or not states.requires_grad


def test_compute_gradient_method():
    """Test compute_gradient convenience method."""
    dt = 0.1
    T = 0.3
    simulator, control_spec, _, dynamics, _ = make_ghm_diff_simulator(dt=dt, T=T)

    batch = 4
    state_dim = dynamics.state_space.dim
    action_dim = control_spec.dim
    num_steps = int(round(T / dt))

    initial_states = torch.ones(batch, state_dim)
    noise = torch.randn(batch, num_steps, state_dim)

    policy = TinyReparamPolicy(state_dim=state_dim, action_dim=action_dim)

    grad = simulator.compute_gradient(policy, initial_states, noise)

    # Gradient should be a 1D tensor
    assert grad.dim() == 1

    # Gradient should have non-zero elements
    assert not torch.allclose(grad, torch.zeros_like(grad))


def test_differentiable_step():
    """Test single differentiable step."""
    dt = 0.01
    T = 0.1
    simulator, control_spec, _, dynamics, _ = make_ghm_diff_simulator(dt=dt, T=T)

    batch = 5
    state_dim = dynamics.state_space.dim
    action_dim = control_spec.dim

    state = torch.ones(batch, state_dim, requires_grad=True)
    action = torch.full((batch, action_dim), 0.1, requires_grad=True)
    noise = torch.randn(batch, state_dim)

    next_state = simulator._differentiable_step(state, action, noise)

    # Next state should have gradients
    assert next_state.requires_grad

    # Test gradient flow
    loss = next_state.sum()
    loss.backward()

    assert state.grad is not None
    assert action.grad is not None


def test_gradient_comparison_with_finite_diff():
    """Test gradient accuracy using finite differences."""
    dt = 0.1
    T = 0.3
    simulator, control_spec, _, dynamics, _ = make_ghm_diff_simulator(dt=dt, T=T)

    batch = 2
    state_dim = dynamics.state_space.dim
    action_dim = control_spec.dim
    num_steps = int(round(T / dt))

    initial_states = torch.ones(batch, state_dim)
    noise = torch.randn(batch, num_steps, state_dim)

    policy = TinyReparamPolicy(state_dim=state_dim, action_dim=action_dim)

    # Compute analytical gradient
    analytical_grad = simulator.compute_gradient(policy, initial_states, noise)

    # Compute finite difference gradient for first parameter
    eps = 1e-4
    param = list(policy.parameters())[0]

    # Positive perturbation
    with torch.no_grad():
        param.add_(eps)
    returns_plus = simulator.simulate(policy, initial_states, noise).mean()

    # Negative perturbation
    with torch.no_grad():
        param.add_(-2 * eps)
    returns_minus = simulator.simulate(policy, initial_states, noise).mean()

    # Restore parameter
    with torch.no_grad():
        param.add_(eps)

    # Finite difference gradient
    fd_grad = (returns_plus - returns_minus) / (2 * eps)

    # Compare first element of analytical gradient
    # (corresponding to first parameter)
    param_size = param.numel()
    analytical_first_param_grad = analytical_grad[:param_size].mean()

    # Allow some numerical error
    relative_error = abs(analytical_first_param_grad - fd_grad) / (abs(fd_grad) + 1e-8)
    assert relative_error < 0.1, f"Gradient mismatch: analytical={analytical_first_param_grad}, fd={fd_grad}"




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
