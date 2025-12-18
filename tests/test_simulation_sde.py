"""
Unit tests for SDE integrator.
"""

import torch
from macro_rl.simulation.sde import SDEIntegrator


def test_euler_maruyama_shapes_and_update():
    """Test basic Euler-Maruyama step with shapes and simple dynamics."""
    integrator = SDEIntegrator(scheme="euler")

    batch, dim = 4, 2
    x = torch.zeros(batch, dim)
    drift = torch.ones(batch, dim)        # constant drift = 1
    diffusion = torch.zeros(batch, dim)   # no diffusion
    dt = 0.1
    noise = torch.zeros(batch, dim)       # no randomness

    x_next = integrator.step(x, drift, diffusion, dt, noise)

    assert x_next.shape == x.shape
    # With drift=1 and no diffusion, x_next = x + dt
    assert torch.allclose(x_next, torch.full_like(x, dt))


def test_euler_maruyama_with_diffusion():
    """Test Euler-Maruyama with diffusion term."""
    integrator = SDEIntegrator(scheme="euler")

    batch, dim = 10, 1
    x = torch.zeros(batch, dim)
    drift = torch.zeros(batch, dim)
    diffusion = torch.ones(batch, dim)
    dt = 0.01
    noise = torch.randn(batch, dim)

    x_next = integrator.step(x, drift, diffusion, dt, noise)

    # With zero drift, x_next = x + σ√dt·ε = σ√dt·ε (since x=0)
    import math
    expected = diffusion * math.sqrt(dt) * noise
    assert torch.allclose(x_next, expected)


def test_batch_simulate_shapes():
    """Test batch simulation produces correct shapes."""
    integrator = SDEIntegrator(scheme="euler")

    batch_size = 100
    state_dim = 2
    n_steps = 50
    dt = 0.01

    x0 = torch.randn(batch_size, state_dim)

    # Simple drift and diffusion functions
    def drift_fn(x):
        return -0.1 * x  # Mean reversion

    def diffusion_fn(x):
        return torch.ones_like(x) * 0.2  # Constant volatility

    trajectory = integrator.batch_simulate(
        x0=x0,
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        dt=dt,
        n_steps=n_steps,
    )

    # Should return (batch, n_steps+1, state_dim)
    assert trajectory.shape == (batch_size, n_steps + 1, state_dim)
    # Initial state should match
    assert torch.allclose(trajectory[:, 0, :], x0)


def test_batch_simulate_with_pre_sampled_noise():
    """Test that pre-sampled noise produces deterministic results."""
    integrator = SDEIntegrator(scheme="euler")

    batch_size = 10
    state_dim = 1
    n_steps = 20
    dt = 0.01

    x0 = torch.zeros(batch_size, state_dim)
    noise = torch.randn(batch_size, n_steps, state_dim)

    def drift_fn(x):
        return torch.zeros_like(x)

    def diffusion_fn(x):
        return torch.ones_like(x)

    # Run twice with same noise
    traj1 = integrator.batch_simulate(x0, drift_fn, diffusion_fn, dt, n_steps, noise=noise)
    traj2 = integrator.batch_simulate(x0, drift_fn, diffusion_fn, dt, n_steps, noise=noise)

    # Should be identical
    assert torch.allclose(traj1, traj2)


def test_euler_maruyama_convergence():
    """Test that Euler-Maruyama converges for simple SDE."""
    integrator = SDEIntegrator(scheme="euler")

    # Test Brownian motion: dx = σ dW
    # Analytical: E[x²] = σ² t
    batch_size = 10000
    state_dim = 1
    sigma = 0.5
    T = 1.0
    dt = 0.01
    n_steps = int(T / dt)

    x0 = torch.zeros(batch_size, state_dim)

    def drift_fn(x):
        return torch.zeros_like(x)

    def diffusion_fn(x):
        return torch.ones_like(x) * sigma

    trajectory = integrator.batch_simulate(x0, drift_fn, diffusion_fn, dt, n_steps)

    # Check variance at final time
    final_state = trajectory[:, -1, 0]
    empirical_var = final_state.var()
    expected_var = sigma**2 * T

    # Allow 10% relative error due to finite sample size
    assert abs(empirical_var - expected_var) / expected_var < 0.1
