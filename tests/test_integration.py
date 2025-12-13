"""Tests for SDE integration methods."""

import torch
import pytest
import math
from macro_rl.numerics.integration import (
    euler_maruyama_step,
    simulate_path,
    milstein_step,
    geometric_brownian_motion,
    ornstein_uhlenbeck,
)


class TestEulerMaruyamaStep:
    """Tests for single Euler-Maruyama step."""

    def test_deterministic_step(self):
        """Test that with σ=0, we get deterministic ODE step."""
        state = torch.tensor([[1.0]])
        drift = torch.tensor([[0.5]])
        diffusion = torch.tensor([[0.0]])
        dt = 0.1

        next_state = euler_maruyama_step(state, drift, diffusion, dt)

        # Should be: x + μ*dt = 1.0 + 0.5*0.1 = 1.05
        expected = torch.tensor([[1.05]])
        assert torch.allclose(next_state, expected, atol=1e-10)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        state = torch.tensor([[1.0, 2.0]])
        drift = torch.tensor([[0.1, -0.2]])
        diffusion = torch.tensor([[0.3, 0.4]])
        dt = 0.01

        torch.manual_seed(42)
        dW1 = torch.randn_like(state) * math.sqrt(dt)
        next_state1 = euler_maruyama_step(state, drift, diffusion, dt, dW=dW1)

        torch.manual_seed(42)
        dW2 = torch.randn_like(state) * math.sqrt(dt)
        next_state2 = euler_maruyama_step(state, drift, diffusion, dt, dW=dW2)

        assert torch.allclose(next_state1, next_state2, atol=1e-10)

    def test_batch_operation(self):
        """Test that batch operations work correctly."""
        batch_size = 100
        state = torch.ones(batch_size, 2)
        drift = torch.ones(batch_size, 2) * 0.05
        diffusion = torch.ones(batch_size, 2) * 0.2
        dt = 0.01

        next_state = euler_maruyama_step(state, drift, diffusion, dt)

        assert next_state.shape == (batch_size, 2)
        # All should be different due to noise
        assert not torch.allclose(next_state[0], next_state[1])


class TestSimulatePath:
    """Tests for full trajectory simulation."""

    def test_deterministic_path(self):
        """Test simulation of deterministic ODE: dx = μx dt."""
        x0 = torch.ones(1, 1)
        mu = 0.1
        T = 1.0
        dt = 0.01

        drift_fn = lambda x, t: mu * x
        diffusion_fn = lambda x, t: torch.zeros_like(x)

        x_T = simulate_path(x0, drift_fn, diffusion_fn, T, dt, seed=42)

        # Analytical solution: x(T) = x0 * exp(μ*T)
        expected = x0 * torch.exp(torch.tensor(mu * T))
        assert torch.allclose(x_T, expected, atol=1e-6)

    def test_return_full_path(self):
        """Test that full path has correct shape."""
        x0 = torch.ones(10, 2)
        T = 1.0
        dt = 0.1
        n_steps = int(T / dt)

        drift_fn = lambda x, t: 0.05 * x
        diffusion_fn = lambda x, t: 0.2 * x

        times, states = simulate_path(
            x0, drift_fn, diffusion_fn, T, dt,
            return_full_path=True, seed=42
        )

        assert times.shape == (n_steps + 1,)
        assert states.shape == (10, n_steps + 1, 2)
        assert torch.allclose(states[:, 0, :], x0)

    def test_reproducibility_with_seed(self):
        """Test that same seed gives same trajectory."""
        x0 = torch.ones(5, 1)
        drift_fn = lambda x, t: 0.05 * x
        diffusion_fn = lambda x, t: 0.2 * x

        x_T1 = simulate_path(x0, drift_fn, diffusion_fn, T=1.0, dt=0.01, seed=42)
        x_T2 = simulate_path(x0, drift_fn, diffusion_fn, T=1.0, dt=0.01, seed=42)

        assert torch.allclose(x_T1, x_T2, atol=1e-10)


class TestGeometricBrownianMotion:
    """Tests for GBM simulation."""

    def test_gbm_mean(self):
        """Test that GBM mean matches theory."""
        x0 = torch.ones(10000, 1)
        mu = 0.05
        sigma = 0.2
        T = 1.0
        dt = 0.01

        x_T = geometric_brownian_motion(x0, mu, sigma, T, dt, seed=42)

        # E[X_T] = X_0 * exp(μ*T)
        expected_mean = torch.exp(torch.tensor(mu * T))
        empirical_mean = x_T.mean()

        # Should be within 2% for 10000 samples
        assert torch.abs(empirical_mean - expected_mean) / expected_mean < 0.02

    def test_gbm_variance(self):
        """Test that GBM variance matches theory."""
        x0 = torch.ones(10000, 1)
        mu = 0.05
        sigma = 0.2
        T = 1.0
        dt = 0.01

        x_T = geometric_brownian_motion(x0, mu, sigma, T, dt, seed=42)

        # Var[X_T] = X_0² * exp(2μT) * (exp(σ²T) - 1)
        expected_var = (
            torch.exp(torch.tensor(2 * mu * T)) *
            (torch.exp(torch.tensor(sigma ** 2 * T)) - 1)
        )
        empirical_var = x_T.var()

        # Should be within 5% for 10000 samples
        assert torch.abs(empirical_var - expected_var) / expected_var < 0.05

    def test_gbm_positive(self):
        """Test that GBM stays positive."""
        x0 = torch.ones(100, 1)
        mu = 0.05
        sigma = 0.2

        x_T = geometric_brownian_motion(x0, mu, sigma, T=1.0, dt=0.01, seed=42)

        assert (x_T > 0).all()

    def test_gbm_full_path(self):
        """Test GBM full path return."""
        x0 = torch.ones(10, 1)
        T = 1.0
        dt = 0.1

        times, states = geometric_brownian_motion(
            x0, mu=0.05, sigma=0.2, T=T, dt=dt,
            return_full_path=True, seed=42
        )

        n_steps = int(T / dt)
        assert times.shape == (n_steps + 1,)
        assert states.shape == (10, n_steps + 1, 1)


class TestOrnsteinUhlenbeck:
    """Tests for Ornstein-Uhlenbeck process."""

    def test_ou_stationary_mean(self):
        """Test that OU process converges to stationary mean."""
        x0 = torch.zeros(10000, 1)
        theta = 2.0  # Fast mean reversion
        mu = 1.5
        sigma = 0.3
        T = 5.0  # Long time
        dt = 0.01

        x_T = ornstein_uhlenbeck(x0, theta, mu, sigma, T, dt, seed=42)

        # For large T, X_T should be close to μ
        empirical_mean = x_T.mean()

        # Should be within 5% of stationary mean
        assert torch.abs(empirical_mean - mu) / mu < 0.05

    def test_ou_stationary_variance(self):
        """Test that OU process variance matches stationary distribution."""
        x0 = torch.ones(10000, 1)
        theta = 1.0
        mu = 0.0
        sigma = 1.0
        T = 10.0  # Long time for convergence
        dt = 0.01

        x_T = ornstein_uhlenbeck(x0, theta, mu, sigma, T, dt, seed=42)

        # Stationary variance: σ² / (2θ)
        expected_var = sigma ** 2 / (2 * theta)
        empirical_var = x_T.var()

        # Should be within 10% for 10000 samples
        assert torch.abs(empirical_var - expected_var) / expected_var < 0.10

    def test_ou_mean_reversion(self):
        """Test that OU process reverts to mean."""
        x0 = torch.ones(100, 1) * 10.0  # Start far from mean
        theta = 1.0
        mu = 0.0
        sigma = 0.1  # Low noise
        T = 5.0
        dt = 0.01

        times, states = ornstein_uhlenbeck(
            x0, theta, mu, sigma, T, dt,
            return_full_path=True, seed=42
        )

        # Check that values are moving toward the mean
        x_0 = states[:, 0, 0].mean()
        x_T = states[:, -1, 0].mean()

        assert torch.abs(x_T - mu) < torch.abs(x_0 - mu)


class TestMilsteinStep:
    """Tests for Milstein scheme."""

    def test_milstein_reduces_to_em_for_constant_diffusion(self):
        """Test that Milstein = EM when σ is constant."""
        state = torch.tensor([[1.0]])
        drift = torch.tensor([[0.5]])
        diffusion = torch.tensor([[0.2]])
        diffusion_derivative = torch.tensor([[0.0]])  # Constant diffusion
        dt = 0.01

        torch.manual_seed(42)
        dW = torch.randn_like(state) * math.sqrt(dt)

        # Milstein step
        next_milstein = milstein_step(
            state, drift, diffusion, diffusion_derivative, dt, dW
        )

        # EM step
        next_em = euler_maruyama_step(state, drift, diffusion, dt, dW)

        assert torch.allclose(next_milstein, next_em, atol=1e-10)

    def test_milstein_correction_term(self):
        """Test that Milstein includes correction for state-dependent diffusion."""
        state = torch.tensor([[1.0]])
        drift = torch.tensor([[0.0]])
        diffusion = state  # σ(x) = x, so ∂σ/∂x = 1
        diffusion_derivative = torch.tensor([[1.0]])
        dt = 0.01

        torch.manual_seed(42)
        dW = torch.randn_like(state) * math.sqrt(dt)

        # Milstein step
        next_milstein = milstein_step(
            state, drift, diffusion, diffusion_derivative, dt, dW
        )

        # EM step
        next_em = euler_maruyama_step(state, drift, diffusion, dt, dW)

        # Correction term: 0.5 * σ * (∂σ/∂x) * (dW² - dt)
        correction = 0.5 * diffusion * diffusion_derivative * (dW ** 2 - dt)

        assert torch.allclose(next_milstein, next_em + correction, atol=1e-10)


class TestNumericalAccuracy:
    """Tests for numerical accuracy and convergence."""

    def test_em_weak_convergence(self):
        """Test weak convergence of Euler-Maruyama (convergence of moments)."""
        # Test with GBM: dX = μX dt + σX dW
        x0 = torch.ones(5000, 1)
        mu = 0.1
        sigma = 0.2
        T = 1.0

        # Analytical solution for mean
        expected_mean = torch.exp(torch.tensor(mu * T))

        # Test convergence with decreasing dt
        dt_values = [0.1, 0.05, 0.01]
        errors = []

        for dt in dt_values:
            x_T = geometric_brownian_motion(x0, mu, sigma, T, dt, seed=42)
            empirical_mean = x_T.mean()
            error = torch.abs(empirical_mean - expected_mean)
            errors.append(error.item())

        # Errors should decrease as dt decreases
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
