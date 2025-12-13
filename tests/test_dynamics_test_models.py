"""
Tests for test models (GBM and OU).

These models have known analytical properties that we can test against.
This validates both the model implementations and the dynamics interface.
"""

import pytest
import torch
import numpy as np

from macro_rl.dynamics import GBMDynamics, OUDynamics


class TestGBMDynamics:
    """Test Geometric Brownian Motion dynamics."""

    @pytest.fixture
    def model(self):
        """Create GBM model with default parameters."""
        return GBMDynamics(mu=0.05, sigma=0.2, x_min=0.01, x_max=10.0)

    def test_state_space(self, model):
        """Test state space specification."""
        ss = model.state_space
        assert ss.dim == 1
        assert torch.allclose(ss.lower, torch.tensor([0.01]))
        assert torch.allclose(ss.upper, torch.tensor([10.0]))
        assert ss.names == ("x",)

    def test_params(self, model):
        """Test parameter dictionary."""
        params = model.params
        assert params["mu"] == 0.05
        assert params["sigma"] == 0.2
        assert abs(params["x_min"] - 0.01) < 1e-6
        assert params["x_max"] == 10.0

    def test_drift_shape(self, model):
        """Test drift returns correct shape."""
        x = torch.rand(100, 1)
        mu = model.drift(x)
        assert mu.shape == (100, 1)

    def test_diffusion_shape(self, model):
        """Test diffusion returns correct shape."""
        x = torch.rand(100, 1)
        sigma = model.diffusion(x)
        assert sigma.shape == (100, 1)

    def test_drift_linearity(self, model):
        """Test that drift(2x) = 2*drift(x) (linearity)."""
        x = torch.rand(100, 1) + 0.1  # Avoid zeros
        mu_x = model.drift(x)
        mu_2x = model.drift(2 * x)

        assert torch.allclose(mu_2x, 2 * mu_x, rtol=1e-5)

    def test_drift_zero_at_zero(self):
        """Test that drift is zero when mu=0."""
        model = GBMDynamics(mu=0.0, sigma=0.2)
        x = torch.tensor([[1.0], [2.0], [3.0]])
        mu = model.drift(x)

        assert torch.allclose(mu, torch.zeros_like(x))

    def test_drift_values(self, model):
        """Test drift at specific points."""
        x = torch.tensor([[1.0], [2.0], [4.0]])
        mu = model.drift(x)

        expected = torch.tensor([[0.05], [0.10], [0.20]])
        assert torch.allclose(mu, expected, atol=1e-6)

    def test_diffusion_linearity(self, model):
        """Test that diffusion(2x) = 2*diffusion(x)."""
        x = torch.rand(100, 1) + 0.1
        sigma_x = model.diffusion(x)
        sigma_2x = model.diffusion(2 * x)

        assert torch.allclose(sigma_2x, 2 * sigma_x, rtol=1e-5)

    def test_diffusion_values(self, model):
        """Test diffusion at specific points."""
        x = torch.tensor([[1.0], [2.0], [5.0]])
        sigma = model.diffusion(x)

        expected = torch.tensor([[0.2], [0.4], [1.0]])
        assert torch.allclose(sigma, expected, atol=1e-6)

    def test_diffusion_positive(self, model):
        """Test that diffusion is always positive."""
        x = model.sample_interior(1000)
        sigma = model.diffusion(x)

        assert (sigma > 0).all()

    def test_diffusion_squared(self, model):
        """Test that diffusion_squared = diffusion^2."""
        x = torch.rand(100, 1)
        sigma = model.diffusion(x)
        sigma_sq = model.diffusion_squared(x)

        assert torch.allclose(sigma_sq, sigma ** 2, rtol=1e-5)

    def test_discount_rate(self, model):
        """Test discount rate."""
        assert model.discount_rate() == 0.03

    def test_sample_interior_bounds(self, model):
        """Test that sampled points are within bounds."""
        samples = model.sample_interior(1000)

        assert (samples >= 0.01).all()
        assert (samples <= 10.0).all()

    def test_sample_boundary(self, model):
        """Test boundary sampling."""
        samples_lower = model.sample_boundary(100, which="lower", dim=0)
        samples_upper = model.sample_boundary(100, which="upper", dim=0)

        assert torch.allclose(samples_lower[:, 0], torch.tensor(0.01))
        assert torch.allclose(samples_upper[:, 0], torch.tensor(10.0))


class TestOUDynamics:
    """Test Ornstein-Uhlenbeck dynamics."""

    @pytest.fixture
    def model(self):
        """Create OU model with default parameters."""
        return OUDynamics(theta=1.0, mu=0.0, sigma=0.5, x_min=-5.0, x_max=5.0)

    def test_state_space(self, model):
        """Test state space specification."""
        ss = model.state_space
        assert ss.dim == 1
        assert torch.allclose(ss.lower, torch.tensor([-5.0]))
        assert torch.allclose(ss.upper, torch.tensor([5.0]))
        assert ss.names == ("x",)

    def test_params(self, model):
        """Test parameter dictionary."""
        params = model.params
        assert params["theta"] == 1.0
        assert params["mu"] == 0.0
        assert params["sigma"] == 0.5
        assert params["x_min"] == -5.0
        assert params["x_max"] == 5.0

    def test_invalid_theta(self):
        """Test that negative theta raises error."""
        with pytest.raises(AssertionError, match="theta must be positive"):
            OUDynamics(theta=-1.0)

    def test_drift_shape(self, model):
        """Test drift returns correct shape."""
        x = torch.rand(100, 1)
        mu = model.drift(x)
        assert mu.shape == (100, 1)

    def test_diffusion_shape(self, model):
        """Test diffusion returns correct shape."""
        x = torch.rand(100, 1)
        sigma = model.diffusion(x)
        assert sigma.shape == (100, 1)

    def test_drift_mean_reversion(self, model):
        """Test that drift(μ) = 0 (mean reversion property)."""
        x = torch.tensor([[model.mu]])
        mu = model.drift(x)

        assert torch.allclose(mu, torch.zeros_like(x), atol=1e-10)

    def test_drift_pulls_toward_mean(self, model):
        """Test that drift pulls values toward mean."""
        # Above mean: drift should be negative
        x_above = torch.tensor([[1.0], [2.0], [3.0]])
        mu_above = model.drift(x_above)
        assert (mu_above < 0).all()

        # Below mean: drift should be positive
        x_below = torch.tensor([[-1.0], [-2.0], [-3.0]])
        mu_below = model.drift(x_below)
        assert (mu_below > 0).all()

    def test_drift_values(self, model):
        """Test drift at specific points."""
        x = torch.tensor([[-2.0], [0.0], [2.0]])
        mu = model.drift(x)

        # θ(μ - x) with θ=1, μ=0
        expected = torch.tensor([[2.0], [0.0], [-2.0]])
        assert torch.allclose(mu, expected, atol=1e-6)

    def test_diffusion_constant(self, model):
        """Test that diffusion is constant."""
        x = torch.linspace(-5, 5, 100).reshape(-1, 1)
        sigma = model.diffusion(x)

        # All values should be equal to sigma parameter
        assert torch.allclose(sigma, torch.tensor([[0.5]]), atol=1e-6)

    def test_diffusion_positive(self, model):
        """Test that diffusion is always positive."""
        x = model.sample_interior(1000)
        sigma = model.diffusion(x)

        assert (sigma > 0).all()

    def test_diffusion_squared(self, model):
        """Test that diffusion_squared = sigma^2."""
        x = torch.rand(100, 1)
        sigma_sq = model.diffusion_squared(x)

        expected = 0.5 ** 2 * torch.ones_like(x)
        assert torch.allclose(sigma_sq, expected, atol=1e-6)

    def test_discount_rate(self, model):
        """Test discount rate."""
        assert model.discount_rate() == 0.03

    def test_stationary_variance(self, model):
        """Test theoretical stationary variance."""
        stat_var = model.stationary_variance()

        # σ²/(2θ) = 0.5²/(2*1.0) = 0.125
        expected = 0.5 ** 2 / (2 * 1.0)
        assert np.isclose(stat_var, expected)

    def test_sample_interior_bounds(self, model):
        """Test that sampled points are within bounds."""
        samples = model.sample_interior(1000)

        assert (samples >= -5.0).all()
        assert (samples <= 5.0).all()

    def test_sample_boundary(self, model):
        """Test boundary sampling."""
        samples_lower = model.sample_boundary(100, which="lower", dim=0)
        samples_upper = model.sample_boundary(100, which="upper", dim=0)

        assert torch.allclose(samples_lower[:, 0], torch.tensor(-5.0))
        assert torch.allclose(samples_upper[:, 0], torch.tensor(5.0))


class TestGBMCustomParameters:
    """Test GBM with various parameter configurations."""

    def test_different_drift(self):
        """Test GBM with different drift parameters."""
        model_pos = GBMDynamics(mu=0.1, sigma=0.2)
        model_neg = GBMDynamics(mu=-0.05, sigma=0.2)

        x = torch.tensor([[1.0]])

        assert model_pos.drift(x) > 0
        assert model_neg.drift(x) < 0

    def test_different_volatility(self):
        """Test GBM with different volatility."""
        model_low = GBMDynamics(mu=0.05, sigma=0.1)
        model_high = GBMDynamics(mu=0.05, sigma=0.5)

        x = torch.tensor([[1.0]])

        assert model_low.diffusion(x) < model_high.diffusion(x)


class TestOUCustomParameters:
    """Test OU with various parameter configurations."""

    def test_different_reversion_speed(self):
        """Test OU with different mean reversion speeds."""
        model_fast = OUDynamics(theta=2.0, mu=0.0, sigma=0.5)
        model_slow = OUDynamics(theta=0.5, mu=0.0, sigma=0.5)

        x = torch.tensor([[1.0]])

        # Faster reversion should have larger magnitude drift
        assert abs(model_fast.drift(x).item()) > abs(model_slow.drift(x).item())

    def test_different_long_run_mean(self):
        """Test OU with different long-run means."""
        model1 = OUDynamics(theta=1.0, mu=1.0, sigma=0.5)
        model2 = OUDynamics(theta=1.0, mu=-1.0, sigma=0.5)

        x = torch.tensor([[0.0]])

        # At x=0, drift should point toward respective means
        assert model1.drift(x) > 0  # Toward mu=1
        assert model2.drift(x) < 0  # Toward mu=-1

    def test_different_volatility(self):
        """Test OU with different volatility."""
        model_low = OUDynamics(theta=1.0, mu=0.0, sigma=0.2)
        model_high = OUDynamics(theta=1.0, mu=0.0, sigma=0.8)

        # Stationary variance should scale with sigma^2
        var_low = model_low.stationary_variance()
        var_high = model_high.stationary_variance()

        assert var_low < var_high
        assert np.isclose(var_high / var_low, (0.8 / 0.2) ** 2)


class TestModelConsistency:
    """Test consistency properties across all models."""

    @pytest.fixture(params=["gbm", "ou"])
    def model(self, request):
        """Parameterized fixture providing both test models."""
        if request.param == "gbm":
            return GBMDynamics()
        else:
            return OUDynamics()

    def test_drift_diffusion_same_shape(self, model):
        """Test that drift and diffusion have same shape."""
        x = model.sample_interior(100)
        mu = model.drift(x)
        sigma = model.diffusion(x)

        assert mu.shape == sigma.shape

    def test_batch_processing(self, model):
        """Test that models handle different batch sizes."""
        for batch_size in [1, 10, 100, 1000]:
            x = model.sample_interior(batch_size)
            mu = model.drift(x)
            sigma = model.diffusion(x)

            assert mu.shape == (batch_size, 1)
            assert sigma.shape == (batch_size, 1)

    def test_gradient_flow(self, model):
        """Test that gradients can flow through drift and diffusion."""
        x = torch.rand(10, 1, requires_grad=True)

        # Drift - all models should support gradients through drift
        mu = model.drift(x)
        loss_mu = mu.sum()
        loss_mu.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Reset gradients
        x.grad.zero_()

        # Diffusion - only test if diffusion depends on x (not constant)
        # OU has constant diffusion, so skip gradient test for it
        sigma = model.diffusion(x)
        # Check if diffusion actually depends on x by seeing if it varies
        x_test = torch.rand(10, 1)
        sigma_test = model.diffusion(x_test)
        if not torch.allclose(sigma, sigma_test, rtol=0.01):
            # Diffusion depends on x, test gradients
            loss_sigma = sigma.sum()
            loss_sigma.backward()
            assert x.grad is not None
            assert not torch.isnan(x.grad).any()

    def test_no_nans_in_bounds(self, model):
        """Test that no NaNs or Infs appear for valid inputs."""
        x = model.sample_interior(1000)

        mu = model.drift(x)
        sigma = model.diffusion(x)
        sigma_sq = model.diffusion_squared(x)

        assert not torch.isnan(mu).any()
        assert not torch.isinf(mu).any()
        assert not torch.isnan(sigma).any()
        assert not torch.isinf(sigma).any()
        assert not torch.isnan(sigma_sq).any()
        assert not torch.isinf(sigma_sq).any()
