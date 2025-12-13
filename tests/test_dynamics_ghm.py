"""
Tests for GHM equity management model.

This module tests:
- Parameter handling
- Drift and diffusion formulas
- Spot-check values from specification
- Integration with numerics module
"""

import pytest
import torch
import numpy as np

from macro_rl.dynamics import GHMEquityDynamics, GHMEquityParams


class TestGHMEquityParams:
    """Test parameter dataclass."""

    def test_default_params(self):
        """Test default parameter values from Table 1."""
        p = GHMEquityParams()

        # Cash flow
        assert p.alpha == 0.18

        # Growth and rates
        assert p.mu == 0.01
        assert p.r == 0.03
        assert p.lambda_ == 0.02

        # Volatility
        assert p.sigma_A == 0.25
        assert p.sigma_X == 0.12
        assert p.rho == -0.2

        # Bounds
        assert p.c_max == 2.0

    def test_custom_params(self):
        """Test custom parameter specification."""
        p = GHMEquityParams(
            alpha=0.20,
            mu=0.02,
            r=0.04,
            lambda_=0.03,
            sigma_A=0.30,
            sigma_X=0.15,
            rho=-0.3,
            c_max=2.5
        )

        assert p.alpha == 0.20
        assert p.mu == 0.02
        assert p.r == 0.04
        assert p.lambda_ == 0.03
        assert p.sigma_A == 0.30
        assert p.sigma_X == 0.15
        assert p.rho == -0.3
        assert p.c_max == 2.5


class TestGHMEquityStateSpace:
    """Test state space specification."""

    @pytest.fixture
    def model(self):
        """Create model with default parameters."""
        return GHMEquityDynamics()

    def test_state_space_dimension(self, model):
        """Test that state space is 1D."""
        assert model.state_space.dim == 1

    def test_state_space_bounds(self, model):
        """Test default state space bounds."""
        ss = model.state_space
        assert torch.allclose(ss.lower, torch.tensor([0.0]))
        assert torch.allclose(ss.upper, torch.tensor([2.0]))

    def test_state_space_names(self, model):
        """Test state variable names."""
        assert model.state_space.names == ("c",)

    def test_custom_bounds(self):
        """Test custom c_max."""
        params = GHMEquityParams(c_max=3.0)
        model = GHMEquityDynamics(params)

        assert torch.allclose(model.state_space.upper, torch.tensor([3.0]))


class TestGHMEquityDrift:
    """Test drift coefficient μ_c(c) = α + c(r - λ - μ)."""

    @pytest.fixture
    def model(self):
        """Create model with default parameters."""
        return GHMEquityDynamics()

    def test_drift_shape(self, model):
        """Test drift returns correct shape."""
        c = torch.rand(100, 1)
        mu = model.drift(c)
        assert mu.shape == (100, 1)

    def test_drift_at_zero(self, model):
        """Test drift at c=0: μ_c(0) = α."""
        c = torch.tensor([[0.0]])
        mu = model.drift(c)

        expected = model.p.alpha  # 0.18
        assert torch.allclose(mu, torch.tensor([[expected]]), atol=1e-10)

    def test_drift_formula(self, model):
        """Test drift formula: μ_c(c) = α + c(r - λ - μ)."""
        c = torch.tensor([[0.5], [1.0], [1.5]])
        mu = model.drift(c)

        # α = 0.18, r - λ - μ = 0.03 - 0.02 - 0.01 = 0.0
        # So μ_c(c) = 0.18 + c * 0.0 = 0.18 for all c
        expected = torch.tensor([[0.18], [0.18], [0.18]])
        assert torch.allclose(mu, expected, atol=1e-10)

    def test_drift_linearity_in_c(self):
        """Test that drift is linear in c."""
        # Use parameters where r - λ - μ ≠ 0
        params = GHMEquityParams(r=0.05, lambda_=0.02, mu=0.01)
        model = GHMEquityDynamics(params)

        c = torch.tensor([[1.0], [2.0], [3.0]])
        mu = model.drift(c)

        # μ_c(c) = 0.18 + c * (0.05 - 0.02 - 0.01) = 0.18 + 0.02c
        expected = torch.tensor([[0.20], [0.22], [0.24]])
        assert torch.allclose(mu, expected, atol=1e-10)

    def test_drift_no_nans(self, model):
        """Test that drift doesn't produce NaNs for valid inputs."""
        c = model.sample_interior(1000)
        mu = model.drift(c)

        assert not torch.isnan(mu).any()
        assert not torch.isinf(mu).any()


class TestGHMEquityDiffusion:
    """Test diffusion coefficient σ_c(c) = sqrt(σ_X²(1-ρ²) + (ρσ_X - cσ_A)²)."""

    @pytest.fixture
    def model(self):
        """Create model with default parameters."""
        return GHMEquityDynamics()

    def test_diffusion_shape(self, model):
        """Test diffusion returns correct shape."""
        c = torch.rand(100, 1)
        sigma = model.diffusion(c)
        assert sigma.shape == (100, 1)

    def test_diffusion_positive(self, model):
        """Test that diffusion is always positive."""
        c = model.sample_interior(1000)
        sigma = model.diffusion(c)

        assert (sigma > 0).all()

    def test_diffusion_squared_at_zero(self, model):
        """Test diffusion_squared at c=0."""
        c = torch.tensor([[0.0]])
        sigma_sq = model.diffusion_squared(c)

        # σ_c(0)² = σ_X²(1-ρ²) + (ρσ_X)²
        # = 0.12² * (1 - 0.04) + (-0.2 * 0.12)²
        # = 0.0144 * 0.96 + 0.000576
        # = 0.013824 + 0.000576 = 0.014400
        # Wait, let me recalculate:
        # σ_X = 0.12, ρ = -0.2
        # σ_X²(1-ρ²) = 0.0144 * (1 - 0.04) = 0.0144 * 0.96 = 0.013824
        # (ρσ_X)² = (-0.024)² = 0.000576
        # Total = 0.013824 + 0.000576 = 0.014400
        # But the spec says 0.01382... let me check
        #
        # From the spec:
        # σ_c(0)² = σ_X²(1-ρ²) + (ρσ_X)²
        # = 0.12²×0.96 + (-0.024)²
        # = 0.01382
        #
        # Let me compute:
        # 0.12² = 0.0144
        # 0.0144 * 0.96 = 0.013824
        # (-0.024)² = 0.000576
        # Sum = 0.013824 + 0.000576 = 0.014400
        #
        # Hmm, there's a discrepancy. Let me check the formula again.
        # The formula is: σ_c(c)² = σ_X²(1-ρ²) + (ρσ_X - cσ_A)²
        # At c=0: σ_c(0)² = σ_X²(1-ρ²) + (ρσ_X)²
        #
        # Actually, I think the spec might have a typo or I'm misreading.
        # Let me just use the actual formula implementation to verify.

        # With default params:
        # σ_X = 0.12, ρ = -0.2, σ_A = 0.25
        # At c=0:
        # linear_term = ρσ_X - 0*σ_A = -0.2 * 0.12 = -0.024
        # σ_c(0)² = σ_X²(1-ρ²) + (-0.024)²
        #         = 0.0144 * 0.96 + 0.000576
        #         = 0.013824 + 0.000576
        #         = 0.014400

        expected = 0.12**2 * (1 - 0.2**2) + (-0.2 * 0.12)**2
        assert torch.allclose(sigma_sq, torch.tensor([[expected]]), atol=1e-6)

    def test_diffusion_squared_spot_values(self, model):
        """Test diffusion_squared at known points using the formula."""
        # Manually verify using formula: σ_c(c)² = σ_X²(1-ρ²) + (ρσ_X - cσ_A)²
        # With defaults: σ_X=0.12, ρ=-0.2, σ_A=0.25
        test_cases = [
            (0.0, 0.01440),   # σ_X²(1-ρ²) + (ρσ_X)² = 0.013824 + 0.000576
            (0.5, 0.03603),   # 0.013824 + (-0.024 - 0.125)² = 0.013824 + 0.022201
            (1.0, 0.08890),   # 0.013824 + (-0.024 - 0.25)² = 0.013824 + 0.075076
        ]

        for c_val, expected_sq in test_cases:
            c = torch.tensor([[c_val]])
            sigma_sq = model.diffusion_squared(c)

            assert torch.allclose(
                sigma_sq, torch.tensor([[expected_sq]]), atol=1e-4
            ), f"Failed at c={c_val}: got {sigma_sq.item():.5f}, expected {expected_sq:.5f}"

    def test_diffusion_formula(self, model):
        """Test explicit diffusion formula."""
        c = torch.tensor([[0.5]])

        # Manual calculation
        sigma_X = model.p.sigma_X  # 0.12
        rho = model.p.rho          # -0.2
        sigma_A = model.p.sigma_A  # 0.25

        const_term = sigma_X**2 * (1 - rho**2)
        linear_term = rho * sigma_X - c.item() * sigma_A
        expected_sq = const_term + linear_term**2

        sigma_sq = model.diffusion_squared(c)
        assert torch.allclose(sigma_sq, torch.tensor([[expected_sq]]), atol=1e-8)

    def test_diffusion_consistency(self, model):
        """Test that diffusion^2 = diffusion_squared."""
        c = torch.rand(100, 1)

        sigma = model.diffusion(c)
        sigma_sq = model.diffusion_squared(c)

        assert torch.allclose(sigma_sq, sigma**2, atol=1e-6)

    def test_diffusion_no_nans(self, model):
        """Test that diffusion doesn't produce NaNs for valid inputs."""
        c = model.sample_interior(1000)
        sigma = model.diffusion(c)

        assert not torch.isnan(sigma).any()
        assert not torch.isinf(sigma).any()


class TestGHMEquityDiscountRate:
    """Test discount rate."""

    def test_default_discount_rate(self):
        """Test discount rate with default parameters."""
        model = GHMEquityDynamics()

        # r - μ = 0.03 - 0.01 = 0.02
        assert abs(model.discount_rate() - 0.02) < 1e-10

    def test_custom_discount_rate(self):
        """Test discount rate with custom parameters."""
        params = GHMEquityParams(r=0.05, mu=0.02)
        model = GHMEquityDynamics(params)

        # r - μ = 0.05 - 0.02 = 0.03
        assert abs(model.discount_rate() - 0.03) < 1e-10


class TestGHMEquityParameters:
    """Test parameter dictionary export."""

    def test_params_dict_complete(self):
        """Test that params dict contains all expected keys."""
        model = GHMEquityDynamics()
        params = model.params

        expected_keys = {
            "alpha", "mu", "r", "lambda",
            "sigma_A", "sigma_X", "rho", "c_max"
        }
        assert set(params.keys()) == expected_keys

    def test_params_dict_values(self):
        """Test that params dict has correct default values."""
        model = GHMEquityDynamics()
        params = model.params

        assert params["alpha"] == 0.18
        assert params["mu"] == 0.01
        assert params["r"] == 0.03
        assert params["lambda"] == 0.02
        assert params["sigma_A"] == 0.25
        assert params["sigma_X"] == 0.12
        assert params["rho"] == -0.2
        assert params["c_max"] == 2.0


class TestGHMEquitySampling:
    """Test sampling methods."""

    @pytest.fixture
    def model(self):
        """Create model with default parameters."""
        return GHMEquityDynamics()

    def test_sample_interior_shape(self, model):
        """Test interior sampling returns correct shape."""
        samples = model.sample_interior(100)
        assert samples.shape == (100, 1)

    def test_sample_interior_bounds(self, model):
        """Test interior samples are within bounds."""
        samples = model.sample_interior(1000)

        assert (samples >= 0.0).all()
        assert (samples <= 2.0).all()

    def test_sample_boundary_lower(self, model):
        """Test lower boundary sampling."""
        samples = model.sample_boundary(100, which="lower", dim=0)

        assert torch.allclose(samples[:, 0], torch.zeros(100))

    def test_sample_boundary_upper(self, model):
        """Test upper boundary sampling."""
        samples = model.sample_boundary(100, which="upper", dim=0)

        assert torch.allclose(samples[:, 0], torch.full((100,), 2.0))


class TestGHMEquityGradients:
    """Test gradient flow through model."""

    @pytest.fixture
    def model(self):
        """Create model with default parameters."""
        return GHMEquityDynamics()

    def test_drift_gradients(self, model):
        """Test that gradients flow through drift."""
        c = torch.rand(10, 1, requires_grad=True)

        mu = model.drift(c)
        loss = mu.sum()
        loss.backward()

        assert c.grad is not None
        assert not torch.isnan(c.grad).any()

    def test_diffusion_gradients(self, model):
        """Test that gradients flow through diffusion."""
        c = torch.rand(10, 1, requires_grad=True)

        sigma = model.diffusion(c)
        loss = sigma.sum()
        loss.backward()

        assert c.grad is not None
        assert not torch.isnan(c.grad).any()

    def test_diffusion_squared_gradients(self, model):
        """Test that gradients flow through diffusion_squared."""
        c = torch.rand(10, 1, requires_grad=True)

        sigma_sq = model.diffusion_squared(c)
        loss = sigma_sq.sum()
        loss.backward()

        assert c.grad is not None
        assert not torch.isnan(c.grad).any()


class TestGHMEquityEdgeCases:
    """Test edge cases and boundary behavior."""

    @pytest.fixture
    def model(self):
        """Create model with default parameters."""
        return GHMEquityDynamics()

    def test_at_lower_bound(self, model):
        """Test behavior exactly at c=0."""
        c = torch.tensor([[0.0]])

        mu = model.drift(c)
        sigma = model.diffusion(c)

        assert not torch.isnan(mu).any()
        assert not torch.isnan(sigma).any()
        assert sigma > 0  # Diffusion should be positive

    def test_at_upper_bound(self, model):
        """Test behavior exactly at c_max."""
        c = torch.tensor([[2.0]])

        mu = model.drift(c)
        sigma = model.diffusion(c)

        assert not torch.isnan(mu).any()
        assert not torch.isnan(sigma).any()
        assert sigma > 0

    def test_very_small_c(self, model):
        """Test behavior for very small c."""
        c = torch.tensor([[1e-6]])

        mu = model.drift(c)
        sigma = model.diffusion(c)

        assert not torch.isnan(mu).any()
        assert not torch.isnan(sigma).any()

    def test_large_batch(self, model):
        """Test with large batch size."""
        c = model.sample_interior(10000)

        mu = model.drift(c)
        sigma = model.diffusion(c)

        assert mu.shape == (10000, 1)
        assert sigma.shape == (10000, 1)
        assert not torch.isnan(mu).any()
        assert not torch.isnan(sigma).any()


class TestGHMEquityIntegration:
    """Test integration with numerics module."""

    @pytest.fixture
    def model(self):
        """Create model with default parameters."""
        return GHMEquityDynamics()

    def test_drift_callable(self, model):
        """Test that drift can be used as a callable."""
        drift_fn = lambda x, t: model.drift(x)

        c = torch.tensor([[1.0]])
        result = drift_fn(c, 0.0)

        assert result.shape == c.shape

    def test_diffusion_callable(self, model):
        """Test that diffusion can be used as a callable."""
        diffusion_fn = lambda x, t: model.diffusion(x)

        c = torch.tensor([[1.0]])
        result = diffusion_fn(c, 0.0)

        assert result.shape == c.shape

    def test_simulate_path_compatibility(self, model):
        """Test compatibility with simulate_path interface."""
        # This is a basic check - full integration test would require
        # the simulate_path function from numerics module
        from macro_rl.numerics import simulate_path

        c0 = torch.ones(10, 1) * 0.5

        c_T = simulate_path(
            c0,
            drift_fn=lambda x, t: model.drift(x),
            diffusion_fn=lambda x, t: model.diffusion(x),
            T=0.1,
            dt=0.01
        )

        assert c_T.shape == c0.shape
        assert not torch.isnan(c_T).any()


# Note: liquidation_value is not part of the core dynamics interface
# and was removed to keep the implementation focused on the SDE specification


class TestGHMEquityReproducibility:
    """Test that results are reproducible."""

    def test_same_input_same_output(self):
        """Test that same input gives same output."""
        model1 = GHMEquityDynamics()
        model2 = GHMEquityDynamics()

        c = torch.tensor([[0.5], [1.0], [1.5]])

        mu1 = model1.drift(c)
        mu2 = model2.drift(c)

        assert torch.allclose(mu1, mu2)

        sigma1 = model1.diffusion(c)
        sigma2 = model2.diffusion(c)

        assert torch.allclose(sigma1, sigma2)

    def test_deterministic_sampling(self):
        """Test that sampling is deterministic with fixed seed."""
        model = GHMEquityDynamics()

        torch.manual_seed(42)
        samples1 = model.sample_interior(100)

        torch.manual_seed(42)
        samples2 = model.sample_interior(100)

        assert torch.allclose(samples1, samples2)
