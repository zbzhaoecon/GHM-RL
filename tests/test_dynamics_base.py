"""
Tests for the base dynamics interface and StateSpace.

This module tests:
- StateSpace dataclass validation
- Abstract interface enforcement
- Base class default implementations
"""

import pytest
import torch

from macro_rl.dynamics.base import StateSpace, ContinuousTimeDynamics


class TestStateSpace:
    """Test StateSpace dataclass."""

    def test_basic_creation(self):
        """Test basic StateSpace creation."""
        ss = StateSpace(
            dim=1,
            lower=torch.tensor([0.0]),
            upper=torch.tensor([1.0]),
            names=("x",)
        )
        assert ss.dim == 1
        assert torch.allclose(ss.lower, torch.tensor([0.0]))
        assert torch.allclose(ss.upper, torch.tensor([1.0]))
        assert ss.names == ("x",)

    def test_multidimensional(self):
        """Test 2D state space."""
        ss = StateSpace(
            dim=2,
            lower=torch.tensor([0.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
            names=("x", "y")
        )
        assert ss.dim == 2
        assert len(ss.lower) == 2
        assert len(ss.upper) == 2
        assert len(ss.names) == 2

    def test_dim_mismatch_lower(self):
        """Test that dimension mismatch in lower bounds raises error."""
        with pytest.raises(AssertionError):
            StateSpace(
                dim=2,
                lower=torch.tensor([0.0]),  # Wrong size
                upper=torch.tensor([1.0, 2.0]),
                names=("x", "y")
            )

    def test_dim_mismatch_upper(self):
        """Test that dimension mismatch in upper bounds raises error."""
        with pytest.raises(AssertionError):
            StateSpace(
                dim=2,
                lower=torch.tensor([0.0, 0.0]),
                upper=torch.tensor([1.0]),  # Wrong size
                names=("x", "y")
            )

    def test_dim_mismatch_names(self):
        """Test that dimension mismatch in names raises error."""
        with pytest.raises(AssertionError):
            StateSpace(
                dim=2,
                lower=torch.tensor([0.0, 0.0]),
                upper=torch.tensor([1.0, 1.0]),
                names=("x",)  # Wrong size
            )

    def test_bounds_validation(self):
        """Test that lower < upper is enforced."""
        with pytest.raises(AssertionError, match="lower must be < upper"):
            StateSpace(
                dim=1,
                lower=torch.tensor([1.0]),
                upper=torch.tensor([0.0]),  # Invalid: upper < lower
                names=("x",)
            )

    def test_bounds_equal(self):
        """Test that lower == upper is rejected."""
        with pytest.raises(AssertionError, match="lower must be < upper"):
            StateSpace(
                dim=1,
                lower=torch.tensor([1.0]),
                upper=torch.tensor([1.0]),  # Invalid: equal
                names=("x",)
            )


class TestContinuousTimeDynamicsInterface:
    """Test abstract interface enforcement."""

    def test_cannot_instantiate_abstract(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            ContinuousTimeDynamics()

    def test_must_implement_abstract_methods(self):
        """Test that subclass must implement all abstract methods."""

        # Missing implementations
        class IncompleteDynamics(ContinuousTimeDynamics):
            pass

        with pytest.raises(TypeError):
            IncompleteDynamics()

    def test_complete_implementation(self):
        """Test that complete implementation works."""

        class CompleteDynamics(ContinuousTimeDynamics):
            @property
            def state_space(self):
                return StateSpace(
                    dim=1,
                    lower=torch.tensor([0.0]),
                    upper=torch.tensor([1.0]),
                    names=("x",)
                )

            @property
            def params(self):
                return {"dummy": 1.0}

            def drift(self, x):
                return x

            def diffusion(self, x):
                return torch.ones_like(x)

            def discount_rate(self):
                return 0.05

        # Should instantiate successfully
        model = CompleteDynamics()
        assert model.state_space.dim == 1
        assert model.params == {"dummy": 1.0}


class TestDefaultImplementations:
    """Test default implementations in base class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing default methods."""

        class SimpleDynamics(ContinuousTimeDynamics):
            @property
            def state_space(self):
                return StateSpace(
                    dim=2,
                    lower=torch.tensor([0.0, -1.0]),
                    upper=torch.tensor([2.0, 1.0]),
                    names=("x", "y")
                )

            @property
            def params(self):
                return {}

            def drift(self, x):
                return torch.zeros_like(x)

            def diffusion(self, x):
                return torch.ones_like(x) * 0.5

            def discount_rate(self):
                return 0.03

        return SimpleDynamics()

    def test_diffusion_squared_default(self, simple_model):
        """Test default diffusion_squared implementation."""
        x = torch.tensor([[1.0, 0.5], [2.0, -0.5]])
        diff = simple_model.diffusion(x)
        diff_sq = simple_model.diffusion_squared(x)

        assert diff_sq.shape == diff.shape
        assert torch.allclose(diff_sq, diff ** 2)

    def test_sample_interior_shape(self, simple_model):
        """Test that sample_interior returns correct shape."""
        n = 100
        samples = simple_model.sample_interior(n)

        assert samples.shape == (n, 2)

    def test_sample_interior_bounds(self, simple_model):
        """Test that sampled points are within bounds."""
        n = 1000
        samples = simple_model.sample_interior(n)

        ss = simple_model.state_space

        # Check all samples are within bounds
        assert (samples[:, 0] >= ss.lower[0]).all()
        assert (samples[:, 0] <= ss.upper[0]).all()
        assert (samples[:, 1] >= ss.lower[1]).all()
        assert (samples[:, 1] <= ss.upper[1]).all()

    def test_sample_boundary_lower(self, simple_model):
        """Test sampling on lower boundary."""
        n = 100
        samples = simple_model.sample_boundary(n, which="lower", dim=0)

        assert samples.shape == (n, 2)
        # First dimension should be at lower bound
        assert torch.allclose(samples[:, 0], torch.zeros(n))

        # Second dimension should vary
        assert samples[:, 1].std() > 0.1

    def test_sample_boundary_upper(self, simple_model):
        """Test sampling on upper boundary."""
        n = 100
        samples = simple_model.sample_boundary(n, which="upper", dim=1)

        assert samples.shape == (n, 2)
        # Second dimension should be at upper bound
        assert torch.allclose(samples[:, 1], torch.ones(n))

        # First dimension should vary
        assert samples[:, 0].std() > 0.1

    def test_sample_boundary_invalid_which(self, simple_model):
        """Test that invalid 'which' parameter raises error."""
        with pytest.raises(ValueError, match="which must be 'lower' or 'upper'"):
            simple_model.sample_boundary(10, which="middle", dim=0)

    def test_sample_interior_device(self, simple_model):
        """Test that samples can be created on specified device."""
        n = 10

        # CPU (default)
        samples_cpu = simple_model.sample_interior(n, device=torch.device("cpu"))
        assert samples_cpu.device.type == "cpu"

        # GPU if available
        if torch.cuda.is_available():
            samples_gpu = simple_model.sample_interior(n, device=torch.device("cuda"))
            assert samples_gpu.device.type == "cuda"

    def test_sample_boundary_device(self, simple_model):
        """Test that boundary samples can be created on specified device."""
        n = 10

        # CPU (default)
        samples_cpu = simple_model.sample_boundary(
            n, which="lower", dim=0, device=torch.device("cpu")
        )
        assert samples_cpu.device.type == "cpu"

        # GPU if available
        if torch.cuda.is_available():
            samples_gpu = simple_model.sample_boundary(
                n, which="lower", dim=0, device=torch.device("cuda")
            )
            assert samples_gpu.device.type == "cuda"


class TestShapeConsistency:
    """Test that all methods maintain consistent tensor shapes."""

    @pytest.fixture
    def model(self):
        """Create a simple 1D model."""

        class TestModel(ContinuousTimeDynamics):
            @property
            def state_space(self):
                return StateSpace(
                    dim=1,
                    lower=torch.tensor([0.0]),
                    upper=torch.tensor([1.0]),
                    names=("x",)
                )

            @property
            def params(self):
                return {}

            def drift(self, x):
                return x * 0.5

            def diffusion(self, x):
                return x * 0.2

            def discount_rate(self):
                return 0.05

        return TestModel()

    @pytest.mark.parametrize("batch_size", [1, 10, 100])
    def test_drift_shape_preservation(self, model, batch_size):
        """Test that drift preserves input shape."""
        x = torch.rand(batch_size, 1)
        mu = model.drift(x)

        assert mu.shape == x.shape

    @pytest.mark.parametrize("batch_size", [1, 10, 100])
    def test_diffusion_shape_preservation(self, model, batch_size):
        """Test that diffusion preserves input shape."""
        x = torch.rand(batch_size, 1)
        sigma = model.diffusion(x)

        assert sigma.shape == x.shape

    @pytest.mark.parametrize("batch_size", [1, 10, 100])
    def test_diffusion_squared_shape_preservation(self, model, batch_size):
        """Test that diffusion_squared preserves input shape."""
        x = torch.rand(batch_size, 1)
        sigma_sq = model.diffusion_squared(x)

        assert sigma_sq.shape == x.shape
