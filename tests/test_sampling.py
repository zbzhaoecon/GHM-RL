"""Tests for state space sampling strategies."""

import torch
import pytest
from scipy import stats
from macro_rl.numerics.sampling import (
    uniform_sample,
    boundary_sample,
    sobol_sample,
    latin_hypercube_sample,
    grid_sample,
    mixed_sample,
)


class TestUniformSample:
    """Tests for uniform random sampling."""

    def test_uniform_bounds(self):
        """Test that all samples are within bounds."""
        lower = torch.tensor([0.0, -1.0])
        upper = torch.tensor([1.0, 2.0])
        n_points = 1000

        samples = uniform_sample(n_points, lower, upper, seed=42)

        assert samples.shape == (n_points, 2)
        assert (samples >= lower).all()
        assert (samples <= upper).all()

    def test_uniform_coverage_1d(self):
        """Test that 1D uniform samples pass Kolmogorov-Smirnov test."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([1.0])
        n_points = 10000

        samples = uniform_sample(n_points, lower, upper, seed=42)

        # KS test for uniformity
        ks_statistic, p_value = stats.kstest(
            samples.numpy().flatten(),
            'uniform',
            args=(0, 1)
        )

        # p-value should be > 0.05 (not rejecting null hypothesis of uniformity)
        assert p_value > 0.05

    def test_uniform_reproducibility(self):
        """Test that same seed gives same samples."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])

        samples1 = uniform_sample(100, lower, upper, seed=42)
        samples2 = uniform_sample(100, lower, upper, seed=42)

        assert torch.allclose(samples1, samples2, atol=1e-10)

    def test_uniform_different_seeds(self):
        """Test that different seeds give different samples."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([1.0])

        samples1 = uniform_sample(100, lower, upper, seed=42)
        samples2 = uniform_sample(100, lower, upper, seed=123)

        assert not torch.allclose(samples1, samples2)

    def test_uniform_high_dimensional(self):
        """Test uniform sampling in high dimensions."""
        dim = 10
        lower = torch.zeros(dim)
        upper = torch.ones(dim)
        n_points = 1000

        samples = uniform_sample(n_points, lower, upper, seed=42)

        assert samples.shape == (n_points, dim)
        assert (samples >= lower).all()
        assert (samples <= upper).all()


class TestBoundarySample:
    """Tests for boundary sampling."""

    def test_boundary_lower(self):
        """Test sampling on lower boundary."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 2.0])
        n_points = 100

        samples = boundary_sample(n_points, lower, upper, which="lower", dim=0, seed=42)

        assert samples.shape == (n_points, 2)
        # First dimension should be at lower bound
        assert torch.allclose(samples[:, 0], torch.zeros(n_points))
        # Second dimension should be in [0, 2]
        assert (samples[:, 1] >= 0.0).all() and (samples[:, 1] <= 2.0).all()

    def test_boundary_upper(self):
        """Test sampling on upper boundary."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 2.0])
        n_points = 100

        samples = boundary_sample(n_points, lower, upper, which="upper", dim=1, seed=42)

        assert samples.shape == (n_points, 2)
        # Second dimension should be at upper bound
        assert torch.allclose(samples[:, 1], torch.ones(n_points) * 2.0)
        # First dimension should be in [0, 1]
        assert (samples[:, 0] >= 0.0).all() and (samples[:, 0] <= 1.0).all()

    def test_boundary_all(self):
        """Test sampling on all boundaries."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])
        n_points = 1000

        samples = boundary_sample(n_points, lower, upper, which="all", seed=42)

        # Should have samples on all 4 boundaries
        # At least some should be on each boundary
        on_x_lower = (torch.abs(samples[:, 0] - 0.0) < 1e-6).sum()
        on_x_upper = (torch.abs(samples[:, 0] - 1.0) < 1e-6).sum()
        on_y_lower = (torch.abs(samples[:, 1] - 0.0) < 1e-6).sum()
        on_y_upper = (torch.abs(samples[:, 1] - 1.0) < 1e-6).sum()

        assert on_x_lower > 0
        assert on_x_upper > 0
        assert on_y_lower > 0
        assert on_y_upper > 0

    def test_boundary_invalid_which(self):
        """Test that invalid 'which' raises error."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([1.0])

        with pytest.raises(ValueError, match="Invalid boundary"):
            boundary_sample(100, lower, upper, which="invalid", dim=0)


class TestSobolSample:
    """Tests for Sobol quasi-random sampling."""

    def test_sobol_bounds(self):
        """Test that Sobol samples are within bounds."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 2.0])
        n_points = 1024

        samples = sobol_sample(n_points, lower, upper, seed=42)

        assert samples.shape == (n_points, 2)
        assert (samples >= lower).all()
        assert (samples <= upper).all()

    def test_sobol_better_coverage(self):
        """Test that Sobol has better coverage than uniform (lower discrepancy)."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])
        n_points = 256

        # Sobol samples
        sobol_samples = sobol_sample(n_points, lower, upper, seed=42)

        # Uniform samples
        uniform_samples = uniform_sample(n_points, lower, upper, seed=42)

        # Compute star discrepancy (simplified version)
        # We'll use a grid-based approximation
        n_grid = 10
        grid_points = torch.linspace(0, 1, n_grid)

        def compute_discrepancy(samples):
            max_disc = 0
            for x in grid_points:
                for y in grid_points:
                    # Count points in [0, x] × [0, y]
                    in_box = ((samples[:, 0] <= x) & (samples[:, 1] <= y)).sum().item()
                    empirical_measure = in_box / len(samples)
                    theoretical_measure = x * y
                    disc = abs(empirical_measure - theoretical_measure)
                    max_disc = max(max_disc, disc)
            return max_disc

        sobol_disc = compute_discrepancy(sobol_samples)
        uniform_disc = compute_discrepancy(uniform_samples)

        # Sobol should have lower discrepancy
        assert sobol_disc < uniform_disc

    def test_sobol_reproducibility(self):
        """Test that Sobol with same seed is reproducible."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])

        samples1 = sobol_sample(100, lower, upper, seed=42)
        samples2 = sobol_sample(100, lower, upper, seed=42)

        assert torch.allclose(samples1, samples2, atol=1e-10)


class TestLatinHypercubeSample:
    """Tests for Latin Hypercube Sampling."""

    def test_lhs_bounds(self):
        """Test that LHS samples are within bounds."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 2.0])
        n_points = 100

        samples = latin_hypercube_sample(n_points, lower, upper, seed=42)

        assert samples.shape == (n_points, 2)
        assert (samples >= lower).all()
        assert (samples <= upper).all()

    def test_lhs_stratification(self):
        """Test that LHS properly stratifies each dimension."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([1.0])
        n_points = 10

        samples = latin_hypercube_sample(n_points, lower, upper, seed=42)

        # Sort samples
        sorted_samples = samples.sort(dim=0)[0].flatten()

        # Each should be in a different decile
        for i in range(n_points):
            expected_min = i / n_points
            expected_max = (i + 1) / n_points
            assert sorted_samples[i] >= expected_min
            assert sorted_samples[i] <= expected_max

    def test_lhs_reproducibility(self):
        """Test that LHS is reproducible with seed."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])

        samples1 = latin_hypercube_sample(100, lower, upper, seed=42)
        samples2 = latin_hypercube_sample(100, lower, upper, seed=42)

        assert torch.allclose(samples1, samples2, atol=1e-10)


class TestGridSample:
    """Tests for regular grid sampling."""

    def test_grid_shape(self):
        """Test that grid has correct shape."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])
        n_points_per_dim = 10

        grid = grid_sample(n_points_per_dim, lower, upper)

        # Should have n^dim total points
        assert grid.shape == (100, 2)

    def test_grid_corners(self):
        """Test that grid includes corners."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 2.0])
        n_points_per_dim = 3

        grid = grid_sample(n_points_per_dim, lower, upper)

        # Check that corners are present
        corners = torch.tensor([
            [0.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 2.0],
        ])

        for corner in corners:
            # Check if this corner is in the grid
            matches = torch.allclose(grid, corner.unsqueeze(0), atol=1e-6)
            has_corner = torch.any(torch.all(
                torch.isclose(grid, corner.unsqueeze(0), atol=1e-6),
                dim=1
            ))
            assert has_corner

    def test_grid_spacing(self):
        """Test that grid has uniform spacing."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([1.0])
        n_points_per_dim = 11

        grid = grid_sample(n_points_per_dim, lower, upper)

        # Sort to check spacing
        sorted_grid = grid.sort(dim=0)[0].flatten()

        # Spacing should be 0.1
        expected_spacing = 0.1
        for i in range(len(sorted_grid) - 1):
            spacing = sorted_grid[i + 1] - sorted_grid[i]
            assert torch.abs(spacing - expected_spacing) < 1e-6

    def test_grid_1d(self):
        """Test 1D grid."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([10.0])
        n_points_per_dim = 11

        grid = grid_sample(n_points_per_dim, lower, upper)

        assert grid.shape == (11, 1)
        assert torch.allclose(grid[0], torch.tensor([[0.0]]))
        assert torch.allclose(grid[-1], torch.tensor([[10.0]]))


class TestMixedSample:
    """Tests for mixed interior and boundary sampling."""

    def test_mixed_sample_count(self):
        """Test that mixed sample has correct number of points."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])
        n_interior = 800
        n_boundary = 200

        samples = mixed_sample(n_interior, n_boundary, lower, upper, seed=42)

        assert samples.shape == (1000, 2)

    def test_mixed_sample_uniform(self):
        """Test mixed sampling with uniform interior."""
        samples = mixed_sample(
            n_interior=100,
            n_boundary=50,
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
            sampler="uniform",
            seed=42
        )

        assert samples.shape == (150, 2)
        assert (samples >= 0.0).all() and (samples <= 1.0).all()

    def test_mixed_sample_sobol(self):
        """Test mixed sampling with Sobol interior."""
        samples = mixed_sample(
            n_interior=100,
            n_boundary=50,
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
            sampler="sobol",
            seed=42
        )

        assert samples.shape == (150, 2)

    def test_mixed_sample_lhs(self):
        """Test mixed sampling with LHS interior."""
        samples = mixed_sample(
            n_interior=100,
            n_boundary=50,
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
            sampler="lhs",
            seed=42
        )

        assert samples.shape == (150, 2)

    def test_mixed_sample_invalid_sampler(self):
        """Test that invalid sampler raises error."""
        with pytest.raises(ValueError, match="Unknown sampler"):
            mixed_sample(
                n_interior=100,
                n_boundary=50,
                lower=torch.tensor([0.0]),
                upper=torch.tensor([1.0]),
                sampler="invalid"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
