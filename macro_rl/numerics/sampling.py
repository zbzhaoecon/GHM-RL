"""Sampling strategies for state space exploration.

This module provides various methods to generate points in the state space
for Monte Carlo training of neural network PDE solvers.
"""

from typing import Optional, List
import torch
from torch import Tensor
import math


def uniform_sample(
    n_points: int,
    lower: Tensor,
    upper: Tensor,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate uniform random samples in a hyperrectangle.

    Args:
        n_points: Number of points to generate.
        lower: Lower bounds for each dimension, shape (dim,).
        upper: Upper bounds for each dimension, shape (dim,).
        seed: Optional random seed for reproducibility.

    Returns:
        Tensor of shape (n_points, dim) with samples in [lower, upper].

    Example:
        >>> lower = torch.tensor([0.0, 0.0])
        >>> upper = torch.tensor([1.0, 2.0])
        >>> samples = uniform_sample(100, lower, upper)
        >>> assert samples.shape == (100, 2)
        >>> assert (samples >= lower).all() and (samples <= upper).all()
    """
    if seed is not None:
        torch.manual_seed(seed)

    dim = lower.shape[0]

    # Generate uniform [0, 1] samples
    u = torch.rand(n_points, dim, device=lower.device, dtype=lower.dtype)

    # Scale to [lower, upper]
    samples = lower + u * (upper - lower)

    return samples


def boundary_sample(
    n_points: int,
    lower: Tensor,
    upper: Tensor,
    which: str = "lower",
    dim: int = 0,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate samples on the boundary of a hyperrectangle.

    Args:
        n_points: Number of points to generate.
        lower: Lower bounds for each dimension, shape (dim,).
        upper: Upper bounds for each dimension, shape (dim,).
        which: Which boundary to sample from: "lower", "upper", or "all".
        dim: Which dimension's boundary to sample (0-indexed).
               Only used if which != "all".
        seed: Optional random seed for reproducibility.

    Returns:
        Tensor of shape (n_points, dim) with samples on the boundary.

    Example:
        >>> lower = torch.tensor([0.0, 0.0])
        >>> upper = torch.tensor([1.0, 2.0])
        >>> # Sample on the lower boundary of dimension 0 (x=0)
        >>> samples = boundary_sample(100, lower, upper, which="lower", dim=0)
        >>> assert torch.allclose(samples[:, 0], torch.zeros(100))
    """
    if seed is not None:
        torch.manual_seed(seed)

    n_dims = lower.shape[0]

    if which == "all":
        # Sample uniformly from all 2*n_dims boundary faces
        samples_list = []
        points_per_face = n_points // (2 * n_dims)
        remainder = n_points % (2 * n_dims)

        for d in range(n_dims):
            # Lower face
            n_face = points_per_face + (1 if d * 2 < remainder else 0)
            if n_face > 0:
                face_samples = uniform_sample(n_face, lower, upper)
                face_samples[:, d] = lower[d]
                samples_list.append(face_samples)

            # Upper face
            n_face = points_per_face + (1 if d * 2 + 1 < remainder else 0)
            if n_face > 0:
                face_samples = uniform_sample(n_face, lower, upper)
                face_samples[:, d] = upper[d]
                samples_list.append(face_samples)

        samples = torch.cat(samples_list, dim=0)

    else:
        # Sample from a specific boundary face
        samples = uniform_sample(n_points, lower, upper, seed=seed)

        if which == "lower":
            samples[:, dim] = lower[dim]
        elif which == "upper":
            samples[:, dim] = upper[dim]
        else:
            raise ValueError(f"Invalid boundary specification: {which}")

    return samples


def sobol_sample(
    n_points: int,
    lower: Tensor,
    upper: Tensor,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate quasi-random Sobol sequence samples.

    Sobol sequences provide better coverage of the space than pure random sampling,
    with lower discrepancy. This can improve convergence for Monte Carlo methods.

    Args:
        n_points: Number of points to generate.
        lower: Lower bounds for each dimension, shape (dim,).
        upper: Upper bounds for each dimension, shape (dim,).
        seed: Optional seed for the Sobol engine.

    Returns:
        Tensor of shape (n_points, dim) with Sobol samples in [lower, upper].

    Example:
        >>> lower = torch.tensor([0.0, 0.0])
        >>> upper = torch.tensor([1.0, 1.0])
        >>> samples = sobol_sample(1024, lower, upper)
        >>> # Sobol samples have better coverage than uniform random
    """
    dim = lower.shape[0]

    # Create Sobol engine
    if seed is not None:
        sobol_engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    else:
        sobol_engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)

    # Generate samples in [0, 1]^dim
    u = sobol_engine.draw(n_points).to(device=lower.device, dtype=lower.dtype)

    # Scale to [lower, upper]
    samples = lower + u * (upper - lower)

    return samples


def latin_hypercube_sample(
    n_points: int,
    lower: Tensor,
    upper: Tensor,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate Latin Hypercube samples.

    Latin Hypercube Sampling (LHS) ensures each dimension is evenly stratified,
    providing better coverage than pure random sampling.

    Args:
        n_points: Number of points to generate.
        lower: Lower bounds for each dimension, shape (dim,).
        upper: Upper bounds for each dimension, shape (dim,).
        seed: Optional random seed.

    Returns:
        Tensor of shape (n_points, dim) with LHS samples in [lower, upper].

    Example:
        >>> lower = torch.tensor([0.0, 0.0])
        >>> upper = torch.tensor([1.0, 1.0])
        >>> samples = latin_hypercube_sample(100, lower, upper)
    """
    if seed is not None:
        torch.manual_seed(seed)

    dim = lower.shape[0]

    # Create stratified samples for each dimension
    samples = torch.zeros(n_points, dim, device=lower.device, dtype=lower.dtype)

    for d in range(dim):
        # Divide [0, 1] into n_points strata
        # Sample uniformly within each stratum
        strata = torch.arange(n_points, device=lower.device, dtype=lower.dtype)
        offsets = torch.rand(n_points, device=lower.device, dtype=lower.dtype)
        u = (strata + offsets) / n_points

        # Randomly permute
        perm = torch.randperm(n_points, device=lower.device)
        u = u[perm]

        # Scale to [lower[d], upper[d]]
        samples[:, d] = lower[d] + u * (upper[d] - lower[d])

    return samples


def grid_sample(
    n_points_per_dim: int,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """
    Generate a regular grid of points.

    Args:
        n_points_per_dim: Number of points per dimension.
        lower: Lower bounds for each dimension, shape (dim,).
        upper: Upper bounds for each dimension, shape (dim,).

    Returns:
        Tensor of shape (n_points_per_dim^dim, dim) with grid points.

    Example:
        >>> lower = torch.tensor([0.0, 0.0])
        >>> upper = torch.tensor([1.0, 1.0])
        >>> grid = grid_sample(10, lower, upper)  # 10x10 = 100 points
        >>> assert grid.shape == (100, 2)
    """
    dim = lower.shape[0]

    # Create 1D grids for each dimension
    grids_1d = []
    for d in range(dim):
        grid_1d = torch.linspace(
            lower[d].item(),
            upper[d].item(),
            n_points_per_dim,
            device=lower.device,
            dtype=lower.dtype
        )
        grids_1d.append(grid_1d)

    # Create meshgrid
    meshgrids = torch.meshgrid(*grids_1d, indexing='ij')

    # Flatten and stack
    points = torch.stack([g.flatten() for g in meshgrids], dim=1)

    return points


def mixed_sample(
    n_interior: int,
    n_boundary: int,
    lower: Tensor,
    upper: Tensor,
    sampler: str = "uniform",
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate a mix of interior and boundary samples.

    This is useful for physics-informed neural networks where both
    interior PDE residuals and boundary conditions need to be satisfied.

    Args:
        n_interior: Number of interior points.
        n_boundary: Number of boundary points.
        lower: Lower bounds for each dimension, shape (dim,).
        upper: Upper bounds for each dimension, shape (dim,).
        sampler: Sampling method for interior points: "uniform", "sobol", or "lhs".
        seed: Optional random seed.

    Returns:
        Tensor of shape (n_interior + n_boundary, dim) with mixed samples.

    Example:
        >>> lower = torch.tensor([0.0, 0.0])
        >>> upper = torch.tensor([1.0, 1.0])
        >>> samples = mixed_sample(800, 200, lower, upper, sampler="sobol")
        >>> assert samples.shape == (1000, 2)
    """
    # Generate interior points
    if sampler == "uniform":
        interior = uniform_sample(n_interior, lower, upper, seed=seed)
    elif sampler == "sobol":
        interior = sobol_sample(n_interior, lower, upper, seed=seed)
    elif sampler == "lhs":
        interior = latin_hypercube_sample(n_interior, lower, upper, seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # Generate boundary points
    boundary = boundary_sample(n_boundary, lower, upper, which="all", seed=seed)

    # Concatenate
    samples = torch.cat([interior, boundary], dim=0)

    return samples
