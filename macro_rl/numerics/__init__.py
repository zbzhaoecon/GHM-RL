"""Numerical methods for differentiation, integration, and sampling."""

from macro_rl.numerics.differentiation import gradient, hessian_diagonal, mixed_partial
from macro_rl.numerics.integration import euler_maruyama_step, simulate_path
from macro_rl.numerics.sampling import uniform_sample, boundary_sample, sobol_sample

__all__ = [
    # Differentiation
    "gradient",
    "hessian_diagonal",
    "mixed_partial",
    # Integration
    "euler_maruyama_step",
    "simulate_path",
    # Sampling
    "uniform_sample",
    "boundary_sample",
    "sobol_sample",
]
