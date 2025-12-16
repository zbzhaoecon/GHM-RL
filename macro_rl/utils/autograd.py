"""
Automatic differentiation utilities for gradient and Hessian computation.
"""

import torch
from torch import Tensor
from typing import Tuple


def compute_gradient(
    output: Tensor,
    input: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute gradient ∂output/∂input.

    Args:
        output: Scalar or vector output (batch,) or (batch, dim_out)
        input: Input tensor (batch, dim_in)
        create_graph: Whether to keep computation graph (for Hessian)

    Returns:
        Gradient (batch, dim_in)

    TODO: Implement gradient computation with proper handling of batch dimensions
    """
    raise NotImplementedError


def compute_hessian(
    output: Tensor,
    input: Tensor,
) -> Tensor:
    """
    Compute Hessian ∂²output/∂input².

    For scalar output and 1D input: returns scalar
    For scalar output and nD input: returns (n, n) matrix

    Args:
        output: Scalar output (batch,)
        input: Input tensor (batch, dim_in)

    Returns:
        Hessian (batch, dim_in, dim_in) or (batch,) for 1D

    TODO: Implement Hessian computation
    """
    raise NotImplementedError


def jacobian_vector_product(
    output: Tensor,
    input: Tensor,
    vector: Tensor,
) -> Tensor:
    """
    Compute Jacobian-vector product J·v efficiently.

    More efficient than computing full Jacobian.

    Args:
        output: Output (batch, dim_out)
        input: Input (batch, dim_in)
        vector: Vector (batch, dim_in)

    Returns:
        J·v (batch, dim_out)

    TODO: Implement JVP
    """
    raise NotImplementedError
