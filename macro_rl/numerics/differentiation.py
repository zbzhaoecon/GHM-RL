"""Automatic differentiation utilities for computing derivatives of neural networks.

This module provides functions to compute gradients and Hessians of scalar-valued
functions with respect to their inputs using PyTorch's autograd functionality.
"""

from typing import Callable, Optional
import torch
from torch import Tensor


def gradient(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """
    Compute the gradient ∇f(x) using automatic differentiation.

    Args:
        f: A function mapping (batch, n) -> (batch, 1).
           The function should return a scalar output for each input.
        x: Input tensor of shape (batch, n).
           Must have requires_grad=True or will be enabled automatically.

    Returns:
        Tensor of shape (batch, n) containing ∇f(x) for each batch element.

    Example:
        >>> def f(x):
        ...     return (x ** 2).sum(dim=1, keepdim=True)
        >>> x = torch.tensor([[2.0, 3.0]], requires_grad=True)
        >>> gradient(f, x)
        tensor([[4., 6.]])
    """
    # Ensure x requires gradients
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)

    # Compute function value
    y = f(x)

    # Ensure output is shape (batch, 1)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    batch_size = x.shape[0]
    n_dims = x.shape[1] if x.dim() > 1 else 1

    # Compute gradient for each batch element
    gradients = []
    for i in range(batch_size):
        # Create gradient vector for this batch element
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1.0

        # Compute gradient
        grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,  # Allow second derivatives
            retain_graph=True,
        )[0]

        gradients.append(grad[i])

    return torch.stack(gradients, dim=0)


def hessian_diagonal(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """
    Compute the diagonal elements of the Hessian [∂²f/∂x₁², ∂²f/∂x₂², ...].

    This computes only the diagonal second derivatives, not the full Hessian matrix.
    This is more efficient and sufficient for many applications (e.g., HJB equations).

    Args:
        f: A function mapping (batch, n) -> (batch, 1).
        x: Input tensor of shape (batch, n).

    Returns:
        Tensor of shape (batch, n) containing diagonal Hessian elements.

    Example:
        >>> def f(x):
        ...     return (x ** 2).sum(dim=1, keepdim=True)
        >>> x = torch.tensor([[2.0, 3.0]], requires_grad=True)
        >>> hessian_diagonal(f, x)
        tensor([[2., 2.]])
    """
    # Ensure x requires gradients
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)

    batch_size = x.shape[0]
    n_dims = x.shape[1] if x.dim() > 1 else 1

    # Compute first derivatives
    grad = gradient(f, x)

    # Compute second derivatives for each dimension
    hess_diag = []
    for dim in range(n_dims):
        # Take gradient of the dim-th component of the gradient
        grad_dim = grad[:, dim:dim+1]

        # Compute second derivative
        second_derivatives = []
        for i in range(batch_size):
            grad_outputs = torch.zeros_like(grad_dim)
            grad_outputs[i] = 1.0

            grad2 = torch.autograd.grad(
                outputs=grad_dim,
                inputs=x,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=True,
            )[0]

            second_derivatives.append(grad2[i, dim])

        hess_diag.append(torch.stack(second_derivatives, dim=0))

    return torch.stack(hess_diag, dim=1)


def mixed_partial(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    i: int,
    j: int
) -> Tensor:
    """
    Compute the mixed partial derivative ∂²f/∂xᵢ∂xⱼ.

    Args:
        f: A function mapping (batch, n) -> (batch, 1).
        x: Input tensor of shape (batch, n).
        i: First dimension index (0-based).
        j: Second dimension index (0-based).

    Returns:
        Tensor of shape (batch, 1) containing ∂²f/∂xᵢ∂xⱼ.

    Example:
        >>> def f(x):
        ...     return (x[:, 0:1] * x[:, 1:2] ** 2)
        >>> x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        >>> mixed_partial(f, x, 0, 1)  # ∂²f/∂x∂y = 2y
        tensor([[4.]])
    """
    # Ensure x requires gradients
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)

    batch_size = x.shape[0]

    # Compute gradient
    grad = gradient(f, x)

    # Take the i-th component
    grad_i = grad[:, i:i+1]

    # Compute derivative with respect to j-th variable
    mixed_derivs = []
    for batch_idx in range(batch_size):
        grad_outputs = torch.zeros_like(grad_i)
        grad_outputs[batch_idx] = 1.0

        grad2 = torch.autograd.grad(
            outputs=grad_i,
            inputs=x,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
        )[0]

        mixed_derivs.append(grad2[batch_idx, j])

    return torch.stack(mixed_derivs, dim=0).unsqueeze(1)


def hessian_matrix(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """
    Compute the full Hessian matrix for a single input.

    This is more expensive than hessian_diagonal but provides all second derivatives.
    Only works with batch_size=1 for simplicity.

    Args:
        f: A function mapping (1, n) -> (1, 1).
        x: Input tensor of shape (1, n).

    Returns:
        Tensor of shape (n, n) containing the full Hessian matrix.

    Example:
        >>> def f(x):
        ...     return (x ** 2).sum(dim=1, keepdim=True)
        >>> x = torch.tensor([[2.0, 3.0]], requires_grad=True)
        >>> hessian_matrix(f, x)
        tensor([[2., 0.],
                [0., 2.]])
    """
    if x.shape[0] != 1:
        raise ValueError("hessian_matrix only supports batch_size=1")

    # Ensure x requires gradients
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)

    n_dims = x.shape[1]

    # Compute gradient
    grad = gradient(f, x).squeeze(0)  # Shape: (n,)

    # Compute Hessian matrix
    hess = []
    for i in range(n_dims):
        # Compute gradient of grad[i] with respect to x
        grad_i = torch.autograd.grad(
            outputs=grad[i],
            inputs=x,
            retain_graph=True,
            create_graph=True,
        )[0]
        hess.append(grad_i.squeeze(0))

    return torch.stack(hess, dim=0)
