"""Tests for automatic differentiation utilities."""

import torch
import pytest
import math
from macro_rl.numerics.differentiation import (
    gradient,
    hessian_diagonal,
    mixed_partial,
    hessian_matrix,
)


class TestGradient:
    """Tests for gradient computation."""

    def test_gradient_quadratic(self):
        """Test gradient of f(x) = x²."""
        def f(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        x = torch.tensor([[2.0, 3.0]], requires_grad=True)
        grad = gradient(f, x)

        expected = torch.tensor([[4.0, 6.0]])
        assert torch.allclose(grad, expected, atol=1e-10)

    def test_gradient_linear(self):
        """Test gradient of f(x) = a·x."""
        a = torch.tensor([2.0, -1.0, 3.0])

        def f(x):
            return (a * x).sum(dim=1, keepdim=True)

        x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        grad = gradient(f, x)

        expected = a.unsqueeze(0)
        assert torch.allclose(grad, expected, atol=1e-10)

    def test_gradient_polynomial(self):
        """Test gradient of f(x, y) = x³ + 2xy + y²."""
        def f(x):
            x_val = x[:, 0:1]
            y_val = x[:, 1:2]
            return x_val ** 3 + 2 * x_val * y_val + y_val ** 2

        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        grad = gradient(f, x)

        # ∂f/∂x = 3x² + 2y = 3(1)² + 2(2) = 7
        # ∂f/∂y = 2x + 2y = 2(1) + 2(2) = 6
        expected = torch.tensor([[7.0, 6.0]])
        assert torch.allclose(grad, expected, atol=1e-10)

    def test_gradient_batch(self):
        """Test gradient with multiple batch elements."""
        def f(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        grad = gradient(f, x)

        expected = 2 * x
        assert torch.allclose(grad, expected, atol=1e-10)

    def test_gradient_neural_net(self):
        """Test gradient of a simple neural network."""
        # Simple MLP: f(x) = W2 * relu(W1 * x)
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1)
        )

        def f(x):
            return net(x)

        x = torch.randn(10, 2, requires_grad=True)
        grad = gradient(f, x)

        # Verify against finite differences
        eps = 1e-4
        grad_fd = torch.zeros_like(grad)

        with torch.no_grad():
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x_plus = x.clone()
                    x_minus = x.clone()
                    x_plus[i, j] += eps
                    x_minus[i, j] -= eps

                    f_plus = f(x_plus)[i]
                    f_minus = f(x_minus)[i]

                    grad_fd[i, j] = (f_plus - f_minus) / (2 * eps)

        assert torch.allclose(grad, grad_fd, atol=1e-3, rtol=1e-2)


class TestHessianDiagonal:
    """Tests for diagonal Hessian computation."""

    def test_hessian_diagonal_quadratic(self):
        """Test Hessian of f(x) = x²."""
        def f(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        x = torch.tensor([[2.0, 3.0]], requires_grad=True)
        hess_diag = hessian_diagonal(f, x)

        expected = torch.tensor([[2.0, 2.0]])
        assert torch.allclose(hess_diag, expected, atol=1e-10)

    def test_hessian_diagonal_quartic(self):
        """Test Hessian of f(x) = x⁴."""
        def f(x):
            return (x ** 4).sum(dim=1, keepdim=True)

        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        hess_diag = hessian_diagonal(f, x)

        # ∂²f/∂x² = 12x²
        expected = torch.tensor([[12.0, 48.0]])
        assert torch.allclose(hess_diag, expected, atol=1e-10)

    def test_hessian_diagonal_polynomial(self):
        """Test Hessian of f(x, y) = x³ + y³."""
        def f(x):
            return (x ** 3).sum(dim=1, keepdim=True)

        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        hess_diag = hessian_diagonal(f, x)

        # ∂²f/∂x² = 6x, ∂²f/∂y² = 6y
        expected = torch.tensor([[6.0, 12.0]])
        assert torch.allclose(hess_diag, expected, atol=1e-10)

    def test_hessian_diagonal_batch(self):
        """Test Hessian diagonal with multiple batch elements."""
        def f(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        hess_diag = hessian_diagonal(f, x)

        expected = torch.ones_like(x) * 2.0
        assert torch.allclose(hess_diag, expected, atol=1e-10)

    def test_hessian_diagonal_neural_net(self):
        """Test Hessian diagonal of a neural network."""
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 1)
        )

        def f(x):
            return net(x)

        x = torch.randn(5, 2, requires_grad=True)
        hess_diag = hessian_diagonal(f, x)

        # Verify against finite differences
        eps = 1e-4
        hess_fd = torch.zeros_like(hess_diag)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_eval = x.clone().detach().requires_grad_(True)

                # Compute gradient at x
                grad = gradient(f, x_eval)
                g_center = grad[i, j]

                # Compute gradient at x + eps
                x_plus = x.clone()
                x_plus[i, j] += eps
                x_plus_eval = x_plus.detach().requires_grad_(True)
                grad_plus = gradient(f, x_plus_eval)
                g_plus = grad_plus[i, j]

                # Compute gradient at x - eps
                x_minus = x.clone()
                x_minus[i, j] -= eps
                x_minus_eval = x_minus.detach().requires_grad_(True)
                grad_minus = gradient(f, x_minus_eval)
                g_minus = grad_minus[i, j]

                # Finite difference approximation
                hess_fd[i, j] = (g_plus - g_minus) / (2 * eps)

        assert torch.allclose(hess_diag, hess_fd, atol=1e-2, rtol=1e-1)


class TestMixedPartial:
    """Tests for mixed partial derivatives."""

    def test_mixed_partial_xy_squared(self):
        """Test ∂²f/∂x∂y for f(x,y) = xy²."""
        def f(x):
            return x[:, 0:1] * x[:, 1:2] ** 2

        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        mixed = mixed_partial(f, x, 0, 1)

        # ∂²f/∂x∂y = 2y
        expected = torch.tensor([[4.0]])
        assert torch.allclose(mixed, expected, atol=1e-10)

    def test_mixed_partial_symmetric(self):
        """Test that ∂²f/∂x∂y = ∂²f/∂y∂x."""
        def f(x):
            return x[:, 0:1] ** 2 * x[:, 1:2] ** 3

        x = torch.randn(5, 2, requires_grad=True)

        mixed_xy = mixed_partial(f, x, 0, 1)
        mixed_yx = mixed_partial(f, x, 1, 0)

        assert torch.allclose(mixed_xy, mixed_yx, atol=1e-6)

    def test_mixed_partial_zero(self):
        """Test that ∂²f/∂x∂y = 0 for separable functions."""
        def f(x):
            return x[:, 0:1] ** 2 + x[:, 1:2] ** 2

        x = torch.randn(3, 2, requires_grad=True)
        mixed = mixed_partial(f, x, 0, 1)

        expected = torch.zeros_like(mixed)
        assert torch.allclose(mixed, expected, atol=1e-10)


class TestHessianMatrix:
    """Tests for full Hessian matrix computation."""

    def test_hessian_matrix_quadratic(self):
        """Test full Hessian of f(x) = x^T A x."""
        # f(x) = x^T A x where A = [[2, 1], [1, 3]]
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

        def f(x):
            return (x @ A @ x.T).unsqueeze(1)

        x = torch.tensor([[1.0, 1.0]], requires_grad=True)
        hess = hessian_matrix(f, x)

        # Hessian = 2*A (for quadratic form)
        expected = 2 * A
        assert torch.allclose(hess, expected, atol=1e-6)

    def test_hessian_matrix_polynomial(self):
        """Test Hessian of f(x,y) = x³ + xy² + y³."""
        def f(x):
            x_val = x[:, 0:1]
            y_val = x[:, 1:2]
            return x_val ** 3 + x_val * y_val ** 2 + y_val ** 3

        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        hess = hessian_matrix(f, x)

        # ∂²f/∂x² = 6x = 6
        # ∂²f/∂x∂y = ∂²f/∂y∂x = 2y = 4
        # ∂²f/∂y² = 2x + 6y = 2 + 12 = 14
        expected = torch.tensor([[6.0, 4.0], [4.0, 14.0]])
        assert torch.allclose(hess, expected, atol=1e-6)

    def test_hessian_matrix_batch_error(self):
        """Test that hessian_matrix raises error for batch_size > 1."""
        def f(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        x = torch.randn(5, 2, requires_grad=True)

        with pytest.raises(ValueError, match="batch_size=1"):
            hessian_matrix(f, x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
