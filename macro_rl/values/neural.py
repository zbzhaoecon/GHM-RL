"""
Neural network value functions.
"""

import torch
import torch.nn as nn
from torch import Tensor
from macro_rl.values.base import ValueFunction


class ValueNetwork(ValueFunction):
    """
    Neural network value function.

    Architecture:
        state → [hidden layers] → scalar value

    Features:
        - Automatic differentiation for gradients/Hessians
        - Flexible architecture
        - Support for HJB residual computation

    Example:
        >>> value_net = ValueNetwork(
        ...     state_dim=1,
        ...     hidden_dims=[64, 64],
        ... )
        >>> V = value_net(state)
        >>> V_grad = value_net.gradient(state)
        >>> V_hess = value_net.hessian(state)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [64, 64],
        activation: str = "tanh",
    ):
        """
        Initialize value network.

        Args:
            state_dim: State dimension
            hidden_dims: Hidden layer sizes
            activation: Activation function

        TODO: Implement network architecture
        - Build fully connected network
        - Initialize weights appropriately
        """
        super().__init__(state_dim)
        # TODO: Build network
        raise NotImplementedError

    def forward(self, state: Tensor) -> Tensor:
        """
        Compute V(s).

        TODO: Implement forward pass
        """
        raise NotImplementedError

    def gradient(self, state: Tensor) -> Tensor:
        """
        Compute ∇V(s) using autograd.

        TODO: Implement gradient computation
        - Use torch.autograd.grad
        - Handle batch dimensions
        """
        raise NotImplementedError

    def hessian(self, state: Tensor) -> Tensor:
        """
        Compute ∇²V(s) using autograd.

        TODO: Implement Hessian computation
        - Compute second derivatives
        - For 1D: just V_cc scalar
        - For nD: full Hessian matrix
        """
        raise NotImplementedError
