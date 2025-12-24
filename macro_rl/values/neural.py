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
        """
        super().__init__(state_dim)

        # Activation function
        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "elu":
            act_fn = nn.ELU
        elif activation == "softplus":
            act_fn = nn.Softplus
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        # Output layer (no activation for value)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: Tensor) -> Tensor:
        """
        Compute V(s).

        Args:
            state: States (batch, state_dim)

        Returns:
            Values (batch,) - scalar value per state
        """
        return self.net(state).squeeze(-1)

    def gradient(self, state: Tensor) -> Tensor:
        """
        Compute ∇V(s) using autograd.

        Args:
            state: States (batch, state_dim)

        Returns:
            Gradients (batch, state_dim)
        """
        # Clone and enable gradients
        state = state.clone().detach().requires_grad_(True)

        # Compute value
        value = self.forward(state)

        # Compute gradient
        grad = torch.autograd.grad(
            outputs=value.sum(),
            inputs=state,
            create_graph=True,
        )[0]

        return grad

    def hessian(self, state: Tensor) -> Tensor:
        """
        Compute value Hessian ∇²V(s).

        Args:
            state: States (batch, state_dim)

        Returns:
            Hessians (batch, state_dim, state_dim) for general case
            or (batch,) for 1D case (diagonal element)
        """
        # Clone and enable gradients
        state = state.clone().detach().requires_grad_(True)

        # Compute value
        value = self.forward(state)

        # Compute first derivative
        V_s = torch.autograd.grad(
            outputs=value.sum(),
            inputs=state,
            create_graph=True,
        )[0]

        # For 1D state space, return single Hessian diagonal element
        if state.shape[-1] == 1:
            V_ss = torch.autograd.grad(
                outputs=V_s.sum(),
                inputs=state,
                create_graph=True,
            )[0]
            return V_ss.squeeze(-1)  # (batch,)

        # For multi-dimensional, compute diagonal Hessian
        batch_size, state_dim = state.shape
        V_ss_diag = []
        for i in range(state_dim):
            grad_i = V_s[:, i]  # (batch,)
            grad_output = torch.autograd.grad(
                outputs=grad_i.sum(),
                inputs=state,
                create_graph=True,
                allow_unused=True,
            )[0]
            if grad_output is not None:
                V_ss_i = grad_output[:, i]  # (batch,)
            else:
                V_ss_i = torch.zeros_like(grad_i)
            V_ss_diag.append(V_ss_i)

        return torch.stack(V_ss_diag, dim=-1)  # (batch, state_dim)
