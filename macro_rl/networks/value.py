from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class ValueNetwork(nn.Module):
    """Value function network with HJB-friendly interface.

    Methods
    -------
    forward(state)
        Returns V(s) with shape (batch,).
    forward_with_grad(state)
        Returns (V, V_s, V_ss_diag), where:
        - V: (batch,)
        - V_s: (batch, state_dim)          # gradient wrt state
        - V_ss_diag: (batch, state_dim)    # diagonal of Hessian wrt state
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "tanh",  # tanh/softplus smoother than ReLU
    ) -> None:
        super().__init__()

        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "softplus":
            act_fn = nn.Softplus
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor) -> Tensor:
        """V(s) as a scalar per batch element, shape (batch,)."""
        return self.net(state).squeeze(-1)

    def forward_with_grad(
        self,
        state: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (V, V_s, V_ss_diag).

        - V: (batch,)
        - V_s: (batch, state_dim)
        - V_ss_diag: (batch, state_dim) [diagonal Hessian entries]
        """
        # We want gradients w.r.t. state
        state = state.clone().detach().requires_grad_(True)
        V = self.forward(state)

        # First derivative wrt state
        V_s = torch.autograd.grad(
            V.sum(), state, create_graph=True
        )[0]

        # Diagonal of Hessian via second derivatives
        V_ss_diag = []
        for i in range(state.shape[-1]):
            grad_i = V_s[:, i]               # (batch,)
            V_ss_i = torch.autograd.grad(
                grad_i.sum(), state, create_graph=True
            )[0][:, i]                        # (batch,)
            V_ss_diag.append(V_ss_i)
        V_ss_diag = torch.stack(V_ss_diag, dim=-1)

        return V, V_s, V_ss_diag
