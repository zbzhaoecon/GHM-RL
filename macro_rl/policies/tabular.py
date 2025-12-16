"""
Tabular policy on discretized state space.

Useful for debugging and small-scale problems.
"""

import torch
from torch import Tensor
from macro_rl.policies.base import Policy


class TabularPolicy(Policy):
    """
    Tabular policy on grid.

    TODO: Implement grid-based policy for debugging
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        grid_size: int = 100,
    ):
        """
        Initialize tabular policy.

        TODO: Implement grid initialization
        """
        super().__init__(state_dim, action_dim)
        self.grid_size = grid_size
        raise NotImplementedError

    def forward(self, state: Tensor) -> Tensor:
        """
        Look up action on grid.

        TODO: Implement grid lookup
        """
        raise NotImplementedError
