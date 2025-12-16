"""
Analytical value functions (when known).
"""

import torch
from torch import Tensor
from macro_rl.values.base import ValueFunction


class AnalyticalValue(ValueFunction):
    """
    Wrapper for analytical value functions.

    Used for:
        - Testing (compare learned vs analytical)
        - Baselines
        - Validation

    TODO: Implement wrapper for analytical solutions
    """

    def __init__(self, value_fn, state_dim: int = 1):
        """
        Initialize with analytical function.

        Args:
            value_fn: Function state -> value
            state_dim: State dimension

        TODO: Implement initialization
        """
        super().__init__(state_dim)
        self.value_fn = value_fn

    def forward(self, state: Tensor) -> Tensor:
        """
        Evaluate analytical function.

        TODO: Implement forward
        """
        return self.value_fn(state)
