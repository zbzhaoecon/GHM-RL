"""
Compare learned solutions with analytical solutions (when available).
"""

import torch
from torch import Tensor
from typing import Callable, Dict


class AnalyticalComparator:
    """
    Compare learned vs analytical solutions.

    TODO: Implement comparison tools
    """

    def __init__(
        self,
        analytical_value: Callable,
        analytical_policy: Callable = None,
    ):
        """
        Initialize comparator.

        Args:
            analytical_value: Analytical V(s)
            analytical_policy: Analytical π(s)

        TODO: Implement initialization
        """
        self.analytical_value = analytical_value
        self.analytical_policy = analytical_policy

    def compare_value_functions(
        self,
        learned_value,
        test_states: Tensor,
    ) -> Dict[str, float]:
        """
        Compare learned vs analytical value functions.

        Returns metrics: MSE, max error, correlation, etc.

        TODO: Implement comparison
        """
        raise NotImplementedError

    def compare_policies(
        self,
        learned_policy,
        test_states: Tensor,
    ) -> Dict[str, float]:
        """
        Compare learned vs analytical policies.

        TODO: Implement comparison
        """
        raise NotImplementedError
