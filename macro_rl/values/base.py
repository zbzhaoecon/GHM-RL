"""
Base class for value functions.
"""

from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor


class ValueFunction(ABC, nn.Module):
    """
    Abstract base class for value functions V(s).

    TODO: Define interface for value functions
    """

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        """
        Compute value V(s).

        Args:
            state: States (batch, state_dim)

        Returns:
            Values (batch,)

        TODO: Implement in subclasses
        """
        pass

    def gradient(self, state: Tensor) -> Tensor:
        """
        Compute value gradient ∇V(s).

        Args:
            state: States (batch, state_dim)

        Returns:
            Gradients (batch, state_dim)

        TODO: Implement using autograd
        """
        raise NotImplementedError

    def hessian(self, state: Tensor) -> Tensor:
        """
        Compute value Hessian ∇²V(s).

        Args:
            state: States (batch, state_dim)

        Returns:
            Hessians (batch, state_dim, state_dim)

        TODO: Implement using autograd
        """
        raise NotImplementedError
