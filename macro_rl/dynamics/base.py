"""
Abstract base class for continuous-time economic models.

This module defines the interface that all economic models must implement.
Solvers query this interface to get drift, diffusion, and discount rates.

Phase 2 implementation - Complete.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch import Tensor


@dataclass
class StateSpace:
    """Description of the state space."""
    dim: int                    # Number of state variables
    lower: Tensor              # Lower bounds (dim,)
    upper: Tensor              # Upper bounds (dim,)
    names: Tuple[str, ...]     # Variable names, e.g., ("c",) or ("eta", "c")

    def __post_init__(self):
        assert len(self.lower) == self.dim
        assert len(self.upper) == self.dim
        assert len(self.names) == self.dim
        assert (self.lower < self.upper).all(), "lower must be < upper"


class ContinuousTimeDynamics(ABC):
    """
    Abstract base class for continuous-time economic models.

    Represents an SDE of the form:
        dX = μ(X) dt + σ(X) dW

    with boundary conditions and discount rate for HJB equations.
    """

    @property
    @abstractmethod
    def state_space(self) -> StateSpace:
        """Return state space specification."""
        pass

    @property
    @abstractmethod
    def params(self) -> Dict[str, float]:
        """Return model parameters as dictionary (for logging/reproducibility)."""
        pass

    @abstractmethod
    def drift(self, x: Tensor) -> Tensor:
        """
        Drift coefficient μ(x).

        Args:
            x: State tensor (batch, state_dim)

        Returns:
            Drift (batch, state_dim)
        """
        pass

    @abstractmethod
    def diffusion(self, x: Tensor) -> Tensor:
        """
        Diffusion coefficient σ(x).

        Args:
            x: State tensor (batch, state_dim)

        Returns:
            Diffusion (batch, state_dim) for diagonal noise
        """
        pass

    @abstractmethod
    def discount_rate(self) -> float:
        """
        Effective discount rate for HJB equation.

        For most models this is (r - μ) where r is interest rate
        and μ is growth rate.
        """
        pass

    def diffusion_squared(self, x: Tensor) -> Tensor:
        """
        Squared diffusion σ(x)² for HJB equation.

        Default implementation squares element-wise.
        Override for correlated noise.
        """
        return self.diffusion(x) ** 2

    def sample_interior(self, n: int, device: torch.device = None) -> Tensor:
        """Sample n points uniformly from interior of state space."""
        ss = self.state_space
        if device is None:
            device = ss.lower.device
        u = torch.rand(n, ss.dim, device=device)
        return ss.lower.to(device) + u * (ss.upper - ss.lower).to(device)

    def sample_boundary(self, n: int, which: str, dim: int = 0, device: torch.device = None) -> Tensor:
        """
        Sample n points on boundary.

        Args:
            n: Number of points
            which: "lower" or "upper"
            dim: Which dimension's boundary
            device: Target device
        """
        samples = self.sample_interior(n, device)
        ss = self.state_space

        if which == "lower":
            samples[:, dim] = ss.lower[dim]
        elif which == "upper":
            samples[:, dim] = ss.upper[dim]
        else:
            raise ValueError(f"which must be 'lower' or 'upper', got {which}")

        return samples
