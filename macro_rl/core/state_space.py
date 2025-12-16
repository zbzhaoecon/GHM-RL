"""
State space representation for continuous-time models.

This module defines the StateSpace dataclass that encapsulates
the dimensionality, bounds, and properties of the state space.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor


@dataclass
class StateSpace:
    """
    Representation of continuous state space.

    Attributes:
        dim: Dimensionality of the state space
        lower: Lower bounds for each dimension
        upper: Upper bounds for each dimension
        names: Optional names for each dimension (for visualization)

    Example:
        >>> # 1D state space for cash level c ∈ [0, 10]
        >>> state_space = StateSpace(
        ...     dim=1,
        ...     lower=torch.tensor([0.0]),
        ...     upper=torch.tensor([10.0]),
        ...     names=("cash",)
        ... )

        >>> # 2D state space for (c, τ) where c ∈ [0, 10], τ ∈ [0, T]
        >>> state_space = StateSpace(
        ...     dim=2,
        ...     lower=torch.tensor([0.0, 0.0]),
        ...     upper=torch.tensor([10.0, 5.0]),
        ...     names=("cash", "time_to_horizon")
        ... )
    """

    dim: int
    lower: Tensor
    upper: Tensor
    names: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        """Validate state space specification."""
        # Convert to tensors if needed
        if not isinstance(self.lower, Tensor):
            self.lower = torch.tensor(self.lower, dtype=torch.float32)
        if not isinstance(self.upper, Tensor):
            self.upper = torch.tensor(self.upper, dtype=torch.float32)

        # Ensure tensors are 1D
        if self.lower.dim() != 1:
            self.lower = self.lower.reshape(-1)
        if self.upper.dim() != 1:
            self.upper = self.upper.reshape(-1)

        # Check dimensions match
        if self.lower.shape[0] != self.dim:
            raise ValueError(f"lower bounds dimension {self.lower.shape[0]} != dim {self.dim}")
        if self.upper.shape[0] != self.dim:
            raise ValueError(f"upper bounds dimension {self.upper.shape[0]} != dim {self.dim}")

        # Check lower < upper
        if not torch.all(self.lower < self.upper):
            raise ValueError("lower bounds must be strictly less than upper bounds")

        # Check names length matches dim if provided
        if self.names is not None and len(self.names) != self.dim:
            raise ValueError(f"names length {len(self.names)} != dim {self.dim}")

    def contains(self, state: Tensor) -> Tensor:
        """
        Check if states are within bounds.

        Args:
            state: Tensor of shape (..., dim)

        Returns:
            Boolean tensor of shape (...,) indicating validity
        """
        # Check lower bounds: state >= lower for all dimensions
        above_lower = torch.all(state >= self.lower, dim=-1)
        # Check upper bounds: state <= upper for all dimensions
        below_upper = torch.all(state <= self.upper, dim=-1)
        # Both conditions must be satisfied
        return above_lower & below_upper

    def sample_uniform(self, n_samples: int) -> Tensor:
        """
        Sample uniformly from state space.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tensor of shape (n_samples, dim)
        """
        # Sample from [0, 1]^dim
        uniform_samples = torch.rand(n_samples, self.dim)
        # Scale to [lower, upper]
        samples = self.lower + uniform_samples * (self.upper - self.lower)
        return samples

    def normalize(self, state: Tensor) -> Tensor:
        """
        Normalize state to [0, 1]^dim.

        Args:
            state: Tensor of shape (..., dim)

        Returns:
            Normalized state in [0, 1]^dim
        """
        # Map from [lower, upper] to [0, 1]
        return (state - self.lower) / (self.upper - self.lower)

    def denormalize(self, state_normalized: Tensor) -> Tensor:
        """
        Denormalize from [0, 1]^dim to original bounds.

        Args:
            state_normalized: Tensor of shape (..., dim) in [0, 1]

        Returns:
            Denormalized state in original bounds
        """
        # Map from [0, 1] to [lower, upper]
        return self.lower + state_normalized * (self.upper - self.lower)

    def clip(self, state: Tensor) -> Tensor:
        """
        Clip states to bounds.

        Args:
            state: Tensor of shape (..., dim)

        Returns:
            Clipped state within bounds
        """
        return torch.clamp(state, min=self.lower, max=self.upper)
