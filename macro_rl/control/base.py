"""
Base class for control specifications.

Control specifications define:
- Dimensionality of control variables
- Bounds on each control
- Whether controls are continuous or singular (impulse)
- Normalization/denormalization utilities
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from abc import ABC, abstractmethod
import torch
from torch import Tensor


@dataclass
class ControlSpec(ABC):
    """
    Abstract base class for control specifications.

    Attributes:
        dim: Number of control variables
        names: Names for each control (for visualization)
        lower: Lower bounds for each control
        upper: Upper bounds for each control
        is_singular: Whether each control is singular/impulse (vs continuous)

    Example:
        >>> # Single continuous control (dividend rate)
        >>> control_spec = ControlSpec(
        ...     dim=1,
        ...     names=("dividend",),
        ...     lower=torch.tensor([0.0]),
        ...     upper=torch.tensor([10.0]),
        ...     is_singular=(False,)
        ... )

        >>> # Two controls (dividend + equity issuance)
        >>> control_spec = GHMControlSpec(
        ...     dim=2,
        ...     names=("dividend", "equity_issuance"),
        ...     lower=torch.tensor([0.0, 0.0]),
        ...     upper=torch.tensor([10.0, 0.5]),
        ...     is_singular=(False, True)
        ... )
    """

    dim: int
    names: Tuple[str, ...]
    lower: Tensor
    upper: Tensor
    is_singular: Tuple[bool, ...]

    def __post_init__(self):
        """Validate control specification."""
        # TODO: Add validation
        # - Check lengths match dim
        # - Check lower < upper
        pass

    def clip(self, action: Tensor) -> Tensor:
        """
        Clip actions to valid bounds.

        Args:
            action: Tensor of shape (..., dim)

        Returns:
            Clipped action within [lower, upper]

        TODO: Implement clipping
        """
        raise NotImplementedError

    def normalize(self, action: Tensor) -> Tensor:
        """
        Normalize action to [0, 1]^dim.

        Args:
            action: Tensor of shape (..., dim) in original bounds

        Returns:
            Normalized action in [0, 1]^dim

        Formula:
            action_norm = (action - lower) / (upper - lower)

        TODO: Implement normalization
        """
        raise NotImplementedError

    def denormalize(self, action_norm: Tensor) -> Tensor:
        """
        Denormalize from [0, 1]^dim to original bounds.

        Args:
            action_norm: Tensor of shape (..., dim) in [0, 1]

        Returns:
            Denormalized action in original bounds

        Formula:
            action = lower + action_norm * (upper - lower)

        TODO: Implement denormalization
        """
        raise NotImplementedError

    @abstractmethod
    def apply_mask(
        self,
        action: Tensor,
        state: Tensor,
        dt: float,
        **kwargs
    ) -> Tensor:
        """
        Apply feasibility masking to actions.

        This is model-specific and must be implemented by subclasses.

        Args:
            action: Raw actions (..., dim)
            state: Current states (..., state_dim)
            dt: Time step (for flow constraints)
            **kwargs: Additional model-specific arguments

        Returns:
            Masked actions satisfying feasibility constraints

        Example constraints:
            - Dividend can't exceed available cash
            - Issuance below threshold → forced to zero
            - Non-negativity constraints

        TODO: Implement in subclasses
        """
        pass

    def sample_uniform(self, n_samples: int) -> Tensor:
        """
        Sample uniformly from action space.

        Args:
            n_samples: Number of samples

        Returns:
            Tensor of shape (n_samples, dim)

        TODO: Implement uniform sampling
        """
        raise NotImplementedError
