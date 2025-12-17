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
class ControlSpec:
    """
    Specification of the control/action space.

    Attributes
    ----------
    dim : int
        Number of control dimensions.
    lower : Tensor
        Lower bounds for each control, shape (dim,).
    upper : Tensor
        Upper bounds for each control, shape (dim,).
    names : Tuple[str, ...]
        Human-readable names for each control dimension, length dim.
    is_singular : Tuple[bool, ...]
        Whether each control is singular/impulse (vs continuous).
    """

    dim: int
    names: Tuple[str, ...]
    lower: Tensor
    upper: Tensor
    is_singular: Tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate control specification."""
        # Basic consistency checks
        assert self.lower.shape == (self.dim,), "lower must have shape (dim,)"
        assert self.upper.shape == (self.dim,), "upper must have shape (dim,)"
        assert len(self.names) == self.dim, "names must have length dim"
        assert len(self.is_singular) == self.dim, "is_singular must have length dim"

    def clip(self, action: Tensor) -> Tensor:
        """
        Clip action to valid range [lower, upper].

        Parameters
        ----------
        action : Tensor
            Raw action tensor of shape (..., dim). The leading dimensions can be
            batch or time; only the last dimension is interpreted as controls.

        Returns
        -------
        Tensor
            Clipped action with same shape as `action`.
        """
        # Broadcast lower/upper to action.shape
        lower = self.lower.to(action.device)
        upper = self.upper.to(action.device)
        while lower.dim() < action.dim():
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)
        return torch.clamp(action, lower, upper)

    def normalize(self, action: Tensor) -> Tensor:
        """
        Normalize actions from [lower, upper] to [0, 1].

        This is useful when the policy outputs are naturally in [0, 1] or when
        we want a normalized representation for logging.

        Parameters
        ----------
        action : Tensor
            Action tensor in physical units, shape (..., dim).

        Returns
        -------
        Tensor
            Normalized actions in [0, 1], same shape as `action`.
        """
        lower = self.lower.to(action.device)
        upper = self.upper.to(action.device)
        while lower.dim() < action.dim():
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)
        return (action - lower) / (upper - lower)

    def denormalize(self, action_norm: Tensor) -> Tensor:
        """
        Denormalize actions from [0, 1] back to [lower, upper].

        Parameters
        ----------
        action_norm : Tensor
            Normalized actions in [0, 1], shape (..., dim).

        Returns
        -------
        Tensor
            Actions in original scale, same shape as `action_norm`.
        """
        lower = self.lower.to(action_norm.device)
        upper = self.upper.to(action_norm.device)
        while lower.dim() < action_norm.dim():
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)
        return action_norm * (upper - lower) + lower

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

    def sample_uniform(self, n_samples: int, device: Optional[str] = None) -> Tensor:
        """
        Sample uniformly from action space.

        Args:
            n_samples: Number of samples
            device: Device to place samples on (default: same as bounds)

        Returns:
            Tensor of shape (n_samples, dim)
        """
        if device is None:
            device = self.lower.device

        # Sample from [0, 1]^dim
        samples_norm = torch.rand(n_samples, self.dim, device=device)

        # Denormalize to [lower, upper]
        return self.denormalize(samples_norm)
