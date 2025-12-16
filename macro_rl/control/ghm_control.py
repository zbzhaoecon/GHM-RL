"""
Control specification for the GHM equity model.

The GHM model has TWO control variables:
    1. a_L: Dividend payout rate (continuous control)
    2. a_E: Equity issuance amount (singular/impulse control)

This corrects the previous single-control formulation.
"""

from typing import Optional
import torch
from torch import Tensor

from macro_rl.control.base import ControlSpec


class GHMControlSpec(ControlSpec):
    """
    Two-control specification for GHM equity model.

    Controls:
        - a_L: Dividend payout rate (flow, ≥ 0)
        - a_E: Equity issuance amount (impulse, ≥ 0)

    Feasibility constraints:
        1. a_L ≤ c / dt (can't pay out more cash than available)
        2. a_E ≥ threshold or a_E = 0 (fixed cost makes small issuances wasteful)

    Parameters:
        a_L_max: Maximum dividend rate
        a_E_max: Maximum equity issuance
        issuance_threshold: Minimum meaningful issuance (as fraction of a_E_max)
        issuance_cost: Fixed cost of issuing equity (λ in model)

    Example:
        >>> control = GHMControlSpec(
        ...     a_L_max=10.0,
        ...     a_E_max=0.5,
        ...     issuance_threshold=0.05,
        ...     issuance_cost=0.1,
        ... )
        >>>
        >>> # Mask actions for feasibility
        >>> state = torch.tensor([[2.0]])  # c = 2.0
        >>> action = torch.tensor([[5.0, 0.2]])  # (a_L, a_E)
        >>> masked = control.apply_mask(action, state, dt=0.01)
        >>> # a_L is clipped to 2.0/0.01 = 200, then to a_L_max
        >>> # a_E = 0.2 is kept (above threshold)
    """

    def __init__(
        self,
        a_L_max: float = 10.0,
        a_E_max: float = 0.5,
        issuance_threshold: float = 0.05,
        issuance_cost: float = 0.0,
    ):
        """
        Initialize GHM control specification.

        Args:
            a_L_max: Maximum dividend rate
            a_E_max: Maximum equity issuance
            issuance_threshold: Minimum issuance (fraction of a_E_max)
            issuance_cost: Fixed cost parameter λ

        TODO: Initialize parent class correctly
        """
        super().__init__(
            dim=2,
            names=("dividend", "equity_issuance"),
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([a_L_max, a_E_max]),
            is_singular=(False, True),
        )
        self.a_L_max = a_L_max
        self.a_E_max = a_E_max
        self.issuance_threshold = issuance_threshold
        self.issuance_cost = issuance_cost

    def apply_mask(
        self,
        action: Tensor,
        state: Tensor,
        dt: float,
        **kwargs
    ) -> Tensor:
        """
        Apply feasibility masking for GHM controls.

        Args:
            action: Raw actions (batch, 2) where [:, 0] = a_L, [:, 1] = a_E
            state: Current states (batch, state_dim) where [:, 0] = c
            dt: Time step size

        Returns:
            Masked actions (batch, 2) satisfying constraints

        Constraints applied:
            1. Dividend: a_L ∈ [0, min(a_L_max, c/dt)]
            2. Issuance: a_E ∈ {0} ∪ [threshold·a_E_max, a_E_max]

        TODO: Implement masking logic
        - Extract c from state
        - Clip a_L to available cash
        - Apply threshold to a_E (zero if below threshold)
        - Handle batch dimensions correctly
        """
        raise NotImplementedError

    def compute_net_payout(
        self,
        action: Tensor,
    ) -> Tensor:
        """
        Compute net payout to shareholders.

        Formula:
            net_payout = a_L - a_E

        Args:
            action: Actions (batch, 2)

        Returns:
            Net payout (batch,)

        TODO: Implement net payout computation
        """
        raise NotImplementedError

    def issuance_indicator(
        self,
        action: Tensor,
    ) -> Tensor:
        """
        Compute indicator for whether issuance occurred.

        Returns 1 if a_E > 0, else 0.

        Args:
            action: Actions (batch, 2)

        Returns:
            Indicator (batch,) of type float

        TODO: Implement issuance indicator
        """
        raise NotImplementedError

    def total_issuance_cost(
        self,
        action: Tensor,
    ) -> Tensor:
        """
        Compute total cost of equity issuance.

        Formula:
            cost = λ * 𝟙(a_E > 0)

        Args:
            action: Actions (batch, 2)

        Returns:
            Total cost (batch,)

        TODO: Implement issuance cost computation
        """
        raise NotImplementedError


class GHMControlSpecWithBarrier(GHMControlSpec):
    """
    GHM control with forced recapitalization at barrier.

    Extension: If c reaches barrier c_b > 0, force equity issuance
    to bring c back to target c_t.

    This implements the "barrier" policy from Bolton et al.

    TODO: Implement barrier logic (future extension)
    """

    def __init__(
        self,
        a_L_max: float = 10.0,
        a_E_max: float = 0.5,
        barrier: float = 0.5,
        target: float = 2.0,
        **kwargs
    ):
        """
        Initialize with barrier policy.

        Args:
            a_L_max: Maximum dividend rate
            a_E_max: Maximum equity issuance
            barrier: Cash level triggering recapitalization
            target: Target cash level after recapitalization
            **kwargs: Other arguments

        TODO: Implement initialization
        """
        super().__init__(a_L_max, a_E_max, **kwargs)
        self.barrier = barrier
        self.target = target

    def apply_mask(
        self,
        action: Tensor,
        state: Tensor,
        dt: float,
        **kwargs
    ) -> Tensor:
        """
        Apply masking with barrier enforcement.

        If c < barrier:
            - Force a_E = target - c
            - Force a_L = 0

        Otherwise, apply standard masking.

        Args:
            action: Raw actions
            state: Current state
            dt: Time step

        Returns:
            Masked actions with barrier enforcement

        TODO: Implement barrier logic
        """
        raise NotImplementedError
