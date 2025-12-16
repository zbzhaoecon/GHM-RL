"""
Terminal value functions for various boundary conditions.

Terminal values represent the payoff when:
    - Time horizon T is reached
    - State reaches boundary (e.g., c = 0)
    - Early termination occurs
"""

from typing import Callable, Optional
import torch
from torch import Tensor


class TerminalValue:
    """
    Container for terminal value specifications.

    Supports:
        - Constant terminal value
        - State-dependent terminal value
        - Time-dependent terminal value
        - Learned terminal value (from neural network)

    Example:
        >>> # Constant liquidation value
        >>> terminal = TerminalValue.constant(value=10.0)
        >>>
        >>> # State-dependent: V_T(c) = c (liquidate at cash value)
        >>> terminal = TerminalValue.from_function(lambda state: state[:, 0])
        >>>
        >>> # Learned terminal value
        >>> terminal_net = nn.Sequential(...)
        >>> terminal = TerminalValue.from_network(terminal_net)
    """

    def __init__(
        self,
        value_fn: Callable[[Tensor], Tensor],
        is_learnable: bool = False,
    ):
        """
        Initialize terminal value.

        Args:
            value_fn: Function mapping states to terminal values
            is_learnable: Whether terminal value has learnable parameters

        TODO: Implement initialization
        """
        self.value_fn = value_fn
        self.is_learnable = is_learnable

    def __call__(self, state: Tensor) -> Tensor:
        """
        Compute terminal value for given states.

        Args:
            state: Terminal states (batch, state_dim)

        Returns:
            Terminal values (batch,)

        TODO: Implement call
        """
        return self.value_fn(state)

    @classmethod
    def constant(cls, value: float) -> "TerminalValue":
        """
        Create constant terminal value.

        Args:
            value: Constant terminal value

        Returns:
            TerminalValue instance

        TODO: Implement constant terminal value
        """
        raise NotImplementedError

    @classmethod
    def from_function(
        cls,
        fn: Callable[[Tensor], Tensor],
    ) -> "TerminalValue":
        """
        Create terminal value from function.

        Args:
            fn: Function state -> value

        Returns:
            TerminalValue instance

        TODO: Implement function-based terminal value
        """
        raise NotImplementedError

    @classmethod
    def from_network(
        cls,
        network: torch.nn.Module,
    ) -> "TerminalValue":
        """
        Create learnable terminal value from neural network.

        Args:
            network: Neural network mapping state -> value

        Returns:
            TerminalValue instance

        TODO: Implement network-based terminal value
        - Set is_learnable=True
        - Wrap network call in value_fn
        """
        raise NotImplementedError

    @classmethod
    def liquidation(
        cls,
        recovery_rate: float,
        expected_flow: float,
        discount_rate: float,
    ) -> "TerminalValue":
        """
        Create GHM liquidation value.

        Formula:
            V_liquidation = ω·α/(r-μ)

        Args:
            recovery_rate: ω
            expected_flow: α
            discount_rate: r - μ

        Returns:
            TerminalValue instance

        TODO: Implement GHM liquidation value
        """
        raise NotImplementedError


class BoundaryCondition:
    """
    Specification of boundary conditions for HJB equation.

    Types:
        - Dirichlet: V(c_boundary) = g(c_boundary)
        - Neumann: V_c(c_boundary) = g'(c_boundary)
        - Robin: αV + βV_c = g

    Example:
        >>> # At c = 0: V(0) = liquidation_value (Dirichlet)
        >>> bc_lower = BoundaryCondition.dirichlet(
        ...     location=0.0,
        ...     value=10.0,
        ... )
        >>>
        >>> # At c = c*: V_c(c*) = 1 (smooth pasting, Neumann)
        >>> bc_upper = BoundaryCondition.neumann(
        ...     location=5.0,
        ...     derivative=1.0,
        ... )
    """

    def __init__(
        self,
        bc_type: str,
        location: float,
        value: Optional[float] = None,
        derivative: Optional[float] = None,
    ):
        """
        Initialize boundary condition.

        Args:
            bc_type: "dirichlet", "neumann", or "robin"
            location: Boundary location in state space
            value: For Dirichlet: V(boundary)
            derivative: For Neumann: V'(boundary)

        TODO: Implement initialization with validation
        """
        raise NotImplementedError

    @classmethod
    def dirichlet(
        cls,
        location: float,
        value: float,
    ) -> "BoundaryCondition":
        """
        Create Dirichlet boundary condition.

        Dirichlet: V(boundary) = value

        Args:
            location: Boundary location
            value: Function value at boundary

        Returns:
            BoundaryCondition instance

        TODO: Implement Dirichlet BC
        """
        raise NotImplementedError

    @classmethod
    def neumann(
        cls,
        location: float,
        derivative: float,
    ) -> "BoundaryCondition":
        """
        Create Neumann boundary condition.

        Neumann: V'(boundary) = derivative

        Args:
            location: Boundary location
            derivative: Function derivative at boundary

        Returns:
            BoundaryCondition instance

        TODO: Implement Neumann BC
        """
        raise NotImplementedError

    @classmethod
    def smooth_pasting(
        cls,
        location: float,
    ) -> "BoundaryCondition":
        """
        Create smooth pasting condition (special Neumann).

        Smooth pasting: V'(c*) = 1
        (Marginal value of cash equals marginal benefit of payout)

        Args:
            location: Barrier location c*

        Returns:
            BoundaryCondition instance

        TODO: Implement smooth pasting (Neumann with derivative=1)
        """
        raise NotImplementedError

    def compute_residual(
        self,
        value: Tensor,
        derivative: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute boundary condition residual.

        For training: minimize ||residual||²

        Args:
            value: V(boundary) from model
            derivative: V'(boundary) from model (if needed)

        Returns:
            Residual (scalar or batch)

        TODO: Implement residual computation based on bc_type
        """
        raise NotImplementedError
