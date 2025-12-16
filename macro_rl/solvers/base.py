"""
Base class for solvers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch


@dataclass
class SolverResult:
    """
    Container for solver results.

    Attributes:
        policy: Learned policy
        value_fn: Learned value function (if applicable)
        diagnostics: Training metrics and logs
    """
    policy: Any
    value_fn: Optional[Any]
    diagnostics: Dict[str, Any]


class Solver(ABC):
    """
    Abstract base class for continuous-time control solvers.

    A solver takes:
        - Dynamics (drift, diffusion)
        - Control specification
        - Reward function

    And produces:
        - Optimal (or near-optimal) policy
        - (Optional) Value function
        - Training diagnostics

    Example:
        >>> solver = PathwiseGradient(policy, simulator, ...)
        >>> result = solver.solve(
        ...     dynamics=ghm_dynamics,
        ...     control_spec=ghm_control,
        ...     reward_fn=ghm_reward,
        ...     n_iterations=10000,
        ... )
        >>> optimal_policy = result.policy
        >>> diagnostics = result.diagnostics
    """

    @abstractmethod
    def solve(
        self,
        dynamics,  # ContinuousTimeDynamics
        control_spec,  # ControlSpec
        reward_fn,  # RewardFunction
        **kwargs,
    ) -> SolverResult:
        """
        Solve for optimal policy.

        Args:
            dynamics: Continuous-time dynamics
            control_spec: Control specification
            reward_fn: Reward function
            **kwargs: Solver-specific arguments

        Returns:
            SolverResult containing policy, value_fn, diagnostics

        TODO: Implement in subclasses
        """
        pass

    def _log_progress(self, iteration: int, metrics: Dict[str, Any]):
        """
        Log training progress.

        Args:
            iteration: Current iteration
            metrics: Dictionary of metrics to log

        TODO: Implement logging (print or tensorboard)
        """
        pass
