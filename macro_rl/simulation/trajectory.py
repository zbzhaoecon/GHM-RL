"""
Trajectory generation for Monte Carlo policy evaluation and optimization.

This module provides tools for rolling out complete trajectories from
known dynamics, computing rewards, and organizing results for policy
gradient estimation.
"""

from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor


@dataclass
class TrajectoryBatch:
    """
    Container for batched trajectory data.

    Attributes:
        states: State trajectory (batch, n_steps+1, state_dim)
        actions: Actions taken (batch, n_steps, action_dim)
        rewards: Per-step rewards (batch, n_steps)
        masks: Active mask (1 if not terminated) (batch, n_steps)
        returns: Discounted cumulative returns (batch,)
        terminal_rewards: Terminal rewards (batch,)

    Example:
        >>> batch = TrajectoryBatch(
        ...     states=torch.randn(100, 51, 1),  # 100 trajectories, 50 steps, 1D state
        ...     actions=torch.randn(100, 50, 2), # 2D actions
        ...     rewards=torch.randn(100, 50),
        ...     masks=torch.ones(100, 50),
        ...     returns=torch.randn(100),
        ...     terminal_rewards=torch.zeros(100),
        ... )
    """

    states: Tensor
    actions: Tensor
    rewards: Tensor
    masks: Tensor
    returns: Tensor
    terminal_rewards: Tensor

    def __post_init__(self):
        """Validate trajectory batch dimensions."""
        # TODO: Add validation
        # - Check states.shape[0] == actions.shape[0] (batch size)
        # - Check states.shape[1] == actions.shape[1] + 1 (n_steps)
        # - Check all batch dimensions match
        pass

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self.states.shape[0]

    @property
    def n_steps(self) -> int:
        """Return number of steps."""
        return self.actions.shape[1]

    def to(self, device: torch.device) -> "TrajectoryBatch":
        """
        Move trajectory batch to device.

        Args:
            device: Target device

        Returns:
            TrajectoryBatch on target device

        TODO: Implement device transfer
        """
        raise NotImplementedError


class TrajectorySimulator:
    """
    Simulate trajectories from known dynamics and policy.

    This is the core engine for Monte Carlo policy gradient methods.
    It generates complete trajectories by:
        1. Sampling actions from policy
        2. Stepping dynamics forward
        3. Computing rewards
        4. Handling termination

    Example:
        >>> from macro_rl.dynamics.ghm_equity import GHMEquityDynamics
        >>> from macro_rl.policies.neural import GaussianPolicy
        >>>
        >>> dynamics = GHMEquityDynamics(params)
        >>> policy = GaussianPolicy(state_dim=1, action_dim=2)
        >>> simulator = TrajectorySimulator(
        ...     dynamics=dynamics,
        ...     control_spec=control_spec,
        ...     reward_fn=reward_fn,
        ...     dt=0.01,
        ...     T=5.0,
        ... )
        >>>
        >>> # Simulate 1000 trajectories
        >>> initial_states = torch.rand(1000, 1) * 10.0
        >>> trajectories = simulator.rollout(policy, initial_states)
        >>> print(trajectories.returns.mean())  # Average return
    """

    def __init__(
        self,
        dynamics,  # ContinuousTimeDynamics
        control_spec,  # ControlSpec
        reward_fn,  # RewardFunction
        dt: float,
        T: float,
        integrator: Optional[object] = None,
    ):
        """
        Initialize trajectory simulator.

        Args:
            dynamics: Continuous-time dynamics (drift, diffusion)
            control_spec: Control specification (bounds, masking)
            reward_fn: Reward function
            dt: Time step size
            T: Time horizon
            integrator: SDE integrator (defaults to Euler-Maruyama)

        TODO: Implement initialization
        - Store all components
        - Compute max_steps = int(T / dt)
        - Initialize default integrator if not provided
        """
        raise NotImplementedError

    def rollout(
        self,
        policy,  # Policy
        initial_states: Tensor,
        noise: Optional[Tensor] = None,
    ) -> TrajectoryBatch:
        """
        Simulate batch of trajectories.

        Args:
            policy: Policy to sample actions from
            initial_states: Initial states (batch, state_dim)
            noise: Optional pre-sampled noise (batch, n_steps, state_dim)

        Returns:
            TrajectoryBatch containing complete trajectories

        Algorithm:
            1. Initialize trajectory storage
            2. For each time step:
                a. Sample action from policy
                b. Apply action masking (control_spec)
                c. Compute reward
                d. Step dynamics forward
                e. Check termination condition
            3. Compute discounted returns
            4. Return TrajectoryBatch

        TODO: Implement trajectory rollout
        - Handle action masking properly
        - Support early termination (e.g., c ≤ 0)
        - Compute discounted returns correctly
        - Handle batch dimensions consistently
        """
        raise NotImplementedError

    def _compute_returns(
        self,
        rewards: Tensor,
        terminal_rewards: Tensor,
        masks: Tensor,
        discount_rate: float,
    ) -> Tensor:
        """
        Compute discounted returns from rewards.

        Args:
            rewards: Per-step rewards (batch, n_steps)
            terminal_rewards: Terminal rewards (batch,)
            masks: Active masks (batch, n_steps)
            discount_rate: Continuous-time discount rate ρ

        Returns:
            Discounted returns (batch,)

        Formula:
            R = Σ_t exp(-ρ·t·dt) · r_t · mask_t + exp(-ρ·T) · r_T

        TODO: Implement discounted return computation
        - Use continuous-time discounting: exp(-ρ·t·dt)
        - Apply masks to handle early termination
        - Add terminal reward with appropriate discount
        """
        raise NotImplementedError

    def _check_termination(self, states: Tensor) -> Tensor:
        """
        Check if states trigger termination.

        Args:
            states: Current states (batch, state_dim)

        Returns:
            Boolean tensor (batch,) indicating termination

        TODO: Implement termination logic
        - For GHM: terminate if c ≤ 0
        - Make this configurable for different models
        """
        raise NotImplementedError
