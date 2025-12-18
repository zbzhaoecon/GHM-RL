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
        batch_size = self.states.shape[0]
        n_steps_states = self.states.shape[1] - 1

        assert self.actions.shape[0] == batch_size, "actions batch size mismatch"
        assert self.rewards.shape[0] == batch_size, "rewards batch size mismatch"
        assert self.masks.shape[0] == batch_size, "masks batch size mismatch"
        assert self.returns.shape[0] == batch_size, "returns batch size mismatch"
        assert self.terminal_rewards.shape[0] == batch_size, "terminal_rewards batch size mismatch"

        assert self.actions.shape[1] == n_steps_states, "actions n_steps mismatch"
        assert self.rewards.shape[1] == n_steps_states, "rewards n_steps mismatch"
        assert self.masks.shape[1] == n_steps_states, "masks n_steps mismatch"

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
        """
        return TrajectoryBatch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            masks=self.masks.to(device),
            returns=self.returns.to(device),
            terminal_rewards=self.terminal_rewards.to(device),
        )


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
        """
        import math
        from macro_rl.simulation.sde import SDEIntegrator

        self.dynamics = dynamics
        self.control_spec = control_spec
        self.reward_fn = reward_fn
        self.dt = float(dt)
        self.T = float(T)

        # Compute number of steps
        steps = self.T / self.dt
        self.max_steps = int(round(steps))
        if not math.isclose(self.max_steps * self.dt, self.T, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"T={self.T} is not an integer multiple of dt={self.dt}.")

        # Initialize integrator
        self.integrator = integrator if integrator is not None else SDEIntegrator(scheme="euler")

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
        """
        import math

        batch_size, state_dim = initial_states.shape
        action_dim = self.control_spec.dim
        device = initial_states.device

        # Pre-allocate storage
        states = torch.zeros(batch_size, self.max_steps + 1, state_dim, device=device)
        actions = torch.zeros(batch_size, self.max_steps, action_dim, device=device)
        rewards = torch.zeros(batch_size, self.max_steps, device=device)
        masks = torch.ones(batch_size, self.max_steps, device=device)

        # Initialize
        states[:, 0, :] = initial_states

        # Generate or use provided noise
        if noise is None:
            noise = torch.randn(batch_size, self.max_steps, state_dim, device=device)
        else:
            assert noise.shape == (batch_size, self.max_steps, state_dim), (
                f"noise must have shape ({batch_size}, {self.max_steps}, {state_dim}), got {noise.shape}"
            )
            noise = noise.to(device)

        # Track which trajectories are still active
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        # No gradients for Monte Carlo rollout
        with torch.no_grad():
            for t in range(self.max_steps):
                # Sample action from policy
                actions[:, t, :] = policy.act(states[:, t, :])

                # Apply control masking (feasibility constraints)
                actions[:, t, :] = self.control_spec.apply_mask(
                    actions[:, t, :],
                    states[:, t, :],
                    self.dt
                )

                # Clip to bounds
                actions[:, t, :] = self.control_spec.clip(actions[:, t, :])

                # Compute drift and diffusion
                drift = self.dynamics.drift(states[:, t, :], actions[:, t, :])
                diffusion = self.dynamics.diffusion(states[:, t, :])

                # Step dynamics using integrator
                states[:, t + 1, :] = self.integrator.step(
                    states[:, t, :],
                    drift,
                    diffusion,
                    self.dt,
                    noise[:, t, :]
                )

                # Compute reward (step reward needs next_state for some reward functions)
                rewards[:, t] = self.reward_fn.step_reward(
                    states[:, t, :],
                    actions[:, t, :],
                    states[:, t + 1, :],
                    self.dt
                )

                # Check termination
                terminated = self._check_termination(states[:, t + 1, :])

                # Update active mask
                masks[:, t] = active.to(dtype=masks.dtype)

                # Zero out rewards for terminated trajectories
                rewards[:, t] = rewards[:, t] * masks[:, t]

                # Update active status
                active = active & (~terminated)

                # Early exit if all terminated
                if not active.any():
                    # Set remaining masks to zero
                    if t + 1 < self.max_steps:
                        masks[:, t + 1:] = 0
                    break

        # Compute terminal rewards
        terminal_mask = (~active).to(dtype=rewards.dtype)
        terminal_rewards = self.reward_fn.terminal_reward(states[:, -1, :], terminal_mask)

        # Compute discounted returns
        discount_rate = self.dynamics.discount_rate()
        returns = self._compute_returns(rewards, terminal_rewards, masks, discount_rate)

        return TrajectoryBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            masks=masks,
            returns=returns,
            terminal_rewards=terminal_rewards,
        )

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
        """
        batch_size, n_steps = rewards.shape
        device = rewards.device

        returns = torch.zeros(batch_size, device=device)

        # Compute discounted sum of per-step rewards
        for t in range(n_steps):
            discount = torch.exp(torch.tensor(-discount_rate * t * self.dt, device=device))
            returns = returns + discount * rewards[:, t] * masks[:, t]

        # Add terminal reward with appropriate discount
        # Terminal time is n_steps * dt (or earlier if terminated)
        terminal_discount = torch.exp(torch.tensor(-discount_rate * n_steps * self.dt, device=device))
        returns = returns + terminal_discount * terminal_rewards

        return returns

    def _check_termination(self, states: Tensor) -> Tensor:
        """
        Check if states trigger termination.

        Args:
            states: Current states (batch, state_dim)

        Returns:
            Boolean tensor (batch,) indicating termination
        """
        # For GHM model: terminate if cash c ≤ 0
        # Cash is the first state component
        return states[:, 0] <= 0.0
