"""
Base class for reward functions.

Reward functions define the objective to be maximized in the control problem.
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    In continuous-time control, rewards consist of:
        1. Flow rewards: r(s, a) accumulated over time
        2. Terminal rewards: R(s_T) at final time/termination

    Example:
        >>> class DividendReward(RewardFunction):
        ...     def step_reward(self, state, action, next_state, dt):
        ...         return action[:, 0] * dt  # Dividend payout
        ...
        ...     def terminal_reward(self, state, terminated):
        ...         return torch.zeros_like(state[:, 0])
    """

    @abstractmethod
    def step_reward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        dt: float,
    ) -> Tensor:
        """
        Compute per-step reward.

        Args:
            state: Current states (batch, state_dim)
            action: Actions taken (batch, action_dim)
            next_state: Next states (batch, state_dim)
            dt: Time step size

        Returns:
            Rewards (batch,)

        Note: For flow rewards r(s,a), the accumulated reward over dt is:
            reward = r(s, a) * dt
        """
        pass

    @abstractmethod
    def terminal_reward(
        self,
        state: Tensor,
        terminated: Tensor,
    ) -> Tensor:
        """
        Compute terminal reward.

        Args:
            state: Terminal states (batch, state_dim)
            terminated: Boolean mask (batch,) indicating termination

        Returns:
            Terminal rewards (batch,)

        Note: Often model liquidation value or scrap value.
        """
        pass

    def trajectory_return(
        self,
        rewards: Tensor,
        terminal_rewards: Tensor,
        masks: Tensor,
        discount_rate: float,
        dt: float,
    ) -> Tensor:
        """
        Compute discounted return for complete trajectories.

        Args:
            rewards: Per-step rewards (batch, n_steps)
            terminal_rewards: Terminal rewards (batch,)
            masks: Active masks (batch, n_steps) - 1 if not terminated
            discount_rate: Continuous-time discount rate ρ
            dt: Time step size

        Returns:
            Discounted returns (batch,)

        Formula:
            R = Σ_t e^(-ρ·t·dt) · r_t · mask_t + e^(-ρ·T) · r_T
        """
        batch_size, T = rewards.shape
        device = rewards.device

        returns = torch.zeros(batch_size, device=device)

        # Compute discounted sum of per-step rewards
        for t in range(T):
            discount = torch.exp(torch.tensor(-discount_rate * t * dt, device=device))
            returns = returns + discount * rewards[:, t] * masks[:, t]

        # Add terminal reward with appropriate discount
        # Terminal time is T * dt
        terminal_discount = torch.exp(torch.tensor(-discount_rate * T * dt, device=device))
        returns = returns + terminal_discount * terminal_rewards

        return returns

    def cumulative_reward(
        self,
        rewards: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """
        Compute undiscounted cumulative reward.

        Args:
            rewards: Per-step rewards (batch, n_steps)
            masks: Active masks (batch, n_steps)

        Returns:
            Cumulative rewards (batch,)
        """
        # Apply masks and sum over time dimension
        masked_rewards = rewards * masks
        return masked_rewards.sum(dim=1)
