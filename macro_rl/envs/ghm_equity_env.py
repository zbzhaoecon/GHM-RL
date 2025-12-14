"""
Gymnasium environment for 1D GHM equity model.

This module implements a continuous-action RL environment for the Géczy-Hackbarth-Mauer
equity management problem, where agents learn optimal dividend policies.

Phase 3 implementation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Optional, Tuple, Dict

from macro_rl.dynamics import GHMEquityDynamics, GHMEquityParams
from .base import ContinuousTimeEnv


class GHMEquityEnv(ContinuousTimeEnv):
    """
    Gymnasium environment for 1D GHM equity model.

    State: c ∈ [0, c_max] (cash/earnings ratio)
    Action: a ∈ [0, a_max] (dividend payout rate)

    Dynamics: dc = (μ_c(c) - a) dt + σ_c(c) dW
    Reward: a * dt (dividends paid)
    Termination: c ≤ 0 (liquidation)

    The agent learns to balance:
    - Paying dividends (immediate reward)
    - Retaining cash (avoid liquidation)
    - Managing stochastic cash flows

    Optimal policy approximates barrier control: pay excess above c*.

    Example:
        >>> env = GHMEquityEnv()
        >>> obs, info = env.reset(seed=42)
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> print(f"Cash: {obs[0]:.3f}, Reward: {reward:.4f}")

    Args:
        params: GHM model parameters (default: Table 1 values)
        dt: Time discretization step
        max_steps: Maximum episode length
        a_max: Maximum dividend rate (action upper bound)
        liquidation_penalty: Penalty when c hits 0
        seed: Random seed
    """

    def __init__(
        self,
        params: Optional[GHMEquityParams] = None,
        dt: float = 0.01,
        max_steps: int = 1000,
        a_max: float = 10.0,
        liquidation_penalty: float = 5.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize GHM equity environment.

        Args:
            params: GHM model parameters (default: Table 1 values)
            dt: Time discretization step (default: 0.01)
            max_steps: Maximum episode length (default: 1000, T=10)
            a_max: Maximum dividend rate (default: 10.0)
            liquidation_penalty: Penalty when c hits 0 (default: 5.0)
            seed: Random seed for reproducibility
        """
        # Create dynamics
        dynamics = GHMEquityDynamics(params)
        super().__init__(dynamics, dt, max_steps, seed)

        self.a_max = a_max
        self.liquidation_penalty = liquidation_penalty

        # Action space: dividend rate in [0, a_max]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([a_max], dtype=np.float32),
            dtype=np.float32
        )

        # Store dynamics for easy access
        self._dynamics = dynamics

    def _sample_initial_state(self) -> np.ndarray:
        """
        Sample initial state from middle of state space.

        Avoids boundaries to give agent room to learn.
        Samples uniformly from [0.2 * c_max, 0.8 * c_max].

        Returns:
            Initial state [c]
        """
        ss = self.dynamics.state_space
        # Start from middle region, avoid boundaries
        low = 0.2 * ss.upper.numpy()
        high = 0.8 * ss.upper.numpy()
        return self.np_random.uniform(low, high).astype(np.float32)

    def _apply_action_and_evolve(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply dividend action and evolve state.

        Modified dynamics: dc = (μ_c(c) - a) dt + σ_c(c) dW

        Args:
            action: Dividend payout rate [a]

        Returns:
            new_state: Updated state [c']
            reward: Immediate reward (dividends paid)
        """
        # Current state as tensor (batch dimension for dynamics interface)
        c = torch.tensor(self._state, dtype=torch.float32).unsqueeze(0)

        # Compute drift μ_c(c) (without action)
        drift = self._dynamics.drift(c).numpy().flatten()

        # Modify drift by subtracting dividend payout
        # dc/dt = μ_c(c) - a
        drift_modified = drift - action

        # Compute diffusion σ_c(c)
        diffusion = self._dynamics.diffusion(c).numpy().flatten()

        # Sample Brownian increment dW ~ N(0, dt)
        dW = self.np_random.standard_normal(size=self._state.shape).astype(np.float32)
        dW = dW * np.sqrt(self.dt)

        # Euler-Maruyama step: c' = c + (μ_c(c) - a)*dt + σ_c(c)*dW
        new_state = self._state + drift_modified * self.dt + diffusion * dW

        # Clip to valid range [0, c_max]
        # Note: We allow soft clipping here; liquidation is checked via termination
        ss = self.dynamics.state_space
        new_state = np.clip(new_state, ss.lower.numpy(), ss.upper.numpy())

        # Reward = dividends paid this step
        reward = float(action[0] * self.dt)

        return new_state.astype(np.float32), reward

    def _get_terminated(self) -> bool:
        """
        Check for liquidation.

        Episode terminates if cash hits zero (c ≤ 0).

        Returns:
            True if liquidated, False otherwise
        """
        return self._state[0] <= 0.0

    def _get_terminal_reward(self) -> float:
        """
        Apply liquidation penalty.

        Returns:
            Negative penalty if liquidated, 0 otherwise
        """
        if self._state[0] <= 0.0:
            return -self.liquidation_penalty
        return 0.0

    def get_expected_discount_factor(self) -> float:
        """
        Compute appropriate RL discount factor γ.

        For continuous-time discounting at rate ρ = r - μ,
        the discrete equivalent is γ = exp(-ρ * dt).

        Returns:
            Recommended γ for this environment
        """
        rho = self._dynamics.discount_rate()
        gamma = np.exp(-rho * self.dt)
        return float(gamma)

    def get_dynamics_info(self, state: np.ndarray) -> Dict[str, float]:
        """
        Get dynamics information at a given state (for debugging).

        Args:
            state: State [c] to query

        Returns:
            Dictionary with drift, diffusion, and diffusion_sq
        """
        c = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        drift = self._dynamics.drift(c).item()
        diffusion = self._dynamics.diffusion(c).item()
        diffusion_sq = self._dynamics.diffusion_squared(c).item()

        return {
            "drift": drift,
            "diffusion": diffusion,
            "diffusion_sq": diffusion_sq,
        }
