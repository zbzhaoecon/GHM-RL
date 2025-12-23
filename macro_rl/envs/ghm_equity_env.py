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
    Action: [dividend_amount, equity_gross_amount] (impulse controls)

    Dynamics: dc = μ_c(c) dt + σ_c(c) dW - dL + dE/p - φ·𝟙(dE>0)
    Reward: dividend_amount - equity_gross_amount (net payout to shareholders)
    Termination: c ≤ 0 (liquidation)

    The agent learns to balance:
    - Paying dividends (positive reward)
    - Avoiding equity issuance (negative reward, dilution)
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
        dividend_max: Maximum dividend amount per step
        equity_max: Maximum equity issuance amount per step
        seed: Random seed
    """

    def __init__(
        self,
        params: Optional[GHMEquityParams] = None,
        dt: float = 0.01,
        max_steps: int = 1000,
        dividend_max: float = 2.0,
        equity_max: float = 2.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize GHM equity environment.

        Args:
            params: GHM model parameters (default: Table 1 values)
            dt: Time discretization step (default: 0.01)
            max_steps: Maximum episode length (default: 1000, T=10)
            dividend_max: Maximum dividend amount (default: 2.0)
            equity_max: Maximum equity issuance amount (default: 2.0)
            seed: Random seed for reproducibility
        """
        # Create dynamics
        dynamics = GHMEquityDynamics(params)
        super().__init__(dynamics, dt, max_steps, seed)

        self.dividend_max = dividend_max
        self.equity_max = equity_max

        # Action space: [dividend_amount, equity_gross_amount]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([dividend_max, equity_max], dtype=np.float32),
            dtype=np.float32
        )

        # Store dynamics and parameters for easy access
        self._dynamics = dynamics
        self._params = dynamics.p
        self._liquidated = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment and liquidation flag.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            Initial observation and info dictionary
        """
        self._liquidated = False
        return super().reset(seed=seed, options=options)

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
        Apply impulse controls then evolve via SDE.

        Order of operations:
            1. Apply dividend payout (c decreases instantly)
            2. Apply equity issuance (c increases instantly, with costs)
            3. Evolve via uncontrolled SDE dynamics

        Args:
            action: [dividend_amount, equity_gross_amount]

        Returns:
            new_state: Updated state [c']
            reward: Immediate reward (net payout to shareholders)
        """
        c = float(self._state[0])
        dividend = float(action[0])
        equity_gross = float(action[1])

        reward = 0.0

        # Step 1: Apply dividend (cannot pay more than available)
        dividend_actual = min(dividend, max(c, 0.0))
        c = c - dividend_actual
        reward += dividend_actual  # Positive reward to shareholders

        # Step 2: Apply equity issuance with costs
        if equity_gross > 0:
            net_proceeds = equity_gross / self._params.p - self._params.phi
            net_proceeds = max(net_proceeds, 0.0)
            c = c + net_proceeds
            reward -= equity_gross  # Dilution cost (negative)

        # Step 3: Evolve via uncontrolled SDE
        c_tensor = torch.tensor([[c]], dtype=torch.float32)
        drift = self._dynamics.drift(c_tensor).item()
        diffusion = self._dynamics.diffusion(c_tensor).item()

        dW = self.np_random.standard_normal() * np.sqrt(self.dt)
        c_new = c + drift * self.dt + diffusion * dW

        # Clip to bounds
        ss = self.dynamics.state_space
        c_new = np.clip(c_new, ss.lower.numpy()[0], ss.upper.numpy()[0])

        return np.array([c_new], dtype=np.float32), reward

    def _get_terminated(self) -> bool:
        """
        Check for liquidation.

        Episode terminates if cash hits zero (c ≤ 0).

        Returns:
            True if liquidated, False otherwise
        """
        if self._state[0] <= 0.0:
            self._liquidated = True
            return True
        return False

    def _get_terminal_reward(self) -> float:
        """
        Liquidation value: ω·α/(r-μ)

        This is POSITIVE - shareholders receive recovery value.

        Returns:
            Positive liquidation value if liquidated, 0 otherwise
        """
        if self._state[0] <= 0.0:
            return self._dynamics.liquidation_value()
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
