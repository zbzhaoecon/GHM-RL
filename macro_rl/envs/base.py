"""
Base Gymnasium environment for continuous-time economic models.

This module provides the abstract base class for wrapping ContinuousTimeDynamics
as Gymnasium environments suitable for RL training.

Phase 3 implementation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from macro_rl.dynamics.base import ContinuousTimeDynamics


class ContinuousTimeEnv(gym.Env):
    """
    Base Gymnasium environment for continuous-time models.

    Wraps a ContinuousTimeDynamics object and discretizes for RL.

    Subclasses must implement:
        - _apply_action_and_evolve(): Apply action and evolve state
        - _get_terminated(): Check termination condition
        - _get_terminal_reward(): Compute terminal reward (optional, default 0)
        - _sample_initial_state(): Sample initial state (optional)

    The action_space must be defined in the subclass __init__.

    Example:
        >>> class MyEnv(ContinuousTimeEnv):
        ...     def __init__(self, dynamics, **kwargs):
        ...         super().__init__(dynamics, **kwargs)
        ...         self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        ...
        ...     def _apply_action_and_evolve(self, action):
        ...         # Custom logic here
        ...         return new_state, reward
        ...
        ...     def _get_terminated(self):
        ...         return self._state[0] <= 0
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dynamics: ContinuousTimeDynamics,
        dt: float = 0.01,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dynamics: The continuous-time model specification
            dt: Time step for discretization
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.dynamics = dynamics
        self.dt = dt
        self.max_steps = max_steps

        # State space from dynamics
        ss = dynamics.state_space
        self.observation_space = spaces.Box(
            low=ss.lower.numpy(),
            high=ss.upper.numpy(),
            dtype=np.float32
        )

        # Action space (to be defined by subclass)
        self.action_space = None  # Override in subclass

        # Internal state
        self._state = None
        self._step_count = 0

        # Initialize RNG if seed provided
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self._np_random = None

    @property
    def np_random(self):
        """Get numpy random generator, initializing if needed."""
        if self._np_random is None:
            self._np_random, _ = gym.utils.seeding.np_random(None)
        return self._np_random

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for this episode
            options: Optional dict that may contain 'initial_state'

        Returns:
            observation: Initial observation
            info: Additional information dict
        """
        super().reset(seed=seed)

        # Reset RNG if seed provided
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        # Default: sample from interior of state space
        if options and "initial_state" in options:
            self._state = np.array(options["initial_state"], dtype=np.float32)
        else:
            self._state = self._sample_initial_state()

        self._step_count = 0

        return self._state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.

        Args:
            action: Action to take

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode terminated (e.g., boundary hit)
            truncated: Whether episode was truncated (e.g., max steps)
            info: Additional information dict
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        # Ensure action is numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        # Apply action and evolve state (implemented by subclass)
        self._state, reward = self._apply_action_and_evolve(action)

        self._step_count += 1

        # Check termination
        terminated = self._get_terminated()
        truncated = self._step_count >= self.max_steps

        # Apply terminal reward if needed
        if terminated:
            reward += self._get_terminal_reward()

        info = {"step": self._step_count, "state": self._state.copy()}

        return self._state.copy(), float(reward), terminated, truncated, info

    def _sample_initial_state(self) -> np.ndarray:
        """
        Sample initial state.

        Default implementation samples uniformly from state space.
        Override for custom initialization distributions.

        Returns:
            Initial state as numpy array
        """
        ss = self.dynamics.state_space
        low = ss.lower.numpy()
        high = ss.upper.numpy()
        return self.np_random.uniform(low, high).astype(np.float32)

    def _apply_action_and_evolve(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply action, evolve state via SDE, and compute reward.

        This method should:
        1. Modify drift/diffusion based on action (if applicable)
        2. Take Euler-Maruyama step to evolve state
        3. Compute immediate reward
        4. Return (new_state, reward)

        Args:
            action: Action taken by agent

        Returns:
            new_state: Updated state after action and dynamics
            reward: Immediate reward for this transition
        """
        raise NotImplementedError("Subclass must implement _apply_action_and_evolve")

    def _get_terminated(self) -> bool:
        """
        Check if episode should terminate.

        Common termination conditions:
        - State hits boundary (e.g., c <= 0 for liquidation)
        - State violates constraints

        Returns:
            True if episode should end, False otherwise
        """
        raise NotImplementedError("Subclass must implement _get_terminated")

    def _get_terminal_reward(self) -> float:
        """
        Compute reward at termination.

        Examples:
        - Negative penalty for liquidation
        - Terminal payoff in finite-horizon problems

        Returns:
            Terminal reward (0 by default)
        """
        return 0.0

    def render(self):
        """Render environment (optional, for human visualization)."""
        if self._state is None:
            return

        state_str = ", ".join(
            f"{name}={val:.4f}"
            for name, val in zip(self.dynamics.state_space.names, self._state)
        )
        print(f"Step {self._step_count}: {state_str}")

    def close(self):
        """Clean up resources."""
        pass
