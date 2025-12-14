"""
Gymnasium environments for continuous-time economic models.

This module wraps continuous-time dynamics as RL environments suitable for
training with standard libraries (Stable-Baselines3, CleanRL, etc.).

Phase 3 implementation.
"""

from macro_rl.envs.base import ContinuousTimeEnv
from macro_rl.envs.ghm_equity_env import GHMEquityEnv

# Register with Gymnasium for gym.make() support
from gymnasium.envs.registration import register

register(
    id="GHMEquity-v0",
    entry_point="macro_rl.envs.ghm_equity_env:GHMEquityEnv",
    max_episode_steps=1000,
)

__all__ = [
    "ContinuousTimeEnv",
    "GHMEquityEnv",
]
