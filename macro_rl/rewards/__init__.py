"""
Reward functions for continuous-time control problems.

This module defines how objectives are computed from states and actions,
including per-step rewards and terminal values.
"""

from macro_rl.rewards.base import RewardFunction
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.rewards.terminal import TerminalValue

__all__ = [
    "RewardFunction",
    "GHMRewardFunction",
    "TerminalValue",
]
