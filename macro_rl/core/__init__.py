"""
Core foundational abstractions for the macro_rl library.

This module provides fundamental data structures and utilities used
throughout the model-based RL framework.
"""

from macro_rl.core.state_space import StateSpace
from macro_rl.core.params import ParameterManager

__all__ = [
    "StateSpace",
    "ParameterManager",
]
