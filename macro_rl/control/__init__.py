"""
Control specifications and action masking for continuous-time control problems.

This module defines how control variables (actions) are structured, bounded,
and masked for feasibility.
"""

from macro_rl.control.base import ControlSpec
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.control.masking import ActionMasker

__all__ = [
    "ControlSpec",
    "GHMControlSpec",
    "ActionMasker",
]
