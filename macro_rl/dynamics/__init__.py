"""
Dynamics module for continuous-time economic models.

This module provides:
- Abstract base class for continuous-time dynamics
- StateSpace specification
- GHM equity management model
- Test models (GBM, OU) for validation
"""

from .base import ContinuousTimeDynamics, StateSpace
from .ghm_equity import GHMEquityDynamics, GHMEquityTimeAugmentedDynamics, GHMEquityParams
from .test_models import GBMDynamics, OUDynamics

__all__ = [
    "ContinuousTimeDynamics",
    "StateSpace",
    "GHMEquityDynamics",
    "GHMEquityTimeAugmentedDynamics",
    "GHMEquityParams",
    "GBMDynamics",
    "OUDynamics",
]
