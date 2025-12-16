"""
Simulation engines for continuous-time stochastic differential equations.

This module provides tools for simulating trajectories from known dynamics,
both for Monte Carlo estimation and differentiable pathwise gradient methods.
"""

from macro_rl.simulation.sde import SDEIntegrator
from macro_rl.simulation.trajectory import TrajectorySimulator, TrajectoryBatch
from macro_rl.simulation.differentiable import DifferentiableSimulator

__all__ = [
    "SDEIntegrator",
    "TrajectorySimulator",
    "TrajectoryBatch",
    "DifferentiableSimulator",
]
