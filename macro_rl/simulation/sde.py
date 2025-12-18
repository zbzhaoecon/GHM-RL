"""
Numerical integration schemes for stochastic differential equations.

This module implements various SDE integration schemes for simulating
continuous-time dynamics: dx = μ(x)dt + σ(x)dW
"""

from typing import Optional
import torch
from torch import Tensor


class SDEIntegrator:
    """
    Numerical integration of SDEs with various schemes.

    Supports:
        - Euler-Maruyama (order 0.5 strong convergence)
        - Milstein (order 1.0 strong convergence, if needed)

    Example:
        >>> integrator = SDEIntegrator(scheme="euler")
        >>> x_next = integrator.step(
        ...     x=x_current,
        ...     drift=drift_fn(x_current),
        ...     diffusion=diffusion_fn(x_current),
        ...     dt=0.01,
        ...     noise=torch.randn_like(x_current)
        ... )
    """

    def __init__(self, scheme: str = "euler"):
        """
        Initialize SDE integrator.

        Args:
            scheme: Integration scheme ("euler" or "milstein")

        TODO: Implement scheme selection
        """
        self.scheme = scheme
        if scheme not in ["euler", "milstein"]:
            raise ValueError(f"Unknown scheme: {scheme}")

    def step(
        self,
        x: Tensor,
        drift: Tensor,
        diffusion: Tensor,
        dt: float,
        noise: Tensor,
    ) -> Tensor:
        """
        Single integration step.

        Args:
            x: Current state (batch, state_dim)
            drift: Drift μ(x) at current state (batch, state_dim)
            diffusion: Diffusion σ(x) at current state (batch, state_dim)
            dt: Time step size
            noise: Gaussian noise samples N(0,1) (batch, state_dim)

        Returns:
            Next state x_next (batch, state_dim)

        TODO: Implement integration step based on self.scheme
        """
        if self.scheme == "euler":
            return self._euler_maruyama_step(x, drift, diffusion, dt, noise)
        elif self.scheme == "milstein":
            return self._milstein_step(x, drift, diffusion, dt, noise)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

    def _euler_maruyama_step(
        self,
        x: Tensor,
        drift: Tensor,
        diffusion: Tensor,
        dt: float,
        noise: Tensor,
    ) -> Tensor:
        """
        Euler-Maruyama integration step.

        Formula: x_next = x + μ(x)·dt + σ(x)·√dt·ε

        Args:
            x: Current state
            drift: μ(x)
            diffusion: σ(x)
            dt: Time step
            noise: ε ~ N(0,1)

        Returns:
            Next state
        """
        import math
        assert x.shape == drift.shape == diffusion.shape == noise.shape, (
            "x, drift, diffusion, noise must have the same shape"
        )
        sqrt_dt = math.sqrt(dt)
        return x + drift * dt + diffusion * sqrt_dt * noise

    def _milstein_step(
        self,
        x: Tensor,
        drift: Tensor,
        diffusion: Tensor,
        dt: float,
        noise: Tensor,
    ) -> Tensor:
        """
        Milstein integration step (higher order).

        Formula: x_next = x + μ·dt + σ·√dt·ε + ½σ·∂σ/∂x·(ε²-1)·dt

        Note: Requires diffusion derivative. Only implement if needed
        for high accuracy.

        Args:
            x: Current state
            drift: μ(x)
            diffusion: σ(x)
            dt: Time step
            noise: ε ~ N(0,1)

        Returns:
            Next state

        TODO: Implement Milstein step (optional)
        - Requires automatic differentiation for ∂σ/∂x
        - May not be necessary if dt is small enough
        """
        raise NotImplementedError

    def batch_simulate(
        self,
        x0: Tensor,
        drift_fn: callable,
        diffusion_fn: callable,
        dt: float,
        n_steps: int,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Simulate batch of trajectories.

        Args:
            x0: Initial states (batch, state_dim)
            drift_fn: Function μ(x) returning (batch, state_dim)
            diffusion_fn: Function σ(x) returning (batch, state_dim)
            dt: Time step size
            n_steps: Number of steps to simulate
            noise: Pre-sampled noise (batch, n_steps, state_dim), or None

        Returns:
            Trajectory states (batch, n_steps+1, state_dim)
        """
        batch_size, state_dim = x0.shape
        device = x0.device

        # Pre-allocate trajectory tensor
        trajectory = torch.zeros(batch_size, n_steps + 1, state_dim, device=device, dtype=x0.dtype)
        trajectory[:, 0, :] = x0

        # Generate or use provided noise
        if noise is None:
            noise = torch.randn(batch_size, n_steps, state_dim, device=device, dtype=x0.dtype)
        else:
            assert noise.shape == (batch_size, n_steps, state_dim), (
                f"noise must have shape ({batch_size}, {n_steps}, {state_dim}), got {noise.shape}"
            )
            noise = noise.to(device)

        # Simulate trajectory
        x = x0
        for t in range(n_steps):
            drift = drift_fn(x)
            diffusion = diffusion_fn(x)
            x = self.step(x, drift, diffusion, dt, noise[:, t, :])
            trajectory[:, t + 1, :] = x

        return trajectory
