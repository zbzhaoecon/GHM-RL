"""Stochastic differential equation (SDE) integration methods.

This module provides numerical methods for discretizing and simulating SDEs,
particularly the Euler-Maruyama scheme for approximating solutions to:
    dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
"""

from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor


def euler_maruyama_step(
    state: Tensor,
    drift: Tensor,
    diffusion: Tensor,
    dt: float,
    dW: Optional[Tensor] = None,
) -> Tensor:
    """
    Perform one Euler-Maruyama step for an SDE.

    Implements: X_{t+dt} = X_t + μ(X_t, t) * dt + σ(X_t, t) * √dt * Z

    where Z ~ N(0, I) is standard normal noise.

    Args:
        state: Current state tensor of shape (batch, state_dim).
        drift: Drift term μ(X_t, t) of shape (batch, state_dim).
        diffusion: Diffusion term σ(X_t, t) of shape (batch, state_dim).
        dt: Time step size.
        dW: Optional Brownian increments of shape (batch, state_dim).
            If None, will be sampled as √dt * N(0, I).

    Returns:
        Next state tensor of shape (batch, state_dim).

    Example:
        >>> # Geometric Brownian motion: dX = μX dt + σX dW
        >>> state = torch.ones(100, 1)
        >>> drift = 0.05 * state
        >>> diffusion = 0.2 * state
        >>> next_state = euler_maruyama_step(state, drift, diffusion, dt=0.01)
    """
    # Sample Brownian increments if not provided
    if dW is None:
        sqrt_dt = torch.sqrt(torch.tensor(dt))
        dW = sqrt_dt * torch.randn_like(state)

    # Euler-Maruyama update: X_{t+dt} = X_t + μ*dt + σ*dW
    next_state = state + drift * dt + diffusion * dW

    return next_state


def simulate_path(
    x0: Tensor,
    drift_fn: Callable[[Tensor, float], Tensor],
    diffusion_fn: Callable[[Tensor, float], Tensor],
    T: float,
    dt: float,
    return_full_path: bool = False,
    seed: Optional[int] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Simulate SDE trajectories from t=0 to t=T using Euler-Maruyama.

    Args:
        x0: Initial state of shape (batch, state_dim).
        drift_fn: Function (state, t) -> drift of shape (batch, state_dim).
        diffusion_fn: Function (state, t) -> diffusion of shape (batch, state_dim).
        T: Terminal time.
        dt: Time step size.
        return_full_path: If True, return the entire trajectory.
                         If False, return only the terminal state.
        seed: Optional random seed for reproducibility.

    Returns:
        If return_full_path=False:
            Terminal state of shape (batch, state_dim).
        If return_full_path=True:
            Tuple of (times, states) where:
                - times: Tensor of shape (n_steps+1,) containing time points.
                - states: Tensor of shape (batch, n_steps+1, state_dim).

    Example:
        >>> # Simulate GBM
        >>> x0 = torch.ones(1000, 1)
        >>> drift_fn = lambda x, t: 0.05 * x
        >>> diffusion_fn = lambda x, t: 0.2 * x
        >>> x_T = simulate_path(x0, drift_fn, diffusion_fn, T=1.0, dt=0.01)
        >>> print(x_T.mean())  # Should be close to exp(0.05 * 1.0) ≈ 1.051
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Setup
    batch_size = x0.shape[0]
    state_dim = x0.shape[1] if x0.dim() > 1 else 1
    n_steps = int(T / dt)
    actual_dt = T / n_steps  # Adjust dt to ensure we hit T exactly

    # Initialize state
    state = x0.clone()

    if return_full_path:
        # Preallocate storage for full trajectory
        times = torch.linspace(0, T, n_steps + 1)
        states = torch.zeros(batch_size, n_steps + 1, state_dim, device=x0.device)
        states[:, 0, :] = x0
    else:
        times = None
        states = None

    # Simulate trajectory
    for step in range(n_steps):
        t = step * actual_dt

        # Compute drift and diffusion at current state
        drift = drift_fn(state, t)
        diffusion = diffusion_fn(state, t)

        # Take Euler-Maruyama step
        state = euler_maruyama_step(state, drift, diffusion, actual_dt)

        # Store if returning full path
        if return_full_path:
            states[:, step + 1, :] = state

    if return_full_path:
        return times, states
    else:
        return state


def milstein_step(
    state: Tensor,
    drift: Tensor,
    diffusion: Tensor,
    diffusion_derivative: Tensor,
    dt: float,
    dW: Optional[Tensor] = None,
) -> Tensor:
    """
    Perform one Milstein step for an SDE (higher-order scheme).

    The Milstein scheme includes a correction term for better accuracy:
        X_{t+dt} = X_t + μ*dt + σ*dW + 0.5*σ*(∂σ/∂x)*(dW² - dt)

    This has strong order 1.0 convergence (vs. 0.5 for Euler-Maruyama).

    Args:
        state: Current state of shape (batch, state_dim).
        drift: Drift μ(X_t, t) of shape (batch, state_dim).
        diffusion: Diffusion σ(X_t, t) of shape (batch, state_dim).
        diffusion_derivative: ∂σ/∂x of shape (batch, state_dim).
        dt: Time step size.
        dW: Optional Brownian increments of shape (batch, state_dim).

    Returns:
        Next state tensor of shape (batch, state_dim).

    Note:
        For scalar diffusion (diffusion doesn't depend on state),
        this reduces to Euler-Maruyama since ∂σ/∂x = 0.
    """
    # Sample Brownian increments if not provided
    if dW is None:
        sqrt_dt = torch.sqrt(torch.tensor(dt))
        dW = sqrt_dt * torch.randn_like(state)

    # Milstein correction term: 0.5 * σ * (∂σ/∂x) * (dW² - dt)
    correction = 0.5 * diffusion * diffusion_derivative * (dW ** 2 - dt)

    # Milstein update
    next_state = state + drift * dt + diffusion * dW + correction

    return next_state


def geometric_brownian_motion(
    x0: Tensor,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    return_full_path: bool = False,
    seed: Optional[int] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Simulate geometric Brownian motion: dX = μX dt + σX dW.

    This is a convenience function for the common GBM process used in finance.

    Args:
        x0: Initial values of shape (batch, 1).
        mu: Drift parameter (expected return).
        sigma: Volatility parameter.
        T: Terminal time.
        dt: Time step size.
        return_full_path: If True, return entire trajectory.
        seed: Optional random seed.

    Returns:
        Terminal values or (times, states) if return_full_path=True.

    Example:
        >>> x0 = torch.ones(10000, 1)
        >>> x_T = geometric_brownian_motion(x0, mu=0.05, sigma=0.2, T=1.0, dt=0.01)
        >>> print(f"Mean: {x_T.mean():.3f}, Expected: {math.exp(0.05):.3f}")
    """
    drift_fn = lambda x, t: mu * x
    diffusion_fn = lambda x, t: sigma * x

    return simulate_path(
        x0, drift_fn, diffusion_fn, T, dt, return_full_path, seed
    )


def ornstein_uhlenbeck(
    x0: Tensor,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    return_full_path: bool = False,
    seed: Optional[int] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Simulate Ornstein-Uhlenbeck process: dX = θ(μ - X) dt + σ dW.

    This is a mean-reverting process commonly used for interest rates.

    Args:
        x0: Initial values of shape (batch, 1).
        theta: Mean reversion speed.
        mu: Long-run mean.
        sigma: Volatility.
        T: Terminal time.
        dt: Time step size.
        return_full_path: If True, return entire trajectory.
        seed: Optional random seed.

    Returns:
        Terminal values or (times, states) if return_full_path=True.
    """
    drift_fn = lambda x, t: theta * (mu - x)
    diffusion_fn = lambda x, t: sigma * torch.ones_like(x)

    return simulate_path(
        x0, drift_fn, diffusion_fn, T, dt, return_full_path, seed
    )
