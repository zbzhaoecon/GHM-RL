"""
Test models with known analytical properties.

These models are used to validate the dynamics interface and
integration with numerical solvers. They have simple, well-understood
behavior that can be tested against known solutions.
"""

from typing import Dict
import torch
from torch import Tensor

from .base import ContinuousTimeDynamics, StateSpace


class GBMDynamics(ContinuousTimeDynamics):
    """
    Geometric Brownian Motion (GBM) model.

    SDE:
        dX = μX dt + σX dW

    Known analytical properties:
        - E[X_t | X_0] = X_0 * exp(μt)
        - Var[X_t | X_0] = X_0² * exp(2μt) * (exp(σ²t) - 1)
        - Solution: X_t = X_0 * exp((μ - σ²/2)t + σW_t)

    This model is useful for testing:
        - Linear scaling properties: drift(2x) = 2*drift(x)
        - Integration schemes
        - Positivity preservation (when properly discretized)

    Example:
        >>> model = GBMDynamics(mu=0.05, sigma=0.2)
        >>> x = torch.tensor([[1.0], [2.0], [3.0]])
        >>> drift = model.drift(x)  # [0.05, 0.10, 0.15]
        >>> diff = model.diffusion(x)  # [0.20, 0.40, 0.60]
    """

    def __init__(
        self,
        mu: float = 0.05,
        sigma: float = 0.2,
        x_min: float = 0.01,
        x_max: float = 10.0
    ):
        """
        Initialize GBM dynamics.

        Args:
            mu: Drift coefficient
            sigma: Volatility coefficient
            x_min: Lower bound (avoid zero to prevent degeneracy)
            x_max: Upper bound
        """
        self.mu = mu
        self.sigma = sigma

        self._state_space = StateSpace(
            dim=1,
            lower=torch.tensor([x_min]),
            upper=torch.tensor([x_max]),
            names=("x",)
        )

    @property
    def state_space(self) -> StateSpace:
        """Return state space: x ∈ [x_min, x_max]."""
        return self._state_space

    @property
    def params(self) -> Dict[str, float]:
        """Return model parameters."""
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "x_min": self._state_space.lower[0].item(),
            "x_max": self._state_space.upper[0].item(),
        }

    def drift(self, x: Tensor) -> Tensor:
        """
        Compute drift μX.

        Args:
            x: State of shape (batch, 1)

        Returns:
            Drift of shape (batch, 1)
        """
        return self.mu * x

    def diffusion(self, x: Tensor) -> Tensor:
        """
        Compute diffusion σX.

        Args:
            x: State of shape (batch, 1)

        Returns:
            Diffusion of shape (batch, 1)
        """
        return self.sigma * x

    def discount_rate(self) -> float:
        """
        Return discount rate (arbitrary for test model).

        In a real application, this would be tied to economic
        parameters like risk-free rate.
        """
        return 0.03


class OUDynamics(ContinuousTimeDynamics):
    """
    Ornstein-Uhlenbeck (OU) process.

    SDE:
        dX = θ(μ - X) dt + σ dW

    Known analytical properties:
        - Mean reverts to level μ with speed θ
        - E[X_t | X_0] = μ + (X_0 - μ) * exp(-θt)
        - Stationary distribution: N(μ, σ²/(2θ))
        - Autocorrelation: exp(-θτ) for lag τ

    This model is useful for testing:
        - Mean reversion: drift(μ) = 0
        - Constant diffusion
        - Stationary properties
        - Long-run convergence

    Example:
        >>> model = OUDynamics(theta=1.0, mu=0.0, sigma=0.5)
        >>> x = torch.tensor([[-2.0], [0.0], [2.0]])
        >>> drift = model.drift(x)  # [2.0, 0.0, -2.0]
        >>> diff = model.diffusion(x)  # [0.5, 0.5, 0.5]
    """

    def __init__(
        self,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 0.5,
        x_min: float = -5.0,
        x_max: float = 5.0
    ):
        """
        Initialize OU dynamics.

        Args:
            theta: Mean reversion speed (must be positive)
            mu: Long-run mean
            sigma: Volatility
            x_min: Lower bound
            x_max: Upper bound
        """
        assert theta > 0, "theta must be positive for mean reversion"

        self.theta = theta
        self.mu = mu
        self.sigma = sigma

        self._state_space = StateSpace(
            dim=1,
            lower=torch.tensor([x_min]),
            upper=torch.tensor([x_max]),
            names=("x",)
        )

    @property
    def state_space(self) -> StateSpace:
        """Return state space: x ∈ [x_min, x_max]."""
        return self._state_space

    @property
    def params(self) -> Dict[str, float]:
        """Return model parameters."""
        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "x_min": self._state_space.lower[0].item(),
            "x_max": self._state_space.upper[0].item(),
        }

    def drift(self, x: Tensor) -> Tensor:
        """
        Compute drift θ(μ - X).

        Args:
            x: State of shape (batch, 1)

        Returns:
            Drift of shape (batch, 1)
        """
        return self.theta * (self.mu - x)

    def diffusion(self, x: Tensor) -> Tensor:
        """
        Compute constant diffusion σ.

        Args:
            x: State of shape (batch, 1)

        Returns:
            Diffusion of shape (batch, 1)
        """
        return self.sigma * torch.ones_like(x)

    def diffusion_squared(self, x: Tensor) -> Tensor:
        """
        Compute σ² (constant).

        Override for efficiency since diffusion is constant.

        Args:
            x: State of shape (batch, 1)

        Returns:
            Squared diffusion of shape (batch, 1)
        """
        return (self.sigma ** 2) * torch.ones_like(x)

    def discount_rate(self) -> float:
        """
        Return discount rate (arbitrary for test model).

        For economic applications, this could be related to
        the risk-free rate or preference parameters.
        """
        return 0.03

    def stationary_variance(self) -> float:
        """
        Compute theoretical stationary variance σ²/(2θ).

        This is a convenience method for testing convergence
        to stationary distribution.
        """
        return self.sigma ** 2 / (2 * self.theta)
