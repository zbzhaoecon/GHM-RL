"""
1D Equity Management Model from GHM (Décamps et al.).

This module implements the continuous-time model where a firm manages
cash reserves subject to permanent and transitory productivity shocks.

Reference: GHM_v2.pdf, Section 1
Phase 2 implementation - Complete.
"""

from dataclasses import dataclass
import torch
from torch import Tensor

from .base import ContinuousTimeDynamics, StateSpace


@dataclass
class GHMEquityParams:
    """Parameters for 1D GHM equity model."""
    # Cash flow
    alpha: float = 0.18       # Mean cash flow rate

    # Growth and rates
    mu: float = 0.01          # Growth rate
    r: float = 0.03           # Interest rate
    lambda_: float = 0.02     # Carry cost (lambda is reserved in Python)

    # Volatility
    sigma_A: float = 0.25     # Permanent shock
    sigma_X: float = 0.12     # Transitory shock
    rho: float = -0.2         # Correlation

    # State bounds
    c_max: float = 2.0

    # Control and reward parameters
    omega: float = 0.8        # Liquidation recovery rate
    p: float = 0.5            # Additional parameter (placeholder)
    phi: float = 0.5          # Additional parameter (placeholder)

    def __post_init__(self):
        """Compute derived parameters."""
        # Liquidation value: ω·α/(r-μ)
        if self.r > self.mu:
            self.liquidation_value = self.omega * self.alpha / (self.r - self.mu)
        else:
            self.liquidation_value = 0.0


class GHMEquityDynamics(ContinuousTimeDynamics):
    """
    1D Equity management model from GHM.

    State: c = cash reserves / earnings

    The firm manages cash holdings subject to permanent and transitory
    shocks. At c=0, the firm must either raise equity or liquidate.
    """

    def __init__(self, params: GHMEquityParams = None):
        self.p = params or GHMEquityParams()

        # Precompute constants
        self._drift_const = self.p.alpha
        self._drift_slope = self.p.r - self.p.lambda_ - self.p.mu
        self._discount = self.p.r - self.p.mu

        # For diffusion: σ_c² = σ_X²(1-ρ²) + (ρσ_X - cσ_A)²
        self._vol_const = self.p.sigma_X**2 * (1 - self.p.rho**2)
        self._vol_linear = self.p.rho * self.p.sigma_X
        self._vol_quad = self.p.sigma_A

        # State space
        self._state_space = StateSpace(
            dim=1,
            lower=torch.tensor([0.0]),
            upper=torch.tensor([self.p.c_max]),
            names=("c",)
        )

    @property
    def state_space(self) -> StateSpace:
        return self._state_space

    @property
    def params(self) -> dict:
        return {
            "alpha": self.p.alpha,
            "mu": self.p.mu,
            "r": self.p.r,
            "lambda": self.p.lambda_,
            "sigma_A": self.p.sigma_A,
            "sigma_X": self.p.sigma_X,
            "rho": self.p.rho,
            "c_max": self.p.c_max,
        }

    def drift(self, x: Tensor, action: Tensor = None) -> Tensor:
        """
        μ_c(c, a) = α + c(r - λ - μ) - a_L + a_E

        Args:
            x: State tensor (batch, state_dim)
            action: Action tensor (batch, action_dim) with [:, 0] = a_L, [:, 1] = a_E
                   If None, assumes zero control (uncontrolled dynamics)

        Returns:
            Drift (batch, state_dim)
        """
        # Base drift: α + c(r - λ - μ)
        drift = self._drift_const + x * self._drift_slope

        # Add control effects if actions provided
        if action is not None:
            assert action.shape[-1] == 2, "GHM actions must be 2D: (a_L, a_E)"
            a_L = action[:, 0:1]  # Dividend payout (outflow)
            a_E = action[:, 1:2]  # Equity issuance (inflow)
            drift = drift - a_L + a_E

        return drift

    def diffusion(self, x: Tensor) -> Tensor:
        """σ_c(c) = sqrt(σ_X²(1-ρ²) + (ρσ_X - cσ_A)²)"""
        linear_term = self._vol_linear - x * self._vol_quad
        variance = self._vol_const + linear_term ** 2
        return torch.sqrt(variance)

    def diffusion_squared(self, x: Tensor) -> Tensor:
        """σ_c(c)² — avoid sqrt for HJB."""
        linear_term = self._vol_linear - x * self._vol_quad
        return self._vol_const + linear_term ** 2

    def discount_rate(self) -> float:
        """r - μ"""
        return self._discount
