"""
1D Equity Management Model from GHM (D'ecamps et al.).

This module implements the continuous-time model where a firm manages
cash reserves subject to permanent and transitory productivity shocks.

Reference: GHM_v2.pdf, Section 1
"""

from dataclasses import dataclass
from typing import Dict
import torch
from torch import Tensor

from .base import ContinuousTimeDynamics, StateSpace


@dataclass
class GHMEquityParams:
    """
    Parameters for 1D GHM equity model.
    
    Default values from Table 1 of GHM_v2.pdf.
    
    Attributes:
        alpha: Mean cash flow rate (α)
        mu: Growth rate of productivity (μ)
        r: Interest rate
        lambda_: Carry cost of cash (λ) - underscore because lambda is reserved
        sigma_A: Volatility of permanent shocks (σ_A)
        sigma_X: Volatility of transitory shocks (σ_X)
        rho: Correlation between shocks (ρ)
        c_max: Upper bound on cash/earnings ratio
    """
    # Cash flow parameters
    alpha: float = 0.18       # Mean cash flow rate
    
    # Growth and discount
    mu: float = 0.01          # Growth rate
    r: float = 0.03           # Interest rate
    lambda_: float = 0.02     # Carry cost
    
    # Volatility structure
    sigma_A: float = 0.25     # Permanent shock
    sigma_X: float = 0.12     # Transitory shock
    rho: float = -0.2         # Correlation
    
    # State bounds
    c_max: float = 2.0        # Upper bound on c


class GHMEquityDynamics(ContinuousTimeDynamics):
    """
    1D Equity management model from GHM.
    
    State: c ∈ [0, c_max] representing cash reserves / earnings
    
    SDE (from Equation 3, simplified for 1D):
        dc = μ_c(c) dt + σ_c(c) dW
    
    where:
        μ_c(c) = α + c(r - λ - μ)
        σ_c(c) = √(σ_X²(1-ρ²) + (ρσ_X - cσ_A)²)
    
    The firm manages cash holdings subject to permanent and transitory
    shocks. At c=0, the firm must either raise equity or liquidate.
    
    Example:
        >>> model = GHMEquityDynamics()
        >>> c = torch.tensor([[0.0], [0.5], [1.0]])
        >>> print(model.drift(c))
        >>> print(model.diffusion(c))
    """
    
    def __init__(self, params: GHMEquityParams = None):
        """
        Initialize GHM equity dynamics.
        
        Args:
            params: Model parameters. If None, uses defaults from Table 1.
        """
        self.p = params if params is not None else GHMEquityParams()
        
        # Precompute drift constants: μ_c(c) = α + c(r - λ - μ)
        self._drift_const = self.p.alpha
        self._drift_slope = self.p.r - self.p.lambda_ - self.p.mu
        
        # Effective discount rate for HJB
        self._discount = self.p.r - self.p.mu
        
        # Precompute diffusion constants
        # σ_c(c)² = σ_X²(1-ρ²) + (ρσ_X - cσ_A)²
        # Let a = σ_X²(1-ρ²), b = ρσ_X, d = σ_A
        # Then σ_c(c)² = a + (b - c*d)²
        self._diff_a = self.p.sigma_X**2 * (1 - self.p.rho**2)
        self._diff_b = self.p.rho * self.p.sigma_X
        self._diff_d = self.p.sigma_A
        
        # State space
        self._state_space = StateSpace(
            dim=1,
            lower=torch.tensor([0.0]),
            upper=torch.tensor([self.p.c_max]),
            names=("c",)
        )
    
    @property
    def state_space(self) -> StateSpace:
        """Return state space: c ∈ [0, c_max]."""
        return self._state_space
    
    @property
    def params(self) -> Dict[str, float]:
        """Return all model parameters as dictionary."""
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
    
    def drift(self, x: Tensor) -> Tensor:
        """
        Compute drift μ_c(c) = α + c(r - λ - μ).
        
        Args:
            x: Cash ratio c of shape (batch, 1)
        
        Returns:
            Drift of shape (batch, 1)
        """
        return self._drift_const + x * self._drift_slope
    
    def diffusion(self, x: Tensor) -> Tensor:
        """
        Compute diffusion σ_c(c) = √(σ_X²(1-ρ²) + (ρσ_X - cσ_A)²).
        
        Args:
            x: Cash ratio c of shape (batch, 1)
        
        Returns:
            Diffusion of shape (batch, 1)
        """
        return torch.sqrt(self.diffusion_squared(x))
    
    def diffusion_squared(self, x: Tensor) -> Tensor:
        """
        Compute σ_c(c)² directly (avoids sqrt for HJB).
        
        σ_c(c)² = σ_X²(1-ρ²) + (ρσ_X - cσ_A)²
        
        Args:
            x: Cash ratio c of shape (batch, 1)
        
        Returns:
            Squared diffusion of shape (batch, 1)
        """
        linear_term = self._diff_b - x * self._diff_d
        return self._diff_a + linear_term ** 2
    
    def discount_rate(self) -> float:
        """
        Effective discount rate (r - μ) for HJB equation.
        
        The HJB equation is:
            (r - μ) F(c) = μ_c(c) F'(c) + 0.5 σ_c(c)² F''(c)
        """
        return self._discount
    
    # =========================================================================
    # Convenience methods for analysis
    # =========================================================================
    
    def liquidation_value(self) -> float:
        """
        Compute liquidation value ωα/(r-μ) at c=0.
        
        Note: This requires the omega parameter which is part of the
        boundary condition, not the dynamics. For full implementation,
        this would be moved to a boundary condition class.
        """
        # Default omega from Table 1
        omega = 0.55
        return omega * self.p.alpha / self._discount
