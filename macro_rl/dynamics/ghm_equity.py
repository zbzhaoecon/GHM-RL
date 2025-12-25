"""
Equity Management Model from GHM (Décamps et al.).

This module implements the continuous-time model where a firm manages
cash reserves subject to permanent and transitory productivity shocks.

Includes both:
- GHMEquityDynamics: 1D state (c) for infinite-horizon approximation
- GHMEquityTimeAugmentedDynamics: 2D state (c, τ) for finite-horizon problems

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

    # Equity issuance costs (Table 1 from GHM_v2.pdf)
    p: float = 1.06           # Proportional cost of equity issuance
    phi: float = 0.002        # Fixed cost of equity issuance

    # Liquidation parameters (Table 1)
    omega: float = 0.55       # Liquidation recovery rate

    def __post_init__(self):
        """Compute derived parameters."""
        # Liquidation value: equity holders get nothing in bankruptcy
        # When c ≤ 0, the firm is bankrupt and equity has zero value
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
        Drift with optional action influence for model-based training.

        From equation (3): μ_c(c, a) = α + c(r - λ - μ) - a_L + a_E/p - φ
        For Gym environment: action is ignored (impulse controls)

        Args:
            x: State tensor (batch, state_dim)
            action: Action tensor (batch, 2) with [:, 0] = a_L, [:, 1] = a_E
                   If None, returns uncontrolled drift

        Returns:
            Drift (batch, state_dim)
        """
        # Base drift: α + c(r - λ - μ)
        drift = self._drift_const + x * self._drift_slope

        # Add control effects if actions provided (for model-based training)
        if action is not None and action.shape[-1] == 2:
            a_L = action[:, 0:1]  # Dividend payout rate (dL/A)
            a_E = action[:, 1:2]  # Gross equity issuance rate (dE/A)

            # Per equation (3): + dE/(pA) - dL/A - dΦ/A
            drift = drift - a_L + a_E / self.p.p - self.p.phi

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

    def net_equity_proceeds(self, gross_amount: float) -> float:
        """
        Compute net cash from equity issuance.

        Net proceeds = gross / p - φ  (if gross > 0)

        Args:
            gross_amount: Amount raised from investors

        Returns:
            Net cash added to firm (after costs)
        """
        if gross_amount <= 0:
            return 0.0
        return gross_amount / self.p.p - self.p.phi

    def gross_equity_required(self, target_increase: float) -> float:
        """
        Compute gross equity needed to increase cash by target amount.

        To add Δc to cash, must raise p(Δc + φ) from investors.

        Args:
            target_increase: Desired increase in cash

        Returns:
            Gross equity to raise (cost to existing shareholders)
        """
        return self.p.p * (target_increase + self.p.phi)

    def liquidation_value(self) -> float:
        """Return ω·α/(r-μ)"""
        return self.p.liquidation_value


class GHMEquityTimeAugmentedDynamics(ContinuousTimeDynamics):
    """
    Time-augmented GHM equity model for finite-horizon problems.

    State: (c, τ) where
        - c = cash reserves / earnings
        - τ = time-to-horizon (T - t)

    This version correctly handles finite-horizon problems where the optimal
    policy depends on remaining time. As τ → 0, the agent should behave
    differently than when τ is large.

    Dynamics:
        dc = [α + c(r - λ - μ) - a_L + a_E] dt + σ_c(c) dW
        dτ = -1 dt  (time decreases deterministically)
    """

    def __init__(self, params: GHMEquityParams = None, T: float = 10.0):
        """
        Initialize time-augmented dynamics.

        Args:
            params: GHM parameters
            T: Time horizon (maximum value of τ)
        """
        self.p = params or GHMEquityParams()
        self.T = float(T)

        # Precompute constants (same as 1D version)
        self._drift_const = self.p.alpha
        self._drift_slope = self.p.r - self.p.lambda_ - self.p.mu
        self._discount = self.p.r - self.p.mu

        # For diffusion: σ_c² = σ_X²(1-ρ²) + (ρσ_X - cσ_A)²
        self._vol_const = self.p.sigma_X**2 * (1 - self.p.rho**2)
        self._vol_linear = self.p.rho * self.p.sigma_X
        self._vol_quad = self.p.sigma_A

        # State space: (c, τ)
        self._state_space = StateSpace(
            dim=2,
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([self.p.c_max, self.T]),
            names=("c", "tau")
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
            "T": self.T,
        }

    def drift(self, x: Tensor, action: Tensor = None) -> Tensor:
        """
        Drift for time-augmented state.

        Args:
            x: State tensor (batch, 2) with x[:, 0] = c, x[:, 1] = τ
            action: Action tensor (batch, 2) with [:, 0] = a_L, [:, 1] = a_E

        Returns:
            Drift (batch, 2): [μ_c(c, a), -1]
        """
        batch_size = x.shape[0]
        c = x[:, 0:1]  # Cash component

        # Cash drift: α + c(r - λ - μ) - a_L + a_E/p - φ
        # From equation (3): dC_t = (α+C_t(r−λ−μ))dt + ... + dE_t/(pA_t) − dΦ_t/A_t − dL_t/A_t
        drift_c = self._drift_const + c * self._drift_slope

        # Add control effects if actions provided
        if action is not None and action.shape[-1] == 2:
            a_L = action[:, 0:1]  # Dividend payout rate (dL/A)
            a_E = action[:, 1:2]  # Gross equity issuance rate (dE/A)

            # Correct implementation per equation (3):
            # + dE/(pA): net cash from gross equity issuance (divided by p)
            # - dL/A: dividends paid out
            # - dΦ/A: fixed cost of financing
            drift_c = drift_c - a_L + a_E / self.p.p - self.p.phi

        # Time drift: -1 (time decreases)
        drift_tau = -torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

        return torch.cat([drift_c, drift_tau], dim=1)

    def diffusion(self, x: Tensor) -> Tensor:
        """
        Diffusion for time-augmented state.

        Args:
            x: State tensor (batch, 2) with x[:, 0] = c, x[:, 1] = τ

        Returns:
            Diffusion (batch, 2): [σ_c(c), 0]
        """
        batch_size = x.shape[0]
        c = x[:, 0:1]  # Cash component

        # Cash diffusion: σ_c(c) = sqrt(σ_X²(1-ρ²) + (ρσ_X - cσ_A)²)
        linear_term = self._vol_linear - c * self._vol_quad
        variance = self._vol_const + linear_term ** 2
        diffusion_c = torch.sqrt(variance)

        # Time diffusion: 0 (deterministic)
        diffusion_tau = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)

        return torch.cat([diffusion_c, diffusion_tau], dim=1)

    def diffusion_squared(self, x: Tensor) -> Tensor:
        """
        Squared diffusion for HJB equation.

        Args:
            x: State tensor (batch, 2)

        Returns:
            Squared diffusion (batch, 2): [σ_c(c)², 0]
        """
        batch_size = x.shape[0]
        c = x[:, 0:1]

        linear_term = self._vol_linear - c * self._vol_quad
        diff_sq_c = self._vol_const + linear_term ** 2

        diff_sq_tau = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)

        return torch.cat([diff_sq_c, diff_sq_tau], dim=1)

    def discount_rate(self) -> float:
        """r - μ"""
        return self._discount

    def net_equity_proceeds(self, gross_amount: float) -> float:
        """Same as 1D version."""
        if gross_amount <= 0:
            return 0.0
        return gross_amount / self.p.p - self.p.phi

    def gross_equity_required(self, target_increase: float) -> float:
        """Same as 1D version."""
        return self.p.p * (target_increase + self.p.phi)

    def liquidation_value(self) -> float:
        """Return ω·α/(r-μ)"""
        return self.p.liquidation_value

    def sample_initial_states(self, n: int, device: torch.device = None) -> Tensor:
        """
        Sample initial states for finite-horizon problem.

        Samples c uniformly from [0, c_max] and sets τ = T (start of horizon).

        Args:
            n: Number of states to sample
            device: Device to place tensors on

        Returns:
            Initial states (n, 2) with [:, 0] = c, [:, 1] = T
        """
        if device is None:
            device = torch.device('cpu')

        # Sample cash uniformly
        c_samples = torch.rand(n, 1, device=device) * self.p.c_max

        # Initialize time-to-horizon at T
        tau_samples = torch.full((n, 1), self.T, device=device)

        return torch.cat([c_samples, tau_samples], dim=1)
