# Phase 2: Dynamics Module Development Guide

## Purpose

The `dynamics/` module specifies continuous-time economic models as mathematical objects. It defines:

- State space and bounds
- Drift and diffusion coefficients (the SDE)
- Discount rate
- Boundary conditions

This module is **solver-agnostic**—it knows nothing about neural networks, training, or optimization. Any solver (DGM, value iteration, RL) queries the same dynamics interface.

---

## Module Structure

```
dynamics/
├── __init__.py
├── base.py          # Abstract interface
└── ghm_equity.py    # 1D equity management model
```

---

## 1. base.py — Abstract Interface

### Design Principles

1. **Minimal interface**: Only what solvers actually need
2. **Tensor-native**: All methods accept/return batched tensors
3. **Self-describing**: Models carry their own parameter metadata

### Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch import Tensor


@dataclass
class StateSpace:
    """Description of the state space."""
    dim: int                    # Number of state variables
    lower: Tensor              # Lower bounds (dim,)
    upper: Tensor              # Upper bounds (dim,)
    names: Tuple[str, ...]     # Variable names, e.g., ("c",) or ("eta", "c")


class ContinuousTimeDynamics(ABC):
    """
    Abstract base class for continuous-time economic models.
    
    Represents an SDE of the form:
        dX = μ(X) dt + σ(X) dW
    
    with boundary conditions and discount rate for HJB equations.
    """
    
    @property
    @abstractmethod
    def state_space(self) -> StateSpace:
        """Return state space specification."""
        pass
    
    @property
    @abstractmethod
    def params(self) -> Dict[str, float]:
        """Return model parameters as dictionary (for logging/reproducibility)."""
        pass
    
    @abstractmethod
    def drift(self, x: Tensor) -> Tensor:
        """
        Drift coefficient μ(x).
        
        Args:
            x: State tensor (batch, state_dim)
        
        Returns:
            Drift (batch, state_dim)
        """
        pass
    
    @abstractmethod
    def diffusion(self, x: Tensor) -> Tensor:
        """
        Diffusion coefficient σ(x).
        
        Args:
            x: State tensor (batch, state_dim)
        
        Returns:
            Diffusion (batch, state_dim) for diagonal noise
        """
        pass
    
    @abstractmethod
    def discount_rate(self) -> float:
        """
        Effective discount rate for HJB equation.
        
        For most models this is (r - μ) where r is interest rate
        and μ is growth rate.
        """
        pass
    
    def diffusion_squared(self, x: Tensor) -> Tensor:
        """
        Squared diffusion σ(x)² for HJB equation.
        
        Default implementation squares element-wise.
        Override for correlated noise.
        """
        return self.diffusion(x) ** 2
    
    def sample_interior(self, n: int) -> Tensor:
        """Sample n points uniformly from interior of state space."""
        ss = self.state_space
        u = torch.rand(n, ss.dim)
        return ss.lower + u * (ss.upper - ss.lower)
    
    def sample_boundary(self, n: int, which: str, dim: int = 0) -> Tensor:
        """
        Sample n points on boundary.
        
        Args:
            n: Number of points
            which: "lower" or "upper"
            dim: Which dimension's boundary
        """
        ss = self.state_space
        samples = self.sample_interior(n)
        if which == "lower":
            samples[:, dim] = ss.lower[dim]
        else:
            samples[:, dim] = ss.upper[dim]
        return samples
```

### Notes

- `drift` and `diffusion` are the raw SDE coefficients
- `diffusion_squared` is a convenience for HJB (avoids redundant computation)
- Sampling methods are provided for convenience but can be overridden
- No boundary condition specification yet—add when needed for specific models

---

## 2. ghm_equity.py — 1D Equity Model

### Mathematical Specification

From GHM_v2.pdf, the 1D equity management model has:

**State**: $c \in [0, c_{max}]$ (cash reserves / earnings)

**SDE** (Equation 3, simplified):
$$dc = \mu_c(c) \, dt + \sigma_c(c) \, dW$$

**Drift**:
$$\mu_c(c) = \alpha + c(r - \lambda - \mu)$$

**Diffusion** (combining permanent and transitory shocks):
$$\sigma_c(c) = \sqrt{\sigma_X^2(1-\rho^2) + (\rho\sigma_X - c\sigma_A)^2}$$

**Effective discount rate**:
$$\rho_{eff} = r - \mu$$

**Parameters** (Table 1):

| Symbol | Name | Value |
|--------|------|-------|
| $\alpha$ | Mean cash flow rate | 0.18 |
| $\mu$ | Growth rate | 0.01 |
| $\sigma_A$ | Permanent shock volatility | 0.25 |
| $\sigma_X$ | Transitory shock volatility | 0.12 |
| $\rho$ | Correlation | -0.2 |
| $r$ | Interest rate | 0.03 |
| $\lambda$ | Carry cost | 0.02 |
| $c_{max}$ | Upper bound | 2.0 |

### Implementation

```python
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
    
    def drift(self, x: Tensor) -> Tensor:
        """μ_c(c) = α + c(r - λ - μ)"""
        return self._drift_const + x * self._drift_slope
    
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
```

---

## 3. Testing Strategy

### Simple Test Models

Before testing GHM, validate the interface with simpler models that have known analytical properties.

#### Test Model 1: Geometric Brownian Motion (GBM)

```python
class GBMDynamics(ContinuousTimeDynamics):
    """
    dX = μX dt + σX dW
    
    Known properties:
    - E[X_t] = X_0 * exp(μt)
    - Var[X_t] = X_0² * exp(2μt) * (exp(σ²t) - 1)
    """
    def __init__(self, mu: float = 0.05, sigma: float = 0.2, x_max: float = 10.0):
        self.mu = mu
        self.sigma = sigma
        self._state_space = StateSpace(
            dim=1,
            lower=torch.tensor([0.01]),  # Avoid zero
            upper=torch.tensor([x_max]),
            names=("x",)
        )
    
    def drift(self, x: Tensor) -> Tensor:
        return self.mu * x
    
    def diffusion(self, x: Tensor) -> Tensor:
        return self.sigma * x
    
    def discount_rate(self) -> float:
        return 0.03  # Arbitrary
```

#### Test Model 2: Ornstein-Uhlenbeck (OU)

```python
class OUDynamics(ContinuousTimeDynamics):
    """
    dX = θ(μ - X) dt + σ dW
    
    Known properties:
    - Mean reverts to μ
    - Stationary variance = σ² / (2θ)
    """
    def __init__(self, theta: float = 1.0, mu: float = 0.0, sigma: float = 0.5):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self._state_space = StateSpace(
            dim=1,
            lower=torch.tensor([-5.0]),
            upper=torch.tensor([5.0]),
            names=("x",)
        )
    
    def drift(self, x: Tensor) -> Tensor:
        return self.theta * (self.mu - x)
    
    def diffusion(self, x: Tensor) -> Tensor:
        return self.sigma * torch.ones_like(x)
    
    def discount_rate(self) -> float:
        return 0.03
```

### Test Cases

| Test | Model | Validation |
|------|-------|------------|
| `test_drift_shape` | All | `drift(x).shape == x.shape` |
| `test_diffusion_shape` | All | `diffusion(x).shape == x.shape` |
| `test_diffusion_positive` | All | `diffusion(x) >= 0` for all x |
| `test_state_space_valid` | All | `lower < upper`, correct dim |
| `test_params_complete` | All | All expected keys in `params` dict |
| `test_gbm_drift_linear` | GBM | `drift(2x) == 2 * drift(x)` |
| `test_ou_drift_mean_reversion` | OU | `drift(μ) == 0` |
| `test_ghm_drift_values` | GHM | Spot-check at known points |
| `test_ghm_diffusion_values` | GHM | Spot-check at known points |
| `test_sample_bounds` | All | All samples within state space |

### Spot-Check Values for GHM

With default parameters:

| c | μ_c(c) | σ_c(c)² |
|---|--------|---------|
| 0.0 | 0.18 | 0.01382 |
| 0.5 | 0.18 | 0.01967 |
| 1.0 | 0.18 | 0.03802 |

Derivation for c=0:
- μ_c(0) = α + 0 = 0.18 ✓
- σ_c(0)² = σ_X²(1-ρ²) + (ρσ_X)² = 0.12²×0.96 + (-0.024)² = 0.01382 ✓

---

## 4. Integration with Numerics

The dynamics module integrates with `numerics/integration.py`:

```python
from macro_rl.numerics import simulate_path
from macro_rl.dynamics import GHMEquityDynamics

# Create model
model = GHMEquityDynamics()

# Simulate trajectories
x0 = torch.ones(1000, 1) * 0.5  # Start at c = 0.5

x_T = simulate_path(
    x0,
    drift_fn=lambda x, t: model.drift(x),
    diffusion_fn=lambda x, t: model.diffusion(x),
    T=1.0,
    dt=0.01
)
```

---

## 5. Validation Checkpoint

Before moving to Phase 3 (losses), verify:

- [ ] `StateSpace` dataclass works correctly
- [ ] GBM and OU test models pass all shape tests
- [ ] GHM drift matches expected values at c=0, 0.5, 1.0
- [ ] GHM diffusion_squared matches expected values
- [ ] `sample_interior` returns points within bounds
- [ ] `sample_boundary` returns points exactly on boundary
- [ ] Parameters dict contains all expected keys
- [ ] Integration with `simulate_path` produces reasonable trajectories

---

## 6. Future Extensions

For Phase 2+, the following will be added:

1. **ghm_debt.py**: 1D debt management model (Section 2 of GHM_v2.pdf)
2. **ghm_joint.py**: 2D joint debt-equity model (Section 3)
3. **Boundary condition specification**: For enforcing F'(c*)=1, F''(c*)=0, etc.

These can share the same base interface—the only additions are more state dimensions and more complex drift/diffusion.
