"""
Value Function Iteration (VFI) solver for GHM equity model.

This implements a traditional numerical solution to the HJB equation
using finite difference methods on a grid. Serves as a benchmark
for deep learning approaches.

The HJB equation for the time-augmented problem is:
    (r-μ)V(c,τ) = max_{a_L,a_E} [a_L - a_E + μ_c(c,a)V_c + (-1)V_τ + ½σ²(c)V_cc]

We solve this backward in time using value function iteration.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from tqdm import tqdm


@dataclass
class VFIConfig:
    """Configuration for Value Function Iteration solver."""
    # Grid parameters
    n_c: int = 100                  # Number of cash grid points
    n_tau: int = 100                # Number of time grid points
    c_min: float = 0.0              # Minimum cash
    c_max: float = 2.0              # Maximum cash

    # Action grid parameters
    n_dividend: int = 50            # Number of dividend grid points
    n_equity: int = 30              # Number of equity issuance grid points
    dividend_max: float = 20.0      # Maximum dividend
    equity_max: float = 4.0         # Maximum equity issuance

    # Numerical parameters
    dt: float = 0.1                 # Time step
    T: float = 10.0                 # Time horizon

    # Convergence parameters
    max_iterations: int = 10000     # Maximum VFI iterations
    tolerance: float = 1e-6         # Convergence tolerance

    # Finite difference scheme
    upwind_scheme: bool = True      # Use upwind scheme for drift


class NumericalVFISolver:
    """
    Value Function Iteration solver for time-augmented GHM equity model.

    Solves the HJB equation on a grid using backward induction in time.
    """

    def __init__(self, dynamics, config: VFIConfig = None):
        """
        Initialize VFI solver.

        Args:
            dynamics: GHMEquityTimeAugmentedDynamics instance
            config: VFI configuration
        """
        self.dynamics = dynamics
        self.config = config or VFIConfig()
        self.p = dynamics.p  # Model parameters

        # Discount rate
        self.rho = self.p.r - self.p.mu

        # Create grids
        self._setup_grids()

        # Initialize value function
        self.V = None
        self.policy_dividend = None
        self.policy_equity = None

    def _setup_grids(self):
        """Create state and action grids."""
        # State grids
        self.c_grid = np.linspace(self.config.c_min, self.config.c_max, self.config.n_c)
        self.tau_grid = np.linspace(0, self.config.T, self.config.n_tau)

        self.dc = self.c_grid[1] - self.c_grid[0]
        self.dtau = self.config.dt

        # Action grids
        self.dividend_grid = np.linspace(0, self.config.dividend_max, self.config.n_dividend)
        self.equity_grid = np.linspace(0, self.config.equity_max, self.config.n_equity)

        # Create 2D meshgrid for states
        self.C, self.TAU = np.meshgrid(self.c_grid, self.tau_grid, indexing='ij')

    def compute_drift_diffusion(self, c: np.ndarray, a_L: float, a_E: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute drift and diffusion for given state and actions.

        Args:
            c: Cash level (can be array)
            a_L: Dividend payout rate
            a_E: Equity issuance (gross)

        Returns:
            drift, diffusion_squared
        """
        # Convert to torch tensors for dynamics evaluation
        c_tensor = torch.tensor(c, dtype=torch.float32).reshape(-1, 1)
        action = torch.tensor([[a_L, a_E]], dtype=torch.float32).expand(len(c_tensor.flatten()), 2)

        # Create 2D state (c, τ) - τ doesn't affect drift/diffusion for c
        tau_dummy = torch.zeros_like(c_tensor)
        state = torch.cat([c_tensor, tau_dummy], dim=1)

        with torch.no_grad():
            # Get drift for cash (first component)
            drift_full = self.dynamics.drift(state, action)
            drift_c = drift_full[:, 0].numpy()

            # Get diffusion squared for cash (first component)
            diff_sq_full = self.dynamics.diffusion_squared(state)
            diff_sq_c = diff_sq_full[:, 0].numpy()

        return drift_c.reshape(c.shape), diff_sq_c.reshape(c.shape)

    def compute_derivatives(self, V: np.ndarray, i: int, j: int) -> Tuple[float, float, float]:
        """
        Compute finite difference derivatives of value function.

        Args:
            V: Value function (n_c, n_tau)
            i: Cash index
            j: Time index

        Returns:
            V_c (first derivative), V_cc (second derivative), V_tau (time derivative)
        """
        # First derivative w.r.t. c (central difference when possible)
        if i == 0:
            # Forward difference at lower boundary
            V_c = (V[i+1, j] - V[i, j]) / self.dc
        elif i == self.config.n_c - 1:
            # Backward difference at upper boundary
            V_c = (V[i, j] - V[i-1, j]) / self.dc
        else:
            # Central difference
            V_c = (V[i+1, j] - V[i-1, j]) / (2 * self.dc)

        # Second derivative w.r.t. c
        if i == 0 or i == self.config.n_c - 1:
            V_cc = 0.0  # No curvature at boundaries
        else:
            V_cc = (V[i+1, j] - 2*V[i, j] + V[i-1, j]) / (self.dc ** 2)

        # Time derivative (backward difference) - not used in time-stepping version
        if j == 0:
            V_tau = 0.0  # At τ=0, no time derivative
        else:
            V_tau = (V[i, j] - V[i, j-1]) / self.dtau

        # Replace NaN/Inf with 0
        V_c = 0.0 if not np.isfinite(V_c) else V_c
        V_cc = 0.0 if not np.isfinite(V_cc) else V_cc
        V_tau = 0.0 if not np.isfinite(V_tau) else V_tau

        return V_c, V_cc, V_tau

    def compute_upwind_derivative(self, V: np.ndarray, i: int, j: int, drift: float) -> float:
        """
        Compute upwind scheme derivative based on drift direction.

        Args:
            V: Value function
            i: Cash index
            j: Time index
            drift: Drift value at this point

        Returns:
            Upwind derivative V_c
        """
        if drift > 0:
            # Forward difference
            if i < self.config.n_c - 1:
                return (V[i+1, j] - V[i, j]) / self.dc
            else:
                return (V[i, j] - V[i-1, j]) / self.dc
        else:
            # Backward difference
            if i > 0:
                return (V[i, j] - V[i-1, j]) / self.dc
            else:
                return (V[i+1, j] - V[i, j]) / self.dc

    def optimize_action(self, c: float, V: np.ndarray, i: int, j: int) -> Tuple[float, float, float]:
        """
        Optimize over actions at a given state to solve Bellman equation.

        Args:
            c: Cash level
            V: Current value function
            i: Cash index
            j: Time index

        Returns:
            Optimal (dividend, equity, value)
        """
        best_value = -np.inf
        best_dividend = 0.0
        best_equity = 0.0

        # Get spatial derivatives only (V_c, V_cc)
        # V_tau is handled implicitly by time-stepping scheme
        V_c, V_cc, _ = self.compute_derivatives(V, i, j)

        # Grid search over actions
        for a_L in self.dividend_grid:
            for a_E in self.equity_grid:
                # Compute drift and diffusion
                drift, diff_sq = self.compute_drift_diffusion(np.array([c]), a_L, a_E)
                drift = float(drift[0])
                diff_sq = float(diff_sq[0])

                # Use upwind scheme if enabled
                if self.config.upwind_scheme:
                    V_c_upwind = self.compute_upwind_derivative(V, i, j, drift)
                else:
                    V_c_upwind = V_c

                # Fixed cost indicator
                fixed_cost = self.p.phi if a_E > 1e-6 else 0.0

                # Instantaneous reward: dividend - equity cost
                reward = a_L - a_E

                # HJB right-hand side (spatial part only):
                # Time derivative handled by backward induction scheme
                # rhs = max_a [u(a) + μ(c,a)·V_c + 0.5·σ²(c)·V_cc]
                rhs = reward + drift * V_c_upwind + 0.5 * diff_sq * V_cc

                if rhs > best_value:
                    best_value = rhs
                    best_dividend = a_L
                    best_equity = a_E

        return best_dividend, best_equity, best_value

    def apply_boundary_conditions(self, V: np.ndarray, j: int) -> np.ndarray:
        """
        Apply boundary conditions to value function.

        Args:
            V: Value function at current time step
            j: Time index

        Returns:
            Value function with boundary conditions applied
        """
        V_new = V.copy()

        # Lower boundary (c = 0): Liquidation
        # Equity holders get liquidation_value (set to 0 in model)
        V_new[0, j] = self.p.liquidation_value

        # Upper boundary (c = c_max): Can pay dividends
        # Natural boundary - no additional constraint needed

        # Time boundary (τ = 0): Terminal condition
        # At end of horizon, liquidate for terminal value
        if j == 0:
            V_new[:, 0] = self.p.liquidation_value

        return V_new

    def solve(self, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Solve HJB equation using value function iteration.

        Returns:
            Dictionary with:
                - V: Value function (n_c, n_tau)
                - policy_dividend: Optimal dividend policy (n_c, n_tau)
                - policy_equity: Optimal equity issuance policy (n_c, n_tau)
        """
        # Initialize value function
        V = np.zeros((self.config.n_c, self.config.n_tau))
        policy_dividend = np.zeros((self.config.n_c, self.config.n_tau))
        policy_equity = np.zeros((self.config.n_c, self.config.n_tau))

        # Apply terminal condition
        V = self.apply_boundary_conditions(V, 0)

        # Backward induction in time
        n_time_steps = int(self.config.T / self.dtau)
        time_indices = np.arange(n_time_steps + 1)

        iterator = tqdm(time_indices[1:], desc="VFI Backward Induction") if verbose else time_indices[1:]

        for j_idx, j in enumerate(iterator, start=1):
            # Single pass per time step (standard backward induction)
            # Update value function at each cash level
            for i in range(self.config.n_c):
                c = self.c_grid[i]

                # Optimize over actions using value from previous time
                opt_dividend, opt_equity, rhs = self.optimize_action(c, V, i, j)

                # Update using implicit time-stepping scheme
                # HJB: ρV - ∂V/∂τ = rhs
                # Discretized: ρV(τ) - [V(τ) - V(τ-dt)]/dt = rhs
                # Solve for V(τ): V(τ) = [V(τ-dt)/dt + rhs] / (ρ + 1/dt)

                if j > 0:
                    # Use backward induction from previous time step
                    V_prev_time = V[i, j-1]  # Value at previous time
                    V[i, j] = (V_prev_time / self.dtau + rhs) / (self.rho + 1.0 / self.dtau)

                    # Clip to prevent overflow (value should be bounded)
                    # Max reasonable value: firm value with infinite dividends
                    max_value = 100.0 * self.p.alpha / self.rho if self.rho > 0 else 1000.0
                    V[i, j] = np.clip(V[i, j], -max_value, max_value)
                else:
                    # At τ=0, use terminal condition
                    V[i, j] = self.p.liquidation_value

                policy_dividend[i, j] = opt_dividend
                policy_equity[i, j] = opt_equity

            # Apply boundary conditions
            V = self.apply_boundary_conditions(V, j)

        self.V = V
        self.policy_dividend = policy_dividend
        self.policy_equity = policy_equity

        return {
            'V': V,
            'policy_dividend': policy_dividend,
            'policy_equity': policy_equity,
            'c_grid': self.c_grid,
            'tau_grid': self.tau_grid,
        }

    def get_policy_at_state(self, c: float, tau: float) -> Tuple[float, float]:
        """
        Get optimal policy at a given state using interpolation.

        Args:
            c: Cash level
            tau: Time to horizon

        Returns:
            (dividend, equity)
        """
        # Find nearest grid points
        i = np.argmin(np.abs(self.c_grid - c))
        j = np.argmin(np.abs(self.tau_grid - tau))

        return self.policy_dividend[i, j], self.policy_equity[i, j]

    def get_value_at_state(self, c: float, tau: float) -> float:
        """
        Get value function at a given state using interpolation.

        Args:
            c: Cash level
            tau: Time to horizon

        Returns:
            Value
        """
        # Find nearest grid points (simple nearest neighbor)
        i = np.argmin(np.abs(self.c_grid - c))
        j = np.argmin(np.abs(self.tau_grid - tau))

        return self.V[i, j]
