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
    n_c: int = 50                   # Number of cash grid points (50-100 recommended)
    n_tau: int = 50                 # Number of time grid points (50-100 recommended)
    c_min: float = 0.0              # Minimum cash
    c_max: float = 2.0              # Maximum cash

    # Action grid parameters
    n_dividend: int = 30            # Number of dividend grid points (20-50 recommended)
    n_equity: int = 20              # Number of equity issuance grid points (15-30 recommended)
    dividend_max: float = 1.0       # Maximum dividend rate (should be ~ α, not >> α!)
    equity_max: float = 0.5         # Maximum equity issuance rate

    # Numerical parameters
    dt: float = 0.1                 # Time step (not used - actual dt computed from grid)
    T: float = 10.0                 # Time horizon

    # Convergence parameters
    max_iterations: int = 10000     # Maximum VFI iterations (not used in current implementation)
    tolerance: float = 1e-6         # Convergence tolerance (not used in current implementation)

    # Finite difference scheme
    upwind_scheme: bool = True      # Use upwind scheme for drift (slower but more stable)


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
        self.dtau = self.tau_grid[1] - self.tau_grid[0] if len(self.tau_grid) > 1 else self.config.dt

        # Action grids
        self.dividend_grid = np.linspace(0, self.config.dividend_max, self.config.n_dividend)
        self.equity_grid = np.linspace(0, self.config.equity_max, self.config.n_equity)

        # Create 2D meshgrid for states
        self.C, self.TAU = np.meshgrid(self.c_grid, self.tau_grid, indexing='ij')

    def compute_drift_diffusion(self, c: np.ndarray, a_L, a_E) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute drift and diffusion for given state and actions.

        Args:
            c: Cash level (array)
            a_L: Dividend payout rate (scalar or array)
            a_E: Equity issuance (scalar or array)

        Returns:
            drift, diffusion_squared
        """
        # Ensure arrays
        c = np.atleast_1d(c)
        a_L = np.atleast_1d(a_L)
        a_E = np.atleast_1d(a_E)

        # Broadcast to same length
        n = max(len(c), len(a_L), len(a_E))
        if len(c) == 1:
            c = np.repeat(c, n)
        if len(a_L) == 1:
            a_L = np.repeat(a_L, n)
        if len(a_E) == 1:
            a_E = np.repeat(a_E, n)

        # Convert to torch tensors
        c_tensor = torch.from_numpy(c).float().reshape(-1, 1)
        a_L_tensor = torch.from_numpy(a_L).float().reshape(-1, 1)
        a_E_tensor = torch.from_numpy(a_E).float().reshape(-1, 1)
        action = torch.cat([a_L_tensor, a_E_tensor], dim=1)

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

        return drift_c, diff_sq_c

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

    def optimize_action(self, c: float, V: np.ndarray, i: int, j: int, j_deriv: int = None) -> Tuple[float, float, float]:
        """
        Optimize over actions at a given state to solve Bellman equation.

        Uses first-order conditions (FOC) for bang-bang optimal controls:
        - Dividends: ∂rhs/∂a_L = 1 - V_c → pay max if V_c < 1, else pay 0
        - Equity: ∂rhs/∂a_E = -1 + V_c/p → issue max if V_c > p, else issue 0

        Args:
            c: Cash level
            V: Current value function
            i: Cash index
            j: Time index (current time step being computed)
            j_deriv: Time index to use for derivatives (default: j-1 for semi-implicit scheme)
                     This should be the previous time step where V is already computed.

        Returns:
            Optimal (dividend, equity, value)
        """
        # Use previous time step for derivatives (semi-implicit scheme)
        # V[:, j] hasn't been computed yet, so we must use V[:, j-1] or V[:, j_deriv]
        if j_deriv is None:
            j_deriv = max(0, j - 1)

        # Get spatial derivatives only (V_c, V_cc)
        V_c, V_cc, _ = self.compute_derivatives(V, i, j_deriv)

        # Use FOC with smooth approximation to avoid chattering
        # Dividend FOC: ∂rhs/∂a_L = 1 - V_c
        # - If V_c < 1: marginal value of cash is low, pay dividends
        # - If V_c > 1: marginal value of cash is high, hold cash
        # Use smooth sigmoid: a_L = max * sigmoid((1 - V_c) / smoothing)
        smoothing = 0.2  # Controls transition sharpness (larger = smoother)
        dividend_signal = np.clip((1.0 - V_c) / smoothing, -20, 20)  # Clip to avoid overflow
        opt_dividend = self.config.dividend_max / (1.0 + np.exp(-dividend_signal))

        # Equity FOC: ∂rhs/∂a_E = -1 + V_c/p
        # - If V_c > p: marginal value of cash exceeds issuance cost, issue equity
        # - If V_c < p: issuance too costly, don't issue
        # Use smooth sigmoid: a_E = max * sigmoid((V_c - p) / smoothing)
        equity_signal = np.clip((V_c - self.p.p) / smoothing, -20, 20)  # Clip to avoid overflow
        opt_equity = self.config.equity_max / (1.0 + np.exp(-equity_signal))

        # Constraint: can't pay more dividends than available cash (rate constraint)
        # For continuous time, this is approximately c/dt, but for stability use c
        max_dividend_feasible = max(0.0, c / self.dtau) if self.dtau > 0 else self.config.dividend_max
        opt_dividend = min(opt_dividend, max_dividend_feasible)

        # Compute drift and diffusion for optimal action
        drift, diff_sq = self.compute_drift_diffusion(
            np.array([c]), np.array([opt_dividend]), np.array([opt_equity])
        )
        drift = drift[0]
        diff_sq = diff_sq[0]

        # Compute V_c with upwind scheme if enabled
        if self.config.upwind_scheme:
            if drift > 0:
                V_c_upwind = (V[min(i+1, self.config.n_c-1), j_deriv] - V[i, j_deriv]) / self.dc
            else:
                V_c_upwind = (V[i, j_deriv] - V[max(i-1, 0), j_deriv]) / self.dc
        else:
            V_c_upwind = V_c

        # Compute HJB RHS
        # reward = a_L - a_E (dividends minus dilution cost)
        reward = opt_dividend - opt_equity
        rhs = reward + drift * V_c_upwind + 0.5 * diff_sq * V_cc

        return opt_dividend, opt_equity, rhs

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
        # At end of horizon, shareholders receive remaining cash as final dividend
        # V(c, τ=0) = c (not 0, because cash can be paid out immediately)
        if j == 0:
            V_new[:, 0] = self.c_grid  # Terminal value equals cash holdings

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
        # Iterate over tau grid indices (from 1 to n_tau-1)
        # tau grid has n_tau points indexed 0 to n_tau-1
        time_indices = range(1, self.config.n_tau)

        if verbose:
            iterator = tqdm(time_indices, desc="VFI Backward Induction",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            iterator = time_indices

        for j in iterator:
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

                    # Clip to prevent overflow and ensure non-negativity
                    # Value should be non-negative (equity can't have negative value)
                    # Max reasonable value: firm value with infinite dividends
                    max_value = 100.0 * self.p.alpha / self.rho if self.rho > 0 else 1000.0
                    V[i, j] = np.clip(V[i, j], 0.0, max_value)
                else:
                    # At τ=0, terminal value equals cash (can be paid as final dividend)
                    V[i, j] = self.c_grid[i]

                policy_dividend[i, j] = opt_dividend
                policy_equity[i, j] = opt_equity

            # Apply boundary conditions
            V = self.apply_boundary_conditions(V, j)

            # Apply light smoothing to reduce oscillations (moving average in c direction)
            # Skip boundaries
            for i_smooth in range(1, self.config.n_c - 1):
                V[i_smooth, j] = 0.25 * V[i_smooth-1, j] + 0.5 * V[i_smooth, j] + 0.25 * V[i_smooth+1, j]

            # Re-apply boundary conditions after smoothing
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
