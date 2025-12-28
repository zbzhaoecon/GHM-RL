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
from scipy import sparse
from scipy.sparse.linalg import spsolve


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


class PolicyIterationSolver:
    """
    Policy Iteration (Howard's Algorithm) solver for time-augmented GHM model.

    Unlike VFI which solves a nonlinear HJB at each step, Policy Iteration:
    1. Fixes the policy (dividend, equity decisions)
    2. Solves the resulting LINEAR PDE for the value function
    3. Updates the policy based on the new value function
    4. Repeats until convergence

    This is more stable because step 2 is a linear problem (tridiagonal solve).
    """

    def __init__(self, dynamics, config: VFIConfig = None):
        """
        Initialize Policy Iteration solver.

        Args:
            dynamics: GHMEquityTimeAugmentedDynamics instance
            config: VFI configuration (reuses same config structure)
        """
        self.dynamics = dynamics
        self.config = config or VFIConfig()
        self.p = dynamics.p  # Model parameters

        # Discount rate
        self.rho = self.p.r - self.p.mu

        # Create grids
        self._setup_grids()

        # Initialize value function and policies
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

        # Create 2D meshgrid for states
        self.C, self.TAU = np.meshgrid(self.c_grid, self.tau_grid, indexing='ij')

    def compute_drift_diffusion(self, c: np.ndarray, a_L: np.ndarray, a_E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute drift and diffusion for given states and actions.

        Args:
            c: Cash levels (array)
            a_L: Dividend rates (array, same length as c)
            a_E: Equity issuance rates (array, same length as c)

        Returns:
            drift, diffusion_squared (arrays)
        """
        c = np.atleast_1d(c)
        a_L = np.atleast_1d(a_L)
        a_E = np.atleast_1d(a_E)

        # Convert to torch tensors
        c_tensor = torch.from_numpy(c).float().reshape(-1, 1)
        a_L_tensor = torch.from_numpy(a_L).float().reshape(-1, 1)
        a_E_tensor = torch.from_numpy(a_E).float().reshape(-1, 1)
        action = torch.cat([a_L_tensor, a_E_tensor], dim=1)

        # Create 2D state (c, τ) - τ doesn't affect drift/diffusion for c
        tau_dummy = torch.zeros_like(c_tensor)
        state = torch.cat([c_tensor, tau_dummy], dim=1)

        with torch.no_grad():
            drift_full = self.dynamics.drift(state, action)
            drift_c = drift_full[:, 0].numpy()

            diff_sq_full = self.dynamics.diffusion_squared(state)
            diff_sq_c = diff_sq_full[:, 0].numpy()

        return drift_c, diff_sq_c

    def _build_tridiagonal_matrix(self, drift: np.ndarray, diff_sq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build tridiagonal matrix for the linear PDE solve.

        The discretized PDE at interior points:
        ρV_i - (V_i - V_i^{prev})/dt = reward_i + μ_i * V_c + 0.5*σ²_i * V_cc

        Using central differences:
        V_c = (V_{i+1} - V_{i-1}) / (2*dc)
        V_cc = (V_{i+1} - 2*V_i + V_{i-1}) / dc²

        Rearranging: A_lower * V_{i-1} + A_diag * V_i + A_upper * V_{i+1} = rhs

        Args:
            drift: Drift values at each cash grid point
            diff_sq: Diffusion squared at each grid point

        Returns:
            lower, diag, upper diagonal arrays
        """
        n = self.config.n_c

        # Diffusion coefficient
        D = 0.5 * diff_sq / (self.dc ** 2)

        # Advection coefficient (upwind scheme for stability)
        # Split drift into positive and negative parts
        drift_pos = np.maximum(drift, 0)
        drift_neg = np.minimum(drift, 0)

        # Upwind: use backward diff for positive drift, forward diff for negative
        A_pos = drift_pos / self.dc  # coefficient for V_{i-1} from positive drift
        A_neg = -drift_neg / self.dc  # coefficient for V_{i+1} from negative drift

        # Time discretization coefficient
        T_coef = 1.0 / self.dtau

        # Build diagonals
        # Lower diagonal: coefficient of V_{i-1}
        lower = D[1:] - A_pos[1:]  # Only for i=1 to n-1

        # Main diagonal: coefficient of V_i
        diag = -(2 * D + A_pos + A_neg + self.rho + T_coef)

        # Upper diagonal: coefficient of V_{i+1}
        upper = D[:-1] + A_neg[:-1]  # Only for i=0 to n-2

        return lower, diag, upper

    def _solve_tridiagonal(self, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
                           rhs: np.ndarray) -> np.ndarray:
        """
        Solve tridiagonal system using Thomas algorithm.

        Args:
            lower: Sub-diagonal (length n-1)
            diag: Main diagonal (length n)
            upper: Super-diagonal (length n-1)
            rhs: Right-hand side (length n)

        Returns:
            Solution vector (length n)
        """
        n = len(diag)

        # Forward elimination
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)

        c_prime[0] = upper[0] / diag[0]
        d_prime[0] = rhs[0] / diag[0]

        for i in range(1, n-1):
            denom = diag[i] - lower[i-1] * c_prime[i-1]
            c_prime[i] = upper[i] / denom
            d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i-1]) / denom

        d_prime[n-1] = (rhs[n-1] - lower[n-2] * d_prime[n-2]) / (diag[n-1] - lower[n-2] * c_prime[n-2])

        # Back substitution
        x = np.zeros(n)
        x[n-1] = d_prime[n-1]

        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def policy_evaluation(self, V_prev: np.ndarray, policy_div: np.ndarray,
                          policy_eq: np.ndarray, j: int) -> np.ndarray:
        """
        Evaluate value function for fixed policy at time step j.

        Given fixed policy, solve the LINEAR PDE:
        ρV - (V - V_prev)/dt = a_L - a_E + μ(c,a)V_c + 0.5*σ²V_cc

        Discretizing with implicit scheme and central differences:
        (ρ + 1/dt)V_i - μ_i*(V_{i+1}-V_{i-1})/(2dc) - D_i*(V_{i+1}-2V_i+V_{i-1})/dc² = (a_L - a_E) + V_prev/dt

        Args:
            V_prev: Value function at previous time step (column vector)
            policy_div: Dividend policy at this time step
            policy_eq: Equity policy at this time step
            j: Time index

        Returns:
            Value function at current time step
        """
        n = self.config.n_c
        dc = self.dc
        dt = self.dtau

        # Get drift and diffusion for current policy
        drift, diff_sq = self.compute_drift_diffusion(
            self.c_grid, policy_div, policy_eq
        )

        # Diffusion coefficient D = σ²/2
        D = 0.5 * diff_sq

        # Build sparse matrix A where A @ V = b
        # Interior equation: (ρ + 1/dt)V_i - μ_i*(V_{i+1}-V_{i-1})/(2dc) - D_i*(V_{i+1}-2V_i+V_{i-1})/dc² = rhs
        # Rearranging:
        # V_{i-1} * (μ_i/(2dc) - D_i/dc²) + V_i * (ρ + 1/dt + 2*D_i/dc²) + V_{i+1} * (-μ_i/(2dc) - D_i/dc²) = rhs

        # Coefficients
        a_lower = drift / (2 * dc) - D / (dc ** 2)  # coef of V_{i-1}
        a_diag = self.rho + 1.0 / dt + 2 * D / (dc ** 2)  # coef of V_i
        a_upper = -drift / (2 * dc) - D / (dc ** 2)  # coef of V_{i+1}

        # Build sparse tridiagonal matrix
        diagonals = [a_lower[1:], a_diag, a_upper[:-1]]
        offsets = [-1, 0, 1]
        A = sparse.diags(diagonals, offsets, shape=(n, n), format='csr')

        # RHS: reward + V_prev/dt
        reward = policy_div - policy_eq
        b = reward + V_prev / dt

        # Apply boundary conditions
        # At c=0 (i=0): V = liquidation_value (Dirichlet)
        A = A.tolil()  # Convert to lil for efficient modification
        A[0, :] = 0
        A[0, 0] = 1.0
        b[0] = self.p.liquidation_value

        # At c=c_max (i=n-1): Neumann dV/dc = 1
        # Use backward difference: (V_{n-1} - V_{n-2})/dc = 1
        # => V_{n-1} - V_{n-2} = dc
        A[n-1, :] = 0
        A[n-1, n-1] = 1.0
        A[n-1, n-2] = -1.0
        b[n-1] = dc

        A = A.tocsr()

        # Solve sparse system
        V_new = spsolve(A, b)

        # Clip to reasonable range
        max_value = 100.0 * self.p.alpha / self.rho if self.rho > 0 else 1000.0
        V_new = np.clip(V_new, 0.0, max_value)

        return V_new

    def policy_improvement(self, V: np.ndarray, j: int,
                           old_dividend: np.ndarray = None,
                           old_equity: np.ndarray = None,
                           damping: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update policy based on current value function with damping.

        Using FOC:
        - Dividend: a_L = max if V_c < 1, else 0 (smooth approximation)
        - Equity: a_E = max if V_c > p, else 0 (smooth approximation)

        Args:
            V: Current value function (column at time j)
            j: Time index
            old_dividend: Previous dividend policy (for damping)
            old_equity: Previous equity policy (for damping)
            damping: Blend factor (0 = no change, 1 = full update)

        Returns:
            (new_dividend_policy, new_equity_policy)
        """
        n = self.config.n_c

        # Compute V_c using central differences with smoothing
        V_c = np.zeros(n)
        V_c[0] = (V[1] - V[0]) / self.dc  # Forward diff at boundary
        V_c[-1] = (V[-1] - V[-2]) / self.dc  # Backward diff at boundary
        V_c[1:-1] = (V[2:] - V[:-2]) / (2 * self.dc)  # Central diff interior

        # Light smoothing of V_c to reduce numerical noise (single pass)
        V_c_smooth = V_c.copy()
        V_c_smooth[1:-1] = 0.25 * V_c_smooth[:-2] + 0.5 * V_c_smooth[1:-1] + 0.25 * V_c_smooth[2:]

        # Apply FOC with smooth transition using sigmoid function
        # The transition width is adaptive based on grid resolution
        transition_width = 0.05  # Controls smoothness of policy transition

        # Dividend FOC: pay dividends when V_c < 1 (marginal value of cash is low)
        # Using smooth sigmoid: dividend = div_max * sigmoid((1 - V_c) / width)
        # When V_c < 1: (1-V_c) > 0, sigmoid is high -> pay dividends
        # When V_c > 1: (1-V_c) < 0, sigmoid is low -> don't pay
        div_arg = (1.0 - V_c_smooth) / transition_width
        div_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(div_arg, -20, 20)))
        div_raw = div_sigmoid * self.config.dividend_max

        # Equity FOC: issue equity when V_c > p (marginal value of cash exceeds dilution cost)
        # Using smooth sigmoid: equity = eq_max * sigmoid((V_c - p) / width)
        # When V_c > p: (V_c-p) > 0, sigmoid is high -> issue equity
        # When V_c < p: (V_c-p) < 0, sigmoid is low -> don't issue
        eq_arg = (V_c_smooth - self.p.p) / transition_width
        eq_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(eq_arg, -20, 20)))
        eq_raw = eq_sigmoid * self.config.equity_max

        # Apply damping to prevent oscillations
        if old_dividend is not None:
            new_dividend = (1 - damping) * old_dividend + damping * div_raw
        else:
            new_dividend = div_raw

        if old_equity is not None:
            new_equity = (1 - damping) * old_equity + damping * eq_raw
        else:
            new_equity = eq_raw

        # Feasibility: can't pay more dividends than cash (in rate terms)
        max_div_feasible = np.maximum(0, self.c_grid / self.dtau)
        new_dividend = np.minimum(new_dividend, max_div_feasible)

        return new_dividend, new_equity

    def solve(self, verbose: bool = True, max_policy_iter: int = 20,
              policy_tol: float = 1e-4) -> Dict[str, np.ndarray]:
        """
        Solve HJB equation using policy iteration.

        Args:
            verbose: Whether to print progress
            max_policy_iter: Maximum policy iterations per time step
            policy_tol: Convergence tolerance for policy

        Returns:
            Dictionary with V, policy_dividend, policy_equity, grids
        """
        n_c = self.config.n_c
        n_tau = self.config.n_tau

        # Initialize value function and policies
        V = np.zeros((n_c, n_tau))
        policy_dividend = np.zeros((n_c, n_tau))
        policy_equity = np.zeros((n_c, n_tau))

        # Terminal condition: V(c, τ=0) = c
        V[:, 0] = self.c_grid

        # Initialize policy at terminal (pay all dividends, no equity)
        policy_dividend[:, 0] = self.config.dividend_max * np.ones(n_c)
        policy_equity[:, 0] = np.zeros(n_c)

        # Backward induction in time
        time_indices = range(1, n_tau)

        if verbose:
            iterator = tqdm(time_indices, desc="Policy Iteration Backward",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            iterator = time_indices

        for j in iterator:
            # Initialize policy from previous time step
            policy_div = policy_dividend[:, j-1].copy()
            policy_eq = policy_equity[:, j-1].copy()

            V_prev = V[:, j-1]

            # Policy iteration loop with damping
            for pi_iter in range(max_policy_iter):
                # Policy Evaluation: solve linear PDE for fixed policy
                V_new = self.policy_evaluation(V_prev, policy_div, policy_eq, j)

                # Policy Improvement: update policy based on new value (with damping)
                new_div, new_eq = self.policy_improvement(
                    V_new, j,
                    old_dividend=policy_div,
                    old_equity=policy_eq,
                    damping=0.5  # Blend 50% old, 50% new
                )

                # Check convergence
                div_change = np.max(np.abs(new_div - policy_div))
                eq_change = np.max(np.abs(new_eq - policy_eq))

                policy_div = new_div
                policy_eq = new_eq

                if max(div_change, eq_change) < policy_tol:
                    break

            # Store results
            V[:, j] = V_new
            policy_dividend[:, j] = policy_div
            policy_equity[:, j] = policy_eq

            # Apply boundary condition at c=0
            V[0, j] = self.p.liquidation_value

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
        """Get optimal policy at a given state."""
        i = np.argmin(np.abs(self.c_grid - c))
        j = np.argmin(np.abs(self.tau_grid - tau))
        return self.policy_dividend[i, j], self.policy_equity[i, j]

    def get_value_at_state(self, c: float, tau: float) -> float:
        """Get value function at a given state."""
        i = np.argmin(np.abs(self.c_grid - c))
        j = np.argmin(np.abs(self.tau_grid - tau))
        return self.V[i, j]
