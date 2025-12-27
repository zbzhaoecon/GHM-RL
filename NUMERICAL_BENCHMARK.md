# Numerical Benchmark for GHM Equity Model

This document describes the numerical benchmark implementation for the GHM (Décamps et al. 2017) equity management model. The benchmark provides a traditional finite difference solution to compare against deep learning approaches.

## Overview

The numerical benchmark solves the Hamilton-Jacobi-Bellman (HJB) equation for the time-augmented GHM equity model using **Value Function Iteration (VFI)** on a discrete grid.

### HJB Equation

For the time-augmented problem with state $(c, \tau)$:

```
(r-μ)V(c,τ) = max_{a_L, a_E} [a_L - a_E + μ_c(c,a)V_c + (-1)V_τ + ½σ²(c)V_cc]
```

where:
- `V(c,τ)`: Value function (firm equity value)
- `c`: Cash reserves / earnings ratio
- `τ`: Time-to-horizon (remaining time)
- `a_L`: Dividend payout rate
- `a_E`: Equity issuance (gross)
- `μ_c(c,a)`: Drift of cash process
- `σ²(c)`: Variance of cash process

## Implementation

### Components

1. **`macro_rl/solvers/numerical_vfi.py`**
   - `NumericalVFISolver`: Main VFI solver class
   - Implements backward induction in time
   - Uses finite differences for spatial derivatives
   - Grid search over action space

2. **`macro_rl/solvers/monte_carlo_evaluator.py`**
   - `MonteCarloEvaluator`: Policy evaluation via simulation
   - Computes "realized" value functions
   - Validates numerical and RL solutions

3. **`scripts/run_numerical_benchmark.py`**
   - Main script to run VFI solver
   - Creates visualizations
   - Saves results

4. **`scripts/compare_rl_numerical.py`**
   - Compares RL and numerical solutions
   - Computes error metrics
   - Creates difference plots

### Algorithm

The VFI solver implements the following algorithm:

```
1. Initialize value function V(c, τ=0) = liquidation_value
2. For each time step τ = dt, 2dt, ..., T:
   a. For each cash level c in grid:
      i. For each action (a_L, a_E) in action grid:
         - Compute drift μ_c(c, a)
         - Compute diffusion σ²(c)
         - Compute derivatives V_c, V_cc, V_τ using finite differences
         - Evaluate HJB right-hand side
      ii. Choose action maximizing HJB RHS
      iii. Update V(c, τ)
   b. Apply boundary conditions
   c. Check convergence
3. Return V, optimal policies
```

## Usage

### 1. Run Numerical Benchmark

Basic usage:

```bash
python scripts/run_numerical_benchmark.py \
    --config configs/time_augmented_sparse_config.yaml \
    --n-c 100 \
    --n-tau 100 \
    --output-dir numerical_benchmark_results
```

With Monte Carlo validation:

```bash
python scripts/run_numerical_benchmark.py \
    --config configs/time_augmented_sparse_config.yaml \
    --n-c 100 \
    --n-tau 100 \
    --compute-mc \
    --mc-samples 100 \
    --output-dir numerical_benchmark_results
```

### 2. Compare RL vs Numerical

```bash
python scripts/compare_rl_numerical.py \
    --rl-checkpoint checkpoints/ghm_time_augmented_sparse/policy_step_5000.pt \
    --numerical-solution numerical_benchmark_results/vfi_solution.npz \
    --config configs/time_augmented_sparse_config.yaml \
    --output-dir comparison_results
```

### Command-line Arguments

#### `run_numerical_benchmark.py`

- `--config`: Path to config file (default: `configs/time_augmented_sparse_config.yaml`)
- `--n-c`: Number of cash grid points (default: 100)
- `--n-tau`: Number of time grid points (default: 100)
- `--n-dividend`: Number of dividend action grid points (default: 50)
- `--n-equity`: Number of equity action grid points (default: 30)
- `--tolerance`: Convergence tolerance (default: 1e-6)
- `--compute-mc`: Compute Monte Carlo realized value function
- `--mc-samples`: MC samples per state (default: 100)
- `--output-dir`: Output directory (default: `numerical_benchmark_results`)

#### `compare_rl_numerical.py`

- `--rl-checkpoint`: Path to RL checkpoint (required)
- `--numerical-solution`: Path to VFI solution .npz (required)
- `--config`: Path to config file
- `--output-dir`: Output directory (default: `comparison_results`)
- `--device`: Device for RL evaluation (default: `cpu`)

## Output

### Numerical Benchmark

The benchmark produces:

1. **`vfi_solution.npz`**: Contains
   - `V`: Value function grid (n_c, n_tau)
   - `policy_dividend`: Optimal dividend policy
   - `policy_equity`: Optimal equity issuance policy
   - `c_grid`: Cash grid points
   - `tau_grid`: Time grid points

2. **`mc_realized_values.npz`** (if --compute-mc):
   - `V_realized`: Monte Carlo realized value function
   - `V_std`: Standard errors

3. **`numerical_benchmark.png`**: Visualization with:
   - Policy heatmaps: π(c, τ) for dividend and equity
   - Value function heatmap: V(c, τ)
   - Slices at different time horizons

### Comparison

The comparison produces:

1. **`comparison_metrics.txt`**: Quantitative metrics
   - MAE, RMSE, Max Diff for policies and values
   - Relative errors

2. **`comparison_heatmaps.png`**: Side-by-side heatmaps
   - Row 1: Numerical solution
   - Row 2: RL solution

3. **`difference_plots.png`**: Difference plots (RL - Numerical)
   - Dividend difference
   - Equity difference
   - Value difference

## Interpretation

### Expected Results

1. **Dividend Policy**:
   - Should increase with cash level `c`
   - Should decrease as `τ → 0` (conserve cash near horizon)

2. **Equity Issuance**:
   - Should spike at low `c` (avoid bankruptcy)
   - Should be near-zero at high `c`
   - May increase near `τ = 0` if firm expects future profitability

3. **Value Function**:
   - Should increase with `c` (more cash = higher value)
   - Should have positive time value (higher `τ` often better)

### Comparison Metrics

Good RL performance typically shows:
- **Dividend MAE < 0.5**: Policy closely matches numerical
- **Equity MAE < 0.1**: Issuance decisions align well
- **Value Relative Error < 5%**: Value function approximation is accurate

## Computational Considerations

### Complexity

- **Time complexity**: O(n_c × n_tau × n_dividend × n_equity × n_iter)
- **Space complexity**: O(n_c × n_tau)

### Typical Runtime

For grid sizes:
- 100 × 100 state grid, 50 × 30 action grid: ~5-10 minutes
- 200 × 200 state grid, 100 × 50 action grid: ~30-60 minutes

### Parallelization

The current implementation is sequential. For faster solving:
1. Parallelize action optimization across cash levels
2. Use vectorized operations for derivative computation
3. Consider GPU acceleration for large grids

## Limitations

1. **Curse of Dimensionality**:
   - Grid size grows exponentially with state dimension
   - Current implementation limited to 2D state space

2. **Action Discretization**:
   - Grid search may miss optimal continuous actions
   - Finer grids increase computation time

3. **Boundary Approximation**:
   - Finite differences less accurate at boundaries
   - May require special treatment for c=0 singularity

4. **Convergence**:
   - VFI can be slow to converge
   - May require many iterations for tight tolerance

## References

- Décamps, J. P., Gryglewicz, S., Morellec, E., & Villeneuve, S. (2017). "Corporate policies with permanent and transitory shocks." *Review of Financial Studies*, 30(1), 162-210.

## Future Enhancements

Potential improvements:

1. **Policy Function Iteration (PFI)**: Faster convergence than VFI
2. **Multigrid Methods**: Hierarchical grid refinement
3. **Adaptive Grids**: Concentrate points where value changes rapidly
4. **Parallel Action Search**: GPU-accelerated optimization
5. **Higher-Order Schemes**: WENO, DG methods for better accuracy
6. **Continuous Action Optimization**: Replace grid search with gradient-based methods

## Contact

For questions or issues with the numerical benchmark, please open an issue on GitHub.
