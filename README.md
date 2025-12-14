# MacroRL: Reinforcement Learning for Continuous-Time Macro-Finance Models

A Python library for solving continuous-time corporate finance and macroeconomic models using deep learning methods.

## Overview

MacroRL provides tools to solve Hamilton-Jacobi-Bellman (HJB) equations arising in continuous-time finance, with a focus on:

- **GHM Models**: Equity and debt management with singular controls (D'ecamps et al.)
- **Deep Galerkin Method (DGM)**: Mesh-free PDE solvers using neural networks
- **Reinforcement Learning**: Policy gradient methods for high-dimensional problems

The library bridges the gap between economic theory (SDEs, HJB equations, boundary conditions) and modern deep learning infrastructure (PyTorch, automatic differentiation).

---

## Project Structure

```
macro_rl/
├── macro_rl/
│   ├── numerics/          # Mathematical foundations
│   │   ├── differentiation.py   # Autograd utilities for ∇V, ∇²V
│   │   ├── integration.py       # SDE discretization (Euler-Maruyama, Milstein)
│   │   └── sampling.py          # State space sampling strategies
│   │
│   ├── dynamics/          # Economic model specifications
│   │   ├── base.py              # Abstract SDE interface
│   │   ├── ghm_equity.py        # 1D equity management model
│   │   ├── ghm_debt.py          # 1D debt management model
│   │   └── ghm_joint.py         # 2D joint model
│   │
│   ├── losses/            # Physics-informed loss functions
│   │   ├── hjb.py               # HJB residual loss
│   │   └── boundary.py          # Boundary condition losses
│   │
│   ├── networks/          # Neural network architectures
│   │   └── value_networks.py    # Value function approximators
│   │
│   ├── solvers/           # Solution algorithms
│   │   └── dgm.py               # Deep Galerkin Method solver
│   │
│   └── utils/             # Helpers
│       ├── config.py            # Configuration management
│       └── plotting.py          # Visualization
│
├── configs/               # YAML configuration files
├── tests/                 # Test suite
├── scripts/               # Training and evaluation scripts
└── notebooks/             # Example notebooks
```

---

## Implementation Phases

### Phase 1: Numerical Foundations
Build and test the core mathematical utilities.

| Module | Purpose | Key Functions | Validation |
|--------|---------|---------------|------------|
| `numerics/differentiation` | Compute ∇V, ∇²V via autograd | `gradient()`, `hessian()`, `hessian_diagonal()` | Analytical derivatives of known functions |
| `numerics/integration` | Discretize SDEs | `euler_maruyama_step()`, `simulate_path()` | GBM moments match theory |
| `numerics/sampling` | Generate training points | `uniform_sampler()`, `boundary_sampler()` | Coverage tests |

**Exit Criteria**: All numerical tests pass; gradients match finite differences to 1e-5.

### Phase 2: Model Specification
Implement the GHM dynamics as abstract interfaces.

| Module | Purpose | Key Methods | Validation |
|--------|---------|-------------|------------|
| `dynamics/base` | Abstract SDE class | `drift()`, `diffusion()`, `boundary_conditions()` | Interface compliance |
| `dynamics/ghm_equity` | 1D equity model | Implements base for Table 1 parameters | Drift/diffusion values at known points |

**Exit Criteria**: Model parameters match GHM paper exactly; dynamics produce sensible trajectories.

### Phase 3: Loss Functions
Implement physics-informed losses for training.

| Module | Purpose | Key Functions | Validation |
|--------|---------|---------------|------------|
| `losses/hjb` | HJB residual | `hjb_residual()`, `hjb_loss()` | Zero residual for analytical solutions |
| `losses/boundary` | Boundary conditions | `dirichlet_loss()`, `neumann_loss()` | Gradient conditions enforced |

**Exit Criteria**: Merton problem residual < 1e-6 with known value function.

### Phase 4: Networks and Solver
Build the DGM solver combining all components.

| Module | Purpose | Key Classes | Validation |
|--------|---------|-------------|------------|
| `networks/value_networks` | Value approximation | `ValueNetwork`, `DGMNetwork` | Smooth outputs, finite Hessians |
| `solvers/dgm` | Training loop | `DGMSolver.train()`, `DGMSolver.evaluate()` | Merton convergence |

**Exit Criteria**: Reproduce GHM paper figures (value function shape, policy boundaries).

### Phase 5: Extensions
Add 2D models, RL solvers, distributed training.

---

## Core Interfaces

### Dynamics Interface

All economic models implement this interface:

```python
class ContinuousTimeDynamics(ABC):
    """Specification of a continuous-time economic model."""

    @property
    def state_dim(self) -> int:
        """Dimension of state space."""

    @property
    def state_bounds(self) -> Tuple[Tensor, Tensor]:
        """(lower, upper) bounds for state variables."""

    def drift(self, state: Tensor, t: float = 0) -> Tensor:
        """Drift μ(x,t) of shape (batch, state_dim)."""

    def diffusion(self, state: Tensor, t: float = 0) -> Tensor:
        """Diffusion σ(x,t) of shape (batch, state_dim)."""

    def discount_rate(self) -> float:
        """Effective discount rate (r - μ) for HJB."""

    def hjb_coefficients(self, state: Tensor) -> Dict[str, Tensor]:
        """Return drift, diffusion², discount for HJB construction."""
```

### Solver Interface

All solvers implement this interface:

```python
class Solver(ABC):
    """Base class for PDE/RL solvers."""

    def __init__(self, dynamics: ContinuousTimeDynamics, config: dict):
        ...

    def train(self, n_iterations: int, callback: Optional[Callable] = None) -> Dict:
        """Train the solver, return loss history."""

    def evaluate(self, states: Tensor) -> Tensor:
        """Evaluate value function at given states."""

    def policy(self, states: Tensor) -> Tensor:
        """Extract optimal policy (if applicable)."""

    def save(self, path: str) -> None:
        """Save model checkpoint."""

    def load(self, path: str) -> None:
        """Load model checkpoint."""
```

---

## Key Design Decisions

### 1. Separation of Dynamics and Solvers
The economic model (drift, diffusion, boundaries) is defined independently of the solution method. This allows:
- Same model solved by DGM, value iteration, or RL
- Easy comparison of methods
- Clean testing of each component

### 2. Physics-Informed Training
The HJB residual is a direct loss term, not just a validation metric:
```
L_total = w_interior * L_HJB + w_boundary * L_BC
```
This leverages known PDE structure for faster convergence than pure RL.

### 3. Autograd for Derivatives
We use PyTorch autograd (not finite differences) for computing V', V''. This:
- Is exact (no discretization error)
- Scales to high dimensions
- Enables end-to-end gradient flow

### 4. Batch-First Operations
All functions operate on batches: `(batch_size, dim)`. This enables:
- Efficient GPU utilization
- Monte Carlo estimation of integrals
- Parallel trajectory simulation

---

## Testing Strategy

### Unit Tests
Each module has isolated tests:
```
tests/
├── test_differentiation.py    # Gradient/Hessian correctness
├── test_integration.py        # SDE discretization accuracy
├── test_dynamics.py           # Model specification
├── test_losses.py             # Loss computation
└── test_solver.py             # Training mechanics
```

### Analytical Benchmarks
Known solutions for validation:

| Problem | Analytical Solution | Use |
|---------|-------------------|-----|
| Merton portfolio | Closed-form V(w) and π* | Verify full pipeline |
| GBM simulation | E[X_T], Var[X_T] known | Verify SDE integration |
| Quadratic value | V(x) = ax² + bx + c | Verify HJB residual = 0 |

### Reproduction Tests
Compare against GHM paper:
- Value function shape (Figures 1-2)
- Convergence curves (Figures 3-4)
- Policy boundaries

---

## Configuration

Models and training are configured via YAML:

```yaml
# configs/ghm_equity_1d.yaml
model:
  name: ghm_equity
  params:
    alpha: 0.18      # Mean cash flow rate
    mu: 0.01         # Growth rate
    sigma_A: 0.25    # Permanent shock vol
    sigma_X: 0.12    # Transient shock vol
    rho: -0.2        # Correlation
    r: 0.03          # Interest rate
    lambda_: 0.02    # Carry cost
    c_max: 2.0       # State upper bound

solver:
  type: dgm
  network:
    hidden_dims: [64, 64, 64]
    activation: tanh
  training:
    learning_rate: 1e-3
    batch_size: 4096
    n_iterations: 10000
  loss_weights:
    interior: 1.0
    boundary: 10.0
```

---

## Dependencies

Core:
- Python >= 3.9
- PyTorch >= 2.0
- NumPy

Optional:
- Ray (distributed training)
- Hydra (configuration)
- Matplotlib (plotting)
- pytest (testing)

---

## Getting Started

```bash
# Install
pip install -e .

# Run tests
pytest tests/

# Train SAC agent on GHM equity model
python scripts/train_ghm.py --timesteps 500000 --output models/ghm_equity

# Monitor training
tensorboard --logdir models/ghm_equity/tensorboard

# Validate solution correctness
python scripts/validate.py --model models/ghm_equity/final_model

# Evaluate policy
python scripts/evaluate.py --model models/ghm_equity/final_model
```

---

## References

- D'ecamps, J.P., Gryglewicz, S., Morellec, E., Villeneuve, S. (2017). Corporate Policies with Permanent and Transitory Shocks. *Review of Financial Studies*.
- Sirignano, J., Spiliopoulos, K. (2018). DGM: A deep learning algorithm for solving partial differential equations. *Journal of Computational Physics*.
- GHM_v2.pdf: Model specifications and benchmark results.

---

## Solution Validation

After training, validate that the learned solution satisfies analytical properties from the GHM paper:

```bash
python scripts/validate.py --model models/ghm_equity/final_model
```

The validation script checks:

1. **Smooth Pasting**: F'(c*) = 1 at the payout threshold
2. **Super-Contact**: F''(c*) = 0 at the threshold
3. **HJB Equation**: Residual is small in continuation region
4. **Monotonicity**: F'(c) > 0 everywhere
5. **Concavity**: F''(c) < 0 below threshold
6. **Policy Threshold**: Clear jump from retention to payout

**Outputs:**
- Console report with PASS/FAIL for each criterion
- `validation_plots.png`: Six-panel diagnostic figure
- `value_and_policy.png`: Combined plot matching paper figures
- `validation_data.npz`: Raw numerical data

See [docs/VALIDATION.md](docs/VALIDATION.md) for detailed methodology and troubleshooting.

---

## Development Workflow

1. **Pick a module** from the current phase
2. **Write tests first** based on the validation criteria
3. **Implement** until tests pass
4. **Document** key design choices
5. **PR review** before moving to next module

Current focus: **Phase 4 Complete - RL Training and Validation**

**Completed:**
- ✅ Phase 1: Numerical foundations (differentiation, integration, sampling)
- ✅ Phase 2: Model specifications (GHM equity dynamics)
- ✅ Phase 3: RL environment (GHM equity gymnasium environment)
- ✅ Phase 4: Training scripts and validation tools
