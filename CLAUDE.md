# CLAUDE.md — AI Assistant Guide for GHM-RL (MacroRL)

This file provides context for AI assistants working on this repository.

---

## Project Overview

**MacroRL** (package name: `macro_rl`) is a Python library for solving continuous-time corporate finance models using **model-based reinforcement learning**. It implements the GHM (Gârleanu-Hackbarth-Morellec) equity management model by exploiting known closed-form system dynamics to achieve superior sample efficiency and convergence over traditional model-free RL approaches.

**Key idea:** Instead of model-free RL (PPO, SAC), the library uses exact knowledge of SDE dynamics:
```
dc = μ(c)dt + σ(c)dW    ← known closed-form functions
```
to compute exact pathwise gradients through differentiable simulation.

- **Docs:** https://macrorl.readthedocs.io/
- **License:** MIT
- **Python:** 3.9–3.11

---

## Repository Structure

```
GHM-RL/
├── macro_rl/              # Main package
│   ├── config/            # Configuration management
│   ├── core/              # StateSpace, ParameterManager abstractions
│   ├── dynamics/          # Continuous-time SDE models (GHM equity)
│   ├── simulation/        # SDE integrators, trajectory generation, differentiable sim
│   ├── control/           # Action space specifications and masking
│   ├── rewards/           # Objective/reward functions
│   ├── policies/          # Policy representations (neural, barrier, tabular)
│   ├── values/            # Value functions (neural, analytical)
│   ├── solvers/           # Model-based RL algorithms
│   ├── validation/        # HJB residual checks, analytical comparisons
│   ├── numerics/          # Differentiation, integration, sampling utilities
│   ├── networks/          # Neural network architectures
│   ├── distributions/     # Custom probability distributions
│   ├── envs/              # Gymnasium-compatible environments
│   ├── evaluation/        # Evaluation utilities
│   ├── losses/            # Loss functions
│   ├── utils/             # Autograd helpers, plotting
│   └── scripts/           # Training scripts (train_pathwise, train_mc, train_dgm)
├── configs/               # YAML configuration files (10 configs)
├── scripts/               # Top-level training and utility scripts
├── tests/                 # Unit + integration tests (18+ test files)
├── examples/              # Usage examples
├── docs/                  # Sphinx documentation source
├── setup.py               # Package configuration
├── pytest.ini             # Test configuration (test path: tests/)
├── requirements.txt       # Core dependencies
├── requirements-dev.txt   # Development dependencies
├── README.md              # Project overview and quick start
└── TRAINING_GUIDE.md      # Step-by-step training instructions
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Core ML | PyTorch ≥ 2.0.0 |
| Scientific | NumPy ≥ 1.20.0, SciPy ≥ 1.7.0 |
| Visualization | Matplotlib ≥ 3.3.0 |
| Configuration | PyYAML ≥ 5.4.0 |
| Testing | pytest, pytest-cov |
| Code quality | black, flake8, mypy |
| Optional: RL envs | gymnasium ≥ 1.0.0, stable-baselines3 ≥ 2.0.0 |
| Optional: Distributed | ray ≥ 2.0.0 |
| Optional: Config mgmt | hydra-core ≥ 1.2.0 |

---

## Installation

```bash
# Basic installation
pip install -e .

# Development (includes black, flake8, mypy, pytest)
pip install -e ".[dev]"

# With distributed computing support
pip install -e ".[distributed]"

# With Hydra config management
pip install -e ".[config]"
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_dynamics_ghm.py -v

# Run with coverage report
pytest tests/ --cov=macro_rl --cov-report=html

# Run only fast tests (skip slow/integration markers)
pytest tests/ -v -m "not slow and not integration"
```

Test markers defined: `@pytest.mark.slow`, `@pytest.mark.integration`

Root-level test files (`test_mask_fix.py`, `test_numerical_solver.py`, `test_reward_fix.py`, `test_sparse_dense_equivalence.py`) are integration/regression tests and can be run with `pytest <filename>`.

---

## Training

```bash
# Primary entry point — recommended
python scripts/train_with_config.py --config configs/time_augmented_config.yaml

# Alternative training scripts
python scripts/train_monte_carlo_ghm_time_augmented.py
python scripts/train_actor_critic_ghm_model1.py

# Visualization after training
python scripts/visualize_policy_value.py

# Hyperparameter search
python scripts/hyperparameter_search.py

# Benchmark RL vs numerical methods
python scripts/compare_rl_numerical.py
```

---

## Key Architecture Concepts

### Three Model-Based Solvers

| Solver | File | Description |
|---|---|---|
| **Pathwise Gradient** | `solvers/pathwise.py` | Backprop through differentiable SDE (recommended) |
| **Monte Carlo PG** | `solvers/monte_carlo.py` | REINFORCE with unlimited free simulation |
| **Deep Galerkin Method (DGM)** | `solvers/deep_galerkin.py` | Directly minimize HJB PDE residual |
| **Actor-Critic** | `solvers/actor_critic.py` | Model-based actor-critic |
| **Numerical VFI** | `solvers/numerical_vfi.py` | Classical value function iteration |

### Core Abstractions (all abstract base classes in `base.py`)

- **`ContinuousTimeDynamics`** (`dynamics/base.py`) — SDE interface: drift, diffusion, discount rate
- **`ControlSpec`** (`control/base.py`) — Action space with bounds and masking
- **`RewardFunction`** (`rewards/base.py`) — Step and terminal reward computation
- **`Policy`** (`policies/base.py`) — Stochastic/deterministic policy π(a|s)
- **`ValueFunction`** (`values/base.py`) — State value with gradient/Hessian autograd support
- **`Solver`** (`solvers/base.py`) — Optimization algorithm returning `SolverResult`

### The GHM Model (Two Controls)

The GHM equity model has **two controls** — this is a critical correctness requirement:

```python
action = policy(state)        # (a_L, a_E)
a_L = action[0]               # Dividend rate (continuous, lb=0)
a_E = action[1]               # Equity issuance (singular control)

# Reward function: net payout
reward = a_L * dt - (1 + λ) * a_E

# State dynamics (cash level c)
dc = (α + c*(r - λ - μ) - a_L) * dt + a_E + σ(c) * dW
```

Do **not** implement the GHM model with a single control (dividend only) — the equity issuance channel is essential to match the Bolton et al. paper.

### Typical Workflow

```python
from macro_rl.dynamics import GHMEquityDynamics, GHMEquityParams
from macro_rl.control import GHMControlSpec
from macro_rl.rewards import GHMRewardFunction
from macro_rl.simulation import TrajectorySimulator
from macro_rl.policies import GaussianPolicy
from macro_rl.solvers import PathwiseGradient

params   = GHMEquityParams()
dynamics = GHMEquityDynamics(params)
control  = GHMControlSpec()
reward   = GHMRewardFunction(discount_rate=0.02)

simulator = TrajectorySimulator(dynamics, control, reward)
policy    = GaussianPolicy(state_dim=1, action_dim=2)
solver    = PathwiseGradient(policy, simulator)

result = solver.solve(dynamics, control, reward, n_iterations=5000)
```

---

## Code Conventions

### Style
- **Formatter:** black (line length 88)
- **Linter:** flake8
- **Type checker:** mypy
- All public functions use Python type hints
- Dataclasses for configuration objects (e.g., `GHMEquityParams`)
- Abstract base classes (ABCs) for all core abstractions

### Naming
| Entity | Convention | Example |
|---|---|---|
| Classes | PascalCase | `GaussianPolicy`, `PathwiseGradient` |
| Functions/methods | snake_case | `compute_gradient()` |
| Private methods | `_` prefix | `_step()`, `_compute_loss()` |
| Constants | UPPER_CASE | `DEFAULT_LR` |

### PyTorch Patterns
- All tensors use `torch.Tensor`; no raw NumPy inside core modules
- Batch dimension is always the **first axis**: shape `(batch, ...)`
- Use `.to(device)` for GPU support — operations must be device-agnostic
- Gradients computed via `torch.autograd.grad()` (not `.backward()` in the solver)
- Differentiable simulation paths must not break the autograd graph

### Module Structure
- Every module directory has an `__init__.py` with explicit imports
- Major modules have a `README.md` with implementation guidance
- Concrete implementations go in dedicated files (e.g., `ghm_equity.py`), not in `base.py`

### Testing Conventions
- Tests live in `tests/` and follow `test_<module_name>_<topic>.py` naming
- Use `pytest.fixture` for shared test objects
- Use `@pytest.mark.parametrize` for multiple scenarios
- Integration tests are marked `@pytest.mark.integration`
- Do not import implementation details — test through public APIs

---

## Configuration Files

YAML configs in `configs/` control all experiment parameters:

```yaml
# Key config sections
dynamics:
  alpha: 0.1       # Cash flow rate
  mu: 0.05         # Growth rate
  r: 0.04          # Interest rate
  lambda: 0.02     # Issuance cost
  sigma: 0.2       # Volatility

training:
  lr: 1e-3
  batch_size: 256
  entropy_weight: 0.01
  n_iterations: 10000

model:
  policy_hidden_dims: [64, 64]
  value_hidden_dims: [64, 64]

solver:
  type: pathwise   # pathwise | monte_carlo | dgm | actor_critic

environment:
  T: 10.0          # Time horizon
  dt: 0.01         # Discretization step
```

**Key configs:**
- `time_augmented_config.yaml` — Recommended for finite-horizon problems
- `default_config.yaml` — Standard infinite-horizon GHM
- `quick_test_config.yaml` — Fast testing (reduced iterations)
- `actor_critic_time_augmented_config.yaml` — Actor-critic variant

---

## Validation

After training, verify solution quality using:

```python
from macro_rl.validation import HJBResidualChecker, AnalyticalComparison

# Check HJB equation residuals
checker = HJBResidualChecker(dynamics, value_fn)
residuals = checker.compute(state_grid)

# Compare to analytical solution (when available)
comp = AnalyticalComparison(analytical_fn, learned_fn)
errors = comp.compute(state_grid)
```

---

## Documentation

- **README.md** — Project overview, installation, architecture
- **TRAINING_GUIDE.md** — Training walkthrough and troubleshooting
- **`docs/source/`** — Sphinx docs (concepts, API, tutorials, contributing)
- **Module `README.md` files** — Implementation details per submodule
- **Online:** https://macrorl.readthedocs.io/

Build documentation locally:
```bash
cd docs
pip install -r requirements.txt
make html
# Open docs/build/html/index.html
```

---

## Git Workflow

- **Main branch:** `master`
- **Feature branches:** `claude/<description>-<session-id>` (for AI-generated work)
- No GitHub Actions CI is configured — run tests locally before committing
- ReadTheDocs auto-builds from `master` via `.readthedocs.yaml`
- Commit messages follow imperative mood: `Fix Monte Carlo training instability`

---

## Common Pitfalls

1. **Single control bug:** Never reduce the GHM model to one control. Both dividend rate (`a_L`) and equity issuance (`a_E`) are required.
2. **Autograd graph breakage:** Operations in `DifferentiableSimulator` must remain on the computation graph. Avoid `.detach()`, `.numpy()`, or in-place operations inside differentiable rollouts.
3. **Batch dimension:** Always ensure the first tensor dimension is the batch. Mismatched shapes cause silent broadcasting errors.
4. **Device mismatch:** Always call `.to(device)` on new tensors created inside model methods.
5. **Config loading:** Use `configs/quick_test_config.yaml` for rapid iteration — it significantly reduces training time.
6. **Monte Carlo instability:** Long trajectories can destabilize MC training. See `Fix Monte Carlo training instability with long trajectories` commit for reference fixes.
