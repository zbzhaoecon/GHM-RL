# MacroRL: Model-Based RL for Continuous-Time Finance

[![Documentation Status](https://readthedocs.org/projects/ghm-rl/badge/?version=latest)](https://macrorl.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

A Python library for solving continuous-time corporate finance models using **model-based reinforcement learning**. This project implements the GHM (Gârleanu-Hackbarth-Morellec) equity management model using known dynamics to achieve superior sample efficiency and convergence.

📖 **[Full Documentation](https://macrorl.readthedocs.io/)** | 🚀 **[Quick Start](#quick-start)** | 📚 **[Tutorials](https://macrorl.readthedocs.io/en/latest/tutorials/index.html)** | 📖 **[API Reference](https://macrorl.readthedocs.io/en/latest/api/index.html)**

## What Changed: From Model-Free to Model-Based

### Why Pivot to Model-Based?

The GHM model gives us **exact knowledge of the dynamics**:
```
dc = μ(c)dt + σ(c)dW
```

where drift and diffusion are **known closed-form functions**. Model-free RL (PPO, SAC) ignores this and tries to learn optimal behavior purely from trial-and-error. Model-based RL exploits known dynamics to:

1. **Simulate freely**: Generate unlimited trajectories without environment interaction
2. **Explore completely**: Sample any initial state, not just reachable ones
3. **Reduce variance**: Use exact gradients (pathwise) instead of REINFORCE estimates
4. **Validate rigorously**: Check solutions against HJB equation

### Three Model-Based Approaches

| Method | Gradient Type | Key Idea | Best For |
|--------|---------------|----------|----------|
| **Pathwise Gradient** (Recommended) | Exact via chain rule | Backprop through differentiable simulation | Most use cases |
| Monte Carlo PG | REINFORCE with unlimited samples | Free simulation reduces variance | Baselines, comparison |
| Deep Galerkin Method | PDE residual minimization | Directly solve HJB equation | Advanced, validation |

---

## Architecture Overview

```
macro_rl/
├── core/                  # Foundational abstractions
│   ├── state_space.py     # State space representation
│   └── params.py          # Parameter management
│
├── dynamics/              # Continuous-time models (UNCHANGED - verified correct)
│   ├── base.py            # ContinuousTimeDynamics interface
│   └── ghm_equity.py      # GHM 1D model (drift, diffusion, parameters)
│
├── simulation/            # NEW: SDE simulation engines
│   ├── sde.py             # Numerical integration (Euler-Maruyama)
│   ├── trajectory.py      # Trajectory generation for Monte Carlo
│   └── differentiable.py  # Differentiable simulation for pathwise gradients
│
├── control/               # NEW: Control specifications (TWO controls, not one!)
│   ├── base.py            # ControlSpec interface
│   ├── ghm_control.py     # GHM two-control spec (dividend + equity issuance)
│   └── masking.py         # Action masking utilities
│
├── rewards/               # NEW: Objective functions
│   ├── base.py            # RewardFunction interface
│   ├── ghm_rewards.py     # GHM net payout: dividends - dilution cost
│   └── terminal.py        # Terminal value specifications
│
├── policies/              # NEW: Policy representations
│   ├── base.py            # Policy interface
│   ├── neural.py          # Gaussian and deterministic policies
│   ├── barrier.py         # Barrier/threshold policies (baselines)
│   └── tabular.py         # Grid-based policies (debugging)
│
├── values/                # NEW: Value function representations
│   ├── base.py            # ValueFunction interface
│   ├── neural.py          # Neural value networks (with autograd support)
│   └── analytical.py      # Analytical solutions (when known)
│
├── solvers/               # NEW: Model-based RL algorithms
│   ├── base.py            # Solver interface
│   ├── pathwise.py        # Pathwise gradient (RECOMMENDED)
│   ├── monte_carlo.py     # Monte Carlo policy gradient
│   ├── deep_galerkin.py   # Deep Galerkin Method (HJB-based)
│   └── actor_critic.py    # Model-based actor-critic
│
├── validation/            # NEW: Solution validation
│   ├── hjb_residual.py    # HJB equation residual computation
│   ├── boundary_conditions.py  # Smooth pasting, etc.
│   └── analytical_comparison.py  # Compare with known solutions
│
├── utils/                 # Utilities
│   ├── autograd.py        # Gradient/Hessian computation
│   ├── plotting.py        # Visualization
│   └── logging.py         # Training logs
│
├── envs/                  # Gymnasium environments (for model-free baselines)
│   └── ghm_equity.py      # (To be rewritten with two controls)
│
└── scripts/               # Training scripts
    ├── train_pathwise.py      # Pathwise gradient training (START HERE)
    ├── train_monte_carlo.py   # Monte Carlo training
    └── train_dgm.py           # Deep Galerkin training
```

---

## Critical Fix: Two Controls, Not One

### Previous (Wrong) Formulation:
```python
# Single control: dividend rate only
action = policy(state)  # Scalar
reward = action  # Just dividend
```

**Problems**:
- No equity issuance mechanism
- Can't handle barrier/recapitalization
- Doesn't match Bolton et al. paper

### Correct Formulation:
```python
# Two controls: dividend + equity issuance
action = policy(state)  # (a_L, a_E)
a_L = action[0]  # Dividend rate (continuous)
a_E = action[1]  # Equity issuance (singular)

# Net payout to shareholders
reward = a_L * dt - (1 + λ) * a_E
#        ^^^^^^^^^   ^^^^^^^^^^^^^^^
#        dividend    dilution cost

# State evolution
dc = (α + c(r-λ-μ) - a_L) * dt + a_E + σ(c) * dW
     \_________________/          \____/
         drift with div         issuance
```

**Key insight**: Shareholders care about **net payout** = dividends - equity dilution cost.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/zbzhaoecon/GHM-RL.git
cd GHM-RL
pip install -e .
```


## Key Design Principles

### 1. Exploit Known Dynamics
Unlike model-free RL, we **know** the dynamics. This enables:
- Free simulation (no environment interaction)
- Exact gradients (pathwise derivatives)
- Direct PDE validation (HJB residual)

### 2. Separation of Concerns
```
Dynamics → Simulation → Policies/Values → Solvers → Validation
```
Each component is independently testable and reusable.

### 3. Batched Operations
All operations support `(batch, ...)` dimensions for GPU efficiency and Monte Carlo estimation.

### 4. PyTorch Throughout
- Automatic differentiation for gradients/Hessians
- GPU acceleration
- Consistent interface

---



## Documentation

Complete documentation is available at **[https://ghm-rl.readthedocs.io](https://ghm-rl.readthedocs.io)**

### Documentation Sections

- **[Getting Started](https://ghm-rl.readthedocs.io/en/latest/getting_started.html)**: Installation and quick start guide
- **[Tutorials](https://ghm-rl.readthedocs.io/en/latest/tutorials/index.html)**: Step-by-step tutorials
- **[API Reference](https://ghm-rl.readthedocs.io/en/latest/api/index.html)**: Complete API documentation
- **[Examples](https://ghm-rl.readthedocs.io/en/latest/examples.html)**: Working examples and use cases
- **[Core Concepts](https://ghm-rl.readthedocs.io/en/latest/concepts.html)**: Theoretical foundations

### Building Documentation Locally

```bash
cd docs
pip install -r requirements.txt
make html
```

View the built documentation:

```bash
# Linux/Mac
open build/html/index.html

# Or use Python's built-in server
cd build/html && python -m http.server
```

## Contributing

We welcome contributions! Please see the **[Contributing Guide](https://ghm-rl.readthedocs.io/en/latest/contributing.html)** for details on:

- Setting up development environment
- Code style and testing guidelines
- Submitting pull requests
- Documentation standards

---

## License

MIT

---

## Contact

For questions about implementation or research collaboration, please open an issue or contact the maintainers.
