# Training Scripts

## Overview

This directory contains end-to-end training scripts for different model-based RL algorithms.

## Available Scripts

### 1. `train_pathwise.py` - **RECOMMENDED**

Pathwise gradient method with reparameterization trick.

**Usage**:
```bash
python macro_rl/scripts/train_pathwise.py \
    --n_iterations 5000 \
    --n_trajectories 100 \
    --lr 1e-3 \
    --dt 0.01 \
    --T 5.0
```

**Pros**: Low variance, fast convergence
**Best for**: Most use cases

### 2. `train_monte_carlo.py`

Monte Carlo policy gradient (REINFORCE with known dynamics).

**Usage**:
```bash
python macro_rl/scripts/train_monte_carlo.py \
    --n_iterations 10000 \
    --n_trajectories 1000 \
    --lr 1e-3
```

**Pros**: Simple, works with any policy
**Best for**: Baselines, comparison

### 3. `train_dgm.py`

Deep Galerkin Method (direct HJB solution).

**Usage**:
```bash
python macro_rl/scripts/train_dgm.py \
    --n_iterations 10000 \
    --n_interior 1000 \
    --n_boundary 100 \
    --lr 1e-3
```

**Pros**: No simulation needed
**Best for**: Advanced use, comparison

## TODO for Implementation

- [ ] Implement `train_pathwise.py` complete pipeline
- [ ] Implement `train_monte_carlo.py` complete pipeline
- [ ] Implement `train_dgm.py` complete pipeline
- [ ] Add logging and checkpointing
- [ ] Add visualization of results
- [ ] Add model comparison script

## Expected Workflow

1. **Start here**: `train_pathwise.py` (fastest, most reliable)
2. **Compare**: Run `train_monte_carlo.py` to verify
3. **Validate**: Run `train_dgm.py` for HJB-based solution
4. **Analyze**: Compare all three approaches

## Output Structure

```
results/
├── pathwise/
│   ├── policy.pt
│   ├── metrics.json
│   └── plots/
├── monte_carlo/
│   ├── policy.pt
│   ├── value_fn.pt
│   ├── metrics.json
│   └── plots/
└── dgm/
    ├── value_fn.pt
    ├── policy.pt (extracted)
    ├── metrics.json
    └── plots/
```
