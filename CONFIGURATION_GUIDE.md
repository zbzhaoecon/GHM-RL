# GHM-RL Configuration System - Quick Start Guide

## Overview

The configuration system allows you to control **all** training parameters through YAML/JSON files, making it easy to test different setups without modifying code.

## Quick Usage

### 1. Basic Training

```bash
# Train with default configuration
python scripts/train_with_config.py --config configs/default_config.yaml

# Use GPU
python scripts/train_with_config.py --config configs/default_config.yaml --device cuda
```

### 2. Override Parameters

```bash
# Change learning rate and seed
python scripts/train_with_config.py \
    --config configs/default_config.yaml \
    --lr_policy 1e-3 \
    --seed 456 \
    --device cuda

# Run with more trajectories
python scripts/train_with_config.py \
    --config configs/default_config.yaml \
    --n_trajectories 1000 \
    --n_iterations 20000
```

### 3. Quick Testing

```bash
# Fast test run (1K iterations, small network)
python scripts/train_with_config.py --config configs/quick_test_config.yaml
```

### 4. Different Scenarios

```bash
# Try actor-critic solver
python scripts/train_with_config.py --config configs/actor_critic_config.yaml

# Test high volatility (2x sigma)
python scripts/train_with_config.py --config configs/high_volatility_config.yaml

# Test low issuance costs
python scripts/train_with_config.py --config configs/low_issuance_cost_config.yaml
```

## Available Configuration Files

| File | Description |
|------|-------------|
| `default_config.yaml` | Standard Monte Carlo training |
| `actor_critic_config.yaml` | Actor-Critic with HJB regularization |
| `high_volatility_config.yaml` | 2x volatility scenario |
| `low_issuance_cost_config.yaml` | Cheaper equity financing |
| `quick_test_config.yaml` | Fast debugging config |

## Key Parameters You Can Configure

### Dynamics (GHM Model)
```yaml
dynamics:
  alpha: 0.18        # Cash flow rate
  mu: 0.01          # Growth rate
  r: 0.03           # Interest rate
  sigma_A: 0.25     # Permanent volatility
  sigma_X: 0.12     # Transitory volatility
  p: 1.06           # Equity issuance cost
  omega: 0.55       # Liquidation recovery
```

### Action Bounds
```yaml
action_space:
  dividend_min: 0.0
  dividend_max: 2.0
  equity_min: 0.0
  equity_max: 2.0
```

### Network Architecture
```yaml
network:
  policy_hidden: [64, 64]      # Policy network layers
  value_hidden: [64, 64]       # Value network layers
  hidden_dims: [256, 256]      # Actor-critic layers
```

### Training Parameters
```yaml
training:
  n_iterations: 10000
  n_trajectories: 500
  lr_policy: 0.0003
  lr_baseline: 0.001
  entropy_weight: 0.05
  max_grad_norm: 0.5
```

### Solver Type
```yaml
solver:
  solver_type: monte_carlo    # or "actor_critic"
  critic_loss: mc+hjb         # for actor-critic
  hjb_weight: 0.1
```

## Creating Custom Configurations

### Method 1: Copy and Edit

```bash
# Copy default config
cp configs/default_config.yaml configs/my_config.yaml

# Edit with your favorite editor
nano configs/my_config.yaml

# Train with it
python scripts/train_with_config.py --config configs/my_config.yaml
```

### Method 2: Programmatic

```python
from macro_rl.config import load_config

# Load base config
config = load_config("configs/default_config.yaml")

# Modify parameters
config.update({
    'dynamics.sigma_A': 0.30,           # Increase volatility
    'training.lr_policy': 1e-3,         # Higher learning rate
    'network.policy_hidden': [128, 128, 64],  # Deeper network
    'training.n_trajectories': 1000,     # More samples
})

# Save
config.save("configs/my_config.yaml")
```

## Command-Line Overrides

You can override any parameter from the command line:

```bash
python scripts/train_with_config.py \
    --config configs/default_config.yaml \
    --lr_policy 1e-3 \              # Override policy LR
    --n_iterations 20000 \          # Override iterations
    --entropy_weight 0.1 \          # Override entropy weight
    --seed 789 \                    # Override seed
    --device cuda \                 # Use GPU
    --experiment_name my_experiment # Name your run
```

Available overrides:
- `--lr_policy`, `--lr_baseline`, `--lr`
- `--n_iterations`, `--n_trajectories`
- `--entropy_weight`
- `--seed`, `--device`
- `--resume` (path to checkpoint)
- `--experiment_name`

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Open browser to http://localhost:6006
```

### Checkpoints

Checkpoints are saved to `checkpoints/ghm_rl/` (or your configured directory):
- `step_XXXXXX.pt` - Periodic checkpoints
- `best_model.pt` - Best model based on eval return
- `final_model.pt` - Final model after training

### Resume Training

```bash
python scripts/train_with_config.py \
    --config configs/my_config.yaml \
    --resume checkpoints/ghm_rl/step_5000.pt
```

## Common Workflows

### 1. Debugging New Code

```bash
# Use quick_test_config for fast iterations
python scripts/train_with_config.py --config configs/quick_test_config.yaml
```

### 2. Hyperparameter Search

```bash
# Try different learning rates
for lr in 1e-4 3e-4 1e-3; do
    python scripts/train_with_config.py \
        --config configs/default_config.yaml \
        --lr_policy $lr \
        --experiment_name lr_${lr}
done
```

### 3. Testing Different Dynamics

```bash
# Test different volatility levels
for sigma in 0.15 0.25 0.35; do
    # Create custom config
    python -c "
from macro_rl.config import load_config
config = load_config('configs/default_config.yaml')
config.update({'dynamics.sigma_A': $sigma})
config.save('configs/sigma_${sigma}.yaml')
"
    # Train
    python scripts/train_with_config.py \
        --config configs/sigma_${sigma}.yaml \
        --experiment_name sigma_${sigma}
done
```

### 4. Compare Solvers

```bash
# Monte Carlo
python scripts/train_with_config.py \
    --config configs/default_config.yaml \
    --experiment_name monte_carlo

# Actor-Critic
python scripts/train_with_config.py \
    --config configs/actor_critic_config.yaml \
    --experiment_name actor_critic
```

## Troubleshooting

### Training is Unstable

Try:
- Decrease learning rates: `--lr_policy 1e-4`
- Increase gradient clipping: modify `training.max_grad_norm` in config
- Use advantage normalization (already enabled by default)

### Policy Collapses to Zero

Try:
- Increase entropy weight: `--entropy_weight 0.1`
- Check action bounds in config
- Verify reward function is correct

### Training Too Slow

Try:
- Use `quick_test_config.yaml` for debugging
- Decrease `n_trajectories`
- Use smaller networks
- Try actor-critic solver (more sample efficient)

### Out of Memory (GPU)

Try:
- Decrease `n_trajectories`
- Use smaller networks
- Use CPU: `--device cpu`

## Examples

### Example 1: Test Higher Volatility

```bash
python scripts/train_with_config.py \
    --config configs/default_config.yaml \
    --experiment_name high_vol_test \
    --device cuda
```

Edit `configs/default_config.yaml` and change:
```yaml
dynamics:
  sigma_A: 0.40  # Increase from 0.25
  sigma_X: 0.20  # Increase from 0.12
```

### Example 2: Larger Network

```bash
python -c "
from macro_rl.config import load_config
config = load_config('configs/default_config.yaml')
config.update({
    'network.policy_hidden': [128, 128, 64],
    'network.value_hidden': [128, 128, 64],
})
config.save('configs/large_network.yaml')
"

python scripts/train_with_config.py --config configs/large_network.yaml
```

### Example 3: Quick Sensitivity Test

```bash
# Test different entropy weights
for ent in 0.01 0.05 0.1; do
    python scripts/train_with_config.py \
        --config configs/quick_test_config.yaml \
        --entropy_weight $ent \
        --experiment_name entropy_${ent} &
done
wait
```

## Full Documentation

See `configs/README.md` for complete documentation of all configuration sections and parameters.

## Getting Help

If you encounter issues:
1. Check the configuration file is valid YAML
2. Verify all required sections are present
3. Check the logs in `runs/` directory
4. Use `--config configs/quick_test_config.yaml` to quickly test

Happy training! 🚀
