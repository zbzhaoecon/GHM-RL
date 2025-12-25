# GHM-RL Configuration System

This directory contains configuration files for training GHM-RL models. The configuration system makes it easy to test different parameters, dynamics, and training setups without modifying code.

## Quick Start

### Basic Usage

Train with default configuration:
```bash
python scripts/train_with_config.py
```

Train with a custom configuration:
```bash
python scripts/train_with_config.py --config configs/actor_critic_config.yaml
```

Override specific parameters:
```bash
python scripts/train_with_config.py --config configs/default_config.yaml \
    --lr_policy 1e-3 \
    --n_iterations 20000 \
    --seed 456
```

Resume from checkpoint:
```bash
python scripts/train_with_config.py \
    --config configs/my_config.yaml \
    --resume checkpoints/ghm_rl/step_5000.pt
```

## Available Configurations

### `default_config.yaml`
Default configuration for Monte Carlo Policy Gradient training with standard GHM parameters from the paper.

### `actor_critic_config.yaml`
Actor-Critic solver with HJB regularization. Uses model-based gradients and shared architecture.

**Key differences:**
- `solver.solver_type: actor_critic`
- `solver.critic_loss: mc+hjb` (combines Monte Carlo and HJB losses)
- `network.shared_layers: 1` (shared first layer between actor/critic)
- Fewer trajectories needed (256 vs 500)

### `high_volatility_config.yaml`
Tests policy learning under high market uncertainty (2x volatility).

**Key parameters:**
- `dynamics.sigma_A: 0.50` (doubled from 0.25)
- `dynamics.sigma_X: 0.24` (doubled from 0.12)
- Larger networks (3 layers)
- More trajectories (1000) for variance reduction
- Higher exploration (`entropy_weight: 0.1`)

### `low_issuance_cost_config.yaml`
Tests behavior when equity financing is cheaper.

**Key parameters:**
- `dynamics.p: 1.02` (reduced from 1.06)
- `dynamics.phi: 0.0005` (reduced from 0.002)
- `action_space.equity_max: 3.0` (increased from 2.0)
- Lower issuance threshold

### `quick_test_config.yaml`
Fast configuration for debugging and code testing.

**Key parameters:**
- `training.n_iterations: 1000` (reduced from 10000)
- `training.n_trajectories: 100` (reduced from 500)
- Smaller networks (32x32)
- More frequent logging

## Configuration Structure

Configuration files are organized into sections:

### `dynamics`
GHM model dynamics parameters (cash flow, volatility, costs)

```yaml
dynamics:
  alpha: 0.18        # Mean cash flow rate
  mu: 0.01          # Growth rate
  r: 0.03           # Interest rate
  sigma_A: 0.25     # Permanent volatility
  sigma_X: 0.12     # Transitory volatility
  p: 1.06           # Equity issuance cost
  omega: 0.55       # Liquidation recovery
```

### `action_space`
Bounds for dividend and equity issuance actions

```yaml
action_space:
  dividend_min: 0.0
  dividend_max: 2.0
  equity_min: 0.0
  equity_max: 2.0
```

### `network`
Neural network architecture

```yaml
network:
  policy_hidden: [64, 64]      # Monte Carlo policy network
  value_hidden: [64, 64]       # Monte Carlo value network
  hidden_dims: [256, 256]      # Actor-Critic architecture
  shared_layers: 0             # Shared layers in actor-critic
```

### `training`
Training loop parameters

```yaml
training:
  n_iterations: 10000
  n_trajectories: 500
  lr_policy: 0.0003
  lr_baseline: 0.001
  entropy_weight: 0.05
  max_grad_norm: 0.5
```

### `solver`
Solver-specific configuration

```yaml
solver:
  solver_type: monte_carlo     # or "actor_critic"
  critic_loss: mc+hjb          # for actor-critic
  actor_loss: pathwise         # for actor-critic
  hjb_weight: 0.1             # HJB regularization weight
```

### `logging`
TensorBoard and checkpoint settings

```yaml
logging:
  log_dir: runs/ghm_rl
  ckpt_dir: checkpoints/ghm_rl
  log_freq: 100
  eval_freq: 1000
  ckpt_freq: 5000
```

### `misc`
General settings

```yaml
misc:
  seed: 123
  device: cpu               # or "cuda"
  experiment_name: null     # Optional experiment name
```

## Creating Custom Configurations

### From Scratch

1. Copy `default_config.yaml`:
   ```bash
   cp configs/default_config.yaml configs/my_config.yaml
   ```

2. Edit the parameters you want to change

3. Train with your config:
   ```bash
   python scripts/train_with_config.py --config configs/my_config.yaml
   ```

### Programmatically

```python
from macro_rl.config import ConfigManager

# Load base config
config = ConfigManager.from_file("configs/default_config.yaml")

# Modify parameters
config.update({
    'dynamics.sigma_A': 0.30,
    'training.lr_policy': 1e-3,
    'network.policy_hidden': [128, 128, 64]
})

# Save new config
config.save("configs/custom_config.yaml")
```

## Command-Line Overrides

Override any parameter from the command line:

```bash
python scripts/train_with_config.py \
    --config configs/default_config.yaml \
    --lr_policy 1e-3 \              # Override learning rate
    --n_iterations 20000 \          # Override iterations
    --n_trajectories 1000 \         # Override trajectories
    --entropy_weight 0.1 \          # Override entropy weight
    --seed 456 \                    # Override seed
    --device cuda \                 # Use GPU
    --experiment_name my_exp        # Name the experiment
```

## Parameter Tuning Guide

### For Faster Convergence
- Increase `training.lr_policy` and `training.lr_baseline`
- Increase `training.n_trajectories`
- Add more network capacity (larger `hidden_dims`)

### For Stability
- Decrease learning rates
- Increase `training.max_grad_norm`
- Enable `training.advantage_normalization`

### For Exploration
- Increase `training.entropy_weight`
- Use larger `network.log_std_bounds`

### For Different Dynamics
- Modify `dynamics.*` parameters
- Adjust `action_space.*` bounds accordingly
- Consider larger networks for complex dynamics

## Monitoring Training

Start TensorBoard to monitor training:

```bash
tensorboard --logdir runs/
```

Then open http://localhost:6006 in your browser.

## Troubleshooting

### Policy collapses to zero actions
- Increase `training.entropy_weight`
- Decrease learning rates
- Check action bounds in `action_space`

### Training is unstable
- Decrease learning rates
- Increase `training.max_grad_norm`
- Use advantage normalization

### Too slow to train
- Decrease `training.n_trajectories`
- Use smaller networks
- Try actor-critic solver (more sample efficient)
- Use `quick_test_config.yaml` for debugging
