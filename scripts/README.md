# Training Scripts

This directory contains scripts for training and evaluating RL agents on the GHM equity model.

## Quick Start

### 1. Debug Infrastructure (Pendulum)

First, verify that SAC + Stable-Baselines3 works correctly:

```bash
# Train SAC on Pendulum-v1 (5-10 minutes)
python scripts/train_cartpole.py --timesteps 20000

# Verify training succeeded
python scripts/verify_pendulum.py
```

**Expected Results:**
- Reward improves from ~-1500 to < -500 (ideally < -200)
- Model saved to `models/pendulum_debug/final_model.zip`
- TensorBoard logs in `models/pendulum_debug/tensorboard/`

**Monitor Training:**
```bash
tensorboard --logdir models/pendulum_debug/tensorboard
```

### 2. Debug GHM Environment

Test the environment manually before training:

```bash
python scripts/debug_env.py
```

**Expected Output:**
- All 8 tests pass ✓
- Environment creates successfully
- SB3 env_checker passes
- Dynamics values match expected calculations

### 3. Train on GHM

Once infrastructure is validated:

```bash
# Full training (20-30 minutes)
python scripts/train_ghm.py --timesteps 100000

# Quick test (2-3 minutes)
python scripts/train_ghm.py --timesteps 10000 --output models/ghm_quick
```

**Monitor Training:**
```bash
tensorboard --logdir models/ghm_equity/tensorboard
```

### 4. Evaluate Policy

Analyze the learned policy:

```bash
python scripts/evaluate.py --model models/ghm_equity/final_model
```

**Outputs:**
- `evaluation.png` - Policy and value function plots
- `results.npz` - Raw data (c_grid, actions, values)

**Expected Results:**
- Policy shows threshold behavior: low action when c < c*, higher when c > c*
- Value function is increasing in c
- Estimated c* ≈ 0.3-0.8 (depends on parameters)

## Script Options

### train_cartpole.py

```bash
python scripts/train_cartpole.py \
    --timesteps 20000 \
    --output models/pendulum_debug \
    --seed 42
```

### debug_env.py

No arguments - runs all tests automatically.

### train_ghm.py

```bash
python scripts/train_ghm.py \
    --timesteps 100000 \
    --output models/ghm_equity \
    --n-envs 4 \
    --seed 42 \
    --eval-freq 5000
```

### evaluate.py

```bash
python scripts/evaluate.py \
    --model models/ghm_equity/final_model \
    --n-points 100 \
    --n-episodes 20 \
    --output models/ghm_equity
```

### verify_pendulum.py

```bash
python scripts/verify_pendulum.py \
    --model models/pendulum_debug/final_model \
    --episodes 10
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'gymnasium'"

```bash
pip install gymnasium stable-baselines3
```

### Issue: Pendulum training fails or doesn't improve

- Check TensorBoard logs for learning curves
- Verify reward is improving (should go from ~-1500 toward -200)
- Try longer training: `--timesteps 50000`

### Issue: GHM environment fails debug tests

- Check dynamics values in debug output
- Verify state space bounds are correct
- Ensure actions are being applied (reward should not always be 0)

### Issue: GHM training doesn't converge

- Increase timesteps: `--timesteps 200000`
- Check for instant liquidations (adjust initial state range)
- Verify reward is non-zero (actions being applied)
- Tune `liquidation_penalty` (try 2.0, 5.0, 10.0)

### Issue: No threshold behavior in learned policy

- Train longer
- Increase liquidation penalty
- Check discount factor γ (should be ~0.98-0.99)
- Verify reward structure is correct

## Validation Checklist

- [ ] Pendulum trains successfully (reward > -500)
- [ ] Model saves and loads without errors
- [ ] TensorBoard logs are created
- [ ] `debug_env.py` all tests pass
- [ ] GHM training shows improving reward
- [ ] Learned policy is not constant (shows variation with c)
- [ ] Value function is generally increasing
- [ ] Can identify approximate threshold c*

## Performance Benchmarks

### Pendulum-v1 (20k timesteps)
- Random policy: ~-1500
- Good policy: ~-200
- Expected after training: -200 to -500

### GHM Equity (100k timesteps)
- Episode length: ~1000 steps (often truncated)
- Typical reward: 20-50 (depends on policy)
- Liquidation rate: Should decrease during training

## Next Steps

After successful training:

1. **Sensitivity Analysis**: Test different parameters (α, σ, r, etc.)
2. **Longer Training**: 500k-1M timesteps for convergence
3. **Hyperparameter Tuning**: Tune learning rate, buffer size, etc.
4. **Compare to Theory**: Compare c* to analytical solution (if available)
5. **Multi-dimensional**: Extend to 2D GHM model

## File Outputs

```
models/
├── pendulum_debug/
│   ├── final_model.zip          # Trained model
│   ├── best/
│   │   └── best_model.zip       # Best model during eval
│   ├── logs/
│   │   └── evaluations.npz      # Evaluation results
│   └── tensorboard/
│       └── SAC_*/               # TensorBoard logs
└── ghm_equity/
    ├── final_model.zip
    ├── best/
    ├── checkpoints/
    │   ├── sac_ghm_10000_steps.zip
    │   ├── sac_ghm_20000_steps.zip
    │   └── ...
    ├── logs/
    ├── tensorboard/
    ├── evaluation.png           # Policy/value plots
    └── results.npz              # Evaluation data
```
