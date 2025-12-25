# Training Guide: Time-Augmented GHM Policy

This guide shows how to train the GHM equity model with time-augmented dynamics using the fixed codebase.

## Quick Start

### 1. Train with Default Configuration

```bash
python scripts/train_with_config.py --config configs/time_augmented_config.yaml
```

This will:
- Train a policy π(c, τ) where the agent observes both cash reserves and time-to-horizon
- Use Monte Carlo policy gradient with the **fixed** reward masking
- Set liquidation value to 0 (bankruptcy provides no reward)
- Save checkpoints to `checkpoints/ghm_time_augmented/`
- Log training to `logs/ghm_time_augmented/`

### 2. Monitor Training

**TensorBoard:**
```bash
tensorboard --logdir logs/ghm_time_augmented
```

Then open http://localhost:6006 in your browser.

**Key Metrics to Watch:**
- `eval/return_mean`: Should increase over time (target: 5-7)
- `eval/termination_rate`: Should be near 0% (no bankruptcies!)
- `policy/entropy`: Should decrease gradually (exploration → exploitation)
- `loss/policy`: Should converge
- `loss/baseline`: Should decrease

### 3. Visualizations

Visualizations are automatically saved during training:
```
checkpoints/ghm_time_augmented/visualizations/
  step_000500.png
  step_001000.png
  step_005000.png
  ...
```

**What to Look For:**
- **Dividend Policy π(c, τ)**: Should be 0-2 (NOT 8-10!)
- **Variation with τ**: Policies should differ across time horizons
- **Equity Issuance**: Should be low (mostly 0, with small amounts at low c)
- **Value Function**: Should increase monotonically with c

## Configuration Customization

### Override Parameters via Command Line

```bash
# Use different learning rate
python scripts/train_with_config.py --config configs/time_augmented_config.yaml --lr_policy 1e-3

# Change batch size and entropy
python scripts/train_with_config.py \
  --config configs/time_augmented_config.yaml \
  --n_trajectories 512 \
  --entropy_weight 0.02

# Change random seed
python scripts/train_with_config.py --config configs/time_augmented_config.yaml --seed 123

# Set experiment name
python scripts/train_with_config.py \
  --config configs/time_augmented_config.yaml \
  --experiment_name "ghm_fixed_bugs"
```

### Edit Configuration File

Open `configs/time_augmented_config.yaml` and modify:

**For faster training (less accurate):**
```yaml
training:
  n_trajectories: 128    # Smaller batch
  lr_policy: 5.0e-4      # Higher learning rate
```

**For more stable training (slower):**
```yaml
training:
  n_trajectories: 512    # Larger batch
  lr_policy: 1.0e-4      # Lower learning rate
  entropy_weight: 0.02   # More exploration
```

**For deeper networks:**
```yaml
policy:
  hidden_dims: [512, 512, 256]  # Bigger network

baseline:
  hidden_dims: [512, 512, 256]
```

## Resume from Checkpoint

```bash
python scripts/train_with_config.py \
  --config configs/time_augmented_config.yaml \
  --resume checkpoints/ghm_time_augmented/step_005000.pt
```

## Evaluation Only

After training, evaluate the best model:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/ghm_time_augmented/best_model.pt \
  --n_episodes 1000 \
  --save_trajectories
```

## Expected Results After Bug Fixes

### Before Fixes (OLD BEHAVIOR):
```
Step 7000:
  Dividend Policy: 8-10 (catastrophic!)
  Bankruptcy Rate: 80-90%
  Value Function: Irregular
  Return: 2-3 (poor)
```

### After Fixes (NEW BEHAVIOR):
```
Step 7000:
  Dividend Policy: 0.2-2.0 (reasonable!)
  Bankruptcy Rate: 0-5%
  Value Function: Smooth, monotonic in c
  Return: 5-7 (good)
```

## Troubleshooting

### Issue: Training is very slow

**Solution 1:** Use GPU
```bash
python scripts/train_with_config.py --config configs/time_augmented_config.yaml --device cuda
```

**Solution 2:** Reduce batch size
```bash
python scripts/train_with_config.py --config configs/time_augmented_config.yaml --n_trajectories 128
```

### Issue: Policy not learning (flat returns)

**Possible causes:**
1. Learning rate too low → increase `--lr_policy 5e-4`
2. Not enough exploration → increase `--entropy_weight 0.02`
3. Gradient clipping too aggressive → check `max_grad_norm`

**Debug:**
```bash
# Check if gradients are flowing
tensorboard --logdir logs/ghm_time_augmented
# Look at policy/entropy and loss/policy
```

### Issue: Still seeing bankruptcy behavior

**Check:**
1. Verify liquidation value is 0:
   ```bash
   python -c "from macro_rl.dynamics.ghm_equity import GHMEquityParams; p=GHMEquityParams(); print(p.liquidation_value)"
   # Should output: 0.0
   ```

2. Verify reward masking fix is applied:
   ```bash
   grep -A 5 "Check termination" macro_rl/simulation/trajectory.py
   # Should show: active = active & (~terminated) BEFORE masks[:, t] = ...
   ```

3. Check evaluation metrics:
   - `eval/termination_rate` should be < 10%
   - If high, may need more training or different hyperparameters

### Issue: Value function is negative

**This is normal!** The value function represents discounted future dividends minus costs. It can be negative if:
- Initial cash is very low
- Time horizon is short (near τ=0)
- Issuance costs are high

What matters is that it's **monotonically increasing in c** and **smooth**.

## File Structure After Training

```
GHM-RL/
├── configs/
│   └── time_augmented_config.yaml
├── logs/
│   └── ghm_time_augmented/
│       └── <timestamp>/
│           └── events.out.tfevents.*
├── checkpoints/
│   └── ghm_time_augmented/
│       ├── config.yaml
│       ├── best_model.pt
│       ├── final_model.pt
│       ├── step_001000.pt
│       ├── step_002000.pt
│       └── visualizations/
│           ├── step_000500.png
│           └── ...
└── scripts/
    └── train_with_config.py
```

## Advanced: Custom Configurations

### Standard (Non-Time-Augmented) Dynamics

Create `configs/standard_config.yaml`:
```yaml
dynamics:
  type: "standard"  # 1D state (c only)
  state_dim: 1

training:
  T: 100.0  # Longer horizon for infinite-horizon approximation
```

### Actor-Critic Instead of Monte Carlo

```yaml
solver:
  solver_type: "actor_critic"
  critic_loss: "temporal_difference"
  actor_loss: "policy_gradient"

training:
  lr: 1.0e-3  # Combined learning rate
```

## Next Steps

1. **Train the model** with the fixed configuration
2. **Monitor TensorBoard** to verify learning
3. **Check visualizations** at step 500, 1000, 5000
4. **Compare with old results** to see the improvement
5. **Tune hyperparameters** if needed

## Questions?

See:
- `IMPLEMENTATION_SUMMARY.md` - Details on bug fixes
- `QUICK_FIX_GUIDE.md` - What was fixed and why
- `BUG_ANALYSIS.md` - Deep dive into the bugs
- `CORRECTED_BUG_ANALYSIS.md` - Full explanation with user feedback

Good luck with training! 🚀
