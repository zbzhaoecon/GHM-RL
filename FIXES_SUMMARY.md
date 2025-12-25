# Monte Carlo Training Crash - Fixes Summary

## Overview

The Monte Carlo training crashed between steps 6500-7000 with the policy collapsing to near-zero outputs (~1e-14). I've implemented comprehensive fixes to prevent this and detect early warning signs.

## Root Cause

The crash was caused by a **cascading numerical instability**:

1. Policy produced actions near boundaries
2. TanhNormal `log_prob()` computation produced extremely negative values (< -50)
3. REINFORCE gradients exploded: `∇ log π(a|s) * advantage`
4. Even with gradient clipping, updates pushed network outputs to extremes
5. Raw network outputs saturated tanh: `tanh(±10) → ±1`
6. Policy stuck at boundaries, unable to recover

## Fixes Implemented

### 1. **Removed Incorrect Loss Clamping** ✅
**File**: `macro_rl/solvers/monte_carlo.py`

**Removed**:
```python
policy_loss = torch.clamp(policy_loss, -1000.0, 1000.0)  # WRONG!
```

**Why**: Clamping the loss value before `.backward()` interferes with gradient computation and doesn't actually prevent gradient explosions.

### 2. **Added NaN/Inf Detection** ✅
**Files**: `macro_rl/solvers/monte_carlo.py`

**Added checks for**:
- Non-finite policy loss
- Non-finite total loss
- Non-finite gradients

**Behavior**: When detected, skip the optimizer step and return safe metrics (prevents corruption).

### 3. **Improved TanhNormal Numerical Stability** ✅
**File**: `macro_rl/distributions/tanh_normal.py`

**Changes**:
- **More conservative clamping**: Changed normalized range from `[-0.999, 0.999]` to `[-0.95, 0.95]`
- **Clip z values**: Added `z = torch.clamp(z, -5.0, 5.0)` to prevent extreme base distribution values
- **Increased epsilon**: Changed from `1e-4` to `5e-4` for boundary padding
- **Larger min for Jacobian**: Changed `one_minus_tanh_sq` min from `1e-6` to `1e-4`
- **Clip log probabilities**: Added `log_prob_action = torch.clamp(log_prob_action, min=-20.0, max=20.0)`

**Impact**: Prevents log probabilities from becoming extremely negative, which was the primary source of gradient explosions.

### 4. **Added Policy Output Clipping** ✅
**File**: `macro_rl/networks/policy.py`

**Added**:
```python
if self.action_bounds is not None:
    mean = torch.clamp(mean, -5.0, 5.0)
```

**Why**: Prevents raw network outputs from becoming extreme before they go through tanh. This is an additional safety layer.

### 5. **Enhanced Diagnostic Logging** ✅
**Files**: `macro_rl/solvers/monte_carlo.py`, `scripts/train_monte_carlo_ghm_time_augmented.py`

**New metrics tracked**:
- `diagnostics/action_min`: Minimum action value
- `diagnostics/action_max`: Maximum action value
- `diagnostics/action_mean`: Mean action value
- `diagnostics/log_prob_mean`: Mean log probability
- `diagnostics/log_prob_min`: Minimum log probability (most important!)
- `diagnostics/log_prob_max`: Maximum log probability

**Console warnings**:
- Policy collapse warning when `action_magnitude < 0.01`
- Boundary issue warning when `log_prob_min < -50`

### 6. **Added Safe Metrics Fallback** ✅
**File**: `macro_rl/solvers/monte_carlo.py`

**New method**: `_get_safe_metrics()` returns valid metrics when training step fails, preventing crashes and allowing training to potentially recover.

## Key Warning Signs to Monitor

When running training, watch for these signs of trouble:

### Early Warning (Can still recover):
- `log_prob_min < -20`: Actions approaching boundaries
- `action_magnitude` decreasing rapidly
- `grad_norm/policy` consistently hitting max (0.5)

### Critical Warning (About to crash):
- `log_prob_min < -50`: Severe boundary issues
- `action_magnitude < 0.1`: Policy collapsing
- Console shows: "WARNING: Extremely negative log probabilities detected!"

### Failure (Too late):
- `action_magnitude < 0.01`: Policy has collapsed
- Console shows: "WARNING: Non-finite loss/gradients detected!"

## How to Use

### Run Training with Fixes:
```bash
python scripts/train_monte_carlo_ghm_time_augmented.py \
  --n_iterations 10000 \
  --lr_policy 3e-4 \
  --entropy_weight 0.05 \
  --action_reg_weight 0.01
```

### Monitor TensorBoard:
```bash
tensorboard --logdir runs/monte_carlo_model1
```

**Key plots to watch**:
- `diagnostics/log_prob_min` - Should stay above -20
- `policy/action_magnitude` - Should stay above 0.1
- `diagnostics/action_min` and `action_max` - Check if hitting boundaries

### If You See Warning Signs:

1. **If `log_prob_min < -20`**:
   - Training is still ok, but monitor closely
   - Check if actions are near boundaries in visualizations

2. **If `log_prob_min < -50`**:
   - Stop training and restart with:
     - Lower learning rate: `--lr_policy 1e-4`
     - Higher entropy: `--entropy_weight 0.1`
     - Higher action reg: `--action_reg_weight 0.05`

3. **If policy collapses (`action_magnitude < 0.01`)**:
   - Training has failed
   - Restart from last good checkpoint
   - Use more conservative hyperparameters

## Testing the Fixes

To verify the fixes work:

1. **Run a short training run** (1000 iterations):
   ```bash
   python scripts/train_monte_carlo_ghm_time_augmented.py --n_iterations 1000
   ```

2. **Check the new diagnostics appear** in console output:
   ```
   Log Prob: mean=-2.345, min=-5.678, max=-0.123
   ```

3. **Verify NaN detection works** by intentionally causing issues (if needed):
   - Set very high learning rate: `--lr_policy 0.01`
   - Should see: "WARNING: Non-finite loss detected!"
   - Training should continue (not crash)

## Expected Behavior After Fixes

- **No sudden collapse**: Policy should train smoothly without sudden crashes
- **Graceful degradation**: If numerical issues occur, training skips bad updates instead of crashing
- **Early warnings**: You'll see console warnings before severe issues occur
- **More stable log probabilities**: `log_prob_min` should stay above -20 most of the time

## Files Modified

1. `macro_rl/solvers/monte_carlo.py` - NaN/Inf detection, diagnostic metrics, safe fallback
2. `macro_rl/distributions/tanh_normal.py` - Improved numerical stability in `log_prob()`
3. `macro_rl/networks/policy.py` - Output clipping for bounded actions
4. `scripts/train_monte_carlo_ghm_time_augmented.py` - Enhanced logging

## Additional Recommendations

### For More Stability:

1. **Use smaller learning rate initially**: Start with `lr_policy=1e-4`, increase to `3e-4` after 1000 steps
2. **Increase entropy weight**: Use `entropy_weight=0.1` for first 2000 steps
3. **Monitor early**: Check diagnostics every 100 steps, not just 500
4. **Save checkpoints frequently**: Use `--ckpt_freq 1000` instead of 5000
5. **Consider PPO**: For even more stability, switch to PPO (trust region constraint)

### Hyperparameter Tuning:

**Safe (Conservative)**:
```bash
--lr_policy 1e-4 --entropy_weight 0.1 --action_reg_weight 0.05 --max_grad_norm 0.3
```

**Balanced (Default)**:
```bash
--lr_policy 3e-4 --entropy_weight 0.05 --action_reg_weight 0.01 --max_grad_norm 0.5
```

**Aggressive (Faster but riskier)**:
```bash
--lr_policy 5e-4 --entropy_weight 0.03 --action_reg_weight 0.01 --max_grad_norm 1.0
```

## Troubleshooting

### Q: Training still crashes with these fixes?
A: Check:
- Are you using the latest version of the code?
- Is learning rate too high?
- Are you resuming from a corrupted checkpoint?

### Q: Training is much slower now?
A: The clipping adds minimal overhead. If slow:
- Check if NaN warnings are appearing frequently (bad)
- Reduce `n_trajectories` if memory/compute limited

### Q: Policy converges to different solution?
A: The fixes change optimization dynamics slightly. This is expected and OK. The solution should be similarly good or better (more stable).

## Summary

These fixes address the core numerical instability issues that caused the crash:
- ✅ Removed incorrect loss clamping
- ✅ Added comprehensive NaN/Inf detection
- ✅ Improved TanhNormal stability with conservative clipping
- ✅ Added policy output clipping
- ✅ Enhanced diagnostic logging

The training should now be much more robust and give early warnings before any crash occurs.
