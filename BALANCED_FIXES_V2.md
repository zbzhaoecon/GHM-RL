# Monte Carlo Training - Balanced Fixes (v2)

## Problem with Initial Fixes

The initial fixes (commit a8cc6df) successfully prevented the crash but were **too conservative** and completely stopped learning:

- Policy stuck at boundaries (dividend ~10, equity ~0.5)
- No learning progression across 10,000 steps
- Plots identical from step 500 to step 10,000

## Root Causes of Learning Failure

### 1. **Overly Aggressive Log Probability Clipping**
```python
# TOO RESTRICTIVE - Prevented gradient signals
log_prob_action = torch.clamp(log_prob_action, min=-20.0, max=20.0)
```
- When policy is stuck in bad region, it needs large negative log probs to generate strong gradients
- Capping at -20 prevented policy from escaping local minima
- **This was the main killer**

### 2. **Too Conservative Normalized Clamping**
```python
# TOO RESTRICTIVE - Limited action space
normalized = torch.clamp(normalized, -0.95, 0.95)
```
- Prevented actions from exploring near boundaries
- Limited effective action space
- Combined with log prob clipping, created "dead zones"

### 3. **Bad Initialization**
```python
# PUSHED POLICY TO BOUNDARIES FROM START
nn.init.constant_(self.mean_net[-1].bias, 1.0)
```
- Started policy with positive bias
- After tanh transformation, pushed actions toward upper bounds
- Policy initialized in saturated region with weak gradients

### 4. **Too Restrictive Output Clipping**
```python
# LIMITED POLICY EXPRESSIVENESS
mean = torch.clamp(mean, -5.0, 5.0)
z = torch.clamp(z, -5.0, 5.0)
```
- Double clipping created narrow operating range
- Combined with other constraints, overly limited policy

## Balanced Fixes (v2)

### 1. **Relax Log Probability Clipping** ✅
```python
# OLD: min=-20.0  -> TOO RESTRICTIVE
# NEW: min=-100.0 -> Allows proper gradient signals
log_prob_action = torch.clamp(log_prob_action, min=-100.0, max=20.0)
```
**Rationale**:
- Min -100 allows strong gradients to escape bad regions
- Max 20 still prevents numerical overflow
- Gradient clipping (max_grad_norm=0.5) provides additional safety

### 2. **Relax Normalized Clamping** ✅
```python
# OLD: -0.95, 0.95   -> TOO CONSERVATIVE
# NEW: -0.997, 0.997 -> Allows exploration near boundaries
normalized = torch.clamp(normalized, -0.997, 0.997)
```
**Rationale**:
- 0.997 corresponds to z ≈ ±3.8 (reasonable range)
- Allows actions to explore near boundaries where optimal policy may lie
- Still prevents extreme atanh values (atanh(0.999) would be ~7.6)

### 3. **Relax z Clipping** ✅
```python
# OLD: -5.0, 5.0  -> TOO RESTRICTIVE
# NEW: -7.0, 7.0  -> Allows larger gradient signals
z = torch.clamp(z, -7.0, 7.0)
```
**Rationale**:
- z=7 corresponds to log_prob ≈ -24 for unit Gaussian (manageable)
- Allows proper gradient flow
- Still prevents true overflow (z > 10 would be problematic)

### 4. **Relax Raw Output Clipping** ✅
```python
# OLD: -5.0, 5.0   -> TOO RESTRICTIVE
# NEW: -10.0, 10.0 -> More expressive policy
mean = torch.clamp(mean, -10.0, 10.0)
```
**Rationale**:
- Allows network to output larger values
- z clipping in log_prob provides safety net
- More expressive policy while still bounded

### 5. **Fix Initialization** ✅
```python
# OLD: bias = 1.0  -> PUSHED TO BOUNDARIES
# NEW: bias = 0.0  -> NEUTRAL START
nn.init.constant_(self.mean_net[-1].bias, 0.0)
```
**Rationale**:
- Start at center of action space (after tanh: ≈ 0.5 for [0,1] actions)
- Allows exploration in both directions
- Avoids starting in saturated regions

### 6. **Keep NaN/Inf Detection** ✅
**No changes** - this is working correctly and essential for safety

## Expected Behavior After v2 Fixes

### Should See:
- ✅ Policy actually learning (values changing across steps)
- ✅ Exploration across action space (not stuck at boundaries)
- ✅ Gradual convergence to reasonable policy
- ✅ Log probabilities in reasonable range (-50 to -5 typical)
- ✅ No crashes from NaN/Inf (detection still active)

### Warning Signs to Monitor:

**Healthy Training**:
- `log_prob_min`: -50 to -5 (reasonable range)
- `action_magnitude`: 0.5 to 5.0 (active policy)
- Policy outputs changing visibly across evaluation steps
- Returns improving over time

**Warning Signs** (but training continues):
- `log_prob_min < -80`: Getting close to boundaries
- `action_magnitude` decreasing trend: May need higher entropy weight

**Critical Issues** (training should skip):
- NaN/Inf detected: Skip update (automatic)
- `action_magnitude < 0.01`: Policy collapsed (needs restart)

## Changes Summary

| Component | Old Value | New Value | Why |
|-----------|-----------|-----------|-----|
| Log prob clip min | -20 | -100 | Allow strong gradients to escape bad regions |
| Log prob clip max | 20 | 20 | Keep overflow protection |
| Normalized clamp | ±0.95 | ±0.997 | Allow boundary exploration |
| z clamp | ±5.0 | ±7.0 | Allow larger gradient signals |
| Raw output clamp | ±5.0 | ±10.0 | More expressive policy |
| Init bias | 1.0 | 0.0 | Neutral starting point |
| NaN detection | ✅ | ✅ | **KEEP** - essential safety |
| Warning threshold | -50 | -80 | Match relaxed constraints |

## Testing the Balanced Fixes

### Quick Smoke Test (1000 steps):
```bash
python scripts/train_monte_carlo_ghm_time_augmented.py --n_iterations 1000
```
**Expected**: Policy should visibly change between step 500 and step 1000

### Full Training Run:
```bash
python scripts/train_monte_carlo_ghm_time_augmented.py \
  --n_iterations 10000 \
  --lr_policy 3e-4 \
  --entropy_weight 0.05 \
  --action_reg_weight 0.01
```
**Expected**:
- Smooth learning curve
- No crashes
- Policy converging to reasonable values by step 10000

### If Still Having Issues:

**If policy still stuck at boundaries**:
- Increase entropy: `--entropy_weight 0.1`
- Decrease learning rate: `--lr_policy 1e-4`
- Check initialization (should be bias=0.0 now)

**If training crashes with NaN**:
- Check console for "WARNING: Non-finite" messages
- Training should skip bad updates automatically
- If crashes persist, may need to relax constraints further

**If learning too slow**:
- Current constraints are reasonable for stability
- Consider increasing learning rate: `--lr_policy 5e-4`
- Or increase action reg: `--action_reg_weight 0.05`

## Files Modified (v2)

1. `macro_rl/distributions/tanh_normal.py`:
   - Relaxed normalized clamp: 0.95 → 0.997
   - Relaxed z clamp: 5.0 → 7.0
   - Relaxed log prob clamp: -20 → -100

2. `macro_rl/networks/policy.py`:
   - Fixed initialization: bias 1.0 → 0.0
   - Relaxed output clamp: 5.0 → 10.0

3. `scripts/train_monte_carlo_ghm_time_augmented.py`:
   - Updated warning threshold: -50 → -80
   - Improved warning message

## Key Insight

The challenge is finding the **balance** between:
- **Too conservative** → Prevents crashes BUT stops learning
- **Too permissive** → Allows learning BUT risk of crashes

v2 fixes aim for this balance:
- **Numerical safety**: Keep NaN detection, reasonable clipping bounds
- **Learning freedom**: Relax constraints enough for proper gradient flow
- **Early warnings**: Detect issues before they become catastrophic

The gradient clipping (max_grad_norm=0.5) provides an additional safety layer that works in conjunction with the relaxed value clipping.
