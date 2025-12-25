# Monte Carlo Training Crash - Root Cause Analysis

## Summary
Training crashed between steps 6500-7000 with policy outputs collapsing to ~1e-14 (essentially zero). The plots show:
- Steps 5500-6500: Normal policy behavior
- Step 7000: Complete collapse (policy outputs ~1e-14, all near-zero)
- Step 7500: Still collapsed

## Root Causes Identified

### 1. **CRITICAL: Incorrect Loss Clamping** ⚠️
**Location**: `macro_rl/solvers/monte_carlo.py:191`

```python
policy_loss = torch.clamp(policy_loss, -1000.0, 1000.0)
```

**Problem**:
- Clamping the **loss value** before `.backward()` is incorrect
- This does NOT prevent gradient explosions
- Interferes with proper gradient computation
- Can hide underlying numerical issues

**Fix**: Remove this line completely (gradient clipping is done correctly later at line 209-212)

### 2. **CRITICAL: Log Probability Explosion at Boundaries** ⚠️
**Location**: `macro_rl/distributions/tanh_normal.py:118-169`

**Problem**:
The TanhNormal distribution uses `atanh` to invert actions back to the base Gaussian space:
```python
normalized = 2.0 * (action - self.low) / (self.high - self.low) - 1.0
normalized = torch.clamp(normalized, -0.999, 0.999)
z = torch.atanh(normalized)
```

When actions approach boundaries:
- `normalized → ±0.999`
- `atanh(0.999) ≈ 3.8` (manageable)
- BUT: Jacobian correction `log(1 - tanh²(z))` becomes very negative
- For `z=3.8`: `log(1 - 0.999²) = log(0.002) ≈ -6.2`
- This creates **huge negative log probabilities**

**Mechanism of Crash**:
1. Policy produces actions near boundaries
2. Log prob becomes very negative (e.g., -50 to -100)
3. REINFORCE gradient: `∇ log π(a|s) * advantage` explodes
4. Even with gradient clipping to 0.5 norm, the update pushes network outputs to extreme values
5. Raw network output `z` becomes large (e.g., z > 10)
6. `tanh(10) → 0.99999`, saturating actions at boundaries
7. **Vicious cycle**: boundary actions → huge negative log probs → huge gradients → more extreme outputs → policy collapse

### 3. **No NaN/Inf Detection**
**Location**: Throughout training loop

**Problem**:
- No checks for NaN or Inf in losses, gradients, or outputs
- Once NaN appears, it corrupts all subsequent updates
- Policy collapses irreversibly

### 4. **Insufficient Numerical Stability in TanhNormal**
**Location**: `macro_rl/distributions/tanh_normal.py`

**Current Issues**:
- Clamping to [-0.999, 0.999] is not conservative enough
- No safeguards against extreme raw outputs from policy network
- Jacobian correction can still produce very negative values

### 5. **Weak Action Regularization**
**Location**: `macro_rl/solvers/monte_carlo.py:199`

```python
action_reg_loss = -self.action_reg_weight * action_magnitude  # weight=0.01
```

**Problem**:
- Weight 0.01 is too weak to prevent collapse
- Once policy starts collapsing, this regularization can't recover it

### 6. **Advantage Normalization Edge Case**
**Location**: `macro_rl/solvers/monte_carlo.py:176-184`

```python
if adv_std > 1e-3:
    advantages = (advantages - advantages.mean()) / adv_std
else:
    advantages = advantages - advantages.mean()
```

**Problem**:
- When variance is very low, only centering (not normalizing) can amplify noise
- Could contribute to instability

## Sequence of Events Leading to Crash

1. **Steps 1-6500**: Normal training
   - Policy learns reasonable actions
   - Some actions occasionally near boundaries

2. **Around Step 6500-6800**: Critical transition
   - Random fluctuation or cumulative drift pushes some actions to boundaries
   - Log probabilities become very negative (-50 to -100)
   - Gradient norms hit the clipping threshold (0.5)
   - But clipped gradients still large enough to push network outputs further

3. **Step 7000**: Catastrophic collapse
   - Network raw outputs become extreme (z > 10 or z < -10)
   - `tanh(z)` saturates to ±1
   - Actions stuck at boundaries (~0 or ~10 for dividends, ~0 or ~0.5 for equity)
   - Possibly NaN appears in gradients or losses
   - Policy essentially frozen at bad values

4. **Step 7500+**: Irrecoverable
   - Policy continues outputting near-zero or boundary-saturated values
   - Training cannot recover

## Why This Manifests as "Zero" Policy

Looking at the plots at step 7000, the policy shows ~1e-14 values, which suggests:
- The raw network outputs went to **extreme negative** values (z << -10)
- `tanh(-100) ≈ -1`
- After transformation: `action = low + (high - low) * (tanh(z) + 1) / 2`
- With `z → -∞`: `action → low + (high - low) * 0 / 2 = low`
- For actions with `low ≈ 0`, this produces near-zero outputs

## Evidence from Config

```json
{
  "lr_policy": 0.0003,
  "max_grad_norm": 0.5,
  "entropy_weight": 0.05,
  "action_reg_weight": 0.01
}
```

- Learning rate 3e-4 is reasonable but not tiny
- Grad clipping 0.5 is quite aggressive, but not sufficient to prevent collapse
- Entropy weight 0.05 should encourage exploration, but once saturated at boundaries, entropy goes to zero
- Action reg 0.01 too weak to prevent collapse

## Recommended Fixes

### Immediate (High Priority):

1. **Remove incorrect loss clamping** (monte_carlo.py:191)
2. **Add NaN/Inf detection** in training loop
3. **Improve TanhNormal numerical stability**:
   - More conservative clamping (±0.95 instead of ±0.999)
   - Clip raw network outputs before tanh transformation
   - Better Jacobian correction handling
4. **Add policy collapse detection** and early stopping
5. **Strengthen action regularization** (increase weight to 0.05-0.1)

### Medium Priority:

6. **Add diagnostic logging**:
   - Raw network outputs (before tanh)
   - Action boundary violation rate
   - Log probability statistics
7. **Learning rate scheduling** or adaptive clipping
8. **Better initialization** to avoid boundary regions initially

### Long-term:

9. Consider alternative action parameterizations (Beta distribution, clipped Gaussian)
10. Add trust region methods (PPO) for more stable updates
11. Curriculum learning starting with smaller action spaces

## Testing Strategy

1. Run with fixes on same config
2. Monitor new diagnostics for early warning signs
3. If stable, gradually increase difficulty
4. Compare convergence speed and final performance
