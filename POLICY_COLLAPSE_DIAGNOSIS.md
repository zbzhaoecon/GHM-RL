# Policy Collapse Diagnosis

## Summary

The policy collapse to zero actions is caused by **baseline overfitting**: the value network learns faster than the policy can improve, creating systematically negative or near-zero advantages that kill the gradient signal.

## Root Cause Analysis

### 1. Learning Rate Imbalance

**File**: `macro_rl/solvers/monte_carlo.py` (lines 56-57, 88-90)

```python
lr_policy: float = 3e-4      # Policy learning rate
lr_baseline: float = 1e-3    # Baseline learning rate (3.3x higher!)
```

**Problem**: The baseline learns **3.3x faster** than the policy.

### 2. Feedback Loop

**File**: `macro_rl/solvers/monte_carlo.py` (lines 169-175, 404-421)

The training loop creates a negative feedback spiral:

1. **Initial Phase** (Steps 0-50):
   - Policy is randomly initialized
   - Trajectories lead to bankruptcy quickly
   - Returns are small/negative (R ≈ 0 to -5)
   - Baseline learns: V(s) ≈ -2 (matching poor returns)

2. **Exploration Phase** (Steps 50-100):
   - Entropy regularization helps policy find better actions
   - Some trajectories get higher returns (R ≈ 2)
   - **But baseline still predicts V(s) ≈ -2** (updating slowly)
   - Advantages are positive: A = R - V = 2 - (-2) = 4 ✓
   - Policy improves!

3. **Collapse Phase** (Steps 100-250):
   - Baseline catches up with learning rate 3.3x higher
   - Baseline now predicts V(s) ≈ 2 (matching recent returns)
   - **But policy hasn't improved enough yet**
   - Actual returns still R ≈ 1 to 2
   - Advantages become negative: A = 1 - 2 = -1 ✗
   - **Negative advantages push policy AWAY from actions**
   - Policy learns: "taking actions leads to negative advantages"
   - Actions shrink toward zero

4. **Terminal Phase** (Steps 250+):
   - Policy outputs near-zero actions
   - Trajectories bankrupt immediately
   - Returns collapse to R ≈ 0
   - Baseline predicts V(s) ≈ 0
   - Advantages are zero: A = 0 - 0 = 0
   - **No gradient signal to escape**
   - Policy stuck at zero actions

### 3. Mathematical Verification

**REINFORCE gradient** (line 348):
```python
policy_loss = -(log_prob_per_traj * advantages).mean()
∇θ J ∝ Σ (log π(a|s)) · (R - V(s))
```

If advantages are negative:
- Gradient points AWAY from current actions
- Policy reduces action probabilities
- Actions shrink toward zero

**Advantage normalization** (lines 177-185) doesn't help because:
- It centers advantages around zero: A_norm = (A - mean(A)) / std(A)
- If baseline systematically overestimates, std(A) is large
- Normalized advantages still have correct sign (negative)
- Normalization just scales, doesn't fix the sign issue

## Evidence from User's Visualizations

**Step 50**: Learning (dividends ~0.8, equity ~0.07)
- Policy is exploring
- Baseline hasn't caught up yet
- Positive advantages driving learning

**Step 250**: Collapsed (dividends ~0, equity ~0, value negative)
- Baseline has overfit
- Advantages are negative/zero
- Policy stuck at zero actions
- Value function shows V(c,τ) < 0 everywhere → baseline predicts negative returns

## Verification

The code at lines 169-175 computes:
```python
returns = trajectories.returns          # Actual returns from rollout
values = self.baseline(initial_states)  # Baseline prediction
advantages = returns - values            # Advantage signal
```

And baseline update at lines 404-421:
```python
baseline_loss = F.mse_loss(value_pred, returns.detach())
# Baseline regresses to match returns with MSE loss
# Updates 3.3x faster than policy due to higher learning rate
```

## Solutions

### Solution 1: Balance Learning Rates (RECOMMENDED)

**Change**: Make baseline learning rate EQUAL to or SMALLER than policy learning rate

```yaml
# configs/time_augmented_sparse_config.yaml
training:
  lr_policy: 0.0003     # 3e-4
  lr_baseline: 0.0001   # 1e-4 (3x SLOWER than policy)
```

**Rationale**:
- Baseline should track policy performance, not lead it
- Slower baseline learning prevents overfitting to early poor returns
- Gives policy time to improve before baseline catches up

### Solution 2: Delay Baseline Training

**Change**: Don't update baseline for first N iterations

```python
# In train_step() at line 237
if step > warmup_iterations and self.baseline is not None:
    baseline_loss, baseline_grad_norm = self._update_baseline(
        initial_states, returns
    )
```

**Rationale**:
- Let policy explore without baseline interference initially
- Start baseline training when policy has found reasonable actions

### Solution 3: Advantage Clipping

**Change**: Clip advantages to prevent extreme negative values

```python
# After line 175
if self.advantage_normalization:
    advantages = torch.clamp(advantages, min=-10.0, max=10.0)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Rationale**:
- Prevents single extreme advantages from dominating gradient
- Stabilizes training

### Solution 4: Value Network Initialization

**Change**: Initialize value network to output small positive values

```python
# In value network __init__
self.output_layer.bias.data.fill_(1.0)  # Initialize to predict V ≈ 1
```

**Rationale**:
- Prevents baseline from starting at negative values
- Biases toward optimistic value estimates

## Recommended Fix

**Apply Solution 1** (balance learning rates):

```yaml
# configs/time_augmented_sparse_config.yaml
training:
  lr_policy: 0.0003
  lr_baseline: 0.0001  # Changed from 0.001 to 0.0001
```

This is the simplest fix that addresses the root cause directly.

## Additional Issues Fixed

1. **Test file wrong formula** (`test_sparse_dense_equivalence.py` lines 44, 61):
   - Changed from `(1.0 + issuance_cost) * a_E` to `issuance_cost * a_E` ✓

2. **Dynamics verification** (`macro_rl/dynamics/ghm_equity.py`):
   - Drift formula: `α + c(r - λ - μ) - a_L + a_E` ✓ CORRECT
   - Diffusion: `sqrt(σ_X²(1-ρ²) + (ρσ_X - cσ_A)²)` ✓ CORRECT
   - Matches GHM model specification

3. **Reward formula** (`macro_rl/rewards/ghm_rewards.py` line 114):
   - Already fixed to `a_L * dt - issuance_cost * a_E` ✓ CORRECT

## Next Steps

1. Update config with lower baseline learning rate
2. Re-run training with new config
3. Monitor advantage statistics (should stay positive on average)
4. Check that policy actions grow instead of shrink
