# The Actual Root Cause of Policy Collapse

## Summary

**Advantage normalization destroys the reward signal when returns are small**, causing the policy to learn that "doing nothing" is optimal even though positive rewards are possible.

## What Actually Happened

### Observation from Logs
- Step 50: Return 1.773, Episode Length 77.5 (learning, some bankruptcy)
- Step 150: Return 0.045, Episode Length 100 (surviving but low rewards)
- Step 300+: Return 0.000, Episode Length 98 (surviving by doing nothing)

### The Perverse Learning Dynamic

1. **Initial exploration** (steps 0-50):
   - Policy outputs dividends ≈ α = 0.18
   - Per-step reward: 0.18 * 0.1 = 0.018
   - Over 77.5 steps: return ≈ 1.4
   - Some trajectories bankrupt early due to volatility

2. **Advantage normalization kicks in**:
   - Returns: [0.0, 1.2, 1.5, 1.8] (mix of bankruptcy and success)
   - After centering: advantages = [-0.9, -0.15, 0.15, 0.9]
   - **Return 1.2 gets NEGATIVE advantage!**
   - Policy learns: "avoid actions that led to 1.2"

3. **Downward spiral**:
   - Policy reduces actions to avoid negative advantages
   - Returns become more similar: [0.4, 0.5, 0.5, 0.6]
   - Normalized advantages still centered at 0
   - Gradient signal is weak (all returns similar)
   - Policy drifts toward zero actions

4. **Terminal state**:
   - Policy outputs near-zero dividends and equity
   - Returns are exactly 0.0 for all trajectories
   - Advantages = 0 - 0 = 0 (no gradient)
   - **Stuck at local minimum**

## Why Advantage Normalization Failed Here

### Standard RL Setting (works fine):
- Returns: [-100, 0, 100, 200]
- Range: 300
- After normalization: advantages preserve ordering
- **Negative returns are clearly bad**

### Our Setting (breaks):
- Returns: [0.0, 0.4, 0.5, 0.6]
- Range: 0.6
- After normalization: advantages = [-1.34, -0.13, 0.13, 1.34]
- **All returns are non-negative, but normalization treats 0.4 as "bad"**

The key difference: **when ALL returns are positive but small, normalization removes the information that "any reward > 0 is good"**.

## Mathematical Analysis

### REINFORCE Gradient

```
∇θ J = E[(log π(a|s)) · (R - V(s))]
```

**With normalization** (what we had):
```python
advantages = (returns - baseline - returns.mean()) / returns.std()
```

- Mean of advantages = 0 (by definition)
- If returns are [0.0, 0.5, 0.5, 0.5], std is small
- Noise in returns gets amplified by dividing by small std
- Gradient direction becomes random

**Without normalization** (the fix):
```python
advantages = returns - baseline
```

- If baseline predicts V(s) = 1.0 but returns = [0.0, 0.5, 0.5, 0.5]
- Advantages = [-1.0, -0.5, -0.5, -0.5]
- All negative → policy updates to avoid these actions ✓ CORRECT
- Baseline learns V(s) = 0.4
- New advantages = [0.0, 0.5, 0.5, 0.5]
- All positive or zero → keep exploring ✓ CORRECT

## Why The Config Made It Worse

### 1. Zero Bankruptcy Penalty

```yaml
liquidation_rate: 0.0
liquidation_flow: 0.0
# => liquidation_value = 0.0
```

- Bankruptcy gives return 0.0
- Surviving with no actions also gives return 0.0
- **No incentive to avoid bankruptcy!**

### 2. Stochastic Dynamics

- Cash evolves: `dc = 0.18 dt + σ_c dW - a_L dt`
- Volatility σ_c causes cash to fluctuate randomly
- Any dividend policy (a_L > 0) creates bankruptcy risk
- "Safe" policy: a_L = 0, no bankruptcy, guaranteed 0 return

### 3. Sparse Rewards + Advantage Normalization

- Sparse rewards: compute total return per trajectory
- High variance due to stochasticity
- Advantage normalization amplifies this variance
- **Perfect storm for losing the signal**

## The Fix

### Primary Fix: Disable Advantage Normalization

```yaml
advantage_normalization: false
```

This preserves the absolute scale of returns:
- Return 0.5 is better than return 0.0
- Return 1.0 is better than return 0.5
- Gradient points toward higher absolute returns

### Why This Works

1. **Baseline still reduces variance**:
   - Advantages = returns - baseline
   - Baseline learns to predict average return
   - Variance reduction WITHOUT centering

2. **Preserves absolute signal**:
   - If all returns > 0, all advantages > -baseline
   - As baseline improves, advantages center naturally around 0
   - But the SCALE is preserved

3. **Works with small returns**:
   - Returns: [0.4, 0.5, 0.6]
   - Baseline: 0.5
   - Advantages: [-0.1, 0.0, 0.1]
   - Gradient slightly favors a_L = 0.6 action ✓

## Alternative/Complementary Fixes

### Option 1: Add Bankruptcy Penalty
```yaml
reward:
  liquidation_rate: -10.0  # Strong penalty
```
Makes bankruptcy costly, encourages positive dividends.

### Option 2: Remove Baseline Entirely
```yaml
training:
  use_baseline: false
```
Higher variance but clearer signal for small returns.

### Option 3: Increase Policy Learning Rate
```yaml
training:
  lr_policy: 0.001  # 10x higher
```
Faster learning might escape zero-action attractor.

### Option 4: Reward Shaping
Add positive reward for maintaining high cash reserves to encourage survival while still paying dividends.

## Expected Behavior After Fix

With `advantage_normalization: false`:

1. **Early training**:
   - Returns: [0.0, 1.0, 1.5, 2.0]
   - Baseline: 0.5
   - Advantages: [-0.5, 0.5, 1.0, 1.5]
   - Gradient pushes toward higher dividend actions ✓

2. **Mid training**:
   - Policy improves, fewer bankruptcies
   - Returns: [1.0, 1.5, 2.0, 2.5]
   - Baseline: 1.75
   - Advantages: [-0.75, -0.25, 0.25, 0.75]
   - Still learning from relative performance ✓

3. **Convergence**:
   - Optimal policy found
   - Returns: [8.0, 9.0, 10.0, 11.0] (higher!)
   - Baseline: 9.5
   - Advantages: [-1.5, -0.5, 0.5, 1.5]
   - Fine-tuning around optimum ✓

## Lessons Learned

1. **Advantage normalization is NOT always good**:
   - Good: When returns span large range (e.g., -100 to +100)
   - Bad: When all returns are small and positive (e.g., 0 to 2)

2. **Sparse rewards need careful handling**:
   - Trajectory-level returns have high variance
   - Need strong signal to overcome noise
   - Normalization can destroy this signal

3. **Reward design matters**:
   - Zero bankruptcy penalty made "do nothing" a viable strategy
   - Small rewards + volatility + normalization = collapse

4. **Baseline helps BUT can hurt**:
   - Reduces variance ✓
   - But if combined with normalization, removes scale ✗
   - Use baseline WITHOUT normalization for small returns

## Testing The Fix

Run training with the updated config:
```bash
python scripts/train_with_config.py --config configs/time_augmented_sparse_config.yaml
```

Expected:
- Returns should stay positive and grow over time
- Policy should learn to pay dividends ≈ α = 0.18
- No collapse to zero
