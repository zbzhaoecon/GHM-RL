# Critical Fix: Action Regularization Sign Error

## Problem Identified

The policy was learning consistently but in the **wrong direction** - dividends increasing from ~3 to ~9.5 over training, pushing toward boundaries instead of optimal values.

## Root Cause: Action Regularization Sign Error

**Location**: `macro_rl/solvers/monte_carlo.py:203-206`

**Buggy Code**:
```python
action_reg_loss = -self.action_reg_weight * action_magnitude  # Line 203
total_loss = policy_loss - self.entropy_weight * entropy + action_reg_loss  # Line 206
```

**Expansion**:
```python
total_loss = policy_loss - entropy_weight * entropy - action_reg_weight * action_magnitude
```

**The Bug**:
When minimizing `total_loss` via gradient descent:
- ✅ `policy_loss` term: Correct REINFORCE
- ✅ `-entropy` term: MAXIMIZES entropy (correct - encourages exploration)
- ❌ **`-action_magnitude` term: MAXIMIZES action_magnitude (WRONG!)**

This **actively pushed actions toward boundaries**, which is exactly opposite of the intended "prevent collapse to zero" goal.

## Why This Happened

The original intent was to "prevent policy from collapsing to zero actions" by adding a regularization term. However:

1. **Conceptual Error**: To prevent collapse, you'd want to penalize SMALL actions, not encourage LARGE ones
2. **Implementation Error**: The negative sign made it maximize action magnitude instead
3. **Compounding Effect**: This worked against the learning signal from REINFORCE, pushing policy to boundaries

## Verification of Other Components

**Checked and Confirmed CORRECT**:

1. ✅ **REINFORCE Gradient**:
   ```python
   policy_loss = -(log_prob * advantages).mean()
   ```
   Correct for gradient descent on loss

2. ✅ **Advantage Calculation**:
   ```python
   advantages = returns - values
   ```
   Correct sign (positive advantage = better than expected)

3. ✅ **Reward Function**:
   ```python
   reward = a_L * dt - (1 + λ) * a_E
   ```
   Correct (maximize dividends, penalize issuance cost)

4. ✅ **Entropy Term**:
   ```python
   total_loss = policy_loss - entropy_weight * entropy
   ```
   Correct (minimizing -entropy maximizes entropy)

## The Fix

**Solution**: **REMOVE** the action regularization entirely.

### Why Remove Instead of Fix Sign?

1. **Not Actually Needed**: REINFORCE + entropy bonus is sufficient for learning
2. **Fundamentally Flawed**: The whole idea of "regularizing action magnitude" doesn't make sense for this problem
   - We don't want to encourage large actions
   - We don't want to penalize all small actions
   - Optimal policy might naturally have small actions in some states

3. **Entropy Already Handles Exploration**: The entropy bonus already prevents premature convergence

### Changes Made

**Files Modified**:

1. **`macro_rl/solvers/monte_carlo.py`**:
   - Removed `action_reg_weight` parameter from `__init__`
   - Removed action_reg_loss computation and usage
   - Simplified loss to: `total_loss = policy_loss - entropy_weight * entropy`
   - Removed action_reg_loss from metrics
   - Updated `_get_safe_metrics` to remove action_reg_loss

2. **`scripts/train_monte_carlo_ghm_time_augmented.py`**:
   - Removed `action_reg_weight` from `TrainConfig`
   - Removed from argparse
   - Removed from solver initialization
   - Removed from print statements
   - Removed from TensorBoard logging

## Expected Behavior After Fix

**Before Fix**:
- Step 500: Dividend ~3-5
- Step 2000: Dividend ~5-9
- Step 3500: Dividend ~5-9.5
- Step 5000: Dividend ~5-9.5
- **Direction**: Consistently increasing dividends toward upper bound (WRONG)

**After Fix**:
- Policy should learn based on actual rewards
- Dividends should vary across state space (not uniformly high)
- Should learn optimal balance between dividends and cash reserves
- No systematic push toward boundaries

## Testing the Fix

Run training:
```bash
python scripts/train_monte_carlo_ghm_time_augmented.py \
  --n_iterations 10000 \
  --lr_policy 3e-4 \
  --entropy_weight 0.05
```

**What to Monitor**:
1. Dividend policy should vary across cash reserve states (not flat high values)
2. Returns should improve over time
3. No systematic drift toward boundaries
4. Policy should stabilize at reasonable values

**Healthy Signs**:
- Dividend policy shows state-dependent behavior
- Lower dividends when cash reserves are low
- Higher dividends when cash reserves are high
- Returns increasing over training

**Warning Signs**:
- Dividends still pushing uniformly toward 10
- No variation across state space
- Returns not improving

## Simplified Loss Function

**Final Loss Function**:
```python
total_loss = policy_loss - entropy_weight * entropy
```

Where:
- `policy_loss = -(log_prob * advantages).mean()` (REINFORCE)
- `entropy = policy.entropy(states).mean()` (exploration bonus)
- `entropy_weight = 0.05` (default)

This is the **standard policy gradient + entropy bonus** formulation - clean, simple, and theoretically sound.

## Key Lesson

**Don't add regularization terms without understanding their gradient effects!**

When adding a term `α * f(x)` to a loss function:
- Minimizing the loss will **minimize** `f(x)` if `α > 0`
- Minimizing the loss will **maximize** `f(x)` if `α < 0`

The buggy code had:
- `action_reg_loss = -action_reg_weight * action_magnitude` (α < 0)
- This **maximized** action_magnitude, opposite of intended effect

## Related Issues Ruled Out

Based on user's suggestions, we checked but ruled out:

- ❌ **Advantage sign reversed**: Confirmed correct (`advantages = returns - values`)
- ❌ **Entropy sign wrong**: Confirmed correct (minimizing `-entropy` maximizes entropy)
- ❌ **Reward sign issue**: Confirmed correct (`reward = dividends - issuance_cost`)
- ❌ **Policy loss sign bug**: Confirmed correct (`-(log_prob * adv)`)
- ❌ **Adapter mismatch**: Not relevant (API correct)

The ONLY issue was the action regularization term.
