# GHM-RL Validation Issues - Root Cause Analysis

## Executive Summary

The validation failures stem from **three critical issues**:

1. **CRITICAL: Wrong discount factor** (γ = 0.99 vs correct 0.9998)
2. **Insufficient training** (100k timesteps insufficient for convergence)
3. **Value estimation noise** amplified by numerical differentiation

## Issue 1: INCORRECT DISCOUNT FACTOR ⚠️ CRITICAL

### The Problem

**Location**: `scripts/train_ghm.py:77`

```python
model = SAC(
    ...
    gamma=0.99,  # ❌ WRONG!
    ...
)
```

### Why This is Wrong

For continuous-time discounting:
- Continuous discount rate: ρ = r - μ = 0.03 - 0.01 = **0.02** (2% per year)
- Time step: dt = **0.01**
- Correct discrete discount: γ = exp(-ρ × dt) = exp(-0.0002) = **0.9998**

**But the code uses γ = 0.99**, which implies:
- Implied ρ = -ln(0.99) / 0.01 = **1.005** (100.5% per year!)
- This makes the agent **50x more myopic** than it should be

### Impact

1. **Agent optimizes wrong objective**:
   - Should maximize: E[∫₀^∞ e^(-0.02t) a(t) dt]
   - Actually maximizes: E[∫₀^∞ e^(-1.005t) a(t) dt]

2. **Value function distorted**:
   - Correct value at c*: F(c*) ≈ c*/0.02 = c*/0.02
   - Wrong value with γ=0.99: F(c*) ≈ c*/1.005 (50x smaller!)

3. **Policy becomes too aggressive**:
   - Agent pays dividends immediately to avoid discounting
   - Doesn't properly value maintaining cash buffer
   - Leads to more liquidations

### Validation Mismatch

The validation script uses the **correct** gamma:

```python
# validate.py:73
gamma = env.get_expected_discount_factor()  # Returns 0.9998
```

This creates inconsistency:
- Training optimizes for γ = 0.99
- Validation evaluates with γ = 0.9998
- Value estimates don't match learned policy

## Issue 2: INSUFFICIENT TRAINING

### The Problem

Training for only **100k timesteps** is likely insufficient:

```bash
python scripts/train_ghm.py --timesteps 100000  # Too short!
```

### Why More Training is Needed

1. **Continuous control is hard**: Unlike discrete action spaces, continuous actions require more samples

2. **Stochastic dynamics**: The GHM model has two sources of randomness:
   - Permanent shocks (σ_A = 0.25)
   - Transitory shocks (σ_X = 0.12)
   - High variance in returns

3. **Threshold behavior**: The optimal policy has a sharp threshold c*:
   - Below c*: minimal dividends
   - Above c*: pay excess above c*
   - Learning this discontinuity requires many samples

4. **No convergence check**: The script doesn't verify convergence before stopping

### Evidence from Validation

Looking at the plots:
- Policy has correct qualitative shape but wrong quantitative values
- Value function has large standard deviation (±1 std bands are wide)
- Derivatives are extremely noisy

Typical training times for similar problems: **500k - 2M timesteps**

## Issue 3: VALUE ESTIMATION NOISE

### The Problem

Value estimation in `scripts/validate.py:29-89`:

```python
def estimate_value_function(..., n_episodes: int = 50):
    # Run 50 episodes from each state and average
    for ep in range(n_episodes):
        # Rollout and accumulate discounted rewards
        ...
```

### Sources of Noise

1. **Finite samples**: Only 50 episodes per grid point
   - Stochastic environment → high variance in returns
   - Standard error ∝ 1/√50 ≈ 14%

2. **Monte Carlo estimation**: Each episode has different random shocks
   - Long rollouts accumulate variance
   - Early termination (liquidation) truncates returns

3. **Numerical differentiation**: Amplifies noise
   - First derivative: noise increases ~2x
   - Second derivative: noise increases ~4x
   - This explains the extremely spiky F''(c) plot

### Impact on Validation

From the results:
- F'(c*) = -0.5338 (should be 1.0) → error = 1.53
- F''(c*) = 1009.9 (should be 0.0) → error = 1010
- Min F' = -14.48 (should be > 0)
- F'' oscillates wildly between -875 and +1000

The validation conditions are:
- |F'(c*) - 1| < 0.2 → FAIL (error = 1.53)
- |F''(c*)| < 1.0 → FAIL (error = 1010)
- Mean HJB residual < 0.5 → FAIL (3.84)

## Issue 4: POTENTIAL REWARD STRUCTURE ISSUES

### Current Implementation

```python
# ghm_equity_env.py:146
reward = float(action[0] * self.dt)
```

This is **theoretically correct**: reward = dividends paid per time step.

### Potential Issues

1. **Scale**: Rewards are small (≈ 0.01 - 0.1 per step)
   - May need reward scaling for better learning
   - Though SAC should handle this reasonably well

2. **Terminal penalty**:
   ```python
   liquidation_penalty=5.0
   ```
   - Is this the right magnitude?
   - Should scale with the value function

3. **No explicit incentive for threshold behavior**:
   - Agent must discover c* through trial and error
   - Could add shaping rewards (though this changes the problem)

## RECOMMENDED FIXES

### Priority 1: Fix Discount Factor ⭐⭐⭐

**What to do**:

```python
# scripts/train_ghm.py
# Remove hardcoded gamma, use environment's value

# Option A: Compute directly
env_instance = GHMEquityEnv(dt=0.01, ...)
gamma = env_instance.get_expected_discount_factor()

model = SAC(
    ...,
    gamma=gamma,  # Use correct value (≈ 0.9998)
    ...
)

# Option B: Let environment control discount
# (Requires modifying to expose gamma as property)
```

**Expected impact**:
- Correct objective function
- Better value function estimates
- Policy closer to theoretical optimum

### Priority 2: Train Much Longer ⭐⭐⭐

**What to do**:

```bash
# Train for at least 500k steps, preferably 1M
python scripts/train_ghm.py --timesteps 1000000

# Use checkpointing to monitor progress
# Check tensorboard logs for convergence
tensorboard --logdir models/ghm_equity/tensorboard
```

**Watch for**:
- Episode reward should stabilize
- Policy entropy should decrease
- Eval callback should show improvement plateau

### Priority 3: Improve Value Estimation ⭐⭐

**Option A: Use more episodes**

```python
# Increase from 50 to 200-500 episodes
python scripts/validate.py --model models/ghm_equity/final_model --n-episodes 200
```

**Option B: Use critic network directly** (better approach)

Instead of Monte Carlo rollouts, use the SAC critic network:
```python
# Approximate V(c) ≈ Q(c, π(c)) from learned Q-functions
V = model.critic(obs, model.actor(obs))
```

This is much less noisy than rollouts.

**Option C: Smooth before differentiating**

Fit a smooth function (polynomial, spline) to value estimates before computing derivatives.

### Priority 4: Verify Hyperparameters ⭐

**Check**:
1. Learning rate (3e-4 is standard)
2. Buffer size (100k may be too small for 1M steps)
3. Network architecture (default MLP may need tuning)
4. Liquidation penalty (5.0 seems reasonable)

## THEORETICAL BENCHMARKS

For comparison with theoretical solution (from GHM paper):

### Expected Values (approximate)

- **Threshold**: c* ≈ 0.4-0.5
- **Value at threshold**: F(c*) ≈ c*/ρ ≈ 0.45/0.02 = 22.5
- **Smooth pasting**: F'(c*) = 1 exactly
- **Super-contact**: F''(c*) = 0 exactly
- **Policy**:
  - Below c*: a(c) ≈ 0 (minimal dividends)
  - Above c*: a(c) large (pay excess)
  - Ratio: > 10x

### Current Results vs Theory

| Metric | Expected | Actual | Status |
|--------|----------|---------|--------|
| c* | 0.4-0.5 | 0.43 | ✓ Good |
| F'(c*) | 1.0 | -0.53 | ✗ Wrong sign! |
| F''(c*) | 0.0 | 1010 | ✗ Way off |
| Action ratio | >10x | 5.6x | ✗ Too low |
| F' > 0 | Always | No (min=-14.5) | ✗ Not monotonic |

The fact that F'(c*) is **negative** is especially concerning - this suggests fundamental issues with the learned policy.

## IMPLEMENTATION PLAN

### Step 1: Fix and Retrain

1. Fix discount factor in `train_ghm.py`
2. Train for 1M timesteps
3. Monitor tensorboard for convergence
4. Save checkpoints every 100k steps

### Step 2: Improved Validation

1. Modify `validate.py` to use critic network for V(c)
2. Add smoothing before differentiation
3. Add theoretical benchmarks to validation output
4. Plot training curves (reward, Q-values over time)

### Step 3: Debugging Tools

1. Add episode analysis: track liquidation rate
2. Add state visitation histogram
3. Plot learned Q-function surface
4. Compare to finite-difference solution (if available)

## CONCLUSION

The validation failures are **expected given the wrong discount factor**. The agent is solving a fundamentally different problem than intended.

**Action items**:
1. ✅ Fix gamma (critical - do first)
2. ✅ Retrain with 1M steps
3. ✅ Improve validation (use critic network)
4. ✅ Add convergence monitoring

Once fixed, we should see:
- F'(c*) ≈ 1.0
- F''(c*) ≈ 0.0
- F'(c) > 0 everywhere
- F''(c) < 0 in continuation region
- HJB residual < 0.5
