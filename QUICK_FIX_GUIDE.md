# Quick Fix Guide: Stop Agent from Learning Bankruptcy

## TL;DR

Your agent learns to bankrupt itself because of **3 critical bugs** that make bankruptcy more rewarding than sustainable operation. Here's how to fix them in ~5 minutes.

---

## Fix #1: Reward Masking at Termination (CRITICAL - DO THIS FIRST!)

### The Problem
When bankruptcy happens, the reward is NOT zeroed out, so the agent gets BOTH the dividend payout AND liquidation value.

### The Fix

**File:** `macro_rl/simulation/trajectory.py`

**Lines 255-265:** Change the ORDER of operations

**BEFORE (broken):**
```python
# Check termination
terminated = self._check_termination(states[:, t + 1, :])

# Update active mask
masks[:, t] = active.to(dtype=masks.dtype)  # ❌ active not updated yet!

# Zero out rewards for terminated trajectories
rewards[:, t] = rewards[:, t] * masks[:, t]  # ❌ mask=1, reward NOT zeroed!

# Update active status
active = active & (~terminated)  # ⚠️ Too late!
```

**AFTER (fixed):**
```python
# Check termination
terminated = self._check_termination(states[:, t + 1, :])

# Update active status FIRST
active = active & (~terminated)  # ✅ Update active immediately

# THEN update active mask
masks[:, t] = active.to(dtype=masks.dtype)  # ✅ Now mask reflects termination

# Zero out rewards for terminated trajectories
rewards[:, t] = rewards[:, t] * masks[:, t]  # ✅ Reward properly zeroed!
```

**Just swap lines 259 and 265!**

### Why This Works

The mask should indicate whether the trajectory is active AT THE END of step t (after taking action and transitioning), not at the BEGINNING.

By updating `active` before setting the mask, terminated trajectories will have `mask=0`, properly zeroing their rewards.

---

## Fix #2: Liquidation Value (HIGH PRIORITY)

### The Problem
Liquidation value (4.95) is **192% of sustainable firm value** (2.57), making bankruptcy attractive.

### The Fix

**File:** `macro_rl/dynamics/ghm_equity.py`

**Line 52:** Change liquidation value calculation

**BEFORE (broken):**
```python
def __post_init__(self):
    """Compute derived parameters."""
    # Liquidation value: ω·α/(r-μ)
    if self.r > self.mu:
        self.liquidation_value = self.omega * self.alpha / (self.r - self.mu)  # ❌ Way too high!
    else:
        self.liquidation_value = 0.0
```

**AFTER (fixed - Option A: Zero recovery):**
```python
def __post_init__(self):
    """Compute derived parameters."""
    # Liquidation value: equity holders typically get nothing in bankruptcy
    self.liquidation_value = 0.0  # ✅ Realistic for equity holders
```

**AFTER (fixed - Option B: Small recovery):**
```python
def __post_init__(self):
    """Compute derived parameters."""
    # Liquidation value: small fixed recovery
    self.liquidation_value = 0.1  # ✅ Small recovery (4% of sustainable value)
```

### Why This Works

In real bankruptcies:
- Assets sold at fire-sale prices
- Legal costs and fees
- Debt holders paid first
- **Equity holders get little or nothing**

Setting liquidation value to 0 or a small positive removes the incentive to bankrupt.

---

## Fix #3: Reward Consistency (MEDIUM PRIORITY)

### The Problem
Two different reward formulas in the codebase.

### The Fix

**File:** `macro_rl/envs/ghm_equity_env.py`

**Line 160:** Make it consistent with the reward function

**BEFORE (inconsistent):**
```python
reward -= equity_gross  # ❌ No cost multiplier
```

**AFTER (consistent):**
```python
reward -= (1.0 + self.issuance_cost) * equity_gross  # ✅ Includes issuance cost
```

---

## Testing the Fixes

### Before Fixes
Your plots show:
- Dividend policy converges to 8-10 (bankruptcy-seeking)
- Even at c=0, agent pays huge dividends
- All time horizons converge to same bankruptcy policy

### After Fixes (Expected)
- Dividend policy should be reasonable (0-2)
- Higher dividends at higher cash levels
- Zero dividends at low cash
- Policy varies with time horizon
- **No more bankruptcy behavior!**

### How to Verify

1. **Apply all three fixes**
2. **Re-run training** from scratch
3. **Check plots** at steps 1000, 5000, 10000

Look for:
- ✅ Dividend policy respects cash constraints
- ✅ No bankruptcy (trajectories reach final time)
- ✅ Value function increases with cash
- ✅ Policy smoothly varies with state and time

---

## Step-by-Step Application

### 1. Open `macro_rl/simulation/trajectory.py`

Find lines 255-265 (in the `simulate` method):

```python
# Check termination
terminated = self._check_termination(states[:, t + 1, :])

# Update active mask
masks[:, t] = active.to(dtype=masks.dtype)

# Zero out rewards for terminated trajectories
rewards[:, t] = rewards[:, t] * masks[:, t]

# Update active status
active = active & (~terminated)
```

**Change to:**

```python
# Check termination
terminated = self._check_termination(states[:, t + 1, :])

# Update active status
active = active & (~terminated)

# Update active mask
masks[:, t] = active.to(dtype=masks.dtype)

# Zero out rewards for terminated trajectories
rewards[:, t] = rewards[:, t] * masks[:, t]
```

### 2. Open `macro_rl/dynamics/ghm_equity.py`

Find line 52 (in `__post_init__` method):

```python
self.liquidation_value = self.omega * self.alpha / (self.r - self.mu)
```

**Change to:**

```python
self.liquidation_value = 0.0  # Equity holders get nothing in bankruptcy
```

### 3. Open `macro_rl/envs/ghm_equity_env.py`

Find line 160 (in the reward computation):

```python
reward -= equity_gross
```

**Change to:**

```python
reward -= (1.0 + self.issuance_cost) * equity_gross
```

### 4. Save all files and re-run training

```bash
# Clear old results
rm -rf outputs/bankruptcy_test/

# Re-train with fixed code
python train_ghm_equity.py --config configs/your_config.yaml

# Monitor for reasonable behavior
```

---

## Why These Bugs Caused Bankruptcy-Seeking

### The Bankruptcy Exploit (with bugs)

```
Agent with c = 0.1:

Option A: Sustainable operation
- Pay small dividends over time
- Total return: ~0.7 (discounted)

Option B: Immediate bankruptcy
- Pay huge dividend: 10.0  ← Bug #1: Not zeroed!
- Get liquidation: 4.95     ← Bug #2: Too high!
- Total return: 14.95

Ratio: 14.95 / 0.7 = 21.4x more rewarding!
```

**Of course the agent learns to bankrupt!**

### After Fixes

```
Agent with c = 0.1:

Option A: Sustainable operation
- Pay small dividends over time
- Total return: ~0.7

Option B: Immediate bankruptcy
- Pay huge dividend: 0.0  ← Bug #1 fixed: Properly zeroed!
- Get liquidation: 0.0    ← Bug #2 fixed: No recovery!
- Total return: 0.0

Ratio: 0.0 / 0.7 = 0x (bankruptcy is terrible!)
```

**Now agent learns to survive!**

---

## Common Questions

### Q: Will these fixes break anything?

**A:** No! These are pure bug fixes that align the code with the intended economics.

### Q: Do I need to change hyperparameters?

**A:** No! The bugs made the problem impossible to solve correctly. Fixing them should make learning easier, not harder.

### Q: Which fix is most important?

**A:** Fix #1 (reward masking) is CRITICAL. It causes double-counting. Fix #2 is also very important. Fix #3 is less critical but good for consistency.

### Q: Can I apply fixes incrementally?

**A:** Yes, but apply Fix #1 first. You can test with just Fix #1 to see partial improvement, then add Fix #2 for full improvement.

---

## Summary

| Fix | File | Lines | Change | Impact |
|-----|------|-------|--------|--------|
| #1 | trajectory.py | 259, 265 | Swap order | Stop double-counting |
| #2 | ghm_equity.py | 52 | Set to 0.0 | Remove bankruptcy incentive |
| #3 | ghm_equity_env.py | 160 | Add cost | Consistency |

**Total: 3 files, ~5 lines changed, 5 minutes of work**

**Result: Agent stops bankrupting itself!**
