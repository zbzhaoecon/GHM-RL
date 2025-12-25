# Bug Analysis: Why Agent Learns to Bankrupt Itself

## Summary

The agent consistently learns to pay huge dividends (>10) even when cash reserves are near zero, leading to immediate bankruptcy. This behavior emerges from **multiple critical bugs** that make bankruptcy more rewarding than sustainable operation.

---

## 🔴 CRITICAL BUG #1: Reward Masking at Termination

**File:** `macro_rl/simulation/trajectory.py:259-265`

### The Bug

When bankruptcy occurs, the reward earned at the bankruptcy step is **NOT zeroed out**.

### Code Flow

```python
# Line 248-253: Compute reward for step t
rewards[:, t] = self.reward_fn.step_reward(
    states[:, t, :],      # Current state
    actions[:, t, :],     # Action taken
    states[:, t + 1, :],  # Next state (may be bankrupt!)
    self.dt
)

# Line 256: Check if next state is bankrupt
terminated = self._check_termination(states[:, t + 1, :])  # c <= 0?

# Line 259: Set mask based on CURRENT active status
masks[:, t] = active.to(dtype=masks.dtype)  # ❌ active = True (not updated yet!)

# Line 262: Zero out rewards using mask
rewards[:, t] = rewards[:, t] * masks[:, t]  # ❌ mask = 1, so reward NOT zeroed!

# Line 265: NOW update active status for next iteration
active = active & (~terminated)  # ⚠️ Too late! Reward already kept
```

### Why This Is Wrong

The mask at step `t` is set BEFORE updating the active status based on termination at step `t`. This means:

1. At step `t`, trajectory is active (active=True)
2. Action at step `t` causes bankruptcy (c → 0 at step t+1)
3. Mask is set: `masks[t] = active = 1.0`
4. Reward is "masked": `rewards[t] = rewards[t] * 1.0 = rewards[t]` (unchanged!)
5. Active is updated: `active = False`

**Result:** The reward at the bankruptcy step is kept, not zeroed!

### Concrete Example

```
Initial state: c = 0.1
Action: Pay dividend rate a_L = 100.0
Time step: dt = 0.1

Step-by-step execution:
1. Reward = a_L * dt = 100.0 * 0.1 = 10.0
2. Next state: c' = c + (alpha - a_L) * dt = 0.1 + (0.18 - 100.0) * 0.1 = -9.89
3. Termination check: c' <= 0? YES → terminated = True
4. Mask assignment: masks[t] = active = 1.0 (still True!)
5. Reward masking: rewards[t] = 10.0 * 1.0 = 10.0 (NOT zeroed!)
6. Active update: active = True & ~True = False

Agent receives:
- Dividend reward: 10.0
- Terminal reward (liquidation): 4.95
- Total return: 14.95

Expected (correct):
- Dividend reward: 0.0 (should be zeroed)
- Terminal reward: 4.95
- Total return: 4.95
```

### Economic Impact

The agent sees this payoff structure:

| Strategy | Rewards Received | Total Return |
|----------|------------------|--------------|
| Survive (stay solvent) | Small dividends over time | ~5-7 (discounted) |
| Bankrupt immediately | Huge dividend + Liquidation | ~10-15 |

**Bankruptcy becomes optimal!**

---

## 🟡 HIGH PRIORITY BUG #2: Liquidation Value Too Large

**File:** `macro_rl/dynamics/ghm_equity.py:52`

### The Bug

Liquidation value is computed as:

```python
self.liquidation_value = self.omega * self.alpha / (self.r - self.mu)
```

With default parameters:
- omega = 0.55 (recovery rate)
- alpha = 0.18 (normal operating cash flow)
- r = 0.07 (discount rate)
- mu = 0.05 (drift)

```
liquidation_value = 0.55 * 0.18 / (0.07 - 0.05) = 0.099 / 0.02 = 4.95
```

### Why This Is Wrong

**Economic interpretation:**
- Sustainable dividend rate: `a_L = alpha = 0.18`
- Present value of perpetual dividends: `PV = alpha / r = 0.18 / 0.07 = 2.57`
- Liquidation value: **4.95**

**The liquidation value is 192% of the sustainable firm value!**

This is economically nonsensical because:

1. **Alpha is NOT liquidation value**: Alpha is the normal operating cash flow, not the recovery value of assets
2. **Liquidation destroys value**: Assets sold in fire sales, legal costs, loss of going-concern value
3. **Equity holders are last**: In bankruptcy, debt holders get paid first; equity holders typically get nothing

### Real-World Analogies

If this were a real company:
- Normal operations: Generate $180k/year → PV = $2.57M
- Go bankrupt: Receive $4.95M

**This is backwards!** No company worth $2.57M alive should be worth $4.95M dead.

### What Should It Be?

For equity holders in bankruptcy:
```python
self.liquidation_value = 0.0  # Most realistic (equity holders get nothing)
```

Or with some recovery:
```python
self.liquidation_value = 0.1  # Small fixed recovery (4% of sustainable value)
```

---

## 🟡 HIGH PRIORITY BUG #3: Inconsistent Reward Computation

**Files:**
- `macro_rl/rewards/ghm_rewards.py:113`
- `macro_rl/envs/ghm_equity_env.py:160`

### The Bug

Two different formulas for computing rewards:

**Reward function** (used in training):
```python
# Line 113 in ghm_rewards.py
return a_L * dt - (1.0 + self.issuance_cost) * a_E
```

**Environment** (used in... somewhere):
```python
# Line 160 in ghm_equity_env.py
reward -= equity_gross  # No cost multiplier!
```

### Why This Matters

The two formulas give **different rewards** for the same action:

Example with `issuance_cost = 0.1`:
- Action: `a_E = 0.5` (issue equity)
- Reward function: `-1.1 * 0.5 = -0.55`
- Environment: `-0.5`

**10% difference in rewards!**

This creates inconsistency between:
- Training dynamics (uses reward function)
- Evaluation/simulation (might use environment)

### The Fix

Use the same formula everywhere. The reward function version is correct (includes issuance cost).

---

## Combined Effect: The Bankruptcy Trap

When all three bugs combine:

```
Scenario: Agent with c = 0.1 at time τ = 5.0

Option A: Sustainable dividend policy
- Pay a_L = 0.18 (sustainable rate)
- Reward per step: 0.18 * 0.1 = 0.018
- Over 50 steps to τ = 10: ~0.018 * 50 = 0.9
- Discounted: ~0.7
- Total return: ~0.7

Option B: Bankruptcy policy (EXPLOITS ALL BUGS)
- Pay a_L = 100.0 (way above cash)
- Immediate bankruptcy
- BUG #1: Dividend reward NOT zeroed: 100.0 * 0.1 = 10.0 ✓
- BUG #2: Get huge liquidation value: 4.95 ✓
- Total return: 10.0 + 4.95 = 14.95

Ratio: 14.95 / 0.7 = 21.4x more rewarding!
```

**No wonder the agent learns to bankrupt itself!**

---

## Evidence from Your Plots

### Step 500 (Early Training)
- Dividend policy: 4-5 (somewhat reasonable)
- Agent hasn't fully learned the bankruptcy exploit yet

### Step 1000 (Learning)
- Dividend policy: 5.5-7.5 (increasing)
- Agent starting to discover higher dividends → higher rewards

### Step 7000 (Converged to Bug)
- Dividend policy: 7-10 (catastrophic!)
- **ALL time horizons converge to ~10** (the bankruptcy policy)
- Even at c=0, paying dividends of 8+ (impossible without bankruptcy)
- Agent has fully learned: "Pay maximum dividend → bankrupt → collect both rewards"

---

## How to Verify These Bugs

### Test 1: Manual Trace

Add print statements in `trajectory.py:255-265`:

```python
print(f"Step {t}:")
print(f"  Before: active={active[0].item()}, terminated={terminated[0].item()}")
print(f"  Reward: {rewards[0, t].item()}")
print(f"  Mask: {masks[0, t].item()}")
print(f"  After: active={active[0].item()}")
```

Run with a bankruptcy-inducing policy. You'll see:
```
Step 0:
  Before: active=1, terminated=1
  Reward: 10.0
  Mask: 1.0  ← Should be 0!
  After: active=0
```

### Test 2: Check Liquidation Value

```python
from macro_rl.dynamics.ghm_equity import GHMEquityParams

params = GHMEquityParams()
print(f"Liquidation value: {params.liquidation_value}")
# Output: 4.95

sustainable_pv = params.alpha / params.r
print(f"Sustainable PV: {sustainable_pv}")
# Output: 2.57

print(f"Ratio: {params.liquidation_value / sustainable_pv}")
# Output: 1.92 (192%!)
```

### Test 3: Compare Reward Formulas

Look at:
- `macro_rl/rewards/ghm_rewards.py:113`
- `macro_rl/envs/ghm_equity_env.py:160`

They're different!

---

## Recommended Fixes (Priority Order)

### 1. Fix Reward Masking (CRITICAL - DO FIRST)

**File:** `macro_rl/simulation/trajectory.py`

**Change lines 256-265 from:**
```python
terminated = self._check_termination(states[:, t + 1, :])
masks[:, t] = active.to(dtype=masks.dtype)
rewards[:, t] = rewards[:, t] * masks[:, t]
active = active & (~terminated)
```

**To:**
```python
terminated = self._check_termination(states[:, t + 1, :])
active = active & (~terminated)  # Update active FIRST
masks[:, t] = active.to(dtype=masks.dtype)  # THEN set mask
rewards[:, t] = rewards[:, t] * masks[:, t]
```

**Why:** This ensures rewards at bankruptcy are properly zeroed.

### 2. Fix Liquidation Value (HIGH PRIORITY)

**File:** `macro_rl/dynamics/ghm_equity.py`

**Change line 52 from:**
```python
self.liquidation_value = self.omega * self.alpha / (self.r - self.mu)
```

**To:**
```python
self.liquidation_value = 0.0  # Equity holders get nothing in bankruptcy
```

**Why:** Removes the bankruptcy incentive entirely.

### 3. Fix Reward Consistency (HIGH PRIORITY)

**File:** `macro_rl/envs/ghm_equity_env.py`

**Change line 160 to match the reward function formula.**

---

## Expected Outcome After Fixes

After fixing just Bug #1:
- Agent should no longer receive double rewards
- Bankruptcy becomes less attractive (only liquidation value)
- But might still bankrupt if liquidation value is high

After fixing Bugs #1 and #2:
- Bankruptcy provides zero value
- Agent should learn to avoid bankruptcy
- Dividend policy should respect cash constraints
- Policy should look like: higher dividends at higher cash levels

After fixing all three bugs:
- Consistent economic model
- Training matches theory
- Realistic firm behavior

---

## Summary Table

| Bug | Severity | Impact | Fix Effort |
|-----|----------|--------|------------|
| #1: Reward masking | CRITICAL | Double-counting rewards at bankruptcy | 1 line |
| #2: Liquidation value | HIGH | Makes bankruptcy rewarding | 1 line |
| #3: Reward inconsistency | MEDIUM | Training/eval mismatch | Few lines |

**Total fix effort: ~3 lines of code changed!**

But the impact is **massive**: these bugs completely change the learned policy from "sustainable firm management" to "rush to bankruptcy."
