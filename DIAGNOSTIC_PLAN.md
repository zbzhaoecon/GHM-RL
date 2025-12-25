# Diagnostic Plan for Bankruptcy-Seeking Bug

## Critical Bugs Identified

### 🔴 BUG #1: Reward Masking at Termination (CRITICAL)
**File:** `macro_rl/simulation/trajectory.py:259-265`

**Problem:** When bankruptcy occurs at step `t`, the reward earned at step `t` is NOT zeroed out.

**Why:** The mask is set BEFORE updating the active status:
```python
masks[:, t] = active.to(dtype=masks.dtype)  # active=True (not updated yet!)
rewards[:, t] = rewards[:, t] * masks[:, t]  # mask=1, so reward NOT zeroed
active = active & (~terminated)  # NOW active becomes False
```

**Economic Impact:**
- Agent receives BOTH the dividend payout AND liquidation value
- Example: Pay 0.1 dividend → bankrupt → get 4.95 liquidation = 5.05 total
- This makes bankruptcy the optimal strategy!

**Fix:**
```python
# Option A: Update active BEFORE setting mask
terminated = self._check_termination(states[:, t + 1, :])
active = active & (~terminated)
masks[:, t] = active.to(dtype=masks.dtype)
rewards[:, t] = rewards[:, t] * masks[:, t]

# Option B: Use terminated flag directly
masks[:, t] = active.to(dtype=masks.dtype)
rewards[:, t] = rewards[:, t] * masks[:, t] * (~terminated).to(dtype=rewards.dtype)
active = active & (~terminated)
```

---

### 🟡 BUG #2: Liquidation Value Too Large (HIGH PRIORITY)
**File:** `macro_rl/dynamics/ghm_equity.py:52`

**Problem:** Liquidation value is computed as:
```python
liquidation_value = omega * alpha / (r - mu) = 0.55 * 0.18 / 0.02 = 4.95
```

This is **larger than sustainable dividend streams**, making bankruptcy attractive.

**Why It's Wrong:**
- `alpha` is normal operating cash flow, NOT recovery value
- In reality, liquidation destroys value (fire-sale discounts, legal costs, etc.)
- omega=0.55 suggests 55% recovery, but recovery of WHAT?

**Fix:** Set liquidation value to near-zero or explicit small constant:
```python
self.liquidation_value = 0.0  # Or small positive like 0.1
```

---

### 🟡 BUG #3: Inconsistent Reward Computation (HIGH PRIORITY)
**Files:**
- `macro_rl/rewards/ghm_rewards.py:113`
- `macro_rl/envs/ghm_equity_env.py:160`

**Problem:** Two different formulas:

**Reward function:**
```python
return a_L * dt - (1.0 + self.issuance_cost) * a_E  # Includes cost
```

**Environment:**
```python
reward -= equity_gross  # No cost multiplier!
```

**Fix:** Use consistent formula everywhere (the reward function version is correct).

---

### 🟠 BUG #4: Terminal Reward Uses Wrong States (MEDIUM PRIORITY)
**File:** `macro_rl/simulation/trajectory.py:277`

**Problem:**
```python
terminal_rewards = self.reward_fn.terminal_reward(
    states[:, -1, :],  # Uses FINAL state for all trajectories
    terminal_mask,
    value_function=self.value_function
)
```

For trajectories that terminate early, should use termination state, not final state.

**Fix:** Track termination states and use those instead.

---

## Diagnostic Tests

### Test 1: Verify Reward Masking Bug
**Goal:** Confirm that bankruptcy rewards are not zeroed out.

**Approach:**
1. Create simple 2-step environment
2. Force bankruptcy at step 1
3. Track reward and mask values
4. Verify that reward at bankruptcy step is non-zero

**Expected:** Bug confirmed if reward ≠ 0

---

### Test 2: Liquidation Value Impact
**Goal:** Measure how liquidation value affects learned policy.

**Approach:**
1. Train with different liquidation values: [0.0, 0.1, 1.0, 4.95]
2. Compare learned policies
3. Measure bankruptcy rate

**Expected:** Lower liquidation → lower bankruptcy rate

---

### Test 3: Trivial Environment - Constant Rewards
**Goal:** Test if RL can learn a simple policy without economic complexity.

**Approach:**
1. Create env where reward = action (no bankruptcy, no constraints)
2. Optimal policy: action = max value
3. Train and check convergence

**Expected:** Should converge to max action

---

### Test 4: Two-Choice Environment
**Goal:** Test if RL prefers higher rewards.

**Approach:**
1. Two discrete actions: A (reward=1) vs B (reward=5)
2. No state dynamics
3. Train for 1000 steps

**Expected:** Should learn to always choose B

---

### Test 5: Constraint Violation Logging
**Goal:** Check if agent tries to violate dividend constraints.

**Approach:**
1. Log pre-clipping and post-clipping actions
2. Measure frequency of violations
3. Track gradient flow through clipping

**Expected:** Frequent violations indicate policy not learning constraints

---

## Implementation Priority

### Phase 1: Critical Bug Fix (DO THIS FIRST!)
1. ✅ Fix reward masking bug in `trajectory.py`
2. ✅ Add unit test to verify fix
3. ✅ Re-run training to see if bankruptcy behavior disappears

### Phase 2: Diagnostic Tests
1. Run Test 1 (verify bug) on BOTH old and new code
2. Run Test 3 (trivial env) to verify basic RL works
3. Run Test 4 (two-choice) to verify value learning

### Phase 3: Hyperparameter Fixes
1. Set liquidation_value = 0.0
2. Fix inconsistent reward computation
3. Re-run training

### Phase 4: Advanced Diagnostics
1. Test 2 (liquidation value sweep)
2. Test 5 (constraint violation logging)
3. Analyze gradient flow

---

## Quick Verification Commands

```bash
# Check current liquidation value
grep -n "liquidation_value" macro_rl/dynamics/ghm_equity.py

# Check reward masking logic
grep -A 10 "Zero out rewards" macro_rl/simulation/trajectory.py

# Run simple test
python -m pytest tests/ -v -k "test_trajectory"
```

---

## Files to Modify

1. **CRITICAL:** `macro_rl/simulation/trajectory.py` (lines 259-265)
2. **HIGH:** `macro_rl/dynamics/ghm_equity.py` (line 52)
3. **HIGH:** `macro_rl/envs/ghm_equity_env.py` (line 160)
4. **MEDIUM:** `macro_rl/simulation/trajectory.py` (line 277)

---

## Expected Outcomes After Fixes

After fixing Bug #1 (reward masking):
- Agent should NO LONGER learn to bankrupt immediately
- Dividend policy should respect cash constraints
- Value function should be monotonic in cash reserves

After fixing Bug #2 (liquidation value):
- Further reduction in bankruptcy-seeking
- Policy should prefer survival over liquidation

After fixing Bug #3 (reward consistency):
- Training should match true economic model
- Equity issuance should be properly penalized
