# Bugs Found: Agent Learns to Bankrupt Itself

## Problem Statement

The RL agent consistently learns to pay huge dividends (>10) even when cash reserves are near zero, leading to immediate bankruptcy. This behavior emerges from **3 critical bugs**.

## Files to Review

1. **DIAGNOSTIC_PLAN.md** - Detailed analysis of bugs with test plans
2. **BUG_ANALYSIS.md** - Deep dive into each bug with concrete examples
3. **QUICK_FIX_GUIDE.md** - Step-by-step instructions to fix the bugs
4. **diagnostic_tests/** - Python scripts to verify the bugs (requires dependencies)

## The 3 Critical Bugs

### 🔴 Bug #1: Reward Masking at Termination (CRITICAL)
- **File:** `macro_rl/simulation/trajectory.py:259-265`
- **Problem:** Rewards at bankruptcy are NOT zeroed out
- **Impact:** Agent gets BOTH dividend AND liquidation value
- **Fix:** Swap lines 259 and 265 (update `active` before setting `masks`)
- **Effort:** 2 lines reordered

### 🟡 Bug #2: Liquidation Value Too Large (HIGH)
- **File:** `macro_rl/dynamics/ghm_equity.py:52`
- **Problem:** Liquidation value (4.95) > Sustainable value (2.57)
- **Impact:** Makes bankruptcy 192% more valuable than survival
- **Fix:** Set `self.liquidation_value = 0.0`
- **Effort:** 1 line changed

### 🟡 Bug #3: Reward Inconsistency (MEDIUM)
- **Files:** `macro_rl/rewards/ghm_rewards.py:113`, `macro_rl/envs/ghm_equity_env.py:160`
- **Problem:** Two different reward formulas
- **Impact:** Training doesn't match economics
- **Fix:** Use consistent formula with issuance cost
- **Effort:** 1 line changed

## Quick Start

**Want to fix it right now?**

→ Read `QUICK_FIX_GUIDE.md` (5 minute fix)

**Want to understand the bugs deeply?**

→ Read `BUG_ANALYSIS.md` (detailed explanation)

**Want to design better tests?**

→ Read `DIAGNOSTIC_PLAN.md` (testing strategy)

## Evidence

Your plots clearly show the progression:

- **Step 500:** Dividend policy ~4-5 (learning)
- **Step 1000:** Dividend policy ~5-7 (discovering exploit)
- **Step 7000:** Dividend policy ~8-10 (fully exploiting bugs)

At step 7000:
- All time horizons converge to same policy (bankruptcy)
- Even at c=0, agent tries to pay 8+ in dividends
- This is only "optimal" because of the bugs!

## Expected Outcome After Fixes

✅ Dividend policy respects cash constraints
✅ Higher dividends at higher cash levels
✅ Policy varies with time horizon
✅ No bankruptcy behavior
✅ Value function monotonic in cash

## Next Steps

1. **Apply the fixes** (see QUICK_FIX_GUIDE.md)
2. **Re-run training** from scratch
3. **Verify behavior** improved
4. **Add unit tests** to prevent regression

If you want to verify the bugs exist before fixing, you can run the diagnostic tests (but they require installing dependencies first).

## Contact

If you have questions about these bugs or the fixes, see the detailed analysis in the other markdown files.

---

**Created:** 2024-12-25
**Analysis by:** Claude Code (Anthropic)
