# Discount Rate Inconsistency Fix Plan

## Summary of Issues

We identified critical inconsistencies in how discount rates are handled across the codebase:

### Issue 1: Dual Storage of Discount Rate
- **Reward Function**: Stores `discount_rate_value` (from config or r-μ)
- **Dynamics**: Computes via `discount_rate()` method (always r-μ)
- **Problem**: Two sources of truth with no guarantee they match

### Issue 2: Inconsistent Usage
- **Liquidation value computation** uses `reward_fn.discount_rate_value`
- **Actual discounting** (dense & sparse) uses `dynamics.discount_rate()`
- **Problem**: If these differ, liquidation value is mathematically wrong

### Issue 3: Mathematical Error in Liquidation Value
Current calculation:
```
PV = e^(-ρ₂·T_term) · (ω·α / ρ₁)
```
where ρ₁ = `reward_fn.discount_rate_value`, ρ₂ = `dynamics.discount_rate()`

Correct calculation:
```
PV = e^(-ρ·T_term) · (ω·α / ρ)
```
where ρ is the **same** discount rate everywhere.

**Impact**: If ρ₁ ≠ ρ₂, errors can be 30-50% in liquidation value!

## Root Cause

The discount rate ρ = r - μ is fundamentally a **dynamics parameter** (derived from interest rate and growth rate), but the reward function needs it for liquidation value computation. The current architecture allows these to diverge.

## Proposed Solution

### Design Principle: **Single Source of Truth**

The discount rate should come from **dynamics only**, since r and μ are dynamics parameters.

### Architecture Changes

```
BEFORE:
┌─────────────────┐     ┌──────────────────┐
│ Dynamics        │     │ RewardFunction   │
│ - discount_rate()│    │ - discount_rate_value │
│   returns r-μ   │     │   (from config)  │
└─────────────────┘     └──────────────────┘
         │                       │
         │                       │ (used for liquidation_value)
         │                       │
         └───────┬───────────────┘
                 │
         ┌───────▼────────┐
         │ Simulator      │
         │ uses dynamics  │
         │ discount_rate()│
         └────────────────┘
         INCONSISTENT! ρ₁ ≠ ρ₂

AFTER:
┌─────────────────┐
│ Dynamics        │
│ - discount_rate()│ ◄─── SINGLE SOURCE
│   returns r-μ   │
└────────┬────────┘
         │
         │ (used everywhere)
         │
    ┌────┴─────────────────┐
    │                      │
┌───▼──────────┐   ┌──────▼─────────┐
│ RewardFn     │   │ Simulator      │
│ (no storage) │   │                │
└──────────────┘   └────────────────┘
```

## Implementation Plan

### Phase 1: Preparation (Safe to Merge)

**1.1. Add deprecation warning to GHMRewardFunction**
```python
def __init__(
    self,
    discount_rate: float = None,  # DEPRECATED
    ...
):
    if discount_rate is not None:
        warnings.warn(
            "discount_rate parameter is deprecated. "
            "Discount rate will be obtained from dynamics.",
            DeprecationWarning
        )
    self._discount_rate = discount_rate  # Store temporarily
```

**1.2. Add helper method to compute liquidation value**
```python
def compute_liquidation_value(
    self,
    discount_rate: float
) -> float:
    """Compute liquidation value using given discount rate."""
    if discount_rate > 0:
        return self.liquidation_rate * self.liquidation_flow / discount_rate
    return 0.0
```

**1.3. Update setup_utils.py to pass dynamics discount rate**
```python
# Get discount rate from dynamics (single source of truth)
discount_rate = params.r - params.mu

reward_fn = GHMRewardFunction(
    discount_rate=discount_rate,  # Will be deprecated
    liquidation_rate=config.reward.liquidation_rate,
    liquidation_flow=config.reward.liquidation_flow,
    fixed_cost=fixed_cost,
    proportional_cost=params.p,
)

# Validate consistency
assert abs(reward_fn.discount_rate_value - dynamics.discount_rate()) < 1e-9, \
    f"Discount rate mismatch: reward={reward_fn.discount_rate_value}, dynamics={dynamics.discount_rate()}"
```

### Phase 2: Refactor GHMRewardFunction (Breaking Change)

**2.1. Remove discount_rate parameter**
```python
def __init__(
    self,
    liquidation_rate: float = 1.0,
    liquidation_flow: float = 0.0,
    fixed_cost: float = 0.0,
    proportional_cost: float = 1.06,
):
    """
    Initialize GHM reward function.

    Note: Discount rate is obtained from dynamics, not stored here.
    Liquidation value is computed on demand.
    """
    self.proportional_cost = proportional_cost
    self.fixed_cost = fixed_cost
    self.liquidation_rate = liquidation_rate
    self.liquidation_flow = liquidation_flow
    self.issuance_cost = 1.0

    # Don't precompute liquidation_value - compute on demand
```

**2.2. Update terminal_reward to accept discount_rate**
```python
def terminal_reward(
    self,
    state: Tensor,
    terminated: Tensor,
    discount_rate: float,  # NEW: required parameter
    value_function=None,
) -> Tensor:
    """
    Compute terminal reward.

    Args:
        discount_rate: Discount rate ρ = r - μ (from dynamics)
    """
    batch_size = state.shape[0]
    device = state.device

    # Compute liquidation value using provided discount rate
    if discount_rate > 0:
        liquidation_value = self.liquidation_rate * self.liquidation_flow / discount_rate
    else:
        liquidation_value = 0.0

    terminal_rewards = torch.full(
        (batch_size,),
        liquidation_value,
        dtype=torch.float32,
        device=device
    )

    return terminal_rewards
```

**2.3. Update trajectory simulator to pass discount_rate**
```python
# In rollout() method, line 278
discount_rate = self.dynamics.discount_rate()  # Get once

terminal_rewards = self.reward_fn.terminal_reward(
    states[:, -1, :],
    terminal_mask,
    discount_rate=discount_rate,  # Pass to reward function
    value_function=self.value_function
)
```

**2.4. Update setup_utils.py**
```python
# Remove discount_rate calculation for reward_fn
reward_fn = GHMRewardFunction(
    # discount_rate removed!
    liquidation_rate=config.reward.liquidation_rate,
    liquidation_flow=config.reward.liquidation_flow,
    fixed_cost=fixed_cost,
    proportional_cost=params.p,
)
```

### Phase 3: Update Base Class (Optional but Recommended)

**3.1. Update RewardFunction base class**
```python
@abstractmethod
def terminal_reward(
    self,
    state: Tensor,
    terminated: Tensor,
    discount_rate: float,  # Make this required in base class
) -> Tensor:
    """
    Compute terminal reward.

    Args:
        state: Terminal states (batch, state_dim)
        terminated: Boolean mask (batch,)
        discount_rate: Discount rate ρ (from dynamics)
    """
    pass
```

### Phase 4: Configuration Changes

**4.1. Remove discount_rate from reward config schema**
```yaml
# OLD (configs/*.yaml)
reward:
  discount_rate: 0.02  # REMOVE THIS
  liquidation_rate: 0.0
  liquidation_flow: 0.0

# NEW
reward:
  liquidation_rate: 0.0
  liquidation_flow: 0.0
  # discount_rate removed - always uses dynamics.r - dynamics.mu
```

**4.2. Add validation in config_manager.py**
```python
def validate_config(self):
    """Validate configuration consistency."""
    # Ensure discount rate not in reward config
    if hasattr(self.config.reward, 'discount_rate'):
        warnings.warn(
            "config.reward.discount_rate is deprecated. "
            "Discount rate is computed as dynamics.r - dynamics.mu"
        )
```

### Phase 5: Update Tests

**5.1. Update all test files that create GHMRewardFunction**
```python
# OLD
reward_fn = GHMRewardFunction(
    discount_rate=0.02,  # REMOVE
    liquidation_rate=0.0,
    liquidation_flow=0.0,
)

# NEW
reward_fn = GHMRewardFunction(
    liquidation_rate=0.0,
    liquidation_flow=0.0,
)
```

**5.2. Update test calls to terminal_reward**
```python
# OLD
terminal_reward = reward_fn.terminal_reward(state, terminated)

# NEW
discount_rate = dynamics.discount_rate()
terminal_reward = reward_fn.terminal_reward(state, terminated, discount_rate)
```

**5.3. Add consistency test**
```python
def test_discount_rate_consistency():
    """Verify discount rate is used consistently everywhere."""
    # Setup
    params = GHMEquityParams(r=0.03, mu=0.01)
    dynamics = GHMEquityDynamics(params)
    reward_fn = GHMRewardFunction(liquidation_rate=0.55, liquidation_flow=0.18)

    discount_rate = dynamics.discount_rate()
    assert discount_rate == 0.02  # r - mu = 0.03 - 0.01

    # Compute terminal reward
    state = torch.zeros(10, 1)
    terminated = torch.ones(10)
    terminal_rewards = reward_fn.terminal_reward(state, terminated, discount_rate)

    # Verify liquidation value uses correct discount rate
    expected_liquidation = 0.55 * 0.18 / 0.02  # ω·α / ρ
    assert torch.allclose(terminal_rewards, torch.tensor(expected_liquidation))
```

## Files to Modify

### Core Changes
1. `macro_rl/rewards/ghm_rewards.py` - Remove discount_rate parameter, update terminal_reward
2. `macro_rl/rewards/base.py` - Update terminal_reward signature
3. `macro_rl/simulation/trajectory.py` - Pass discount_rate to terminal_reward
4. `macro_rl/config/setup_utils.py` - Remove discount_rate from reward_fn creation

### Test Updates
5. `tests/test_ghm_rewards.py` - Update all tests
6. `tests/test_rewards_base.py` - Update base class tests
7. `tests/test_simulation_trajectory.py` - Update trajectory tests
8. `test_sparse_dense_equivalence.py` - Verify still equivalent
9. Add new test: `tests/test_discount_rate_consistency.py`

### Configuration
10. `configs/time_augmented_config.yaml` - Remove reward.discount_rate
11. `configs/time_augmented_sparse_config.yaml` - Remove reward.discount_rate
12. `configs/actor_critic_time_augmented_config.yaml` - Remove reward.discount_rate

### Documentation
13. `SPARSE_REWARDS_IMPLEMENTATION.md` - Update with correct formulas
14. `macro_rl/rewards/README.md` - Document discount rate handling
15. This plan document

## Migration Path for Users

### For users with configs:
```bash
# Old config (will show deprecation warning)
reward:
  discount_rate: 0.02  # This will be ignored

# New config (recommended)
reward:
  # discount_rate removed - uses dynamics.r - dynamics.mu automatically
```

### For users creating reward functions programmatically:
```python
# Old (deprecated)
reward_fn = GHMRewardFunction(discount_rate=0.02, ...)

# New (required)
reward_fn = GHMRewardFunction(...)  # No discount_rate parameter
# When calling terminal_reward:
terminal_rewards = reward_fn.terminal_reward(state, terminated, discount_rate=dynamics.discount_rate())
```

## Testing Strategy

1. **Unit tests**: Verify each component independently
2. **Integration tests**: Verify sparse/dense equivalence maintained
3. **Regression tests**: Compare results before/after on saved trajectories
4. **Numerical tests**: Verify liquidation value formulas are correct

## Rollout Plan

### Step 1: Add validation (immediate, non-breaking)
- Add assertion in setup_utils.py to catch mismatches
- Run on all configs to ensure consistency

### Step 2: Deprecation warnings (next PR)
- Add warnings when discount_rate passed to GHMRewardFunction
- Update documentation

### Step 3: Breaking change (major version bump)
- Remove discount_rate parameter
- Update all configs and tests
- Update base class

## Expected Benefits

1. ✅ **Correctness**: Liquidation values will be mathematically correct
2. ✅ **Consistency**: Single source of truth for discount rate
3. ✅ **Clarity**: Clear that ρ = r - μ comes from dynamics
4. ✅ **Maintainability**: Less code duplication, fewer parameters
5. ✅ **Debuggability**: Easier to trace where discount rate comes from

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Deprecation period with warnings |
| Breaking existing configs | Validation + clear migration guide |
| Changing numerical results | Regression tests to verify |
| Users with custom reward functions | Update base class carefully, provide examples |

## Timeline Estimate

- **Phase 1** (Validation): 1-2 hours
- **Phase 2** (Refactor): 3-4 hours
- **Phase 3** (Base class): 1-2 hours
- **Phase 4** (Config): 1 hour
- **Phase 5** (Tests): 2-3 hours
- **Total**: 8-12 hours of development + testing

## Success Criteria

- [ ] All tests pass
- [ ] Sparse and dense rewards still mathematically equivalent
- [ ] No discount rate mismatches possible
- [ ] Liquidation value computed with consistent discount rate
- [ ] Clear error messages if something goes wrong
- [ ] Documentation updated and clear
