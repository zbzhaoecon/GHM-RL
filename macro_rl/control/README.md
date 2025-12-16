# Control Module

## Purpose

The `control` module defines how control variables (actions) are structured, bounded, and masked for feasibility in continuous-time control problems.

## Why This Module Is Critical

The previous implementation had a **fundamental error**: it treated GHM as a single-control problem (dividend only). The correct formulation has **two controls**:

1. **a_L**: Dividend payout rate (continuous, flow)
2. **a_E**: Equity issuance amount (singular, impulse)

This module fixes that and provides proper infrastructure for multi-control problems.

## Components

### 1. ControlSpec (`base.py`)

**Purpose**: Abstract base class for control specifications.

**Key Features**:
- Define control dimensions and bounds
- Distinguish continuous vs singular (impulse) controls
- Normalize/denormalize actions
- Abstract `apply_mask()` for model-specific feasibility

**Usage Pattern**:

```python
class MyControlSpec(ControlSpec):
    def __init__(self):
        super().__init__(
            dim=2,
            names=("control_1", "control_2"),
            lower=torch.tensor([0.0, -1.0]),
            upper=torch.tensor([10.0, 1.0]),
            is_singular=(False, True),
        )

    def apply_mask(self, action, state, dt):
        # Model-specific masking logic
        return masked_action
```

**TODO for Implementation**:
- [ ] Implement `clip()` - torch.clamp with proper broadcasting
- [ ] Implement `normalize()`/`denormalize()` for [0,1] mapping
- [ ] Implement `sample_uniform()` for exploration
- [ ] Add validation in `__post_init__`
- [ ] Add unit tests for all methods

### 2. GHMControlSpec (`ghm_control.py`)

**Purpose**: Two-control specification for GHM equity model.

**The Two Controls**:

| Control | Type | Bounds | Constraint |
|---------|------|--------|------------|
| a_L (dividend) | Continuous | [0, a_L_max] | a_L ≤ c/dt |
| a_E (issuance) | Singular | [0, a_E_max] | a_E ∈ {0} ∪ [threshold, a_E_max] |

**Why Dividend Constraint**: Can't pay out more cash than available.
- Cash evolves as: dc = (α + c(r-λ-μ) - a_L)dt + a_E + σ(c)dW
- Must have a_L·dt ≤ c to avoid negative cash (before issuance)

**Why Issuance Threshold**: Fixed cost λ makes tiny issuances wasteful.
- Cost of issuing: (1 + λ)·a_E
- If a_E too small, cost > benefit
- Force a_E = 0 if below threshold, else a_E ≥ threshold

**Implementation Guidance**:

```python
def apply_mask(self, action, state, dt):
    batch_size = action.shape[0]
    c = state[:, 0]  # Cash level

    a_L = action[:, 0]
    a_E = action[:, 1]

    # Constraint 1: Dividend ≤ available cash
    max_dividend = torch.clamp(c / dt, min=0.0, max=self.a_L_max)
    a_L_masked = torch.clamp(a_L, 0.0, max_dividend)

    # Constraint 2: Issuance threshold
    threshold_value = self.issuance_threshold * self.a_E_max
    a_E_masked = torch.where(
        a_E >= threshold_value,
        torch.clamp(a_E, threshold_value, self.a_E_max),
        torch.zeros_like(a_E)
    )

    return torch.stack([a_L_masked, a_E_masked], dim=-1)
```

**TODO for Implementation**:
- [ ] Implement `apply_mask()` as above
- [ ] Implement `compute_net_payout()` - return a_L - a_E
- [ ] Implement `issuance_indicator()` - return 1 if a_E > 0
- [ ] Implement `total_issuance_cost()` - return λ·𝟙(a_E > 0)
- [ ] Add unit tests:
  - Test dividend clipping with various c values
  - Test issuance threshold zeroing
  - Test batch dimensions
  - Test edge cases (c=0, a_L=0, etc.)

### 3. ActionMasker (`masking.py`)

**Purpose**: General utilities for action constraints.

**Features**:
- Composable constraints
- Threshold masking
- Soft clipping (for differentiable simulation)

**Usage Pattern**:

```python
# Create masker
masker = ActionMasker()

# Add dividend constraint
def dividend_constraint(action, state, dt):
    c = state[:, 0]
    max_div = c / dt
    action[:, 0] = torch.minimum(action[:, 0], max_div)
    return action

masker.add_constraint(0, dividend_constraint)

# Apply
masked = masker.apply(action, state, dt)
```

**TODO for Implementation**:
- [ ] Implement `add_constraint()` - store constraints in list
- [ ] Implement `apply()` - apply all constraints sequentially
- [ ] Implement `threshold_mask()` - {0} ∪ [threshold, max]
- [ ] Implement `box_clip()` - standard clipping
- [ ] Implement `soft_clip()` - differentiable alternative
  - Use: `lower + (upper - lower) * sigmoid((x - mid) / temp)`
- [ ] Implement `create_ghm_masker()` factory
- [ ] Add tests for each masking type

## Design Principles

1. **Separate Specification from Masking**: ControlSpec defines what, masking defines how
2. **Support Both Hard and Soft Constraints**: Hard for simulation, soft for differentiability
3. **Batch-Friendly**: All operations vectorized
4. **Model-Agnostic Base**: Specific models subclass ControlSpec

## GHM Model Correction

### Previous (Wrong) Formulation:
```python
# Single control: dividend rate only
action = policy(state)  # Scalar
reward = action  # Just dividend
```

Problems:
- No equity issuance mechanism
- Can't handle barrier/recapitalization
- Doesn't match Bolton et al. paper

### Correct Formulation:
```python
# Two controls: dividend + issuance
action = policy(state)  # (a_L, a_E)
reward = a_L * dt - a_E  # Dividends minus dilution
```

### Impact on Dynamics:
```python
# State evolution
dc = (α + c(r - λ - μ) - a_L) * dt + a_E + σ(c) * dW
     \_________________/          \____/
         drift with div         issuance
```

## Testing Strategy

Create `tests/control/`:

### `test_control_spec.py`
- Test base class methods (clip, normalize, etc.)
- Test custom subclasses

### `test_ghm_control.py`
- **Critical**: Test dividend constraint with c/dt bound
- Test issuance threshold behavior
- Test net payout computation
- Test batch operations
- Test edge cases:
  ```python
  # Edge case 1: Zero cash
  state = torch.tensor([[0.0]])
  action = torch.tensor([[10.0, 0.0]])  # Try to pay dividend
  masked = control.apply_mask(action, state, dt=0.01)
  assert masked[0, 0] == 0.0  # Dividend forced to 0

  # Edge case 2: Below issuance threshold
  action = torch.tensor([[0.0, 0.01]])  # Small issuance
  masked = control.apply_mask(action, state, dt=0.01)
  assert masked[0, 1] == 0.0  # Issuance forced to 0

  # Edge case 3: Batch with mixed conditions
  states = torch.tensor([[0.0], [5.0], [10.0]])
  actions = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
  masked = control.apply_mask(actions, states, dt=0.01)
  # First: dividend = 0 (no cash)
  # Second: dividend = 1.0 (enough cash)
  # Third: dividend = 1.0 (enough cash)
  ```

### `test_masking.py`
- Test threshold masking
- Test soft vs hard clipping
- Test constraint composition

## Integration with Other Modules

```
control/ (defines action space)
    ↓
simulation/ (uses masking during rollouts)
    ↓
policies/ (output must match control dims)
    ↓
solvers/ (optimize over control space)
```

## Common Pitfalls

1. **Forgetting dt in dividend constraint**: a_L ≤ c/dt, not a_L ≤ c
   - Dividend is a *rate* (per unit time)
   - Over dt, payout is a_L·dt
   - Must have a_L·dt ≤ c

2. **Hard masking in differentiable sim**: Use soft_clip for gradients

3. **Issuance cost vs threshold**: Threshold prevents tiny issuances, cost λ affects value

4. **Batch dimension handling**: Ensure masking works for (batch, dim)

## Future Extensions

- [ ] Add `GHMControlSpecWithBarrier` implementation
  - Force recapitalization at barrier c_b
  - Jump to target c_t
- [ ] Add support for bounded controls (e.g., a_L ∈ [-max, max] for loans)
- [ ] Add inequality constraints (e.g., a_L + a_E ≤ budget)
- [ ] Add visualization tools for feasible action regions

## References

- Bolton et al.: "Executive Compensation and Short-termism"
  - Section on equity issuance and recapitalization
  - Two-control formulation
- Sundaresan & Wang: "On the design of contingent capital"
  - Barrier policies
