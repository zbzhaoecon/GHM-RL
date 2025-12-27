# Control Module

## Purpose

The `control` module defines how control variables (actions) are structured, bounded, and masked for feasibility in continuous-time control problems.

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

### Spcific ActionMasker (`masking.py`)

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

