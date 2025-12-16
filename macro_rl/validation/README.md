# Validation Module

## Purpose

Provide rigorous validation tools to verify that learned solutions actually solve the control problem correctly.

## Why Validation Matters

Unlike supervised learning (where you can check test accuracy), in RL you need to verify:
1. **Optimality**: Does the policy maximize the objective?
2. **Correctness**: Does the value function satisfy the HJB equation?
3. **Boundary conditions**: Are boundary conditions met?

## Components

### 1. HJBValidator (`hjb_residual.py`)

**Purpose**: Compute HJB residual to verify value function correctness.

**Usage**:
```python
validator = HJBValidator(dynamics, control_spec)
residuals = validator.compute_residual(value_fn, test_states)
stats = validator.compute_statistics(value_fn)
print(f"Mean residual: {stats['mean']:.6f}")
print(f"Max residual: {stats['max']:.6f}")
validator.plot_residual(value_fn)
```

**Acceptance Criteria**:
- Mean residual < 0.01
- Max residual < 0.1
- Residual should be uniformly small (not localized errors)

### 2. BoundaryValidator (`boundary_conditions.py`)

**Purpose**: Check boundary conditions.

**For GHM**:
- At c = 0: V(0) = ω·α/(r-μ)
- At dividend barrier c*: V_c(c*) = 1

### 3. AnalyticalComparator (`analytical_comparison.py`)

**Purpose**: Compare with known solutions (when available).

## TODO for Implementation

- [ ] Implement HJB residual computation
- [ ] Implement boundary condition checks
- [ ] Add visualization tools
- [ ] Add unit tests with known solutions

## Testing Strategy

Test validators using:
1. Toy problems with known solutions
2. Deliberately wrong value functions (should fail validation)
3. Approximate solutions (should have small but non-zero residuals)
