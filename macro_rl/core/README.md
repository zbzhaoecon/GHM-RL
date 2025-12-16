# Core Module

## Purpose

The `core` module provides foundational abstractions used throughout the model-based RL framework. These components are framework-agnostic and can be reused across different continuous-time models.

## Components

### 1. StateSpace (`state_space.py`)

**Purpose**: Encapsulate state space properties including dimensionality, bounds, and names.

**Key Features**:
- Bounds checking and validation
- Uniform sampling for exploration
- Normalization/denormalization utilities
- Clipping operations

**Implementation Guidance**:

```python
# State space for GHM 1D model (cash only)
state_space = StateSpace(
    dim=1,
    lower=torch.tensor([0.0]),
    upper=torch.tensor([10.0]),
    names=("cash",)
)

# State space for 2D model with time-to-horizon
state_space = StateSpace(
    dim=2,
    lower=torch.tensor([0.0, 0.0]),
    upper=torch.tensor([10.0, 5.0]),
    names=("cash", "tau")
)
```

**TODO for Implementation**:
- [ ] Implement `contains()` - check if states within bounds
- [ ] Implement `sample_uniform()` - uniform sampling over bounds
- [ ] Implement `normalize()`/`denormalize()` - map to/from [0,1]^d
- [ ] Implement `clip()` - enforce bounds
- [ ] Add validation in `__post_init__`
- [ ] Add unit tests with edge cases (boundary values, batched inputs)

### 2. ParameterManager (`params.py`)

**Purpose**: Manage model parameters with validation, serialization, and sweep capabilities.

**Key Features**:
- Parameter validation (positivity, economic constraints)
- JSON serialization/deserialization
- Parameter sweep generation for sensitivity analysis

**Implementation Guidance**:

```python
# Usage with GHM parameters
params = GHMEquityParams(r=0.05, mu=0.02, ...)
manager = ParameterManager(params)

# Validate parameters
manager.validate()  # Checks r > 0, σ > 0, r > μ, etc.

# Save/load
manager.to_json("configs/baseline.json")
loaded = ParameterManager.from_json("configs/baseline.json", GHMEquityParams)

# Generate parameter sweep
configs = manager.sweep("r", [0.03, 0.05, 0.07])
```

**TODO for Implementation**:
- [ ] Implement `validate()` with model-specific constraints
  - Positivity: r, μ, λ, σ_X, σ_A > 0
  - Economic: r > μ (else infinite value)
  - Numerical: Check for stability bounds
- [ ] Implement `to_dict()`/`to_json()` handling torch.Tensor
- [ ] Implement `from_json()` with proper type conversion
- [ ] Implement `sweep()` for grid search
- [ ] Add support for nested parameter structures
- [ ] Add unit tests for serialization round-trips

## Design Principles

1. **Framework Agnostic**: Core components don't depend on specific models
2. **Type Safe**: Use dataclasses and type hints throughout
3. **Batched Operations**: All operations support batch dimensions
4. **PyTorch Native**: Use torch.Tensor for all numerical operations
5. **Validation First**: Fail fast with clear error messages

## Testing Strategy

Create `tests/core/` with:
- `test_state_space.py` - Test all StateSpace methods
- `test_params.py` - Test ParameterManager with mock parameters

Key test cases:
- Boundary conditions (exactly at bounds)
- Batch operations (various batch shapes)
- Invalid inputs (out of bounds, wrong dimensions)
- Serialization round-trips
- Parameter validation edge cases

## Integration with Other Modules

```
core/
  ↓ (used by)
dynamics/     - StateSpace for defining state bounds
  ↓
simulation/   - StateSpace for sampling, ParameterManager for configs
  ↓
solvers/      - StateSpace for interior point sampling
```

## Future Extensions

- [ ] Add `StateSpace.sample_boundary()` for boundary point sampling
- [ ] Add `StateSpace.grid()` for creating uniform grids
- [ ] Add `ParameterManager.random_search()` for random parameter sampling
- [ ] Add support for parameter priors (Bayesian optimization)
