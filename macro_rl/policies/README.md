# Policies Module

## Purpose

Provide policy architectures for continuous-time control, supporting both stochastic and deterministic policies.

## Key Components

### 1. GaussianPolicy (`neural.py`)

**Purpose**: Stochastic policy with Gaussian action distribution.

**Features**:
- Learnable mean and std
- Reparameterization trick for pathwise gradients
- Entropy computation for exploration

**Usage**:
```python
policy = GaussianPolicy(state_dim=1, action_dim=2, hidden_dims=[64, 64])
action = policy.sample(state)  # Stochastic
log_prob = policy.log_prob(state, action)  # For REINFORCE
action_reparam = policy.reparameterize(state, noise)  # For pathwise
```

### 2. BarrierPolicy (`barrier.py`)

**Purpose**: Classical barrier policy from corporate finance literature.

**TODO**: Implement barrier logic for comparison with learned policies.

## TODO for Implementation

- [ ] Implement GaussianPolicy network architecture
- [ ] Implement reparameterization trick
- [ ] Add proper weight initialization
- [ ] Add unit tests for each policy type
- [ ] Implement BarrierPolicy for baselines

## Testing

Create tests/policies/ with:
- test_gaussian_policy.py
- test_reparameterization.py (verify gradients)
- test_barrier_policy.py
