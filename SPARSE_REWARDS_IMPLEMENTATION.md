# Sparse Rewards Implementation

## Motivation

The policy collapse issue persists even after fixing the mask logic bug. The problem may be related to **dense rewards** causing conflicting gradient signals.

### Dense Rewards Problem

**Current (Dense) Approach:**
- Reward computed at every timestep: `r_t = a_L·dt - (1+λ)·a_E`
- Return = sum of discounted per-step rewards + terminal reward
- **Issue**: An aggressive dividend action gets immediate positive reward, but might cause bankruptcy soon after (negative terminal value)
- **Result**: Conflicting gradient signals - "this action gave +reward but the trajectory got -return"
- This can cause high variance in gradients and unstable learning

### Sparse Rewards Solution

**New (Sparse) Approach:**
- No per-step rewards during rollout
- Compute total trajectory return directly at the end:
  ```
  R = ∫_0^T e^(-ρt) (dL_t - (1+λ)dE_t) dt + terminal_value
    = Σ_t e^(-ρt·dt) (a_L[t]·dt - (1+λ)·a_E[t]) · mask[t] + terminal
  ```
- **Advantage**: Clearer gradient signal - entire trajectory gets same credit/blame
- **Benefit**: Reduces gradient variance, improves stability

## Implementation

### 1. TrajectorySimulator Changes

Added `use_sparse_rewards` flag to `TrajectorySimulator`:

```python
def __init__(
    self,
    dynamics,
    control_spec,
    reward_fn,
    dt: float,
    T: float,
    integrator: Optional[object] = None,
    value_function: Optional[object] = None,
    use_sparse_rewards: bool = False,  # NEW
):
    ...
    self.use_sparse_rewards = use_sparse_rewards
```

### 2. Sparse Return Computation

Added new method `_compute_sparse_returns()`:

```python
def _compute_sparse_returns(
    self,
    states: Tensor,
    actions: Tensor,
    terminal_rewards: Tensor,
    masks: Tensor,
    discount_rate: float,
) -> Tensor:
    """
    Compute trajectory returns using sparse rewards.

    Instead of accumulating per-step rewards, directly compute the total
    discounted payout for each trajectory.
    """
    batch_size, n_steps = actions.shape[0], actions.shape[1]
    device = actions.device

    returns = torch.zeros(batch_size, device=device)

    # Compute discounted sum of net payouts directly from actions
    for t in range(n_steps):
        discount = torch.exp(torch.tensor(-discount_rate * t * self.dt, device=device))

        # Net payout at time t
        a_L = actions[:, t, 0]  # Dividend rate
        a_E = actions[:, t, 1]  # Equity issuance
        net_payout = a_L * self.dt - (1.0 + self.reward_fn.issuance_cost) * a_E

        # Add discounted net payout (only if trajectory was active)
        returns = returns + discount * net_payout * masks[:, t]

    # Add discounted terminal reward
    termination_times = masks.sum(dim=1)
    terminal_discount = torch.exp(-discount_rate * termination_times * self.dt)
    returns = returns + terminal_discount * terminal_rewards

    return returns
```

### 3. Configuration Support

Added `use_sparse_rewards` to training config:

```yaml
training:
  use_sparse_rewards: true  # Enable sparse rewards
```

Created new config file: `configs/time_augmented_sparse_config.yaml`

### 4. Setup Utils Integration

Modified `macro_rl/config/setup_utils.py` to pass flag to simulator:

```python
use_sparse_rewards = config.training.get('use_sparse_rewards', False)
simulator = TrajectorySimulator(
    ...
    use_sparse_rewards=use_sparse_rewards,
)
```

## Usage

### Training with Sparse Rewards

```bash
python scripts/train_with_config.py --config configs/time_augmented_sparse_config.yaml
```

### Training with Dense Rewards (Original)

```bash
python scripts/train_with_config.py --config configs/time_augmented_config.yaml
```

## Mathematical Equivalence

**Important**: Sparse and dense rewards are mathematically equivalent in terms of the total return:

**Dense:**
```
R = Σ_t e^(-ρt·dt) · r_t · mask_t + terminal
where r_t = a_L[t]·dt - (1+λ)·a_E[t]
```

**Sparse:**
```
R = Σ_t e^(-ρt·dt) · (a_L[t]·dt - (1+λ)·a_E[t]) · mask_t + terminal
```

These are identical! The difference is in **when** the computation happens:
- **Dense**: Compute `r_t` during rollout, accumulate in `_compute_returns()`
- **Sparse**: Skip per-step rewards, compute total return in `_compute_sparse_returns()`

## Expected Benefits

1. **Reduced gradient variance**: Simpler credit assignment
2. **Improved stability**: No conflicting signals between per-step rewards and terminal values
3. **Better learning**: Clearer policy gradient signal

## Files Modified

1. `macro_rl/simulation/trajectory.py` - Added sparse rewards support
2. `macro_rl/config/setup_utils.py` - Pass sparse rewards flag
3. `configs/time_augmented_sparse_config.yaml` - New config with sparse rewards enabled

## Testing

Run training with both configs and compare:
- Policy collapse behavior
- Gradient magnitudes
- Training stability
- Final performance

## Notes

- Mathematically equivalent to dense rewards
- Main benefit is **variance reduction** in gradient estimation
- Can throw more compute at it (more trajectories) for better estimates
- No performance overhead - actually slightly faster (skips per-step reward computation during rollout)
