# Rewards Module

## Purpose

The `rewards` module defines objective functions for continuous-time control problems, including per-step flow rewards and terminal values.

## Why This Module Matters

The reward function must correctly reflect the economic objective. For GHM:

**Previous (Wrong)**:
```python
reward = dividend_rate  # Only dividends
```

**Correct**:
```python
reward = dividend_flow - issuance_cost
       = a_L * dt - (1 + λ) * a_E
```

The firm maximizes **net payout** to shareholders, accounting for both:
1. **Benefits**: Dividend payments
2. **Costs**: Equity dilution (with fixed cost λ)

## Components

### 1. RewardFunction (`base.py`)

**Purpose**: Abstract base class for reward functions.

**Key Methods**:
- `step_reward()`: Flow reward r(s, a)·dt
- `terminal_reward()`: Terminal value R(s_T)
- `trajectory_return()`: Discounted cumulative reward

**Implementation Pattern**:

```python
class MyReward(RewardFunction):
    def step_reward(self, state, action, next_state, dt):
        # Compute instantaneous reward
        return reward * dt  # Scale by dt for flow rewards

    def terminal_reward(self, state, terminated):
        # Compute terminal value
        return value * terminated.float()  # Mask by termination
```

**TODO for Implementation**:
- [ ] Implement `trajectory_return()` with continuous-time discounting
  ```python
  def trajectory_return(self, rewards, terminal_rewards, masks, discount_rate, dt):
      batch_size, n_steps = rewards.shape
      returns = torch.zeros(batch_size)
      discount_factors = torch.exp(-discount_rate * torch.arange(n_steps) * dt)

      returns = (rewards * masks * discount_factors).sum(dim=1)
      final_discount = torch.exp(-discount_rate * n_steps * dt)
      returns += final_discount * terminal_rewards

      return returns
  ```
- [ ] Implement `cumulative_reward()` (undiscounted sum)
- [ ] Add unit tests for both methods

### 2. GHMRewardFunction (`ghm_rewards.py`)

**Purpose**: Reward function for GHM equity model.

**Objective**:
```
V(c) = E[ ∫_0^τ e^(-ρt) (dL_t - (1+λ)dE_t) + e^(-ρτ) V_liquidation ]
```

where:
- ρ = r - μ (discount rate)
- dL_t = a_L·dt (dividend flow)
- dE_t = a_E (equity issuance, impulse)
- λ (issuance cost parameter)
- V_liquidation = ω·α/(r-μ)

**Implementation Guidance**:

```python
def step_reward(self, state, action, next_state, dt):
    a_L = action[:, 0]  # Dividend rate
    a_E = action[:, 1]  # Equity issuance

    # Dividend flow over dt
    dividend_flow = a_L * dt

    # Issuance cost (multiplicative)
    issuance_cost = (1 + self.issuance_cost) * a_E

    # Net payout
    return dividend_flow - issuance_cost

def terminal_reward(self, state, terminated):
    # Liquidation value when c reaches 0
    reward = torch.zeros_like(state[:, 0])
    reward[terminated] = self.liquidation_value
    return reward
```

**TODO for Implementation**:
- [ ] Implement `step_reward()` as above
- [ ] Implement `terminal_reward()` as above
- [ ] Implement `net_payout()` utility method
- [ ] Implement `total_issuance_cost()` utility method
- [ ] Add parameter validation (discount_rate > 0, etc.)
- [ ] Add unit tests:
  ```python
  # Test case 1: Only dividends
  action = torch.tensor([[1.0, 0.0]])
  reward = reward_fn.step_reward(state, action, state, dt=0.01)
  assert torch.isclose(reward, torch.tensor([0.01]))  # 1.0 * 0.01

  # Test case 2: Only issuance
  action = torch.tensor([[0.0, 0.5]])
  reward = reward_fn.step_reward(state, action, state, dt=0.01)
  expected = -(1 + issuance_cost) * 0.5
  assert torch.isclose(reward, torch.tensor([expected]))

  # Test case 3: Both
  action = torch.tensor([[1.0, 0.5]])
  reward = reward_fn.step_reward(state, action, state, dt=0.01)
  expected = 1.0 * 0.01 - (1 + issuance_cost) * 0.5
  assert torch.isclose(reward, torch.tensor([expected]))
  ```

### 3. TerminalValue (`terminal.py`)

**Purpose**: Flexible terminal value specifications.

**Use Cases**:
- Constant terminal value
- State-dependent terminal value
- Learned terminal value (neural network)
- GHM liquidation value

**Implementation Guidance**:

```python
@classmethod
def constant(cls, value):
    return cls(value_fn=lambda state: torch.full((state.shape[0],), value))

@classmethod
def liquidation(cls, recovery_rate, expected_flow, discount_rate):
    liquidation_value = recovery_rate * expected_flow / discount_rate
    return cls.constant(liquidation_value)

@classmethod
def from_network(cls, network):
    return cls(
        value_fn=lambda state: network(state).squeeze(-1),
        is_learnable=True
    )
```

**TODO for Implementation**:
- [ ] Implement all class methods
- [ ] Implement `BoundaryCondition` class
- [ ] Implement `smooth_pasting()` factory
- [ ] Add tests for each boundary condition type

## Design Principles

1. **Separate Flow and Terminal**: Flow rewards r(s,a)·dt vs terminal R(s_T)
2. **Continuous-Time Semantics**: Use dt explicitly for flow rewards
3. **Economic Correctness**: Net payout = dividends - dilution cost
4. **Support Extensions**: Easy to add penalties, constraints, etc.

## GHM Model Correction

### Why Two Terms?

**Dividends** (a_L):
- Flow control: pays out continuously
- Benefit to shareholders: +a_L·dt
- Reduces cash: dc = ... - a_L·dt + ...

**Equity Issuance** (a_E):
- Singular control: happens at discrete times
- Cost to shareholders: -(1+λ)·a_E (dilution + fixed cost)
- Increases cash: dc = ... + a_E + ...

**Net Effect**:
```
Shareholder value = PV(dividends) - PV(dilution)
                  = E[∫ e^(-ρt) a_L dt] - E[Σ e^(-ρt_i) (1+λ)a_E_i]
```

### Fixed Cost λ

The parameter λ represents issuance costs:
- Underwriting fees
- Regulatory costs
- Adverse selection costs
- Market impact

Typical range: λ ∈ [0.05, 0.20] (5-20% of issuance)

This creates threshold behavior:
- Small issuances are wasteful (cost > benefit)
- Only issue if a_E large enough to overcome fixed cost

## Testing Strategy

Create `tests/rewards/`:

### `test_base.py`
- Test `trajectory_return()` with various discount rates
- Test with early termination (partial masks)
- Test batch dimensions

### `test_ghm_rewards.py`
- **Critical**: Test net payout formula
- Test with zero dividends (only issuance)
- Test with zero issuance (only dividends)
- Test with both controls
- Test terminal reward with liquidation
- Test batch operations
- Test edge cases:
  ```python
  # Edge case: Zero everything
  action = torch.tensor([[0.0, 0.0]])
  reward = reward_fn.step_reward(state, action, state, dt=0.01)
  assert reward == 0.0

  # Edge case: Large issuance cost
  reward_fn = GHMRewardFunction(discount_rate=0.03, issuance_cost=0.5)
  action = torch.tensor([[0.0, 1.0]])
  reward = reward_fn.step_reward(state, action, state, dt=0.01)
  assert reward == -1.5  # -(1 + 0.5) * 1.0
  ```

### `test_terminal.py`
- Test constant terminal value
- Test state-dependent terminal value
- Test learned terminal value (with mock network)
- Test boundary condition residuals

## Integration with Other Modules

```
control/ (defines actions a_L, a_E)
    ↓
rewards/ (computes r(s, a))
    ↓
simulation/ (accumulates rewards in trajectories)
    ↓
solvers/ (optimizes policy to maximize E[return])
```

## Common Pitfalls

1. **Forgetting dt for flow rewards**: r·dt, not just r
   - Dividends are a *rate* (per unit time)
   - Must scale by dt for discrete-time sum

2. **Sign errors**: Issuance is a *cost* to shareholders
   - Reward = +dividend - dilution
   - Not: +dividend + issuance

3. **Fixed cost vs multiplicative cost**:
   - Fixed: cost = λ·𝟙(a_E > 0)
   - Multiplicative: cost = λ·a_E
   - GHM uses: (1+λ)·a_E = a_E + λ·a_E (both)

4. **Discounting**: Use continuous-time e^(-ρt), not discrete (1-ρ)^t

## Sensitivity to Parameters

| Parameter | Effect on Policy |
|-----------|------------------|
| λ (issuance cost) | Higher → less frequent issuance, larger amounts |
| ω (recovery rate) | Higher → less aggressive dividend policy |
| ρ (discount rate) | Higher → more myopic, higher dividends |

## Future Extensions

- [ ] Add `GHMRewardWithDebt` for debt+equity model
- [ ] Add risk-sensitive rewards (mean-variance, CVaR)
- [ ] Add constraint penalties (leverage limits, etc.)
- [ ] Add multi-agent rewards (shareholder vs bondholder)

## References

- Bolton et al.: Net payout formulation
- DeMarzo & Sannikov: Continuous-time agency model
- Hugonnier et al.: Corporate cash management
