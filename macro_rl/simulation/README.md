# Simulation Module

## Purpose

The `simulation` module provides engines for simulating continuous-time stochastic differential equations (SDEs) with known dynamics. This is the foundation for model-based RL methods.

## Why This Module Exists

Model-free RL (PPO, SAC) treats the environment as a black box:
```
s_t → [Environment] → s_{t+1}, r_t
         ↑ (opaque)
```

Model-based RL exploits known dynamics:
```
s_t, a_t → [Known: dx = μ(x,a)dt + σ(x)dW] → s_{t+1}, r_t
                    ↑ (transparent)
```

This allows:
1. **Free simulation**: Generate unlimited trajectories
2. **State space exploration**: Sample any initial state
3. **Low-variance gradients**: Use pathwise derivatives
4. **Validation**: Check against PDE solutions

## Components

### 1. SDEIntegrator (`sde.py`)

**Purpose**: Numerical integration of SDEs.

**Schemes**:
- **Euler-Maruyama**: x_{n+1} = x_n + μ(x_n)Δt + σ(x_n)√Δt·ε_n
  - Order 0.5 strong convergence
  - Sufficient for most applications

- **Milstein** (optional): Adds correction term ½σ·∂σ/∂x·(ε²-1)·Δt
  - Order 1.0 strong convergence
  - Only needed for very high accuracy

**Implementation Guidance**:

```python
# Basic usage
integrator = SDEIntegrator(scheme="euler")

# Single step
x_next = integrator.step(
    x=current_state,
    drift=drift_fn(current_state),
    diffusion=diffusion_fn(current_state),
    dt=0.01,
    noise=torch.randn_like(current_state)
)

# Batch simulation
x0 = torch.rand(1000, 1)  # 1000 initial states
trajectory = integrator.batch_simulate(
    x0=x0,
    drift_fn=lambda x: dynamics.drift(x, action),
    diffusion_fn=lambda x: dynamics.diffusion(x),
    dt=0.01,
    n_steps=500,
)
# Returns: (1000, 501, 1) - includes initial state
```

**TODO for Implementation**:
- [ ] Implement `_euler_maruyama_step()`
  - Handle both scalar diffusion: dx = μdt + σdW
  - Handle diagonal diffusion: dx_i = μ_i dt + σ_i dW_i
  - Support batch dimensions properly
- [ ] Implement `batch_simulate()`
  - Pre-allocate trajectory tensor for speed
  - Option for pre-sampled noise (reproducibility)
  - Support early termination via callback
- [ ] (Optional) Implement `_milstein_step()`
  - Requires autodiff for ∂σ/∂x
  - Only if Euler-Maruyama accuracy insufficient
- [ ] Add unit tests against analytical solutions
  - Brownian motion: dx = σdW → test variance growth
  - Ornstein-Uhlenbeck: dx = -θxdt + σdW → test mean reversion
  - Geometric Brownian: dx = μxdt + σxdW → test log-normality

### 2. TrajectorySimulator (`trajectory.py`)

**Purpose**: Generate complete trajectories for Monte Carlo policy gradient.

**Key Class**: `TrajectoryBatch`
```python
@dataclass
class TrajectoryBatch:
    states: Tensor       # (batch, n_steps+1, state_dim)
    actions: Tensor      # (batch, n_steps, action_dim)
    rewards: Tensor      # (batch, n_steps)
    masks: Tensor        # (batch, n_steps) - 1 if active
    returns: Tensor      # (batch,) - discounted sum
    terminal_rewards: Tensor  # (batch,)
```

**Usage Pattern**:

```python
# Setup
simulator = TrajectorySimulator(
    dynamics=ghm_dynamics,
    control_spec=ghm_control,
    reward_fn=ghm_reward,
    dt=0.01,
    T=5.0,
)

# Rollout trajectories
initial_states = torch.rand(1000, 1) * 10.0
trajectories = simulator.rollout(policy, initial_states)

# Use for policy gradient
advantages = trajectories.returns - baseline(initial_states)
loss = -(policy.log_prob(trajectories.states, trajectories.actions) * advantages).mean()
```

**Implementation Guidance**:

```python
def rollout(self, policy, initial_states, noise=None):
    batch_size = initial_states.shape[0]
    n_steps = self.max_steps

    # Pre-allocate storage
    states = torch.zeros(batch_size, n_steps+1, self.state_dim)
    actions = torch.zeros(batch_size, n_steps, self.action_dim)
    rewards = torch.zeros(batch_size, n_steps)
    masks = torch.ones(batch_size, n_steps)

    states[:, 0] = initial_states

    for t in range(n_steps):
        # Sample action
        actions[:, t] = policy.sample(states[:, t])

        # Apply masking (e.g., can't overdraw cash)
        actions[:, t] = self.control_spec.apply_mask(
            actions[:, t],
            states[:, t],
            self.dt
        )

        # Compute reward
        rewards[:, t] = self.reward_fn.step_reward(
            states[:, t],
            actions[:, t],
            self.dt
        )

        # Step dynamics
        drift = self.dynamics.drift(states[:, t], actions[:, t])
        diffusion = self.dynamics.diffusion(states[:, t])
        noise_t = noise[:, t] if noise is not None else torch.randn_like(states[:, t])
        states[:, t+1] = states[:, t] + drift * self.dt + diffusion * sqrt(self.dt) * noise_t

        # Check termination
        terminated = self._check_termination(states[:, t+1])
        masks[:, t][terminated] = 0

    # Compute returns
    terminal_rewards = self.reward_fn.terminal_reward(states[:, -1], masks[:, -1] == 0)
    returns = self._compute_returns(rewards, terminal_rewards, masks, self.dynamics.discount_rate())

    return TrajectoryBatch(states, actions, rewards, masks, returns, terminal_rewards)
```

**TODO for Implementation**:
- [ ] Implement `rollout()` following above pattern
- [ ] Implement `_compute_returns()` with continuous-time discounting
  - R = Σ_t exp(-ρ·t·dt) · r_t + exp(-ρ·T) · r_T
- [ ] Implement `_check_termination()` model-specifically
  - For GHM: c ≤ 0
  - Make configurable via callback
- [ ] Add support for time-varying state (c, τ)
- [ ] Add logging/diagnostics (trajectory lengths, termination rates)
- [ ] Test with dummy policy to verify shapes

### 3. DifferentiableSimulator (`differentiable.py`)

**Purpose**: Fully differentiable simulation for pathwise gradients.

**Key Concept**: Reparameterization trick

Standard policy gradient:
```
a ~ π_θ(·|s)  ← Stochastic, can't backprop through sampling
∇J = E[∇log π_θ(a|s) · R]  ← High variance
```

Reparameterized:
```
a = μ_θ(s) + σ_θ(s) · ε, ε ~ N(0,1)  ← Deterministic given ε
∇J = ∇E[R(τ(ε; θ))]  ← Low variance, direct gradient
```

**Implementation Guidance**:

```python
def simulate(self, policy, initial_states, noise, return_trajectory=False):
    batch_size = initial_states.shape[0]
    n_steps = self.max_steps

    # Storage (all differentiable)
    states = torch.zeros(batch_size, n_steps+1, self.state_dim)
    states[:, 0] = initial_states

    total_return = torch.zeros(batch_size)
    discount = 1.0

    for t in range(n_steps):
        # Reparameterized action sampling
        action = policy.reparameterize(states[:, t], noise[:, t])

        # Differentiable reward
        reward = self.reward_fn.step_reward(states[:, t], action, self.dt)

        # Accumulate return (differentiable)
        total_return += discount * reward

        # Differentiable dynamics step
        states[:, t+1] = self._differentiable_step(
            states[:, t],
            action,
            noise[:, t]
        )

        # Soft masking for differentiability
        mask = self._soft_termination_mask(states[:, t+1])
        total_return *= mask  # Smooth termination
        discount *= torch.exp(-self.dynamics.discount_rate() * self.dt)

    # Terminal reward
    terminal_reward = self.reward_fn.terminal_reward(states[:, -1])
    total_return += discount * terminal_reward

    if return_trajectory:
        return total_return, states, actions
    return total_return
```

**TODO for Implementation**:
- [ ] Implement `simulate()` with full gradient tracking
- [ ] Implement `_differentiable_step()` using Euler-Maruyama
- [ ] Implement `_soft_termination_mask()` using sigmoid
  - Hard: mask = (c > 0).float() (not differentiable)
  - Soft: mask = sigmoid(α * c) (differentiable)
  - Tune α for sharpness
- [ ] Add gradient checking via finite differences
- [ ] Test that gradients flow correctly
- [ ] Compare gradient variance with REINFORCE

## Design Principles

1. **Batched by Default**: All operations support (batch, ...) dimensions
2. **Reproducible**: Support pre-sampled noise for deterministic trajectories
3. **Efficient**: Pre-allocate tensors, avoid Python loops when possible
4. **Differentiable When Needed**: Differentiable simulator preserves gradients
5. **Modular**: Simulators work with any Dynamics + Control + Reward

## Testing Strategy

Create `tests/simulation/`:

### `test_sde_integrator.py`
- Test Euler-Maruyama against analytical solutions
- Test batch dimensions
- Test with pre-sampled vs fresh noise
- Verify convergence rate (weak and strong)

### `test_trajectory_simulator.py`
- Test with dummy policy
- Verify shapes of TrajectoryBatch
- Test early termination
- Test return computation
- Test action masking integration

### `test_differentiable_simulator.py`
- Test gradient flow (non-None gradients)
- Compare gradients with finite differences
- Test soft vs hard masking
- Verify reparameterization trick

## Integration with Other Modules

```
dynamics/ (provides drift, diffusion, discount)
    ↓
simulation/ (uses dynamics to simulate)
    ↓
solvers/ (uses simulation for policy optimization)
```

## Performance Considerations

1. **Pre-allocate tensors**: Don't grow lists in Python loops
2. **Vectorize**: Use tensor operations, not Python loops
3. **GPU-friendly**: All operations should work on CUDA tensors
4. **Memory vs Speed**: Trade-off between storing full trajectories vs computing on-the-fly

**Typical Performance Targets**:
- 10,000 trajectories of 500 steps: ~1 second on GPU
- Gradient computation: ~2x simulation time

## Common Pitfalls

1. **Forgetting sqrt(dt) in diffusion term**: σ√dt·ε, not σ·dt·ε
2. **Continuous vs discrete discounting**: exp(-ρ·dt) not (1-ρ·dt)
3. **Batch dimension handling**: Ensure broadcasting works correctly
4. **Hard termination in differentiable sim**: Use soft masking
5. **Not detaching noise in pathwise**: Noise must be fixed (not learnable)

## Future Extensions

- [ ] Add support for time-inhomogeneous dynamics
- [ ] Add multi-dimensional diffusion (full covariance matrix)
- [ ] Add adaptive time stepping
- [ ] Add trajectory visualization tools
- [ ] Add importance sampling for rare events
