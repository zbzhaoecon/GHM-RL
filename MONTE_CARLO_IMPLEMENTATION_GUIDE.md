# Monte Carlo Policy Gradient Implementation Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Specifications](#component-specifications)
3. [Parallelism Strategy](#parallelism-strategy)
4. [Logging & Monitoring](#logging--monitoring)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Testing Strategy](#testing-strategy)
7. [Integration Points](#integration-points)

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Monte Carlo Training Loop                     │
│                                                                  │
│  1. Sample initial states from state space                      │
│  2. Rollout trajectories (parallel simulation)                  │
│  3. Compute returns & advantages                                │
│  4. Estimate policy gradient (REINFORCE)                        │
│  5. Update policy parameters                                    │
│  6. Update baseline (value function)                            │
│  7. Log metrics & visualize policy/value                        │
│  8. Save checkpoints                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Components

```
monte_carlo/
├── GaussianPolicy              # Stochastic policy (mean + std)
├── ValueNetwork                # Baseline for variance reduction
├── TrajectorySimulator         # Sequential rollouts
├── ParallelTrajectorySimulator # Parallel rollouts (multi-process)
├── MonteCarloPolicyGradient    # Main solver
└── MonteCarloTrainer           # Training script with logging
```

---

## Component Specifications

### 1. Gaussian Policy (`macro_rl/policies/neural.py`)

#### Purpose
Stochastic policy that outputs a Gaussian distribution over continuous actions.

#### Requirements
- **Input**: State `s ∈ ℝ^state_dim`
- **Output**: Distribution `π(·|s) = N(μ_θ(s), Σ_θ(s))`
- **Methods**:
  - `forward(states)` → Distribution
  - `act(states, deterministic=False)` → Actions (sampled or mean)
  - `log_prob(states, actions)` → Log-probabilities (for REINFORCE)
  - `entropy(states)` → Entropy (for monitoring)

#### Architecture

```python
class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous control.

    Network architecture:
        state → [hidden layers] → mean, log_std

    For GHM:
        - state_dim = 1 (cash c)
        - action_dim = 2 (dividend a_L, equity issuance a_E)

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer sizes (e.g., [64, 64])
        activation: Activation function (default: Tanh)
        log_std_init: Initial log std (default: 0.0 → std=1.0)
        state_dependent_std: If True, learn std as function of state
        min_std: Minimum std for numerical stability
        max_std: Maximum std to prevent excessive exploration
    """
```

**Design Decisions**:

1. **Mean Network**:
   ```
   state → Linear(state_dim, hidden_dims[0]) → Activation
        → Linear(hidden_dims[0], hidden_dims[1]) → Activation
        → Linear(hidden_dims[-1], action_dim) → Softplus (ensure ≥ 0)
   ```

2. **Standard Deviation**:
   - **Option A (Simple)**: Fixed learnable parameter per action
     ```python
     self.log_std = nn.Parameter(torch.zeros(action_dim))
     ```
   - **Option B (State-dependent)**: Separate network
     ```python
     self.std_net = MLP(state_dim, action_dim, hidden_dims)
     std = torch.exp(torch.clamp(self.std_net(state), -20, 2))
     ```

   **Recommendation**: Start with Option A, upgrade to B if needed.

3. **Action Constraints**:
   - Apply softplus to mean output: `μ = softplus(μ_raw)` ensures `μ ≥ 0`
   - Clip sampled actions: `a = torch.clamp(a_sampled, min=0.0)`
   - Control spec handles additional masking

#### Key Methods

```python
def forward(self, states: Tensor) -> Distribution:
    """
    Compute policy distribution.

    Args:
        states: (batch, state_dim)

    Returns:
        Normal distribution with mean (batch, action_dim) and
        std (batch, action_dim)
    """

def act(self, states: Tensor, deterministic: bool = False) -> Tensor:
    """
    Sample actions from policy.

    Args:
        states: (batch, state_dim)
        deterministic: If True, return mean; else sample

    Returns:
        actions: (batch, action_dim) clipped to [0, ∞)
    """

def log_prob(self, states: Tensor, actions: Tensor) -> Tensor:
    """
    Compute log π(a|s) for given state-action pairs.

    Critical for REINFORCE gradient estimation!

    Args:
        states: (batch, state_dim)
        actions: (batch, action_dim)

    Returns:
        log_probs: (batch,) - sum over action dimensions
    """

def evaluate_actions(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Comprehensive action evaluation for logging.

    Args:
        states: (batch, state_dim)
        actions: (batch, action_dim)

    Returns:
        values: (batch,) - always None for policy-only
        log_probs: (batch,) - log π(a|s)
        entropy: (batch,) - H[π(·|s)]
    """
```

---

### 2. Value Network (`macro_rl/values/neural.py`)

#### Purpose
Baseline function `V(s)` to reduce variance in policy gradient estimation.

#### Requirements
- **Input**: State `s ∈ ℝ^state_dim`
- **Output**: Value estimate `V(s) ∈ ℝ`
- **Training**: Regression to match actual returns

#### Architecture

```python
class ValueNetwork(nn.Module):
    """
    Value function approximator.

    Network architecture:
        state → [hidden layers] → value (scalar)

    Args:
        state_dim: Dimension of state space
        hidden_dims: List of hidden layer sizes (e.g., [64, 64])
        activation: Activation function (default: Tanh)
    """
```

**Design**:
```
state → Linear(state_dim, hidden_dims[0]) → Activation
     → Linear(hidden_dims[0], hidden_dims[1]) → Activation
     → Linear(hidden_dims[-1], 1) → value (no activation)
```

#### Key Methods

```python
def forward(self, states: Tensor) -> Tensor:
    """
    Predict value for given states.

    Args:
        states: (batch, state_dim)

    Returns:
        values: (batch,) - predicted V(s)
    """
```

**No gradient computation needed** (unlike HJB validation) - just value prediction.

---

### 3. Monte Carlo Policy Gradient Solver (`macro_rl/solvers/monte_carlo.py`)

#### Purpose
Main training algorithm - REINFORCE with baseline and known dynamics.

#### Algorithm

```python
"""
Monte Carlo Policy Gradient (REINFORCE with Known Dynamics)

For iteration = 1 to N:
    1. Sample initial states: s₀ ~ Uniform(state_space)

    2. Rollout trajectories using policy:
       τ = {s₀, a₀, r₀, s₁, ..., sₜ, aₜ, rₜ, ...}

    3. Compute returns:
       R(s₀) = Σₜ exp(-ρ·t·dt)·rₜ + exp(-ρ·T)·r_T

    4. Compute baseline:
       b(s₀) = V_ψ(s₀)

    5. Compute advantages:
       A(s₀) = R(s₀) - b(s₀)
       A = (A - mean(A)) / (std(A) + ε)  # Normalize

    6. Policy gradient:
       ∇_θ J ≈ (1/B) Σᵢ Σₜ ∇_θ log π_θ(aₜⁱ|sₜⁱ) · Aⁱ

    7. Update policy:
       θ ← θ + α_policy · ∇_θ J

    8. Update baseline:
       ψ ← ψ - α_value · ∇_ψ MSE(V_ψ(s₀), R(s₀))
"""
```

#### Implementation Details

##### Initialization

```python
def __init__(
    self,
    policy: GaussianPolicy,
    simulator: Union[TrajectorySimulator, ParallelTrajectorySimulator],
    baseline: Optional[ValueNetwork] = None,
    n_trajectories: int = 1000,
    lr_policy: float = 3e-4,
    lr_baseline: float = 1e-3,
    batch_size: int = 1000,
    advantage_normalization: bool = True,
    max_grad_norm: float = 0.5,
):
    """
    Initialize Monte Carlo solver.

    Args:
        policy: Gaussian policy to optimize
        simulator: Trajectory simulator (parallel or sequential)
        baseline: Value network for variance reduction (optional)
        n_trajectories: Number of trajectories per iteration
        lr_policy: Learning rate for policy
        lr_baseline: Learning rate for baseline
        batch_size: Batch size for initial state sampling
        advantage_normalization: Whether to normalize advantages
        max_grad_norm: Gradient clipping threshold
    """
```

##### Core Training Step

```python
def train_step(self) -> Dict[str, float]:
    """
    Single training iteration.

    Returns:
        metrics: Dictionary of training metrics

    Steps:
        1. Sample initial states
        2. Rollout trajectories (parallel)
        3. Compute advantages
        4. Update policy (REINFORCE)
        5. Update baseline (regression)
        6. Collect metrics
    """
```

**Detailed breakdown**:

```python
# 1. Sample initial states
s0 = self._sample_initial_states(self.n_trajectories)

# 2. Rollout trajectories
with torch.no_grad():
    trajectories = self.simulator.rollout(self.policy, s0)
    # trajectories.states: (B, T+1, state_dim)
    # trajectories.actions: (B, T, action_dim)
    # trajectories.rewards: (B, T)
    # trajectories.returns: (B,)

# 3. Compute advantages
returns = trajectories.returns  # (B,)
if self.baseline is not None:
    values = self.baseline(s0).squeeze(-1)  # (B,)
    advantages = returns - values.detach()
else:
    advantages = returns

if self.advantage_normalization:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# 4. Compute policy loss
# Flatten trajectories: (B, T, ...) → (B*T, ...)
B, T = trajectories.actions.shape[0], trajectories.actions.shape[1]
states_flat = trajectories.states[:, :-1, :].reshape(B * T, -1)
actions_flat = trajectories.actions.reshape(B * T, -1)
masks_flat = trajectories.masks.reshape(B * T)

# Compute log π(aₜ|sₜ) for all timesteps
log_probs_flat = self.policy.log_prob(states_flat, actions_flat)  # (B*T,)
log_probs = log_probs_flat.reshape(B, T)  # (B, T)

# REINFORCE loss (negative for maximization)
# Broadcast advantages over time, weighted by masks
policy_loss = -(log_probs * trajectories.masks * advantages.unsqueeze(-1)).sum() / B

# 5. Update policy
self.policy_optimizer.zero_grad()
policy_loss.backward()
nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
self.policy_optimizer.step()

# 6. Update baseline
if self.baseline is not None:
    value_pred = self.baseline(s0).squeeze(-1)
    baseline_loss = F.mse_loss(value_pred, returns.detach())

    self.baseline_optimizer.zero_grad()
    baseline_loss.backward()
    nn.utils.clip_grad_norm_(self.baseline.parameters(), self.max_grad_norm)
    self.baseline_optimizer.step()

# 7. Collect metrics
metrics = {
    'loss/policy': policy_loss.item(),
    'loss/baseline': baseline_loss.item() if self.baseline else 0.0,
    'return/mean': returns.mean().item(),
    'return/std': returns.std().item(),
    'advantage/mean': advantages.mean().item(),
    'advantage/std': advantages.std().item(),
    'episode_length/mean': trajectories.masks.sum(dim=-1).mean().item(),
}

return metrics
```

---

## Parallelism Strategy

### Why Parallelism?

Monte Carlo needs **many trajectories** per update (typically 100-1000) for low-variance gradient estimates. Sequential simulation is a bottleneck.

### Existing Infrastructure

You already have `ParallelTrajectorySimulator` in `macro_rl/simulation/parallel.py`!

#### Architecture

```
Main Process (Training)
    ↓
    ├─→ Worker 1 (rollouts 1-N/4)
    ├─→ Worker 2 (rollouts N/4-N/2)
    ├─→ Worker 3 (rollouts N/2-3N/4)
    └─→ Worker 4 (rollouts 3N/4-N)
    ↓
Aggregate trajectories → compute gradients
```

### Integration

```python
# In training script
if args.use_parallel:
    simulator = ParallelTrajectorySimulator(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        dt=args.dt,
        T=args.T,
        n_workers=args.n_workers,  # Default: cpu_count()
    )
else:
    simulator = TrajectorySimulator(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        dt=args.dt,
        T=args.T,
    )

solver = MonteCarloPolicyGradient(
    policy=policy,
    simulator=simulator,  # Works with both!
    baseline=baseline,
    n_trajectories=args.n_trajectories,
)
```

### Performance Considerations

1. **Worker Count**:
   - Default: `n_workers = min(cpu_count(), n_trajectories // 10)`
   - Too many workers: Overhead dominates
   - Too few: Underutilized CPU

2. **Batch Distribution**:
   - Split trajectories evenly across workers
   - Each worker simulates `n_trajectories // n_workers` rollouts

3. **Serialization**:
   - Policy must be pickleable (PyTorch modules are)
   - Dynamics, control_spec, reward_fn must be pickleable

4. **When to Use**:
   - ✅ n_trajectories ≥ 100
   - ✅ Trajectory length is long (T/dt ≥ 100 steps)
   - ❌ Very short trajectories (overhead not worth it)
   - ❌ GPU simulation (stay on single device)

---

## Logging & Monitoring

### Logging Framework

Use structured logging with multiple levels:

1. **Console logging**: High-level progress
2. **File logging**: Detailed metrics
3. **TensorBoard**: Visualizations
4. **Checkpointing**: Model snapshots

### Metrics to Track

#### Standard Metrics (Every Iteration)

```python
metrics = {
    # Returns
    'return/mean': float,              # Average return
    'return/std': float,               # Return standard deviation
    'return/min': float,               # Worst trajectory
    'return/max': float,               # Best trajectory

    # Losses
    'loss/policy': float,              # Policy gradient loss
    'loss/baseline': float,            # Value function MSE

    # Advantages
    'advantage/mean': float,           # Should be ~0 after normalization
    'advantage/std': float,            # Should be ~1 after normalization
    'advantage/max': float,            # Largest advantage
    'advantage/min': float,            # Smallest advantage

    # Episode statistics
    'episode_length/mean': float,      # Average trajectory length
    'episode_length/std': float,       # Trajectory length variance
    'termination_rate': float,         # Fraction hitting c ≤ 0

    # Policy statistics
    'policy/mean_action_L': float,     # Average dividend rate
    'policy/mean_action_E': float,     # Average equity issuance
    'policy/std_action_L': float,      # Dividend std
    'policy/std_action_E': float,      # Equity issuance std
    'policy/entropy': float,           # Policy entropy H[π]

    # Training
    'grad_norm/policy': float,         # Policy gradient norm
    'grad_norm/baseline': float,       # Baseline gradient norm
    'learning_rate/policy': float,     # Current policy LR
    'learning_rate/baseline': float,   # Current baseline LR
}
```

#### Policy/Value Checks (Every N Iterations)

This is the **key feature** - understanding what the model is learning.

```python
def evaluate_policy_value(
    policy: GaussianPolicy,
    baseline: Optional[ValueNetwork],
    dynamics: ContinuousTimeDynamics,
    n_eval_points: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate policy and value function across state space.

    Purpose: Understand what the model has learned

    Returns:
        - State grid evaluations
        - Policy means and stds at each state
        - Value predictions at each state
        - Action distributions
        - Comparison with analytical solutions (if available)
    """

    # Create state grid
    state_space = dynamics.state_space
    c_grid = torch.linspace(
        state_space.lower[0],
        state_space.upper[0],
        n_eval_points
    ).reshape(-1, 1)  # (n_eval_points, 1)

    with torch.no_grad():
        # Evaluate policy
        dist = policy(c_grid)
        mean_actions = dist.mean  # (n_eval_points, action_dim)
        std_actions = dist.stddev  # (n_eval_points, action_dim)

        # Evaluate value function
        if baseline is not None:
            values = baseline(c_grid).squeeze(-1)  # (n_eval_points,)
        else:
            values = None

    results = {
        'states': c_grid.cpu().numpy(),  # Cash levels

        # Policy
        'policy/mean_dividend': mean_actions[:, 0].cpu().numpy(),
        'policy/mean_equity_issuance': mean_actions[:, 1].cpu().numpy(),
        'policy/std_dividend': std_actions[:, 0].cpu().numpy(),
        'policy/std_equity_issuance': std_actions[:, 1].cpu().numpy(),

        # Value
        'value/predictions': values.cpu().numpy() if values is not None else None,

        # Derived quantities
        'policy/exploration_level': std_actions.mean(dim=0).cpu().numpy(),
    }

    return results
```

**Visualization** (save plots every N iterations):

```python
def plot_policy_value(
    eval_results: Dict[str, Any],
    iteration: int,
    save_dir: str,
):
    """
    Create comprehensive visualization of learned policy and value.

    Plots:
        1. Policy mean actions vs state
        2. Policy std (exploration) vs state
        3. Value function vs state
        4. Action distribution at key states
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    states = eval_results['states']

    # Plot 1: Mean actions
    ax = axes[0, 0]
    ax.plot(states, eval_results['policy/mean_dividend'], label='Dividend (a_L)', color='blue')
    ax.plot(states, eval_results['policy/mean_equity_issuance'], label='Equity Issuance (a_E)', color='red')
    ax.set_xlabel('Cash (c)')
    ax.set_ylabel('Mean Action')
    ax.set_title(f'Policy Mean Actions (Iter {iteration})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Policy std (exploration)
    ax = axes[0, 1]
    ax.plot(states, eval_results['policy/std_dividend'], label='Dividend Std', color='blue', linestyle='--')
    ax.plot(states, eval_results['policy/std_equity_issuance'], label='Equity Std', color='red', linestyle='--')
    ax.set_xlabel('Cash (c)')
    ax.set_ylabel('Action Std Dev')
    ax.set_title(f'Exploration Level (Iter {iteration})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Value function
    if eval_results['value/predictions'] is not None:
        ax = axes[1, 0]
        ax.plot(states, eval_results['value/predictions'], color='green')
        ax.set_xlabel('Cash (c)')
        ax.set_ylabel('V(c)')
        ax.set_title(f'Value Function (Iter {iteration})')
        ax.grid(True, alpha=0.3)

    # Plot 4: Net payout (dividend - issuance cost)
    ax = axes[1, 1]
    lambda_dilution = 0.1  # From GHM params
    net_payout = eval_results['policy/mean_dividend'] - (1 + lambda_dilution) * eval_results['policy/mean_equity_issuance']
    ax.plot(states, net_payout, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Cash (c)')
    ax.set_ylabel('Net Payout to Shareholders')
    ax.set_title(f'Expected Net Payout (Iter {iteration})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/policy_value_iter_{iteration:06d}.png', dpi=150)
    plt.close()
```

### Logging Implementation

```python
class MonteCarloLogger:
    """
    Comprehensive logging for Monte Carlo training.

    Features:
        - Console progress (tqdm)
        - CSV metrics file
        - TensorBoard integration
        - Policy/value visualizations
        - Checkpointing
    """

    def __init__(
        self,
        log_dir: str,
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 1000,
        use_tensorboard: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.plot_dir = self.log_dir / 'plots'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

        # Initialize loggers
        self.csv_file = open(self.log_dir / 'metrics.csv', 'w', newline='')
        self.csv_writer = None  # Initialized on first log

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.log_dir / 'tensorboard')
        else:
            self.tb_writer = None

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

    def log_iteration(
        self,
        iteration: int,
        metrics: Dict[str, float],
    ):
        """Log metrics for current iteration."""

        # Console (every log_interval)
        if iteration % self.log_interval == 0:
            print(f"[Iter {iteration:6d}] "
                  f"Return: {metrics['return/mean']:7.3f} ± {metrics['return/std']:6.3f} | "
                  f"Policy Loss: {metrics['loss/policy']:8.4f} | "
                  f"Baseline Loss: {metrics.get('loss/baseline', 0.0):8.4f}")

        # CSV
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=['iteration'] + list(metrics.keys()))
            self.csv_writer.writeheader()

        row = {'iteration': iteration, **metrics}
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, iteration)

    def log_evaluation(
        self,
        iteration: int,
        policy: GaussianPolicy,
        baseline: Optional[ValueNetwork],
        dynamics: ContinuousTimeDynamics,
    ):
        """Evaluate and visualize policy/value."""

        eval_results = evaluate_policy_value(policy, baseline, dynamics)
        plot_policy_value(eval_results, iteration, self.plot_dir)

        # Log to TensorBoard
        if self.tb_writer is not None:
            # Add images
            img = plt.imread(f'{self.plot_dir}/policy_value_iter_{iteration:06d}.png')
            self.tb_writer.add_image('policy_value', img, iteration, dataformats='HWC')

            # Add scalars
            self.tb_writer.add_scalar(
                'eval/exploration_dividend',
                eval_results['policy/exploration_level'][0],
                iteration
            )
            self.tb_writer.add_scalar(
                'eval/exploration_equity',
                eval_results['policy/exploration_level'][1],
                iteration
            )

    def save_checkpoint(
        self,
        iteration: int,
        policy: GaussianPolicy,
        baseline: Optional[ValueNetwork],
        policy_optimizer: torch.optim.Optimizer,
        baseline_optimizer: Optional[torch.optim.Optimizer],
    ):
        """Save training checkpoint."""

        checkpoint = {
            'iteration': iteration,
            'policy_state_dict': policy.state_dict(),
            'policy_optimizer_state_dict': policy_optimizer.state_dict(),
        }

        if baseline is not None:
            checkpoint['baseline_state_dict'] = baseline.state_dict()
            checkpoint['baseline_optimizer_state_dict'] = baseline_optimizer.state_dict()

        path = self.checkpoint_dir / f'checkpoint_iter_{iteration:06d}.pt'
        torch.save(checkpoint, path)

        # Keep symlink to latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(path.name)

    def close(self):
        """Close all file handles."""
        self.csv_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
```

---

## Implementation Roadmap

### Phase 1: Core Components (Week 1)

**Goal**: Implement and test individual components.

#### Tasks

1. **GaussianPolicy** (`macro_rl/policies/neural.py`)
   - [ ] Implement network architecture
   - [ ] Implement `forward()`, `act()`, `log_prob()`
   - [ ] Add entropy computation
   - [ ] Unit tests: test shapes, gradients, log-prob correctness

2. **ValueNetwork** (`macro_rl/values/neural.py`)
   - [ ] Implement network architecture
   - [ ] Implement `forward()`
   - [ ] Unit tests: test shapes, gradients

3. **Testing**
   - [ ] Test policy samples are non-negative
   - [ ] Test log-prob gradients flow correctly
   - [ ] Test value network regression on synthetic data

**Validation**:
```bash
pytest tests/test_policies_neural.py
pytest tests/test_values_neural.py
```

---

### Phase 2: Monte Carlo Solver (Week 2)

**Goal**: Complete training algorithm with logging.

#### Tasks

1. **MonteCarloPolicyGradient** (`macro_rl/solvers/monte_carlo.py`)
   - [ ] Implement `_sample_initial_states()`
   - [ ] Implement `_estimate_policy_gradient()`:
     - Flatten trajectories
     - Compute log-probs
     - Compute REINFORCE loss
   - [ ] Implement `_update_baseline()`:
     - MSE regression
   - [ ] Implement `solve()` training loop:
     - Call simulator
     - Compute advantages
     - Update policy and baseline
     - Collect metrics

2. **MonteCarloLogger**
   - [ ] Implement CSV logging
   - [ ] Implement TensorBoard integration
   - [ ] Implement policy/value evaluation
   - [ ] Implement checkpointing

3. **Testing**
   - [ ] Test on toy problem (deterministic dynamics, σ=0)
   - [ ] Verify policy improves over iterations
   - [ ] Check gradient norms are reasonable

**Validation**:
```bash
pytest tests/test_solvers_monte_carlo.py
```

---

### Phase 3: Training Script & Parallelism (Week 3)

**Goal**: End-to-end training on GHM model.

#### Tasks

1. **Training Script** (`macro_rl/scripts/train_monte_carlo.py`)
   - [ ] Parse command-line arguments
   - [ ] Initialize GHM dynamics, control spec, reward function
   - [ ] Create policy and baseline networks
   - [ ] Create simulator (parallel or sequential)
   - [ ] Create solver and logger
   - [ ] Run training loop
   - [ ] Save final results

2. **Parallelism Integration**
   - [ ] Test with `ParallelTrajectorySimulator`
   - [ ] Benchmark: sequential vs parallel
   - [ ] Tune number of workers

3. **Hyperparameter Tuning**
   - [ ] Learning rates (policy, baseline)
   - [ ] Network sizes
   - [ ] Number of trajectories per iteration
   - [ ] Advantage normalization

**Validation**:
```bash
# Sequential
python macro_rl/scripts/train_monte_carlo.py \
    --n_iterations 5000 \
    --n_trajectories 500 \
    --lr_policy 3e-4 \
    --lr_baseline 1e-3

# Parallel
python macro_rl/scripts/train_monte_carlo.py \
    --n_iterations 5000 \
    --n_trajectories 500 \
    --use_parallel \
    --n_workers 8
```

---

### Phase 4: Analysis & Comparison (Week 4)

**Goal**: Validate results and compare with other methods.

#### Tasks

1. **Solution Validation**
   - [ ] Check HJB residual for learned policy/value
   - [ ] Compare with analytical solution (if available)
   - [ ] Verify boundary conditions

2. **Comparison with Other Solvers**
   - [ ] Train with Pathwise Gradient solver
   - [ ] Train with Actor-Critic solver
   - [ ] Compare:
     - Final return
     - Sample efficiency (iterations to convergence)
     - Computational cost (wall-clock time)
     - HJB residual

3. **Sensitivity Analysis**
   - [ ] Vary n_trajectories (100, 500, 1000, 2000)
   - [ ] Vary baseline architecture
   - [ ] Vary learning rates
   - [ ] Vary advantage normalization

**Deliverables**:
- Comparison table (Monte Carlo vs Pathwise vs Actor-Critic)
- Plots: Learning curves, final policies, value functions
- Technical report

---

## Testing Strategy

### Unit Tests

```python
# tests/test_policies_neural.py
def test_gaussian_policy_shapes():
    """Test output shapes are correct."""
    policy = GaussianPolicy(state_dim=1, action_dim=2, hidden_dims=[32, 32])
    states = torch.randn(10, 1)

    # Test forward
    dist = policy(states)
    assert dist.mean.shape == (10, 2)
    assert dist.stddev.shape == (10, 2)

    # Test act
    actions = policy.act(states)
    assert actions.shape == (10, 2)
    assert (actions >= 0).all()  # Non-negative

    # Test log_prob
    log_probs = policy.log_prob(states, actions)
    assert log_probs.shape == (10,)

def test_gaussian_policy_gradient():
    """Test gradients flow through log_prob."""
    policy = GaussianPolicy(state_dim=1, action_dim=2, hidden_dims=[32])
    states = torch.randn(5, 1)
    actions = torch.randn(5, 2).abs()  # Non-negative

    log_probs = policy.log_prob(states, actions)
    loss = -log_probs.mean()
    loss.backward()

    # Check gradients exist
    for param in policy.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

def test_gaussian_policy_deterministic():
    """Test deterministic mode returns mean."""
    policy = GaussianPolicy(state_dim=1, action_dim=2, hidden_dims=[32])
    policy.eval()

    states = torch.randn(1, 1)

    # Sample multiple times (deterministic)
    actions1 = policy.act(states, deterministic=True)
    actions2 = policy.act(states, deterministic=True)

    assert torch.allclose(actions1, actions2)
```

```python
# tests/test_solvers_monte_carlo.py
def test_monte_carlo_training_improves():
    """Test that policy improves over training."""
    # Create toy problem
    dynamics = SimpleDynamics()  # Linear dynamics for testing
    policy = GaussianPolicy(state_dim=1, action_dim=1, hidden_dims=[16])
    baseline = ValueNetwork(state_dim=1, hidden_dims=[16])
    simulator = TrajectorySimulator(dynamics, control_spec, reward_fn, dt=0.1, T=1.0)

    solver = MonteCarloPolicyGradient(
        policy=policy,
        simulator=simulator,
        baseline=baseline,
        n_trajectories=50,
    )

    # Initial performance
    with torch.no_grad():
        initial_states = torch.randn(100, 1)
        trajectories_before = simulator.rollout(policy, initial_states)
        return_before = trajectories_before.returns.mean()

    # Train
    for _ in range(100):
        solver.train_step()

    # Final performance
    with torch.no_grad():
        trajectories_after = simulator.rollout(policy, initial_states)
        return_after = trajectories_after.returns.mean()

    # Should improve
    assert return_after > return_before
```

### Integration Tests

```python
def test_end_to_end_ghm():
    """Test full training pipeline on GHM model."""
    # Initialize GHM
    from macro_rl.dynamics.ghm_equity import GHMEquityDynamics
    from macro_rl.control.ghm_control import GHMControlSpec
    from macro_rl.rewards.ghm_rewards import GHMReward

    params = GHMEquityParams()
    dynamics = GHMEquityDynamics(params)
    control_spec = GHMControlSpec()
    reward_fn = GHMReward(params)

    # Create components
    policy = GaussianPolicy(state_dim=1, action_dim=2, hidden_dims=[64, 64])
    baseline = ValueNetwork(state_dim=1, hidden_dims=[64, 64])
    simulator = TrajectorySimulator(dynamics, control_spec, reward_fn, dt=0.01, T=5.0)

    solver = MonteCarloPolicyGradient(
        policy=policy,
        simulator=simulator,
        baseline=baseline,
        n_trajectories=100,
    )

    # Train for a few iterations
    for iteration in range(10):
        metrics = solver.train_step()
        assert 'return/mean' in metrics
        assert 'loss/policy' in metrics
        assert not np.isnan(metrics['return/mean'])
```

---

## Integration Points

### With Existing Codebase

1. **Dynamics** (`macro_rl/dynamics/ghm_equity.py`)
   - ✅ Already implemented and tested
   - Use `GHMEquityDynamics` directly

2. **Simulation** (`macro_rl/simulation/`)
   - ✅ `TrajectorySimulator` ready
   - ✅ `ParallelTrajectorySimulator` ready
   - Just pass policy to `rollout()`

3. **Control** (`macro_rl/control/ghm_control.py`)
   - ✅ `GHMControlSpec` handles masking
   - Applied automatically in simulator

4. **Rewards** (`macro_rl/rewards/ghm_rewards.py`)
   - ✅ `GHMReward` computes net payout
   - Used automatically in simulator

5. **Validation** (`macro_rl/validation/`)
   - Use `HJBResidualValidator` to check learned solution
   - Compare with analytical solutions if available

### With Actor-Critic Solver

The `ActorCritic` solver already uses similar components. Key differences:

| Component | Monte Carlo | Actor-Critic |
|-----------|-------------|--------------|
| Policy update | REINFORCE (score function) | Pathwise or REINFORCE |
| Value update | MC returns | MC + TD + HJB |
| Network | Separate policy/value | Joint ActorCritic module |
| Parallelism | Via simulator | Via simulator |

**Reuse**:
- Same `GaussianPolicy` architecture
- Same `ValueNetwork` architecture
- Same logging patterns
- Same evaluation functions

---

## Command-Line Interface

### Training Script Arguments

```bash
python macro_rl/scripts/train_monte_carlo.py \
    # Training
    --n_iterations 10000 \
    --n_trajectories 1000 \
    --lr_policy 3e-4 \
    --lr_baseline 1e-3 \
    --batch_size 1000 \
    --max_grad_norm 0.5 \
    \
    # Network architecture
    --policy_hidden_dims 64 64 \
    --baseline_hidden_dims 64 64 \
    --state_dependent_std \
    \
    # Simulation
    --dt 0.01 \
    --T 5.0 \
    --use_parallel \
    --n_workers 8 \
    \
    # Logging
    --log_dir results/monte_carlo/run_001 \
    --log_interval 10 \
    --eval_interval 100 \
    --save_interval 1000 \
    \
    # GHM parameters (optional overrides)
    --mu 0.1 \
    --sigma 0.2 \
    --r 0.05 \
    --lambda_dilution 0.1 \
    \
    # Other
    --seed 42 \
    --device cuda \
    --no_baseline  # Train without baseline (for comparison)
```

---

## Expected Outputs

After training, you should have:

```
results/monte_carlo/run_001/
├── metrics.csv                    # All metrics over training
├── checkpoints/
│   ├── checkpoint_iter_001000.pt
│   ├── checkpoint_iter_002000.pt
│   └── checkpoint_latest.pt       # Symlink to latest
├── plots/
│   ├── policy_value_iter_000100.png
│   ├── policy_value_iter_000200.png
│   └── ...
├── tensorboard/
│   └── events.out.tfevents...
└── config.json                     # Saved hyperparameters
```

### Final Analysis

After training completes, run analysis:

```bash
# Evaluate final policy
python scripts/evaluate.py \
    --checkpoint results/monte_carlo/run_001/checkpoints/checkpoint_latest.pt \
    --n_episodes 1000

# Validate against HJB
python scripts/validate.py \
    --checkpoint results/monte_carlo/run_001/checkpoints/checkpoint_latest.pt

# Compare with other methods
python scripts/compare_solvers.py \
    --monte_carlo results/monte_carlo/run_001 \
    --pathwise results/pathwise/run_001 \
    --actor_critic results/actor_critic/run_001
```

---

## Summary Checklist

Before implementation, ensure you understand:

- [ ] **Theory**: REINFORCE gradient estimator with baseline
- [ ] **Architecture**: Policy (Gaussian) + Value (baseline) + Simulator
- [ ] **Parallelism**: Multi-process rollouts via `ParallelTrajectorySimulator`
- [ ] **Logging**: Metrics, visualizations, checkpoints
- [ ] **Policy/Value Monitoring**: Grid evaluation + plotting
- [ ] **Integration**: Works with existing dynamics, control, rewards
- [ ] **Testing**: Unit tests, integration tests, validation

**Key Implementation Files**:
1. `macro_rl/policies/neural.py` - GaussianPolicy
2. `macro_rl/values/neural.py` - ValueNetwork
3. `macro_rl/solvers/monte_carlo.py` - MonteCarloPolicyGradient (fill in TODOs)
4. `macro_rl/scripts/train_monte_carlo.py` - Training script with logging

**Success Criteria**:
- Policy returns improve over training
- Baseline MSE decreases over training
- Learned policy matches intuition (higher dividends at higher cash)
- HJB residual is small (<1e-3)
- Comparable or better than pathwise gradient
