# Environments Module

Phase 3: Gymnasium environments for continuous-time economic models.

## Overview

This module wraps continuous-time dynamics as Gymnasium environments, enabling RL training with standard libraries (Stable-Baselines3, CleanRL, RLlib, etc.).

**Key Features:**
- Gymnasium-compatible interface
- Automatic time discretization via Euler-Maruyama
- Vectorization support for parallel sampling
- Economic reward structures
- Proper handling of boundary conditions

## Quick Start

```python
from macro_rl.envs import GHMEquityEnv
import numpy as np

# Create environment
env = GHMEquityEnv(dt=0.01, max_steps=1000)

# Run episode
obs, info = env.reset(seed=42)
done = False

while not done:
    action = env.action_space.sample()  # Random policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Final cash: {obs[0]:.3f}")
```

## Available Environments

### GHMEquity-v0

1D equity management problem from Géczy-Hackbarth-Mauer.

**State:** `c ∈ [0, c_max]` (cash/earnings ratio)
**Action:** `a ∈ [0, a_max]` (dividend payout rate)
**Dynamics:** `dc = (μ_c(c) - a) dt + σ_c(c) dW`
**Reward:** `a * dt` (dividends paid)
**Termination:** `c ≤ 0` (liquidation, penalty applied)

```python
import gymnasium as gym

# Via registration
env = gym.make("GHMEquity-v0")

# Direct instantiation
from macro_rl.envs import GHMEquityEnv
from macro_rl.dynamics import GHMEquityParams

params = GHMEquityParams(alpha=0.2, r=0.05)
env = GHMEquityEnv(params=params, dt=0.01)
```

## Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `params` | `GHMEquityParams()` | Economic model parameters |
| `dt` | 0.01 | Time discretization step |
| `max_steps` | 1000 | Maximum episode length (T = dt × max_steps) |
| `a_max` | 10.0 | Maximum dividend rate |
| `liquidation_penalty` | 5.0 | Penalty when c hits 0 |
| `seed` | None | Random seed |

## Vectorized Environments

For parallel sampling (required by most RL libraries):

```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Same process (simple, good for debugging)
vec_env = DummyVecEnv([lambda: GHMEquityEnv() for _ in range(4)])

# Multiple processes (faster training)
vec_env = SubprocVecEnv([lambda: GHMEquityEnv() for _ in range(8)])
```

## Training with Stable-Baselines3

```python
from stable_baselines3 import SAC
from macro_rl.envs import GHMEquityEnv

# Create environment
env = GHMEquityEnv(dt=0.01, max_steps=500)

# Compute appropriate discount factor
gamma = env.get_expected_discount_factor()  # ≈ 0.9998

# Train SAC agent
model = SAC(
    "MlpPolicy",
    env,
    gamma=0.99,  # Or use computed gamma
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    verbose=1
)

model.learn(total_timesteps=100000)

# Test learned policy
obs, info = env.reset()
for _ in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Extending to Custom Environments

Create custom environments by subclassing `ContinuousTimeEnv`:

```python
from macro_rl.envs.base import ContinuousTimeEnv
from gymnasium import spaces
import numpy as np

class MyCustomEnv(ContinuousTimeEnv):
    def __init__(self, dynamics, **kwargs):
        super().__init__(dynamics, **kwargs)

        # Define action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _apply_action_and_evolve(self, action):
        # Custom state evolution logic
        # ...
        return new_state, reward

    def _get_terminated(self):
        # Custom termination logic
        return self._state[0] <= 0

    def _get_terminal_reward(self):
        # Custom terminal payoff
        return -10.0 if self._state[0] <= 0 else 0.0
```

## Testing

```python
# Run test suite
pytest tests/test_envs.py -v

# Check SB3 compatibility
from stable_baselines3.common.env_checker import check_env
from macro_rl.envs import GHMEquityEnv

env = GHMEquityEnv()
check_env(env)  # Should pass without errors
```

## Design Principles

1. **Continuous-time fidelity:** Discretization respects underlying SDE
2. **Economic rewards:** Directly from cash flows, not shaped
3. **Gymnasium standard:** Works with all Gym-compatible libraries
4. **Reproducibility:** Seeding ensures deterministic trajectories
5. **Vectorization:** Efficient parallel sampling

## Hyperparameter Guidelines (SAC)

Based on `dt=0.01`, `ρ=0.02`:

| Parameter | Suggested | Notes |
|-----------|-----------|-------|
| `gamma` | 0.99 | Approx. exp(-ρ·dt), or use `.get_expected_discount_factor()` |
| `learning_rate` | 3e-4 | SB3 default |
| `buffer_size` | 100000 | Sufficient for 1D |
| `batch_size` | 256 | Standard |
| `tau` | 0.005 | Soft update |
| `ent_coef` | "auto" | Let SB3 tune |

## Future Extensions

- **Finite horizon:** Add time to observation
- **Equity issuance:** Allow jumps at c=0 with cost
- **Multi-dimensional:** Extend to debt+equity models
- **Custom rewards:** Value function shaping

## References

- Géczy, C., Hackbarth, D., & Mauer, D. (GHM) - Equity management model
- Gymnasium documentation: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
