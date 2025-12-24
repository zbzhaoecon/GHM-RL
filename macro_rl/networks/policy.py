from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from macro_rl.distributions import TanhNormal


class GaussianPolicy(nn.Module):
    """Gaussian policy with reparameterization and (optional) bounds.

    If action_bounds is provided, actions are squashed into [low, high]
    via a tanh transform with proper Jacobian correction for log probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        action_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> None:
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.action_bounds = None
        if action_bounds is not None:
            low, high = action_bounds
            # Store as buffers to move with .to(device)
            self.register_buffer("action_low", low.float())
            self.register_buffer("action_high", high.float())
            self.action_bounds = (self.action_low, self.action_high)

        # Mean network
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mean_net = nn.Sequential(*layers)

        # Better initialization: bias toward positive raw actions
        # This prevents the policy from collapsing to zero actions
        nn.init.uniform_(self.mean_net[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.mean_net[-1].bias, 1.0)

        # Log std parameters (state-independent for now)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    # ---- Core helpers ----

    def _get_mean_log_std(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.mean_net(state)
        log_std = self.log_std.clamp(*self.log_std_bounds)
        return mean, log_std

    def get_distribution(self, state: Tensor):
        """Get distribution over actions.

        Returns TanhNormal if action_bounds are set, otherwise Normal.
        """
        mean, log_std = self._get_mean_log_std(state)
        std = log_std.exp()

        if self.action_bounds is not None:
            low, high = self.action_bounds
            return TanhNormal(mean, std, low, high)
        else:
            return Normal(mean, std)

    # ---- Public API ----

    def forward(self, state: Tensor) -> Tensor:
        """Return mean action (deterministic mode)."""
        dist = self.get_distribution(state)
        if self.action_bounds is not None:
            # For TanhNormal, return the mode (transformed mean)
            return dist.mode
        else:
            return dist.mean

    def sample(
        self,
        state: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sample an action and return (action, log_prob).

        Uses rsample() for reparameterization when stochastic.
        Now properly handles bounded actions with TanhNormal distribution.
        """
        dist = self.get_distribution(state)

        if deterministic:
            # Use mode for bounded, mean for unbounded
            if self.action_bounds is not None:
                action = dist.mode
            else:
                action = dist.mean
        else:
            # Stochastic sampling with reparameterization
            action = dist.rsample()

        # Compute log probability (now correct for bounded actions!)
        log_prob = dist.log_prob(action)

        return action, log_prob

    def sample_with_noise(
        self,
        state: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Reparameterized sample using explicit noise.

        noise ~ N(0, I), shape (batch, action_dim).
        Returns *action only*; caller can recompute log_prob if needed.
        """
        mean, log_std = self._get_mean_log_std(state)
        std = log_std.exp()
        raw_action = mean + std * noise

        if self.action_bounds is not None:
            # Apply tanh squashing for bounded actions
            low, high = self.action_bounds
            tanh_raw = torch.tanh(raw_action)
            action = low + (high - low) * (tanh_raw + 1.0) / 2.0
        else:
            action = raw_action

        return action

    def log_prob_and_entropy(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute log probability and entropy for given actions.

        Now correctly computes log probabilities for bounded actions
        using TanhNormal with proper Jacobian correction.
        """
        dist = self.get_distribution(state)

        # TanhNormal.log_prob already returns summed log prob (batch,)
        # Normal.log_prob returns per-dimension (batch, action_dim)
        log_prob = dist.log_prob(action)
        if not self.action_bounds:
            log_prob = log_prob.sum(dim=-1)

        # Entropy is also summed over action dimensions
        entropy = dist.entropy()

        return log_prob, entropy

    # Convenience for TrajectorySimulator interface
    def act(self, state: Tensor) -> Tensor:
        """Alias for sampling an action (no log_prob)."""
        action, _ = self.sample(state, deterministic=False)
        return action
