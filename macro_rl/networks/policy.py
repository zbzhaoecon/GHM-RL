from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    """Gaussian policy with reparameterization and (optional) bounds.

    If action_bounds is provided, actions are squashed into [low, high]
    via a sigmoid transform.
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

        # Log std parameters (state-independent for now)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    # ---- Core helpers ----

    def _get_mean_log_std(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.mean_net(state)
        log_std = self.log_std.clamp(*self.log_std_bounds)
        return mean, log_std

    def get_distribution(self, state: Tensor) -> Normal:
        mean, log_std = self._get_mean_log_std(state)
        std = log_std.exp()
        return Normal(mean, std)

    # ---- Public API ----

    def forward(self, state: Tensor) -> Tensor:
        """Return mean action (unbounded)."""
        mean, _ = self._get_mean_log_std(state)
        return mean

    def sample(
        self,
        state: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sample an action and return (action, log_prob).

        Uses rsample() for reparameterization when stochastic.
        """
        dist = self.get_distribution(state)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        action = raw_action
        if self.action_bounds is not None:
            low, high = self.action_bounds
            # Squash with sigmoid into [low, high]
            action = low + (high - low) * torch.sigmoid(raw_action)

            # NOTE: for exact log_probs under squashing we would need
            # to include the Jacobian of the transform. For now, we
            # keep log_prob under the base Gaussian only, which is
            # usually fine for advantage-based updates.
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

        action = raw_action
        if self.action_bounds is not None:
            low, high = self.action_bounds
            action = low + (high - low) * torch.sigmoid(raw_action)
        return action

    def log_prob_and_entropy(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute log probability and entropy for given actions.

        For now we treat `action` as if it lived in the unconstrained
        Gaussian space; this is an approximation if you've squashed
        actions. For advantage-based methods it's usually acceptable.
        """
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    # Convenience for TrajectorySimulator interface
    def act(self, state: Tensor) -> Tensor:
        """Alias for sampling an action (no log_prob)."""
        action, _ = self.sample(state, deterministic=False)
        return action
