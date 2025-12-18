from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor

from .policy import GaussianPolicy
from .value import ValueNetwork


class MLP(nn.Module):
    """Simple MLP used as a shared feature trunk."""

    def __init__(self, input_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Joint actor–critic network for continuous control.

    - If shared_layers > 0, the first `shared_layers` hidden layers are
      shared between actor and critic (via a common MLP trunk).
    - If shared_layers == 0, actor and critic each get their own MLPs,
      i.e. no parameter sharing.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        shared_layers: int = 0,   # 0 => no shared trunk
        action_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> None:
        super().__init__()

        if shared_layers > len(hidden_dims):
            raise ValueError("shared_layers cannot exceed len(hidden_dims)")

        self.shared: Optional[MLP] = None
        feature_dim = state_dim

        if shared_layers > 0:
            shared_hidden = hidden_dims[:shared_layers]
            self.shared = MLP(state_dim, shared_hidden)
            feature_dim = shared_hidden[-1]

        # Actor and critic use the remaining layers (possibly empty)
        actor_hidden = hidden_dims[shared_layers:]
        critic_hidden = hidden_dims[shared_layers:]

        self.actor = GaussianPolicy(
            input_dim=feature_dim,
            output_dim=action_dim,
            hidden_dims=actor_hidden,
            action_bounds=action_bounds,
        )
        self.critic = ValueNetwork(
            input_dim=feature_dim,
            hidden_dims=critic_hidden,
        )

    def _features(self, state: Tensor) -> Tensor:
        if self.shared is not None:
            return self.shared(state)
        return state

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (mean_action, value) for logging/debugging."""
        feat = self._features(state)
        mean_action = self.actor(feat)
        value = self.critic(feat)
        return mean_action, value

    def act(self, state: Tensor, deterministic: bool = False) -> Tensor:
        """Sample action from policy (no log_prob)."""
        feat = self._features(state)
        action, _ = self.actor.sample(feat, deterministic=deterministic)
        return action

    def evaluate(self, state: Tensor) -> Tensor:
        """Value estimate V(s)."""
        feat = self._features(state)
        return self.critic(feat)

    def evaluate_actions(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate state–action pairs.

        Returns:
            value   : V(s)
            log_prob: log π(a|s)
            entropy : H(π(·|s))
        """
        feat = self._features(state)
        value = self.critic(feat)
        log_prob, entropy = self.actor.log_prob_and_entropy(feat, action)
        return value, log_prob, entropy

    def sample_with_noise(self, state: Tensor, noise: Tensor) -> Tensor:
        """Reparameterized sampling with explicit noise (for differentiable simulation).

        Args:
            state: State tensor
            noise: Noise tensor for reparameterization

        Returns:
            action: Sampled action
        """
        feat = self._features(state)
        return self.actor.sample_with_noise(feat, noise)
