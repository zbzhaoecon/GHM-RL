"""
Neural network policy architectures.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from typing import Optional

from macro_rl.policies.base import Policy


class GaussianPolicy(Policy):
    """
    Gaussian policy with learnable mean and std.

    π(a|s) = N(μ_θ(s), σ_θ(s))

    This policy supports:
        - Stochastic sampling
        - Log probability computation (for REINFORCE)
        - Reparameterization (for pathwise gradients)

    Architecture:
        state → [shared layers] → [mu_head]  → μ(s)
                                 ↘ [sigma_head] → log σ(s)

    Example:
        >>> policy = GaussianPolicy(
        ...     state_dim=1,
        ...     action_dim=2,
        ...     hidden_dims=[64, 64],
        ... )
        >>>
        >>> state = torch.randn(100, 1)
        >>> action = policy.sample(state)  # (100, 2)
        >>> log_prob = policy.log_prob(state, action)  # (100,)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        activation: str = "tanh",
        log_std_bounds: tuple = (-20.0, 2.0),
        state_dependent_std: bool = False,
        min_std: float = 1e-6,
        max_std: float = 10.0,
    ):
        """
        Initialize Gaussian policy.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer sizes
            activation: Activation function ("relu", "tanh", "elu")
            log_std_bounds: Bounds for log σ
            state_dependent_std: If True, learn std as function of state
            min_std: Minimum std for numerical stability
            max_std: Maximum std to prevent excessive exploration
        """
        super().__init__(state_dim, action_dim)
        self.log_std_bounds = log_std_bounds
        self.state_dependent_std = state_dependent_std
        self.min_std = min_std
        self.max_std = max_std

        # Activation function
        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build mean network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        # Use Softplus to ensure non-negative actions (for GHM model)
        layers.append(nn.Softplus())
        self.mean_net = nn.Sequential(*layers)

        # Build std network (state-independent by default)
        if state_dependent_std:
            std_layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                std_layers.append(nn.Linear(prev_dim, hidden_dim))
                std_layers.append(act_fn())
                prev_dim = hidden_dim
            std_layers.append(nn.Linear(prev_dim, action_dim))
            self.log_std_net = nn.Sequential(*std_layers)
        else:
            # State-independent learnable log_std
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: Tensor) -> Normal:
        """
        Compute policy distribution.

        Args:
            state: States (batch, state_dim)

        Returns:
            Normal distribution over actions
        """
        # Compute mean
        mean = self.mean_net(state)

        # Compute std
        if self.state_dependent_std:
            log_std = self.log_std_net(state)
        else:
            # Broadcast log_std to batch size
            log_std = self.log_std.expand_as(mean)

        # Clamp log_std to bounds
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        std = torch.exp(log_std)

        # Clamp std for stability
        std = torch.clamp(std, self.min_std, self.max_std)

        return Normal(mean, std)

    def act(self, state: Tensor, deterministic: bool = False) -> Tensor:
        """
        Sample actions from policy (for TrajectorySimulator interface).

        Args:
            state: States (batch, state_dim)
            deterministic: If True, return mean; else sample

        Returns:
            Actions (batch, action_dim) clipped to [0, ∞)
        """
        dist = self.forward(state)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample()

        # Ensure non-negative (already handled by softplus in mean_net, but clip samples)
        actions = torch.clamp(actions, min=0.0)
        return actions

    def sample(self, state: Tensor) -> Tensor:
        """
        Sample action from N(μ(s), σ(s)).

        Args:
            state: States (batch, state_dim)

        Returns:
            Sampled actions (batch, action_dim)
        """
        dist = self.forward(state)
        actions = dist.sample()
        # Ensure non-negative
        actions = torch.clamp(actions, min=0.0)
        return actions

    def log_prob(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Compute log π(a|s) under Gaussian.

        Args:
            state: States (batch, state_dim)
            action: Actions (batch, action_dim)

        Returns:
            Log probabilities (batch,) - sum over action dimensions
        """
        dist = self.forward(state)
        # Sum log probs over action dimensions (assuming independence)
        return dist.log_prob(action).sum(dim=-1)

    def reparameterize(self, state: Tensor, noise: Tensor) -> Tensor:
        """
        Reparameterized sampling: a = μ(s) + σ(s) · ε

        Args:
            state: States (batch, state_dim)
            noise: ε ~ N(0, I) (batch, action_dim)

        Returns:
            Actions with gradients w.r.t. policy parameters
        """
        dist = self.forward(state)
        actions = dist.mean + dist.stddev * noise
        # Ensure non-negative
        actions = torch.clamp(actions, min=0.0)
        return actions

    def entropy(self, state: Tensor) -> Tensor:
        """
        Compute entropy of Gaussian policy.

        Formula: H = 0.5 * log((2πe)^k det(Σ))
               = 0.5 * k * (1 + log(2π)) + sum(log σ_i)

        Args:
            state: States (batch, state_dim)

        Returns:
            Entropy (batch,) - sum over action dimensions
        """
        dist = self.forward(state)
        # Entropy of multivariate Gaussian with diagonal covariance
        # is sum of entropies of individual dimensions
        return dist.entropy().sum(dim=-1)

    def evaluate_actions(self, state: Tensor, action: Tensor) -> tuple[Optional[Tensor], Tensor, Tensor]:
        """
        Comprehensive action evaluation for logging.

        Args:
            state: States (batch, state_dim)
            action: Actions (batch, action_dim)

        Returns:
            values: None (policy-only, no value function)
            log_probs: (batch,) - log π(a|s)
            entropy: (batch,) - H[π(·|s)]
        """
        log_probs = self.log_prob(state, action)
        entropy = self.entropy(state)
        return None, log_probs, entropy


class DeterministicPolicy(Policy):
    """
    Deterministic policy π(s) = μ_θ(s).

    Used for:
        - Evaluation (no exploration)
        - Deterministic policy gradient
        - Debugging

    Architecture:
        state → [hidden layers] → [output layer] → action

    Example:
        >>> policy = DeterministicPolicy(
        ...     state_dim=1,
        ...     action_dim=2,
        ...     hidden_dims=[64, 64],
        ... )
        >>> action = policy(state)  # Deterministic output
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        activation: str = "relu",
        output_activation: str = "tanh",
    ):
        """
        Initialize deterministic policy.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer sizes
            activation: Hidden activation
            output_activation: Output activation (for bounded actions)

        TODO: Implement network architecture
        """
        super().__init__(state_dim, action_dim)
        # TODO: Build network
        raise NotImplementedError

    def forward(self, state: Tensor) -> Tensor:
        """
        Compute deterministic action.

        Args:
            state: States (batch, state_dim)

        Returns:
            Actions (batch, action_dim)

        TODO: Implement forward pass
        """
        raise NotImplementedError

    def sample(self, state: Tensor) -> Tensor:
        """For deterministic policy, sample = forward."""
        return self.forward(state)

    def reparameterize(self, state: Tensor, noise: Tensor) -> Tensor:
        """Deterministic policy ignores noise."""
        return self.forward(state)
