"""
Neural network policy architectures.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

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
    ):
        """
        Initialize Gaussian policy.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer sizes
            activation: Activation function ("relu", "tanh", "elu")
            log_std_bounds: Bounds for log σ

        TODO: Implement network architecture
        - Build shared trunk
        - Build mu_head (state_dim -> action_dim)
        - Build log_sigma_head (state_dim -> action_dim)
        - Initialize weights properly
        """
        super().__init__(state_dim, action_dim)
        self.log_std_bounds = log_std_bounds

        # TODO: Build networks
        raise NotImplementedError

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute mean and log std.

        Args:
            state: States (batch, state_dim)

        Returns:
            (mu, log_sigma) tuple

        TODO: Implement forward pass
        - Pass through shared trunk
        - Compute mu from mu_head
        - Compute log_sigma from log_sigma_head
        - Clamp log_sigma to bounds
        """
        raise NotImplementedError

    def sample(self, state: Tensor) -> Tensor:
        """
        Sample action from N(μ(s), σ(s)).

        TODO: Implement sampling
        """
        mu, log_sigma = self.forward(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        return dist.sample()

    def log_prob(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Compute log π(a|s) under Gaussian.

        TODO: Implement log probability
        - Compute mu, sigma from state
        - Create Normal distribution
        - Return log_prob(action).sum(dim=-1)
        """
        raise NotImplementedError

    def reparameterize(self, state: Tensor, noise: Tensor) -> Tensor:
        """
        Reparameterized sampling: a = μ(s) + σ(s) · ε

        Args:
            state: States (batch, state_dim)
            noise: ε ~ N(0, I) (batch, action_dim)

        Returns:
            Actions with gradients

        TODO: Implement reparameterization trick
        """
        raise NotImplementedError

    def entropy(self, state: Tensor) -> Tensor:
        """
        Compute entropy of Gaussian policy.

        Formula: H = 0.5 * log((2πe)^k det(Σ))
               = 0.5 * k * (1 + log(2π)) + sum(log σ_i)

        TODO: Implement entropy computation
        """
        raise NotImplementedError


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
