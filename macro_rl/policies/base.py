"""
Base class for policy representations.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor


class Policy(ABC, nn.Module):
    """
    Abstract base class for policies.

    A policy maps states to actions: π: S → A

    For stochastic policies, we need:
        - sample(s): Sample action from π(·|s)
        - log_prob(s, a): Compute log π(a|s)
        - reparameterize(s, ε): For pathwise gradients

    For deterministic policies:
        - forward(s): Return deterministic action
    """

    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize policy.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space

        TODO: Implement initialization
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass (state -> action).

        For stochastic policies, this typically returns parameters
        (e.g., mean and std for Gaussian).

        Args:
            state: States (batch, state_dim)

        Returns:
            Action or distribution parameters

        TODO: Implement in subclasses
        """
        pass

    def sample(self, state: Tensor) -> Tensor:
        """
        Sample action from policy.

        Args:
            state: States (batch, state_dim)

        Returns:
            Sampled actions (batch, action_dim)

        TODO: Implement in subclasses
        """
        raise NotImplementedError

    def log_prob(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Compute log probability log π(a|s).

        Args:
            state: States (batch, state_dim)
            action: Actions (batch, action_dim)

        Returns:
            Log probabilities (batch,)

        TODO: Implement in stochastic policy subclasses
        """
        raise NotImplementedError

    def reparameterize(self, state: Tensor, noise: Tensor) -> Tensor:
        """
        Reparameterized sampling for pathwise gradients.

        Instead of a ~ π(·|s), compute a = f(s, ε; θ) where ε ~ N(0,I).

        Args:
            state: States (batch, state_dim)
            noise: Pre-sampled noise (batch, noise_dim)

        Returns:
            Actions (batch, action_dim) with gradients w.r.t. θ

        TODO: Implement in subclasses for pathwise gradient
        """
        raise NotImplementedError

    def entropy(self, state: Tensor) -> Tensor:
        """
        Compute policy entropy H[π(·|s)].

        Args:
            state: States (batch, state_dim)

        Returns:
            Entropy (batch,)

        TODO: Implement in stochastic policy subclasses
        """
        raise NotImplementedError
