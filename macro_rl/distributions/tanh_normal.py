"""
TanhNormal distribution for bounded continuous actions.

This implements a squashed Gaussian distribution where actions are bounded
via a tanh transformation with proper log probability correction.
"""

import math
import torch
from torch import Tensor
from torch.distributions import Normal, Distribution


class TanhNormal(Distribution):
    """
    Tanh-squashed Normal distribution for bounded actions.

    This distribution applies a tanh transformation to a Gaussian base
    distribution, properly accounting for the change of variables in the
    log probability.

    The transformation is:
        action = low + (high - low) * (tanh(z) + 1) / 2
    where z ~ N(mean, std)

    The log probability includes the Jacobian correction:
        log π(action) = log N(z; mean, std) - log|da/dz|
        where log|da/dz| = log((high-low)/2) + 2*log(cosh(z))
                         = log((high-low)/2) + log(1 - tanh²(z))

    Example:
        >>> import torch
        >>> from macro_rl.distributions import TanhNormal
        >>>
        >>> loc = torch.zeros(10, 2)  # Mean of base Gaussian
        >>> scale = torch.ones(10, 2)  # Std of base Gaussian
        >>> low = torch.tensor([0.0, 0.0])
        >>> high = torch.tensor([10.0, 0.5])
        >>>
        >>> dist = TanhNormal(loc, scale, low, high)
        >>> actions = dist.rsample()  # Sample with reparameterization
        >>> log_probs = dist.log_prob(actions)  # Correct log probabilities
    """

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        low: Tensor,
        high: Tensor,
        epsilon: float = 1e-6,
    ):
        """
        Initialize TanhNormal distribution.

        Args:
            loc: Mean of base Gaussian (batch, action_dim)
            scale: Std of base Gaussian (batch, action_dim)
            low: Lower action bounds (action_dim,)
            high: Upper action bounds (action_dim,)
            epsilon: Small constant for numerical stability
        """
        self.loc = loc
        self.scale = scale
        self.low = low.to(loc.device)
        self.high = high.to(loc.device)
        self.epsilon = epsilon

        # Base Gaussian distribution
        self.base_dist = Normal(loc, scale)

        # Validate shapes
        assert loc.shape == scale.shape, "loc and scale must have same shape"
        assert low.shape == high.shape, "low and high must have same shape"
        assert loc.shape[-1] == low.shape[-1], "action dimension mismatch"

        super().__init__(batch_shape=loc.shape[:-1], event_shape=loc.shape[-1:])

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """
        Sample using reparameterization trick.

        Args:
            sample_shape: Shape of samples to draw

        Returns:
            Squashed actions (sample_shape, batch, action_dim)
        """
        # Sample from base Gaussian
        z = self.base_dist.rsample(sample_shape)

        # Apply tanh squashing
        tanh_z = torch.tanh(z)

        # Scale to [low, high]
        action = self.low + (self.high - self.low) * (tanh_z + 1.0) / 2.0

        return action

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """
        Sample without gradients.

        Args:
            sample_shape: Shape of samples to draw

        Returns:
            Squashed actions (sample_shape, batch, action_dim)
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, action: Tensor) -> Tensor:
        """
        Compute log probability of actions with Jacobian correction.

        Args:
            action: Actions in [low, high] (batch, action_dim)

        Returns:
            Log probabilities (batch,) - summed over action dimensions
        """
        # Clamp actions to valid range for numerical stability
        action = torch.clamp(action, self.low + self.epsilon, self.high - self.epsilon)

        # Inverse transform: action -> z
        # action = low + (high - low) * (tanh(z) + 1) / 2
        # => (action - low) / (high - low) = (tanh(z) + 1) / 2
        # => 2*(action - low)/(high - low) - 1 = tanh(z)
        # => z = arctanh(2*(action - low)/(high - low) - 1)

        normalized = 2.0 * (action - self.low) / (self.high - self.low) - 1.0
        # Clamp to valid tanh range
        normalized = torch.clamp(normalized, -1.0 + self.epsilon, 1.0 - self.epsilon)
        z = torch.atanh(normalized)

        # Log probability of z under base Gaussian
        log_prob_z = self.base_dist.log_prob(z)  # (batch, action_dim)

        # Jacobian correction: log|da/dz|
        # da/dz = (high - low) / 2 * (1 - tanh²(z))
        # log|da/dz| = log((high - low) / 2) + log(1 - tanh²(z))

        tanh_z = torch.tanh(z)
        jacobian_correction = torch.log((self.high - self.low) / 2.0 + self.epsilon) + \
                             torch.log(1.0 - tanh_z.pow(2) + self.epsilon)

        # Log probability with change of variables
        # log π(action) = log π(z) - log|da/dz|
        log_prob_action = log_prob_z - jacobian_correction  # (batch, action_dim)

        # Sum over action dimensions
        return log_prob_action.sum(dim=-1)  # (batch,)

    def entropy(self) -> Tensor:
        """
        Compute entropy of the distribution.

        This is an approximation since the true entropy of the transformed
        distribution is intractable. We use the entropy of the base Gaussian
        as an approximation.

        Returns:
            Entropy (batch,) - summed over action dimensions
        """
        # Use base Gaussian entropy as approximation
        # True entropy would require: H = E[-log π(action)]
        return self.base_dist.entropy().sum(dim=-1)

    @property
    def mode(self) -> Tensor:
        """
        Return the mode (deterministic action) of the distribution.

        Returns:
            Mode actions (batch, action_dim)
        """
        # Mode of base Gaussian
        z_mode = self.loc
        tanh_z = torch.tanh(z_mode)
        return self.low + (self.high - self.low) * (tanh_z + 1.0) / 2.0

    @property
    def mean(self) -> Tensor:
        """
        Approximate mean of the transformed distribution.

        For computational efficiency, we use the mode as an approximation.
        The true mean is intractable for this distribution.

        Returns:
            Approximate mean (batch, action_dim)
        """
        return self.mode

    @property
    def stddev(self) -> Tensor:
        """
        Return the standard deviation of the base distribution.

        Note: This is the std of the base Gaussian, not the transformed distribution.

        Returns:
            Standard deviation (batch, action_dim)
        """
        return self.scale
