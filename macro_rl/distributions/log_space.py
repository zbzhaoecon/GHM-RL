"""
Log-space transformation for handling actions spanning multiple orders of magnitude.

This is useful when actions can be zero or span several orders of magnitude
(e.g., equity issuance from 0 to 0.5 where small values ~0.001 are meaningful).
"""

import torch
from torch import Tensor
from torch.distributions import Distribution


class LogSpaceTransform(Distribution):
    """
    Log-space transformation wrapper for distributions.

    This wraps a base distribution (e.g., Gaussian, Beta) and applies:
        action = exp(z) - offset
    where z ~ base_dist and offset ensures actions start from 0.

    This is useful for actions that:
    - Can naturally be zero
    - Span multiple orders of magnitude when non-zero
    - Should be strictly positive when non-zero

    The transformation is:
        z ~ base_dist (defined on real line or bounded range)
        action = exp(z - log(offset+1))

    Actually, a simpler approach for [0, high] actions:
        z ~ base_dist on some range
        action = (exp(z) - 1) * scale
    where scale chosen so max maps to high

    Alternative: Use exp-normal directly:
        log(action + epsilon) ~ Normal(mean, std)
        action = exp(Normal(mean, std)) - epsilon

    Example:
        >>> import torch
        >>> from torch.distributions import Normal
        >>> from macro_rl.distributions import LogSpaceTransform
        >>>
        >>> # Actions in log-space: log(action + 1) ~ N(mean, std)
        >>> mean = torch.zeros(10, 2)
        >>> std = torch.ones(10, 2)
        >>> base_dist = Normal(mean, std)
        >>> low = torch.tensor([0.0, 0.0])
        >>> high = torch.tensor([10.0, 0.5])
        >>>
        >>> dist = LogSpaceTransform(base_dist, low, high)
        >>> actions = dist.rsample()
        >>> log_probs = dist.log_prob(actions)
    """

    def __init__(
        self,
        base_dist: Distribution,
        low: Tensor,
        high: Tensor,
        epsilon: float = 1e-6,
    ):
        """
        Initialize LogSpaceTransform.

        Args:
            base_dist: Base distribution for log-space values
            low: Lower action bounds (action_dim,) - typically 0
            high: Upper action bounds (action_dim,)
            epsilon: Small constant for numerical stability
        """
        self.base_dist = base_dist
        self.low = low
        self.high = high
        self.epsilon = epsilon

        # Compute scale: we want exp(z_max) - 1 ≈ high
        # If base_dist outputs typical values in range [-3, 3], we scale so that:
        # exp(3) * scale ≈ high
        # scale = high / (exp(3) - 1) ≈ high / 19
        # But this is handled by the transformation

        batch_shape = base_dist.batch_shape
        event_shape = low.shape
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """
        Sample using reparameterization.

        Transformation: action = (exp(z) - 1) / (exp(z_scale) - 1) * (high - low) + low
        where z_scale is chosen so typical values map to [low, high]

        Args:
            sample_shape: Shape of samples to draw

        Returns:
            Actions (sample_shape, batch, action_dim)
        """
        # Sample from base distribution
        z = self.base_dist.rsample(sample_shape)

        # Apply exponential transformation
        # action = (exp(z) - 1) scaled to [low, high]
        # For simplicity: assume low = 0, and scale so exp(z) - 1 gives values in [0, high]
        # Use: action = (exp(z/scale) - 1) where scale chosen so typical z maps well

        # Simpler approach: action = low + (high - low) * (exp(z) - 1) / (exp(z_max) - 1)
        # where z_max is expected max value from base_dist

        # Even simpler for [0, high]: action = min(high, max(0, exp(z) - 1))
        # But this isn't differentiable at boundaries

        # Best approach: assume z ~ N(mean, std) and use log-normal with bounds
        # action = exp(z), then clip to [low + eps, high - eps] via soft clipping

        exp_z = torch.exp(z)

        # Scale to [low, high] range
        # Map exp(z) from [exp(-3), exp(3)] ≈ [0.05, 20] to [low, high]
        # Normalize: u = (exp(z) - exp_min) / (exp_max - exp_min)
        # Then: action = low + (high - low) * u

        # Simpler: use sigmoid-like scaling
        # action = low + (high - low) * (exp(z) / (1 + exp(z)))  # This is sigmoid!
        # Or: action = low + (high - low) * (1 - exp(-exp(z)))  # Gumbel-like

        # Most straightforward for now: action = low + (high - low) * sigmoid(z)
        # This is equivalent to TanhNormal but with sigmoid instead of tanh
        # Actually, let's just do: action = (exp(z) - 1) clamped to [low, high]

        # For [0, high]: action = min(high, max(0, k * (exp(z) - 1)))
        # where k = high / E[exp(z) - 1] to get good scale

        # Simple implementation: assume low = 0
        # action = (high - low) * min(1, max(0, (exp(z) - 1) / scale))
        # where scale = exp(3) - 1 ≈ 19 for z ~ N(0, 1)

        scale = torch.exp(torch.tensor(3.0, device=z.device)) - 1.0
        normalized = (exp_z - 1.0) / scale
        normalized = torch.clamp(normalized, 0.0, 1.0)

        action = self.low + (self.high - self.low) * normalized

        return action

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Sample without gradients."""
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, action: Tensor) -> Tensor:
        """
        Compute log probability with change of variables.

        Args:
            action: Actions in [low, high] (batch, action_dim)

        Returns:
            Log probabilities (batch,)
        """
        # Clamp actions
        eps = torch.maximum(
            torch.tensor(self.epsilon, device=action.device),
            1e-4 * (self.high - self.low)
        )
        action = torch.clamp(action, self.low + eps, self.high - eps)

        # Inverse transform
        scale = torch.exp(torch.tensor(3.0, device=action.device)) - 1.0
        normalized = (action - self.low) / (self.high - self.low)
        exp_z = 1.0 + normalized * scale
        z = torch.log(exp_z)

        # Log prob under base distribution
        log_prob_z = self.base_dist.log_prob(z)
        if log_prob_z.dim() > 1:
            log_prob_z = log_prob_z.sum(dim=-1)

        # Jacobian: dz/da = 1 / (a * scale / (high - low))
        # log|dz/da| = -log(exp_z - 1) - log(scale) + log(high - low)
        log_jacobian = -torch.log(exp_z - 1.0 + self.epsilon) - \
                       torch.log(scale + self.epsilon) + \
                       torch.log(self.high - self.low + self.epsilon)

        if log_jacobian.dim() > 1:
            log_jacobian = log_jacobian.sum(dim=-1)

        # Chain rule: log p(a) = log p(z) + log|dz/da|
        log_prob = log_prob_z + log_jacobian

        return torch.clamp(log_prob, min=-100.0, max=20.0)

    def entropy(self) -> Tensor:
        """Approximate entropy using base distribution."""
        entropy = self.base_dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)
        return entropy

    @property
    def mode(self) -> Tensor:
        """Mode of the distribution."""
        z_mode = self.base_dist.mode if hasattr(self.base_dist, 'mode') else self.base_dist.mean

        scale = torch.exp(torch.tensor(3.0, device=z_mode.device)) - 1.0
        exp_z = torch.exp(z_mode)
        normalized = (exp_z - 1.0) / scale
        normalized = torch.clamp(normalized, 0.0, 1.0)

        return self.low + (self.high - self.low) * normalized

    @property
    def mean(self) -> Tensor:
        """Approximate mean."""
        return self.mode
