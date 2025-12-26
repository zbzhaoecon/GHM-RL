"""
ScaledBeta distribution for bounded continuous actions.

Beta distribution is naturally bounded to [0, 1] and can represent zero and
upper bound naturally. We scale it to [low, high] for arbitrary action bounds.
"""

import torch
from torch import Tensor
from torch.distributions import Beta, Distribution


class ScaledBeta(Distribution):
    """
    Beta distribution scaled to arbitrary bounds [low, high].

    This distribution uses a Beta(α, β) base distribution on [0, 1]
    and scales it linearly to [low, high].

    The Beta distribution is particularly useful for actions that:
    - Can naturally be zero (when α < 1)
    - Can naturally be at the upper bound (when β < 1)
    - Should stay strictly positive (when α > 1, β > 1)

    The transformation is:
        action = low + (high - low) * Beta(α, β)

    The log probability includes the Jacobian correction:
        log π(action) = log Beta(u; α, β) - log(high - low)
        where u = (action - low) / (high - low)

    Reparameterization uses the Kumaraswamy approximation for differentiability.

    Example:
        >>> import torch
        >>> from macro_rl.distributions import ScaledBeta
        >>>
        >>> alpha = torch.ones(10, 2) * 2.0  # Shape parameters
        >>> beta = torch.ones(10, 2) * 5.0   # Shape parameters
        >>> low = torch.tensor([0.0, 0.0])
        >>> high = torch.tensor([10.0, 0.5])
        >>>
        >>> dist = ScaledBeta(alpha, beta, low, high)
        >>> actions = dist.rsample()  # Sample with reparameterization
        >>> log_probs = dist.log_prob(actions)  # Correct log probabilities
    """

    def __init__(
        self,
        alpha: Tensor,
        beta: Tensor,
        low: Tensor,
        high: Tensor,
        epsilon: float = 1e-6,
    ):
        """
        Initialize ScaledBeta distribution.

        Args:
            alpha: Shape parameter α (concentration1) (batch, action_dim)
            beta: Shape parameter β (concentration0) (batch, action_dim)
            low: Lower action bounds (action_dim,)
            high: Upper action bounds (action_dim,)
            epsilon: Small constant for numerical stability
        """
        # Handle broadcasting
        if beta.dim() < alpha.dim():
            beta = beta.expand_as(alpha)

        self.alpha = alpha
        self.beta = beta
        self.low = low.to(alpha.device)
        self.high = high.to(alpha.device)
        self.epsilon = epsilon

        # Base Beta distribution
        # Clamp to prevent extreme concentration values
        self.base_dist = Beta(
            torch.clamp(alpha, min=0.1, max=10.0),
            torch.clamp(beta, min=0.1, max=10.0)
        )

        # Validate shapes
        assert alpha.shape == beta.shape, f"alpha and beta must have same shape"
        assert low.shape == high.shape, "low and high must have same shape"
        assert alpha.shape[-1] == low.shape[-1], "action dimension mismatch"

        super().__init__(batch_shape=alpha.shape[:-1], event_shape=alpha.shape[-1:])

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """
        Sample using reparameterization trick.

        Uses the Kumaraswamy distribution as a reparameterizable approximation
        to Beta when α and β are close to 1. Otherwise uses rsample from PyTorch's
        Beta implementation.

        Args:
            sample_shape: Shape of samples to draw

        Returns:
            Scaled actions (sample_shape, batch, action_dim)
        """
        # PyTorch's Beta.rsample uses implicit reparameterization
        u = self.base_dist.rsample(sample_shape)

        # Scale to [low, high]
        action = self.low + (self.high - self.low) * u

        return action

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """
        Sample without gradients.

        Args:
            sample_shape: Shape of samples to draw

        Returns:
            Scaled actions (sample_shape, batch, action_dim)
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
        eps = torch.maximum(
            torch.tensor(self.epsilon, device=action.device, dtype=action.dtype),
            1e-4 * (self.high - self.low)
        )
        action = torch.clamp(action, self.low + eps, self.high - eps)

        # Inverse transform: action -> u
        # action = low + (high - low) * u
        # => u = (action - low) / (high - low)
        u = (action - self.low) / (self.high - self.low)

        # Clamp to valid Beta support (0, 1)
        u = torch.clamp(u, self.epsilon, 1.0 - self.epsilon)

        # Log probability under base Beta
        log_prob_u = self.base_dist.log_prob(u)  # (batch, action_dim)

        # Jacobian correction: log|da/du| = log(high - low)
        jacobian_correction = torch.log(self.high - self.low + self.epsilon)

        # Log probability with change of variables
        # log π(action) = log π(u) - log|da/du|
        log_prob_action = log_prob_u - jacobian_correction  # (batch, action_dim)

        # Clip to prevent extreme values
        log_prob_action = torch.clamp(log_prob_action, min=-100.0, max=20.0)

        # Sum over action dimensions
        return log_prob_action.sum(dim=-1)  # (batch,)

    def entropy(self) -> Tensor:
        """
        Compute entropy of the distribution.

        Returns:
            Entropy (batch,) - summed over action dimensions
        """
        # Entropy of scaled distribution:
        # H[scaled] = H[base] + log(high - low)
        base_entropy = self.base_dist.entropy()  # (batch, action_dim)
        jacobian_term = torch.log(self.high - self.low + self.epsilon)

        return (base_entropy + jacobian_term).sum(dim=-1)  # (batch,)

    @property
    def mode(self) -> Tensor:
        """
        Return the mode of the distribution.

        For Beta(α, β):
        - mode = (α - 1) / (α + β - 2) when α, β > 1
        - mode = 0 when α ≤ 1, β > 1
        - mode = 1 when α > 1, β ≤ 1
        - undefined when α, β ≤ 1 (we return mean in this case)

        Returns:
            Mode actions (batch, action_dim)
        """
        alpha = self.base_dist.concentration1
        beta = self.base_dist.concentration0

        # Compute mode for well-defined cases
        mode_u = (alpha - 1.0) / (alpha + beta - 2.0)

        # Handle edge cases
        mode_u = torch.where(
            (alpha <= 1.0) & (beta > 1.0),
            torch.zeros_like(mode_u),
            mode_u
        )
        mode_u = torch.where(
            (alpha > 1.0) & (beta <= 1.0),
            torch.ones_like(mode_u),
            mode_u
        )
        mode_u = torch.where(
            (alpha <= 1.0) & (beta <= 1.0),
            torch.full_like(mode_u, 0.5),  # Use mean when undefined
            mode_u
        )

        # Clamp to valid range
        mode_u = torch.clamp(mode_u, 0.0, 1.0)

        # Scale to [low, high]
        return self.low + (self.high - self.low) * mode_u

    @property
    def mean(self) -> Tensor:
        """
        Mean of the distribution.

        For Beta(α, β), mean = α / (α + β)

        Returns:
            Mean actions (batch, action_dim)
        """
        alpha = self.base_dist.concentration1
        beta = self.base_dist.concentration0

        mean_u = alpha / (alpha + beta)

        # Scale to [low, high]
        return self.low + (self.high - self.low) * mean_u

    @property
    def concentration1(self) -> Tensor:
        """Return alpha parameter."""
        return self.alpha

    @property
    def concentration0(self) -> Tensor:
        """Return beta parameter."""
        return self.beta

    @property
    def variance(self) -> Tensor:
        """
        Variance of the distribution.

        For Beta(α, β), variance = αβ / ((α+β)²(α+β+1))
        For scaled: Var[a] = (high - low)² * Var[u]

        Returns:
            Variance (batch, action_dim)
        """
        alpha = self.base_dist.concentration1
        beta = self.base_dist.concentration0

        # Variance of base Beta distribution
        var_u = (alpha * beta) / ((alpha + beta).pow(2) * (alpha + beta + 1))

        # Scale variance
        scale = (self.high - self.low).pow(2)
        return scale * var_u

    @property
    def stddev(self) -> Tensor:
        """
        Standard deviation of the distribution.

        Returns:
            Standard deviation (batch, action_dim)
        """
        return self.variance.sqrt()

    # Set arg_constraints to avoid warning
    arg_constraints = {}
