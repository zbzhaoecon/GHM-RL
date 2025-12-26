from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from macro_rl.distributions import TanhNormal, ScaledBeta, LogSpaceTransform


class GaussianPolicy(nn.Module):
    """Flexible policy with multiple distribution options.

    Supports:
    - tanh_normal: Gaussian with tanh squashing (default)
    - beta: Beta distribution (natural for [0, max] actions)
    - log_normal: Log-space Gaussian (for actions spanning orders of magnitude)

    If action_bounds is provided, actions are bounded to [low, high]
    with appropriate transformations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        action_bounds: Optional[Tuple[Tensor, Tensor]] = None,
        distribution_type: str = "tanh_normal",
    ) -> None:
        """
        Initialize policy network.

        Args:
            input_dim: State dimension
            output_dim: Action dimension
            hidden_dims: Hidden layer sizes
            log_std_bounds: Bounds for log std (for Gaussian-based policies)
            action_bounds: (low, high) action bounds
            distribution_type: One of ["tanh_normal", "beta", "log_normal"]
        """
        super().__init__()

        self.output_dim = output_dim
        self.log_std_bounds = log_std_bounds
        self.distribution_type = distribution_type

        self.action_bounds = None
        if action_bounds is not None:
            low, high = action_bounds
            # Store as buffers to move with .to(device)
            self.register_buffer("action_low", low.float())
            self.register_buffer("action_high", high.float())
            self.action_bounds = (self.action_low, self.action_high)

        # Build network based on distribution type
        if distribution_type == "beta":
            # Beta policy outputs alpha and beta parameters
            # Network outputs 2 * output_dim values (alpha, beta for each action)
            net_output_dim = 2 * output_dim
        else:
            # Gaussian-based policies output mean
            net_output_dim = output_dim

        # Parameter network (mean for Gaussian, alpha for Beta)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, net_output_dim))
        self.param_net = nn.Sequential(*layers)

        # Initialization
        if distribution_type == "beta":
            # For Beta: initialize to output alpha=2, beta=2 (unimodal, centered)
            # Output of network is log(alpha - 0.1) and log(beta - 0.1) to ensure alpha, beta > 0.1
            # So to get alpha=2, we want log(2 - 0.1) = log(1.9) ≈ 0.64
            nn.init.uniform_(self.param_net[-1].weight, -0.1, 0.1)
            nn.init.constant_(self.param_net[-1].bias, 0.6)
        else:
            # For Gaussian: start near zero
            nn.init.uniform_(self.param_net[-1].weight, -0.01, 0.01)
            nn.init.constant_(self.param_net[-1].bias, 0.0)

        # Second parameter (log_std for Gaussian, beta for Beta)
        if distribution_type == "beta":
            # Beta parameters come from network, no separate param needed
            pass
        else:
            # Log std parameters (state-independent for Gaussian)
            self.log_std = nn.Parameter(torch.zeros(output_dim))

    # ---- Core helpers ----

    def _get_distribution_params(self, state: Tensor):
        """Get distribution parameters from network output."""
        params = self.param_net(state)

        if self.distribution_type == "beta":
            # Split into alpha and beta
            # params are log(alpha - 0.1) and log(beta - 0.1)
            alpha_raw, beta_raw = torch.chunk(params, 2, dim=-1)

            # Convert to alpha and beta (ensure > 0.1)
            alpha = torch.exp(alpha_raw) + 0.1
            beta = torch.exp(beta_raw) + 0.1

            # Clamp to reasonable range
            alpha = torch.clamp(alpha, 0.1, 10.0)
            beta = torch.clamp(beta, 0.1, 10.0)

            return alpha, beta

        else:
            # Gaussian-based: params is mean
            mean = params

            # SAFETY: Clip raw network outputs to prevent extreme values
            if self.action_bounds is not None:
                mean = torch.clamp(mean, -10.0, 10.0)

            log_std = self.log_std.clamp(*self.log_std_bounds)
            std = log_std.exp()

            return mean, std

    def get_distribution(self, state: Tensor):
        """Get distribution over actions based on distribution type."""
        if self.distribution_type == "beta":
            alpha, beta = self._get_distribution_params(state)
            if self.action_bounds is not None:
                low, high = self.action_bounds
                return ScaledBeta(alpha, beta, low, high)
            else:
                # Beta without bounds: use [0, 1]
                low = torch.zeros(self.output_dim, device=state.device)
                high = torch.ones(self.output_dim, device=state.device)
                return ScaledBeta(alpha, beta, low, high)

        elif self.distribution_type == "log_normal":
            mean, std = self._get_distribution_params(state)
            base_dist = Normal(mean, std)

            if self.action_bounds is not None:
                low, high = self.action_bounds
                return LogSpaceTransform(base_dist, low, high)
            else:
                raise ValueError("log_normal distribution requires action_bounds")

        else:  # tanh_normal (default)
            mean, std = self._get_distribution_params(state)

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
