"""
Differentiable simulation for pathwise gradient methods.

This module implements fully differentiable trajectory simulation,
enabling direct gradient flow through the entire trajectory for
low-variance policy optimization.
"""

from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn


class DifferentiableSimulator:
    """
    Fully differentiable SDE simulator for pathwise gradients.

    Key feature: Gradients flow through the entire simulation,
    enabling direct policy optimization via backpropagation through
    the reparameterization trick.

    Contrast with Monte Carlo:
        - Monte Carlo: ∇J = E[∇log π(a|s) · R]  (high variance)
        - Pathwise:    ∇J = ∇E[R]                (low variance)

    The pathwise gradient requires:
        1. Deterministic policy given noise: a = π(s; θ, ε)
        2. Differentiable dynamics: x_next = f(x, a, ε)
        3. Differentiable reward: r = r(x, a)

    Example:
        >>> # Policy with reparameterization
        >>> class GaussianPolicy(nn.Module):
        ...     def forward(self, state, noise):
        ...         mu = self.mu_net(state)
        ...         sigma = self.sigma_net(state)
        ...         return mu + sigma * noise  # Reparameterized
        >>>
        >>> simulator = DifferentiableSimulator(
        ...     dynamics=dynamics,
        ...     control_spec=control_spec,
        ...     reward_fn=reward_fn,
        ...     dt=0.01,
        ...     T=5.0,
        ... )
        >>>
        >>> # Simulate with gradients
        >>> initial_states = torch.rand(100, 1, requires_grad=True)
        >>> noise = torch.randn(100, 500, 1)  # Fixed noise
        >>> returns = simulator.simulate(policy, initial_states, noise)
        >>> loss = -returns.mean()
        >>> loss.backward()  # Gradients flow through entire trajectory!
    """

    def __init__(
        self,
        dynamics,  # ContinuousTimeDynamics
        control_spec,  # ControlSpec
        reward_fn,  # RewardFunction
        dt: float,
        T: float,
    ):
        """
        Initialize differentiable simulator.

        Args:
            dynamics: Continuous-time dynamics
            control_spec: Control specification
            reward_fn: Reward function
            dt: Time step size
            T: Time horizon
        """
        import math

        self.dynamics = dynamics
        self.control_spec = control_spec
        self.reward_fn = reward_fn
        self.dt = float(dt)
        self.T = float(T)

        # Compute number of steps
        steps = self.T / self.dt
        self.max_steps = int(round(steps))
        if not math.isclose(self.max_steps * self.dt, self.T, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"T={self.T} is not an integer multiple of dt={self.dt}.")

    def simulate(
        self,
        policy: nn.Module,
        initial_states: Tensor,
        noise: Tensor,
        return_trajectory: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Simulate trajectories with gradient tracking.

        Args:
            policy: Policy network (must support reparameterization)
            initial_states: Initial states (batch, state_dim)
            noise: Pre-sampled noise for reparameterization
                  Shape: (batch, n_steps, noise_dim)
            return_trajectory: If True, return full trajectory

        Returns:
            If return_trajectory=False:
                returns: Discounted returns (batch,)
            If return_trajectory=True:
                (returns, states, actions)

        Algorithm:
            1. Initialize state trajectory storage
            2. For each time step:
                a. Sample action via reparameterization: a = π(s, ε)
                b. Compute reward (differentiable)
                c. Step dynamics (differentiable)
            3. Compute discounted returns (differentiable)
            4. Return with gradient graph attached
        """
        import math

        batch_size, state_dim = initial_states.shape
        action_dim = self.control_spec.dim
        device = initial_states.device

        # Validate noise shape
        assert noise.shape[0] == batch_size, "noise batch size mismatch"
        assert noise.shape[1] == self.max_steps, "noise n_steps mismatch"

        # Pre-allocate storage (keep gradients!)
        if return_trajectory:
            states = torch.zeros(batch_size, self.max_steps + 1, state_dim, device=device, dtype=initial_states.dtype)
            actions_storage = torch.zeros(batch_size, self.max_steps, action_dim, device=device, dtype=initial_states.dtype)

        # Pre-compute discount factor to avoid repeated tensor creation
        discount_factor = torch.exp(torch.tensor(-self.dynamics.discount_rate() * self.dt, device=device, dtype=initial_states.dtype))

        state = initial_states
        total_return = torch.zeros(batch_size, device=device, dtype=initial_states.dtype)
        discount = torch.tensor(1.0, device=device, dtype=initial_states.dtype)

        for t in range(self.max_steps):
            # Reparameterized action sampling
            # Assume policy has a method: sample_with_noise(state, noise) or reparameterize(state, noise)
            if hasattr(policy, 'sample_with_noise'):
                action = policy.sample_with_noise(state, noise[:, t, :action_dim])
            elif hasattr(policy, 'reparameterize'):
                action = policy.reparameterize(state, noise[:, t, :action_dim])
            else:
                # Fallback: assume policy.forward does reparameterization
                action = policy(state, noise[:, t, :action_dim])

            # Apply control masking (but keep gradients)
            action = self.control_spec.apply_mask(action, state, self.dt)
            action = self.control_spec.clip(action)

            # Store if needed
            if return_trajectory:
                states[:, t, :] = state
                actions_storage[:, t, :] = action

            # Compute next state (differentiable)
            next_state = self._differentiable_step(state, action, noise[:, t, :])

            # Compute reward (differentiable)
            reward = self.reward_fn.step_reward(state, action, next_state, self.dt)

            # Soft termination masking
            mask = self._soft_termination_mask(next_state)
            reward = reward * mask

            # Accumulate return
            total_return = total_return + discount * reward
            discount = discount * discount_factor

            # Update state
            state = next_state

        # Store final state if needed
        if return_trajectory:
            states[:, self.max_steps, :] = state

        # Terminal reward (differentiable)
        # For differentiable version, use soft mask
        terminal_mask = 1.0 - self._soft_termination_mask(state)  # 1 if terminated
        terminal_reward = self.reward_fn.terminal_reward(state, terminal_mask)
        total_return = total_return + discount * terminal_reward

        if return_trajectory:
            return total_return, states, actions_storage
        return total_return

    def _differentiable_step(
        self,
        state: Tensor,
        action: Tensor,
        noise_step: Tensor,
    ) -> Tensor:
        """
        Single differentiable dynamics step.

        Args:
            state: Current state (batch, state_dim)
            action: Action (batch, action_dim)
            noise_step: Noise for this step (batch, state_dim)

        Returns:
            Next state (batch, state_dim) with gradients
        """
        import math

        # Compute drift and diffusion (differentiable)
        drift = self.dynamics.drift(state, action)
        diffusion = self.dynamics.diffusion(state)

        # Euler-Maruyama step (differentiable)
        sqrt_dt = math.sqrt(self.dt)
        next_state = state + drift * self.dt + diffusion * sqrt_dt * noise_step

        return next_state

    def _soft_termination_mask(self, states: Tensor) -> Tensor:
        """
        Compute soft termination mask for differentiability.

        Instead of hard termination (0 if c ≤ 0), use smooth function
        to preserve gradients.

        Args:
            states: Current states (batch, state_dim)

        Returns:
            Soft mask (batch,) in [0, 1]

        Example:
            Hard: mask = (c > 0).float()
            Soft: mask = torch.sigmoid(α * c)  # Smooth transition
        """
        # For GHM model: soft termination based on cash level
        # Cash is the first state component
        # Use sigmoid with temperature parameter for smooth transition
        alpha = 10.0  # Temperature parameter (higher = sharper transition)
        c = states[:, 0]

        # sigmoid(α * c) → 1 for c >> 0, → 0 for c << 0
        mask = torch.sigmoid(alpha * c)

        return mask

    def compute_gradient(
        self,
        policy: nn.Module,
        initial_states: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """
        Compute pathwise gradient ∇_θ E[R].

        This is a convenience method that:
        1. Simulates trajectories
        2. Computes returns
        3. Takes gradient w.r.t. policy parameters

        Args:
            policy: Policy network
            initial_states: Initial states (batch, state_dim)
            noise: Pre-sampled noise (batch, n_steps, noise_dim)

        Returns:
            Gradient tensor (concatenated gradients of all parameters)
        """
        # Simulate with gradient tracking
        returns = self.simulate(policy, initial_states, noise, return_trajectory=False)

        # Compute mean return (objective to maximize)
        objective = returns.mean()

        # Compute gradients w.r.t. policy parameters
        policy_params = list(policy.parameters())
        gradients = torch.autograd.grad(objective, policy_params, create_graph=False)

        # Concatenate gradients into single tensor
        grad_flat = torch.cat([g.flatten() for g in gradients])

        return grad_flat
