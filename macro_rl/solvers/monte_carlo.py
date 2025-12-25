"""
Monte Carlo Policy Gradient solver.

Uses known dynamics to simulate many trajectories and estimate
policy gradients via REINFORCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from typing import Optional, Dict
import numpy as np

from macro_rl.solvers.base import Solver, SolverResult


class MonteCarloPolicyGradient(Solver):
    """
    Monte Carlo Policy Gradient (REINFORCE with known dynamics).

    Algorithm:
        1. Sample initial states
        2. Simulate N trajectories using known dynamics
        3. Compute returns for each trajectory
        4. Estimate gradient: ∇J ≈ E[∇log π(a|s) · (R - b)]
        5. Update policy

    Key advantage over model-free:
        - Can simulate unlimited trajectories (free)
        - Can sample any initial state (full coverage)
        - Known dynamics = no model error

    Example:
        >>> solver = MonteCarloPolicyGradient(
        ...     policy=policy,
        ...     simulator=simulator,
        ...     n_trajectories=1000,
        ...     lr=1e-3,
        ... )
        >>> result = solver.solve(
        ...     dynamics=dynamics,
        ...     control_spec=control_spec,
        ...     reward_fn=reward_fn,
        ...     n_iterations=10000,
        ... )
    """

    def __init__(
        self,
        policy,  # Policy
        simulator,  # TrajectorySimulator
        n_trajectories: int = 1000,
        baseline: Optional[nn.Module] = None,
        lr_policy: float = 3e-4,
        lr_baseline: float = 1e-3,
        batch_size: int = 1000,
        advantage_normalization: bool = True,
        max_grad_norm: float = 0.5,
        entropy_weight: float = 0.05,  # INCREASED from 0.01 to encourage exploration
    ):
        """
        Initialize Monte Carlo solver.

        Args:
            policy: Policy to optimize
            simulator: TrajectorySimulator for rollouts
            n_trajectories: Number of trajectories per iteration
            baseline: Optional baseline (value function) for variance reduction
            lr_policy: Learning rate for policy
            lr_baseline: Learning rate for baseline
            batch_size: Batch size for initial state sampling
            advantage_normalization: Whether to normalize advantages
            max_grad_norm: Gradient clipping threshold
            entropy_weight: Entropy regularization weight (encourage exploration)
        """
        self.policy = policy
        self.simulator = simulator
        self.n_trajectories = n_trajectories
        self.baseline = baseline
        self.batch_size = batch_size
        self.advantage_normalization = advantage_normalization
        self.max_grad_norm = max_grad_norm
        self.entropy_weight = entropy_weight

        # Optimizers
        self.policy_optimizer = Adam(policy.parameters(), lr=lr_policy)
        if baseline is not None:
            self.baseline_optimizer = Adam(baseline.parameters(), lr=lr_baseline)
        else:
            self.baseline_optimizer = None

    def solve(
        self,
        dynamics,
        control_spec,
        reward_fn,
        n_iterations: int = 10000,
        log_interval: int = 100,
        **kwargs,
    ) -> SolverResult:
        """
        Solve for optimal policy via Monte Carlo.

        Args:
            dynamics: Dynamics model
            control_spec: Control specification
            reward_fn: Reward function
            n_iterations: Number of training iterations
            log_interval: Logging frequency

        Returns:
            SolverResult
        """
        # Store dynamics for state sampling
        self.dynamics = dynamics

        diagnostics = {
            "returns": [],
            "policy_loss": [],
            "baseline_loss": [],
            "advantages": [],
            "grad_norm_policy": [],
            "grad_norm_baseline": [],
        }

        for iteration in range(n_iterations):
            # Perform training step
            metrics = self.train_step()

            # Store diagnostics
            diagnostics["returns"].append(metrics["return/mean"])
            diagnostics["policy_loss"].append(metrics["loss/policy"])
            diagnostics["baseline_loss"].append(metrics.get("loss/baseline", 0.0))
            diagnostics["advantages"].append(metrics["advantage/mean"])
            diagnostics["grad_norm_policy"].append(metrics.get("grad_norm/policy", 0.0))
            if self.baseline is not None:
                diagnostics["grad_norm_baseline"].append(metrics.get("grad_norm/baseline", 0.0))

            if iteration % log_interval == 0:
                self._log_progress(iteration, metrics)

        return SolverResult(
            policy=self.policy,
            value_fn=self.baseline,
            diagnostics=diagnostics,
        )

    def train_step(self) -> Dict[str, float]:
        """
        Single training iteration.

        Returns:
            metrics: Dictionary of training metrics
        """
        # 1. Sample initial states
        initial_states = self._sample_initial_states(self.n_trajectories)

        # 2. Update simulator with current baseline for boundary condition
        if self.baseline is not None:
            self.simulator.value_function = self.baseline

        # 3. Rollout trajectories
        with torch.no_grad():
            trajectories = self.simulator.rollout(self.policy, initial_states)

        # 3. Compute advantages
        returns = trajectories.returns  # (B,)
        if self.baseline is not None:
            with torch.no_grad():
                values = self.baseline(initial_states).detach()  # (B,)
            advantages = returns - values
        else:
            advantages = returns.clone()

        # Adaptive advantage normalization (only normalize if there's real variance)
        if self.advantage_normalization:
            adv_std = advantages.std()
            if adv_std > 1e-3:
                # Normal normalization when there's real variance
                advantages = (advantages - advantages.mean()) / adv_std
            else:
                # Just center when variance is too low (avoids noise amplification)
                advantages = advantages - advantages.mean()

        # 4. Compute policy loss (REINFORCE)
        policy_loss = self._compute_policy_loss(trajectories, advantages)

        # DIAGNOSTIC: Check for NaN/Inf in policy loss
        if not torch.isfinite(policy_loss):
            print(f"WARNING: Non-finite policy loss detected: {policy_loss.item()}")
            print(f"  Advantages: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}")
            print(f"  Returns: min={returns.min():.4f}, max={returns.max():.4f}")
            # Skip this update to prevent corruption
            return self._get_safe_metrics(trajectories, initial_states, returns, advantages)

        # 4b. Add entropy bonus (encourage exploration)
        entropy = self.policy.entropy(initial_states).mean()

        # Total loss combines all objectives
        # REMOVED action regularization - it was pushing actions to boundaries!
        # The negative sign made it MAXIMIZE action magnitude instead of preventing collapse
        total_loss = policy_loss - self.entropy_weight * entropy

        # DIAGNOSTIC: Check for NaN/Inf in total loss
        if not torch.isfinite(total_loss):
            print(f"WARNING: Non-finite total loss detected: {total_loss.item()}")
            print(f"  Policy loss: {policy_loss.item()}, Entropy: {entropy.item()}")
            return self._get_safe_metrics(trajectories, initial_states, returns, advantages)

        # 5. Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients
        policy_grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.max_grad_norm
        )

        # DIAGNOSTIC: Check for NaN/Inf in gradients
        has_nan_grad = False
        for param in self.policy.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"WARNING: Non-finite gradient detected in policy parameters")
                has_nan_grad = True
                break

        if has_nan_grad:
            print("WARNING: Skipping optimizer step due to non-finite gradients")
            self.policy_optimizer.zero_grad()
            return self._get_safe_metrics(trajectories, initial_states, returns, advantages)

        self.policy_optimizer.step()

        # 6. Update baseline (if exists)
        baseline_loss = torch.tensor(0.0)
        baseline_grad_norm = torch.tensor(0.0)
        if self.baseline is not None:
            baseline_loss, baseline_grad_norm = self._update_baseline(
                initial_states,
                returns
            )

        # 7. Collect metrics
        with torch.no_grad():
            # Get policy statistics
            dist = self.policy(initial_states)
            if hasattr(dist, 'mode'):
                # TanhNormal distribution
                mean_actions = dist.mode  # (B, action_dim)
                std_actions = dist.stddev  # (B, action_dim)
            else:
                # Normal distribution
                mean_actions = dist.mean  # (B, action_dim)
                std_actions = dist.stddev  # (B, action_dim)

            # Compute action magnitude for diagnostics only (not used in loss)
            action_magnitude = trajectories.actions.abs().mean()

            metrics = {
                # Returns
                'return/mean': returns.mean().item(),
                'return/std': returns.std().item(),
                'return/min': returns.min().item(),
                'return/max': returns.max().item(),

                # Losses
                'loss/policy': policy_loss.item(),
                'loss/total': total_loss.item(),
                'loss/baseline': baseline_loss if isinstance(baseline_loss, float) else baseline_loss.item(),

                # Advantages
                'advantage/mean': advantages.mean().item(),
                'advantage/std': advantages.std().item(),
                'advantage/max': advantages.max().item(),
                'advantage/min': advantages.min().item(),

                # Episode statistics
                'episode_length/mean': trajectories.masks.sum(dim=-1).mean().item(),
                'episode_length/std': trajectories.masks.sum(dim=-1).std().item(),
                'termination_rate': (trajectories.masks.sum(dim=-1) < trajectories.masks.shape[1]).float().mean().item(),

                # Policy statistics
                'policy/mean_action_0': mean_actions[:, 0].mean().item(),
                'policy/std_action_0': std_actions[:, 0].mean().item(),
                'policy/entropy': self.policy.entropy(initial_states).mean().item(),
                'policy/action_magnitude': action_magnitude.item(),

                # Gradients
                'grad_norm/policy': policy_grad_norm.item() if isinstance(policy_grad_norm, torch.Tensor) else float(policy_grad_norm),
                'grad_norm/baseline': baseline_grad_norm.item() if isinstance(baseline_grad_norm, torch.Tensor) else float(baseline_grad_norm),
            }

            # Add action statistics for multi-dimensional actions
            if mean_actions.shape[1] > 1:
                metrics['policy/mean_action_1'] = mean_actions[:, 1].mean().item()
                metrics['policy/std_action_1'] = std_actions[:, 1].mean().item()

            # DIAGNOSTIC: Add statistics for detecting boundary issues
            actions_sample = trajectories.actions
            metrics['diagnostics/action_min'] = actions_sample.min().item()
            metrics['diagnostics/action_max'] = actions_sample.max().item()
            metrics['diagnostics/action_mean'] = actions_sample.mean().item()

            # Compute log probs for diagnostic purposes
            with torch.no_grad():
                states_first = trajectories.states[:, 0, :]  # First state in each trajectory
                actions_first = trajectories.actions[:, 0, :]  # First action
                log_probs_sample = self.policy.log_prob(states_first, actions_first)
                metrics['diagnostics/log_prob_mean'] = log_probs_sample.mean().item()
                metrics['diagnostics/log_prob_min'] = log_probs_sample.min().item()
                metrics['diagnostics/log_prob_max'] = log_probs_sample.max().item()

        return metrics

    def _compute_policy_loss(self, trajectories, advantages: Tensor) -> Tensor:
        """
        Compute REINFORCE policy loss.

        Args:
            trajectories: TrajectoryBatch from rollout
            advantages: Advantages (B,)

        Returns:
            Policy loss (negative for maximization)
        """
        B, T = trajectories.actions.shape[0], trajectories.actions.shape[1]

        # Flatten trajectories: (B, T, ...) → (B*T, ...)
        states_flat = trajectories.states[:, :-1, :].reshape(B * T, -1)  # Exclude terminal state
        actions_flat = trajectories.actions.reshape(B * T, -1)
        masks_flat = trajectories.masks.reshape(B * T)

        # Compute log π(aₜ|sₜ) for all timesteps
        log_probs_flat = self.policy.log_prob(states_flat, actions_flat)  # (B*T,)
        log_probs = log_probs_flat.reshape(B, T)  # (B, T)

        # REINFORCE loss (negative for maximization)
        # Standard REINFORCE: ∇J = E_τ [ (Σ_t log π(a_t|s_t)) · (G - b) ]
        # where the expectation is over trajectories, not timesteps

        # Sum log probabilities over time for each trajectory
        log_prob_per_traj = (log_probs * trajectories.masks).sum(dim=1)  # (B,)

        # REINFORCE gradient estimator: average over trajectories
        policy_loss = -(log_prob_per_traj * advantages).mean()

        return policy_loss

    def _sample_initial_states(self, n: int) -> Tensor:
        """
        Sample initial states from state space.

        Args:
            n: Number of states to sample

        Returns:
            Initial states (n, state_dim)
        """
        state_space = self.dynamics.state_space

        # Get device from policy parameters
        device = next(self.policy.parameters()).device

        # Check if dynamics has custom sampling (e.g., for time-augmented states)
        if hasattr(self.dynamics, 'sample_initial_states'):
            states = self.dynamics.sample_initial_states(n, device)
            return states

        # Check if state_space has sample method
        if hasattr(state_space, 'sample'):
            states = state_space.sample(n)
            return states.to(device)

        # Otherwise use uniform sampling from bounds
        lower = state_space.lower.to(device)
        upper = state_space.upper.to(device)

        # Sample uniformly
        states = lower + (upper - lower) * torch.rand(
            n, len(lower),
            device=device,
            dtype=lower.dtype
        )

        return states

    def _update_baseline(
        self,
        states: Tensor,
        returns: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Update baseline (value function) via regression.

        Args:
            states: Initial states from trajectories (B, state_dim)
            returns: Actual returns from trajectories (B,)

        Returns:
            (baseline_loss, grad_norm)
        """
        # Predict values
        value_pred = self.baseline(states)  # (B,)

        # Compute MSE loss
        baseline_loss = F.mse_loss(value_pred, returns.detach())

        # Update baseline
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()

        # Clip gradients
        baseline_grad_norm = nn.utils.clip_grad_norm_(
            self.baseline.parameters(),
            self.max_grad_norm
        )

        self.baseline_optimizer.step()

        return baseline_loss, baseline_grad_norm

    def _get_safe_metrics(
        self,
        trajectories,
        initial_states: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> Dict[str, float]:
        """
        Return safe metrics when training step fails due to NaN/Inf.

        This prevents crashes and logs diagnostic information.
        """
        with torch.no_grad():
            dist = self.policy(initial_states)
            if hasattr(dist, 'mode'):
                mean_actions = dist.mode
                std_actions = dist.stddev
            else:
                mean_actions = dist.mean
                std_actions = dist.stddev

            action_magnitude = trajectories.actions.abs().mean()

            return {
                'return/mean': returns.mean().item() if torch.isfinite(returns).all() else 0.0,
                'return/std': returns.std().item() if torch.isfinite(returns).all() else 0.0,
                'return/min': returns.min().item() if torch.isfinite(returns).all() else 0.0,
                'return/max': returns.max().item() if torch.isfinite(returns).all() else 0.0,
                'loss/policy': 0.0,  # Failed to compute
                'loss/total': 0.0,
                'loss/baseline': 0.0,
                'advantage/mean': advantages.mean().item() if torch.isfinite(advantages).all() else 0.0,
                'advantage/std': advantages.std().item() if torch.isfinite(advantages).all() else 0.0,
                'advantage/max': advantages.max().item() if torch.isfinite(advantages).all() else 0.0,
                'advantage/min': advantages.min().item() if torch.isfinite(advantages).all() else 0.0,
                'episode_length/mean': trajectories.masks.sum(dim=-1).mean().item(),
                'episode_length/std': trajectories.masks.sum(dim=-1).std().item(),
                'termination_rate': 0.0,
                'policy/mean_action_0': mean_actions[:, 0].mean().item(),
                'policy/std_action_0': std_actions[:, 0].mean().item(),
                'policy/entropy': 0.0,
                'policy/action_magnitude': action_magnitude.item() if torch.isfinite(action_magnitude) else 0.0,
                'grad_norm/policy': 0.0,
                'grad_norm/baseline': 0.0,
                'diagnostics/action_min': trajectories.actions.min().item(),
                'diagnostics/action_max': trajectories.actions.max().item(),
                'diagnostics/action_mean': trajectories.actions.mean().item(),
                'diagnostics/log_prob_mean': float('nan'),
                'diagnostics/log_prob_min': float('nan'),
                'diagnostics/log_prob_max': float('nan'),
            }

    def _log_progress(self, iteration: int, metrics: Dict[str, float]):
        """
        Log training progress.

        Args:
            iteration: Current iteration
            metrics: Metrics dictionary
        """
        print(f"[Iter {iteration:6d}] "
              f"Return: {metrics['return/mean']:7.3f} ± {metrics['return/std']:6.3f} | "
              f"Policy Loss: {metrics['loss/policy']:8.4f} | "
              f"Baseline Loss: {metrics.get('loss/baseline', 0.0):8.4f}")
