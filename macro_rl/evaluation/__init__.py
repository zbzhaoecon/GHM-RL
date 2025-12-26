"""
Evaluation utilities for GHM-RL policies.

This module provides evaluation functions for different solver types:
- Monte Carlo policy gradient
- Actor-Critic
- Time-augmented vs standard dynamics

Separated from training scripts for better modularity and reusability.
"""

import torch
from typing import Dict, Optional


def evaluate_monte_carlo_policy(
    solver,
    dynamics,
    n_episodes: int = 50,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate Monte Carlo policy on deterministic rollouts.

    Args:
        solver: MonteCarloPolicyGradient solver
        dynamics: Dynamics model (for state space bounds)
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy (mean action)

    Returns:
        Dictionary with evaluation metrics:
        - return_mean: Mean return across episodes
        - return_std: Standard deviation of returns
        - episode_length: Mean episode length
        - termination_rate: Fraction of episodes that terminated early
    """
    solver.policy.eval()

    # Sample initial states
    state_space = dynamics.state_space
    device = next(solver.policy.parameters()).device

    if state_space.dim == 1:
        # Standard dynamics: sample c uniformly
        c_values = torch.rand(n_episodes, 1, device=device) * state_space.upper[0]
        initial_states = c_values
    elif state_space.dim == 2:
        # Time-augmented: sample c uniformly, τ = T (start at beginning)
        c_values = torch.rand(n_episodes, 1, device=device) * state_space.upper[0]
        tau_values = torch.full((n_episodes, 1), state_space.upper[1].item(), device=device)
        initial_states = torch.cat([c_values, tau_values], dim=1)
    else:
        raise ValueError(f"Unsupported state dimension: {state_space.dim}")

    # Create deterministic policy wrapper
    class DeterministicPolicy:
        def __init__(self, policy):
            self.policy = policy.policy if hasattr(policy, 'policy') else policy

        def act(self, state):
            action, _ = self.policy.sample(state, deterministic=deterministic)
            return action

    with torch.no_grad():
        det_policy = DeterministicPolicy(solver.policy)
        trajectories = solver.simulator.rollout(det_policy, initial_states)

    solver.policy.train()

    # Compute metrics
    returns = trajectories.returns
    masks = trajectories.masks  # (batch, steps)

    # Episode length: sum of masks per trajectory
    episode_lengths = masks.sum(dim=1)

    # Termination rate: fraction that didn't reach max_steps
    max_steps = solver.simulator.max_steps
    terminated_early = (episode_lengths < max_steps).float()

    return {
        'return_mean': returns.mean().item(),
        'return_std': returns.std().item(),
        'episode_length': episode_lengths.mean().item(),
        'termination_rate': terminated_early.mean().item(),
    }


def evaluate_actor_critic_policy(
    solver,
    n_episodes: int = 50,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate Actor-Critic policy.

    Args:
        solver: ModelBasedActorCritic solver
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy (mean action)

    Returns:
        Dictionary with evaluation metrics
    """
    solver.ac.eval()

    # Sample initial states
    state_space = solver.dynamics.state_space
    device = next(solver.ac.parameters()).device

    if state_space.dim == 1:
        # Standard dynamics
        c_values = torch.rand(n_episodes, 1, device=device) * state_space.upper[0]
        initial_states = c_values
    elif state_space.dim == 2:
        # Time-augmented
        c_values = torch.rand(n_episodes, 1, device=device) * state_space.upper[0]
        tau_values = torch.full((n_episodes, 1), state_space.upper[1].item(), device=device)
        initial_states = torch.cat([c_values, tau_values], dim=1)
    else:
        raise ValueError(f"Unsupported state dimension: {state_space.dim}")

    # Create deterministic policy wrapper
    class DeterministicPolicy:
        """Wrapper to make ActorCritic act deterministically."""
        def __init__(self, ac):
            self.ac = ac
            # Copy over attributes needed by parallel simulator
            self.state_dim = ac.state_dim
            self.action_dim = ac.action_dim
            self.hidden_dims = ac.hidden_dims
            self.shared_layers = ac.shared_layers
            self.action_bounds = ac.action_bounds

        def act(self, state):
            action, _ = self.ac.sample(state, deterministic=deterministic)
            return action

        def state_dict(self):
            """Delegate to underlying network for serialization."""
            return self.ac.state_dict()

        def load_state_dict(self, state_dict):
            """Delegate to underlying network for deserialization."""
            self.ac.load_state_dict(state_dict)

        def parameters(self):
            """Delegate to underlying network for device detection."""
            return self.ac.parameters()

        def to(self, device):
            """Delegate to underlying network for device movement."""
            self.ac.to(device)
            return self

    with torch.no_grad():
        det_policy = DeterministicPolicy(solver.ac)
        trajectories = solver.simulator.rollout(det_policy, initial_states)

    solver.ac.train()

    # Compute metrics
    returns = trajectories.returns
    masks = trajectories.masks
    episode_lengths = masks.sum(dim=1)
    max_steps = solver.simulator.max_steps
    terminated_early = (episode_lengths < max_steps).float()

    return {
        'return_mean': returns.mean().item(),
        'return_std': returns.std().item(),
        'episode_length': episode_lengths.mean().item(),
        'termination_rate': terminated_early.mean().item(),
    }


def evaluate_policy(
    solver,
    solver_type: str,
    dynamics=None,
    n_episodes: int = 50,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Unified evaluation function that routes to appropriate evaluator.

    Args:
        solver: Solver instance (MonteCarloPolicyGradient or ModelBasedActorCritic)
        solver_type: "monte_carlo" or "actor_critic"
        dynamics: Dynamics model (required for Monte Carlo)
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy

    Returns:
        Dictionary with evaluation metrics
    """
    if solver_type == "monte_carlo":
        if dynamics is None:
            # Try to get dynamics from solver
            dynamics = getattr(solver, 'dynamics', None)
            if dynamics is None:
                raise ValueError("dynamics must be provided for Monte Carlo evaluation")
        return evaluate_monte_carlo_policy(solver, dynamics, n_episodes, deterministic)

    elif solver_type == "actor_critic":
        return evaluate_actor_critic_policy(solver, n_episodes, deterministic)

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
