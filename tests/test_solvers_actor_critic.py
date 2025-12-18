"""
Tests for model-based actor-critic solver.
"""

import pytest
import torch
import numpy as np

from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.networks.actor_critic import ActorCritic
from macro_rl.solvers.actor_critic import ModelBasedActorCritic


@pytest.fixture
def simple_setup():
    """Simple GHM setup with short horizon for fast tests."""
    # Dynamics
    params = GHMEquityParams()
    dynamics = GHMEquityDynamics(params)

    # Control spec
    control_spec = GHMControlSpec()

    # Reward function
    reward_fn = GHMRewardFunction(
        discount_rate=params.r - params.mu,
        issuance_cost=params.lambda_,
        liquidation_rate=params.omega,
        liquidation_flow=params.alpha,
    )

    state_dim = 1  # cash reserves
    action_dim = 2  # dividend, equity issuance

    return dynamics, control_spec, reward_fn, state_dim, action_dim


def test_actor_critic_shared_layers(simple_setup):
    """Test actor-critic solver with shared layers."""
    dynamics, control_spec, reward_fn, state_dim, action_dim = simple_setup

    # Build ActorCritic with shared layers
    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=1,
        shared_hidden_dim=32,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        action_bounds=(control_spec.lower, control_spec.upper),
    )

    # Build solver with very short horizon for fast test
    solver = ModelBasedActorCritic(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        actor_critic=ac,
        dt=0.05,
        T=0.1,  # Very short horizon
        critic_loss="mc+hjb",
        actor_loss="pathwise",
        n_trajectories=8,
        lr=3e-4,
    )

    # Test single training step
    metrics = solver.train_step(n_samples=8)

    # Check that all expected metrics are present
    assert "actor/loss" in metrics
    assert "actor/return" in metrics
    assert "critic/mc" in metrics
    assert "critic/hjb" in metrics
    assert "loss/total" in metrics
    assert "return/mean" in metrics
    assert "return/std" in metrics
    assert "episode_length/mean" in metrics

    # Check that metrics are reasonable
    assert not np.isnan(metrics["loss/total"])
    assert not np.isnan(metrics["return/mean"])
    assert metrics["episode_length/mean"] > 0

    # Test short training loop
    history = solver.train(n_iterations=3, log_freq=1)

    assert "return/mean" in history
    assert len(history["return/mean"]) == 3
    assert "loss/total" in history
    assert len(history["loss/total"]) == 3


def test_actor_critic_no_shared_layers(simple_setup):
    """Test actor-critic solver without shared layers."""
    dynamics, control_spec, reward_fn, state_dim, action_dim = simple_setup

    # Build ActorCritic with NO shared layers
    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=0,  # No shared layers
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        action_bounds=(control_spec.lower, control_spec.upper),
    )

    # Build solver
    solver = ModelBasedActorCritic(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        actor_critic=ac,
        dt=0.05,
        T=0.1,
        critic_loss="mc",
        actor_loss="pathwise",
        n_trajectories=8,
        lr=3e-4,
    )

    # Test single training step
    metrics = solver.train_step(n_samples=8)

    assert "actor/loss" in metrics
    assert "loss/total" in metrics
    assert "return/mean" in metrics

    # Test short training loop
    history = solver.train(n_iterations=3, log_freq=1)

    assert "return/mean" in history
    assert len(history["return/mean"]) == 3


def test_actor_critic_reinforce(simple_setup):
    """Test actor-critic with REINFORCE actor loss."""
    dynamics, control_spec, reward_fn, state_dim, action_dim = simple_setup

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=1,
        shared_hidden_dim=32,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        action_bounds=(control_spec.lower, control_spec.upper),
    )

    solver = ModelBasedActorCritic(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        actor_critic=ac,
        dt=0.05,
        T=0.1,
        critic_loss="mc",
        actor_loss="reinforce",  # Use REINFORCE
        entropy_weight=0.01,
        n_trajectories=8,
        lr=3e-4,
    )

    metrics = solver.train_step(n_samples=8)

    # REINFORCE should have advantage metric
    assert "actor/advantage" in metrics
    assert "actor/entropy" in metrics
    assert "actor/loss" in metrics
    assert "loss/total" in metrics


def test_actor_critic_td_loss(simple_setup):
    """Test actor-critic with TD critic loss."""
    dynamics, control_spec, reward_fn, state_dim, action_dim = simple_setup

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=1,
        shared_hidden_dim=32,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        action_bounds=(control_spec.lower, control_spec.upper),
    )

    solver = ModelBasedActorCritic(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        actor_critic=ac,
        dt=0.05,
        T=0.1,
        critic_loss="td",  # TD loss
        actor_loss="pathwise",
        n_trajectories=8,
        lr=3e-4,
    )

    metrics = solver.train_step(n_samples=8)

    # TD loss should be present
    assert "critic/td" in metrics
    assert "loss/total" in metrics


def test_actor_critic_evaluate(simple_setup):
    """Test evaluation method."""
    dynamics, control_spec, reward_fn, state_dim, action_dim = simple_setup

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=1,
        shared_hidden_dim=32,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        action_bounds=(control_spec.lower, control_spec.upper),
    )

    solver = ModelBasedActorCritic(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        actor_critic=ac,
        dt=0.05,
        T=0.1,
        n_trajectories=8,
    )

    # Evaluate
    eval_metrics = solver.evaluate(n_episodes=10)

    assert "return_mean" in eval_metrics
    assert "return_std" in eval_metrics
    assert "episode_length" in eval_metrics
    assert not np.isnan(eval_metrics["return_mean"])
    assert eval_metrics["episode_length"] > 0


def test_actor_critic_with_custom_gamma(simple_setup):
    """Test actor-critic with custom gamma."""
    dynamics, control_spec, reward_fn, state_dim, action_dim = simple_setup

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=1,
        action_bounds=(control_spec.lower, control_spec.upper),
    )

    # Custom gamma instead of using dynamics discount rate
    solver = ModelBasedActorCritic(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        actor_critic=ac,
        dt=0.05,
        T=0.1,
        gamma=0.99,  # Custom gamma
        n_trajectories=8,
    )

    assert solver.gamma == 0.99

    metrics = solver.train_step(n_samples=8)
    assert "loss/total" in metrics


def test_actor_critic_optimizer_updates_all_params(simple_setup):
    """Test that single optimizer updates both actor and critic."""
    dynamics, control_spec, reward_fn, state_dim, action_dim = simple_setup

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=1,
        shared_hidden_dim=32,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        action_bounds=(control_spec.lower, control_spec.upper),
    )

    solver = ModelBasedActorCritic(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        actor_critic=ac,
        dt=0.05,
        T=0.1,
        n_trajectories=8,
    )

    # Store initial parameters
    initial_actor_params = [p.clone() for p in ac.actor.parameters()]
    initial_critic_params = [p.clone() for p in ac.critic.parameters()]
    if ac.shared_trunk is not None:
        initial_shared_params = [p.clone() for p in ac.shared_trunk.parameters()]

    # Run one training step
    solver.train_step(n_samples=8)

    # Check that actor parameters changed
    actor_changed = False
    for p_initial, p_current in zip(initial_actor_params, ac.actor.parameters()):
        if not torch.allclose(p_initial, p_current):
            actor_changed = True
            break

    # Check that critic parameters changed
    critic_changed = False
    for p_initial, p_current in zip(initial_critic_params, ac.critic.parameters()):
        if not torch.allclose(p_initial, p_current):
            critic_changed = True
            break

    assert actor_changed, "Actor parameters should be updated"
    assert critic_changed, "Critic parameters should be updated"

    # If there's a shared trunk, it should also be updated
    if ac.shared_trunk is not None:
        shared_changed = False
        for p_initial, p_current in zip(initial_shared_params, ac.shared_trunk.parameters()):
            if not torch.allclose(p_initial, p_current):
                shared_changed = True
                break
        assert shared_changed, "Shared trunk parameters should be updated"


def test_actor_critic_gradient_clipping(simple_setup):
    """Test that gradient clipping is applied."""
    dynamics, control_spec, reward_fn, state_dim, action_dim = simple_setup

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=1,
        action_bounds=(control_spec.lower, control_spec.upper),
    )

    solver = ModelBasedActorCritic(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        actor_critic=ac,
        dt=0.05,
        T=0.1,
        n_trajectories=8,
        max_grad_norm=0.5,  # Small clip value
    )

    # This should not raise an error even with gradient clipping
    metrics = solver.train_step(n_samples=8)
    assert "loss/total" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
