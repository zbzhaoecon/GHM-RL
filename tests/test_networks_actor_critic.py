import pytest
import torch

from macro_rl.networks.actor_critic import ActorCritic


def test_actor_critic_no_sharing():
    """Test ActorCritic with shared_layers=0 (no parameter sharing)."""
    state_dim = 3
    action_dim = 2
    batch_size = 4

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        shared_layers=0,
    )

    # Verify no shared trunk
    assert ac.shared is None

    # Test act()
    state = torch.randn(batch_size, state_dim)
    action = ac.act(state)
    assert action.shape == (batch_size, action_dim)

    # Test evaluate()
    value = ac.evaluate(state)
    assert value.shape == (batch_size,)

    # Test evaluate_actions()
    value2, log_prob, entropy = ac.evaluate_actions(state, action)
    assert value2.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)


def test_actor_critic_with_sharing():
    """Test ActorCritic with shared_layers=1 (first layer shared)."""
    state_dim = 3
    action_dim = 2
    batch_size = 4

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        shared_layers=1,
    )

    # Verify shared trunk exists
    assert ac.shared is not None

    # Test act()
    state = torch.randn(batch_size, state_dim)
    action = ac.act(state)
    assert action.shape == (batch_size, action_dim)

    # Test evaluate()
    value = ac.evaluate(state)
    assert value.shape == (batch_size,)

    # Test evaluate_actions()
    value2, log_prob, entropy = ac.evaluate_actions(state, action)
    assert value2.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)


def test_actor_critic_deterministic_action():
    """Test deterministic action sampling."""
    state_dim = 3
    action_dim = 2

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        shared_layers=0,
    )

    state = torch.randn(1, state_dim)

    # Deterministic actions should be identical across calls
    action1 = ac.act(state, deterministic=True)
    action2 = ac.act(state, deterministic=True)
    assert torch.allclose(action1, action2)

    # Stochastic actions should differ (with high probability)
    action3 = ac.act(state, deterministic=False)
    action4 = ac.act(state, deterministic=False)
    # Note: there's a tiny chance they could be equal, but extremely unlikely
    assert not torch.allclose(action3, action4)


def test_actor_critic_with_action_bounds():
    """Test ActorCritic with action bounds."""
    state_dim = 3
    action_dim = 2
    batch_size = 10

    lower = torch.tensor([-1.0, -2.0])
    upper = torch.tensor([1.0, 2.0])

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        shared_layers=1,
        action_bounds=(lower, upper),
    )

    state = torch.randn(batch_size, state_dim)
    action = ac.act(state, deterministic=True)

    # Check that actions are within bounds
    assert torch.all(action >= lower)
    assert torch.all(action <= upper)


def test_actor_critic_forward():
    """Test forward() method returns (mean_action, value)."""
    state_dim = 3
    action_dim = 2
    batch_size = 4

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        shared_layers=0,
    )

    state = torch.randn(batch_size, state_dim)
    mean_action, value = ac.forward(state)

    assert mean_action.shape == (batch_size, action_dim)
    assert value.shape == (batch_size,)


def test_actor_critic_invalid_shared_layers():
    """Test that invalid shared_layers raises ValueError."""
    with pytest.raises(ValueError, match="shared_layers cannot exceed len\\(hidden_dims\\)"):
        ActorCritic(
            state_dim=3,
            action_dim=2,
            hidden_dims=[64, 64],
            shared_layers=3,  # More than len(hidden_dims)
        )


def test_actor_critic_all_layers_shared():
    """Test ActorCritic with all layers shared."""
    state_dim = 3
    action_dim = 2
    batch_size = 4

    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        shared_layers=2,  # All layers shared
    )

    # Verify shared trunk exists
    assert ac.shared is not None

    state = torch.randn(batch_size, state_dim)
    action = ac.act(state)
    value = ac.evaluate(state)

    assert action.shape == (batch_size, action_dim)
    assert value.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
