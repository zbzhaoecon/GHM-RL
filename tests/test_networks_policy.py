import pytest
import torch
from macro_rl.networks.policy import GaussianPolicy


class TestGaussianPolicy:
    """Tests for GaussianPolicy network."""

    def test_init_unbounded(self):
        """Test initialization without action bounds."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64, 64])
        assert policy.action_bounds is None
        assert policy.log_std.shape == (2,)

    def test_init_with_bounds(self):
        """Test initialization with action bounds."""
        low = torch.tensor([-1.0, -2.0])
        high = torch.tensor([1.0, 2.0])
        policy = GaussianPolicy(
            input_dim=4,
            output_dim=2,
            hidden_dims=[64, 64],
            action_bounds=(low, high),
        )
        assert policy.action_bounds is not None
        assert torch.allclose(policy.action_low, low)
        assert torch.allclose(policy.action_high, high)

    def test_forward_shape(self):
        """Test forward pass returns correct shape."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(10, 4)
        mean = policy(state)
        assert mean.shape == (10, 2)

    def test_sample_shape(self):
        """Test sample returns correct shapes."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(10, 4)

        action, log_prob = policy.sample(state)
        assert action.shape == (10, 2)
        assert log_prob.shape == (10,)

    def test_sample_deterministic(self):
        """Test deterministic sampling returns same action."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(1, 4)

        action1, _ = policy.sample(state, deterministic=True)
        action2, _ = policy.sample(state, deterministic=True)
        assert torch.allclose(action1, action2)

    def test_sample_stochastic_different(self):
        """Test stochastic sampling returns different actions."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(1, 4)

        # Set seed for reproducibility in testing
        torch.manual_seed(42)
        action1, _ = policy.sample(state, deterministic=False)
        torch.manual_seed(43)
        action2, _ = policy.sample(state, deterministic=False)
        # Should be different with different seeds
        assert not torch.allclose(action1, action2, atol=1e-6)

    def test_sample_with_noise_shape(self):
        """Test sample_with_noise returns correct shape."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(10, 4)
        noise = torch.randn(10, 2)

        action = policy.sample_with_noise(state, noise)
        assert action.shape == (10, 2)

    def test_sample_with_noise_deterministic(self):
        """Test sample_with_noise is deterministic with same noise."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(1, 4)
        noise = torch.randn(1, 2)

        action1 = policy.sample_with_noise(state, noise)
        action2 = policy.sample_with_noise(state, noise)
        assert torch.allclose(action1, action2)

    def test_sample_with_noise_zero_noise(self):
        """Test sample_with_noise with zero noise returns mean."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(1, 4)
        noise = torch.zeros(1, 2)

        action = policy.sample_with_noise(state, noise)
        mean = policy(state)
        assert torch.allclose(action, mean, atol=1e-5)

    def test_action_bounds_respected(self):
        """Test that actions are within bounds when action_bounds is set."""
        low = torch.tensor([-1.0, -2.0])
        high = torch.tensor([1.0, 2.0])
        policy = GaussianPolicy(
            input_dim=4,
            output_dim=2,
            hidden_dims=[64],
            action_bounds=(low, high),
        )

        state = torch.randn(100, 4)
        action, _ = policy.sample(state)

        # Actions should be within bounds (with some numerical tolerance)
        assert torch.all(action[:, 0] >= low[0] - 1e-5)
        assert torch.all(action[:, 0] <= high[0] + 1e-5)
        assert torch.all(action[:, 1] >= low[1] - 1e-5)
        assert torch.all(action[:, 1] <= high[1] + 1e-5)

    def test_action_bounds_with_noise(self):
        """Test that sample_with_noise respects action bounds."""
        low = torch.tensor([-1.0, -2.0])
        high = torch.tensor([1.0, 2.0])
        policy = GaussianPolicy(
            input_dim=4,
            output_dim=2,
            hidden_dims=[64],
            action_bounds=(low, high),
        )

        state = torch.randn(100, 4)
        noise = torch.randn(100, 2)
        action = policy.sample_with_noise(state, noise)

        # Actions should be within bounds
        assert torch.all(action[:, 0] >= low[0] - 1e-5)
        assert torch.all(action[:, 0] <= high[0] + 1e-5)
        assert torch.all(action[:, 1] >= low[1] - 1e-5)
        assert torch.all(action[:, 1] <= high[1] + 1e-5)

    def test_log_std_bounds(self):
        """Test that log_std is clamped to bounds."""
        policy = GaussianPolicy(
            input_dim=4,
            output_dim=2,
            hidden_dims=[64],
            log_std_bounds=(-5.0, 2.0),
        )

        # Set log_std to extreme values
        policy.log_std.data.fill_(100.0)
        state = torch.randn(1, 4)
        _, log_std = policy._get_mean_log_std(state)

        # Should be clamped to upper bound
        assert torch.allclose(log_std, torch.tensor([2.0, 2.0]))

        # Set to very negative
        policy.log_std.data.fill_(-100.0)
        _, log_std = policy._get_mean_log_std(state)

        # Should be clamped to lower bound
        assert torch.allclose(log_std, torch.tensor([-5.0, -5.0]))

    def test_small_std_deterministic_behavior(self):
        """Test that very small std makes sampling nearly deterministic."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])

        # Make log_std very small
        policy.log_std.data.fill_(-10.0)

        state = torch.randn(1, 4)

        # Sample multiple times with deterministic=True
        action1, _ = policy.sample(state, deterministic=True)
        action2, _ = policy.sample(state, deterministic=True)
        action3, _ = policy.sample(state, deterministic=True)

        # All should be identical
        assert torch.allclose(action1, action2)
        assert torch.allclose(action2, action3)

        # And should be close to the mean
        mean = policy(state)
        assert torch.allclose(action1, mean)

    def test_log_prob_and_entropy_shape(self):
        """Test log_prob_and_entropy returns correct shapes."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(10, 4)
        action = torch.randn(10, 2)

        log_prob, entropy = policy.log_prob_and_entropy(state, action)
        assert log_prob.shape == (10,)
        assert entropy.shape == (10,)

    def test_log_prob_consistency(self):
        """Test that log_prob from sample matches log_prob_and_entropy."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(5, 4)

        # Get action and log_prob from sample
        torch.manual_seed(42)
        dist = policy.get_distribution(state)
        raw_action = dist.rsample()
        log_prob1 = dist.log_prob(raw_action).sum(dim=-1)

        # Get log_prob from log_prob_and_entropy
        log_prob2, _ = policy.log_prob_and_entropy(state, raw_action)

        assert torch.allclose(log_prob1, log_prob2)

    def test_act_method(self):
        """Test act method returns action without log_prob."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(10, 4)

        action = policy.act(state)
        assert action.shape == (10, 2)
        assert isinstance(action, torch.Tensor)

    def test_get_distribution(self):
        """Test get_distribution returns a Normal distribution."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(10, 4)

        dist = policy.get_distribution(state)
        assert isinstance(dist, torch.distributions.Normal)
        assert dist.mean.shape == (10, 2)
        assert dist.stddev.shape == (10, 2)

    def test_different_hidden_dims(self):
        """Test policy works with different hidden layer configurations."""
        # Single hidden layer
        policy1 = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[128])
        state = torch.randn(5, 4)
        action1, _ = policy1.sample(state)
        assert action1.shape == (5, 2)

        # Three hidden layers
        policy2 = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64, 128, 64])
        action2, _ = policy2.sample(state)
        assert action2.shape == (5, 2)

        # Empty hidden dims (direct connection)
        policy3 = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[])
        action3, _ = policy3.sample(state)
        assert action3.shape == (5, 2)

    def test_batch_size_one(self):
        """Test policy works with batch size 1."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(1, 4)

        action, log_prob = policy.sample(state)
        assert action.shape == (1, 2)
        assert log_prob.shape == (1,)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        policy = GaussianPolicy(input_dim=4, output_dim=2, hidden_dims=[64])
        state = torch.randn(5, 4, requires_grad=True)

        # Forward pass
        mean = policy(state)
        loss = mean.sum()
        loss.backward()

        # Check that state has gradients
        assert state.grad is not None
        assert not torch.all(state.grad == 0)

    def test_device_transfer(self):
        """Test that policy can be moved to different devices."""
        policy = GaussianPolicy(
            input_dim=4,
            output_dim=2,
            hidden_dims=[64],
            action_bounds=(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 2.0])),
        )

        # CPU
        state_cpu = torch.randn(5, 4)
        action_cpu, _ = policy.sample(state_cpu)
        assert action_cpu.device.type == 'cpu'

        # If CUDA is available, test GPU transfer
        if torch.cuda.is_available():
            policy_gpu = policy.cuda()
            state_gpu = state_cpu.cuda()
            action_gpu, _ = policy_gpu.sample(state_gpu)
            assert action_gpu.device.type == 'cuda'

            # Bounds should also be on GPU
            assert policy_gpu.action_low.device.type == 'cuda'
            assert policy_gpu.action_high.device.type == 'cuda'
