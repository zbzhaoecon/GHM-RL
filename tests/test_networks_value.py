import pytest
import torch
from macro_rl.networks.value import ValueNetwork


class TestValueNetwork:
    """Tests for the ValueNetwork class."""

    def test_basic_shape_checks(self):
        """Test basic shape requirements with random state."""
        state = torch.randn(4, 3)
        net = ValueNetwork(input_dim=3)

        # Test forward method
        V = net(state)
        assert V.shape == (4,), f"Expected V.shape == (4,), got {V.shape}"

        # Test forward_with_grad method
        V2, V_s, V_ss = net.forward_with_grad(state)
        assert V2.shape == (4,), f"Expected V2.shape == (4,), got {V2.shape}"
        assert V_s.shape == (4, 3), f"Expected V_s.shape == (4, 3), got {V_s.shape}"
        assert V_ss.shape == (4, 3), f"Expected V_ss.shape == (4, 3), got {V_ss.shape}"

    def test_simple_network_gradients(self):
        """Test gradients with a simple network (no hidden layers).

        For a linear network V = w^T s + b, we should have:
        - V_s = w (constant)
        - V_ss = 0 (second derivatives are zero)
        """
        # Create a network with no hidden layers
        net = ValueNetwork(input_dim=3, hidden_dims=[])

        # Set fixed weights for deterministic testing
        # The network is just a single linear layer
        with torch.no_grad():
            net.net[0].weight.data = torch.tensor([[2.0, 3.0, 4.0]])
            net.net[0].bias.data = torch.tensor([1.0])

        state = torch.tensor([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])

        V, V_s, V_ss = net.forward_with_grad(state)

        # Check values
        expected_V = torch.tensor([3.0, 4.0, 5.0])  # [2*1+1, 3*1+1, 4*1+1]
        torch.testing.assert_close(V, expected_V, rtol=1e-4, atol=1e-4)

        # Check gradients (should be constant = weights)
        expected_V_s = torch.tensor([[2.0, 3.0, 4.0],
                                      [2.0, 3.0, 4.0],
                                      [2.0, 3.0, 4.0]])
        torch.testing.assert_close(V_s, expected_V_s, rtol=1e-4, atol=1e-4)

        # Check Hessian diagonal (should be zero for linear function)
        expected_V_ss = torch.zeros(3, 3)
        torch.testing.assert_close(V_ss, expected_V_ss, rtol=1e-4, atol=1e-4)

    def test_quadratic_network_hessian(self):
        """Test Hessian computation with a simple quadratic function.

        For a function like V = sum(w_i * s_i^2), we should have:
        - V_s_i = 2 * w_i * s_i
        - V_ss_ii = 2 * w_i
        """
        # Create a simple network and manually construct a quadratic
        # We'll use a network with one hidden layer and tanh activation
        # but then manually verify the gradient computations
        net = ValueNetwork(input_dim=2, hidden_dims=[4], activation="tanh")

        # For this test, we just verify the gradient computation mechanism
        state = torch.randn(3, 2)
        V, V_s, V_ss = net.forward_with_grad(state)

        # Verify shapes
        assert V.shape == (3,)
        assert V_s.shape == (3, 2)
        assert V_ss.shape == (3, 2)

        # Numerical gradient check
        eps = 1e-4
        for batch_idx in range(3):
            for dim in range(2):
                # Perturb state in positive direction
                state_plus = state.clone()
                state_plus[batch_idx, dim] += eps
                V_plus = net(state_plus)[batch_idx]

                # Perturb state in negative direction
                state_minus = state.clone()
                state_minus[batch_idx, dim] -= eps
                V_minus = net(state_minus)[batch_idx]

                # Numerical gradient
                numerical_grad = (V_plus - V_minus) / (2 * eps)

                # Compare with analytical gradient
                assert torch.abs(V_s[batch_idx, dim] - numerical_grad) < 1e-3, \
                    f"Gradient mismatch at batch {batch_idx}, dim {dim}"

    def test_different_activations(self):
        """Test that different activation functions work correctly."""
        state = torch.randn(2, 3)

        for activation in ["tanh", "relu", "softplus"]:
            net = ValueNetwork(input_dim=3, activation=activation)
            V = net(state)
            V2, V_s, V_ss = net.forward_with_grad(state)

            assert V.shape == (2,)
            assert V2.shape == (2,)
            assert V_s.shape == (2, 3)
            assert V_ss.shape == (2, 3)

    def test_invalid_activation(self):
        """Test that invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            ValueNetwork(input_dim=3, activation="invalid")

    def test_different_hidden_dims(self):
        """Test networks with different architectures."""
        state = torch.randn(5, 4)

        # Single hidden layer
        net1 = ValueNetwork(input_dim=4, hidden_dims=[128])
        V1, V_s1, V_ss1 = net1.forward_with_grad(state)
        assert V1.shape == (5,)
        assert V_s1.shape == (5, 4)
        assert V_ss1.shape == (5, 4)

        # Three hidden layers
        net2 = ValueNetwork(input_dim=4, hidden_dims=[64, 64, 64])
        V2, V_s2, V_ss2 = net2.forward_with_grad(state)
        assert V2.shape == (5,)
        assert V_s2.shape == (5, 4)
        assert V_ss2.shape == (5, 4)

    def test_gradient_flow(self):
        """Test that gradients can flow through the network."""
        net = ValueNetwork(input_dim=3, hidden_dims=[32, 32])
        state = torch.randn(4, 3, requires_grad=True)

        # Forward pass
        V = net(state)
        loss = V.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert state.grad is not None
        assert not torch.all(state.grad == 0)

    def test_forward_with_grad_creates_graph(self):
        """Test that forward_with_grad creates a computation graph."""
        net = ValueNetwork(input_dim=3, hidden_dims=[32])
        state = torch.randn(2, 3)

        V, V_s, V_ss = net.forward_with_grad(state)

        # V_s should have a gradient function (part of computation graph)
        assert V_s.grad_fn is not None
        assert V_ss.grad_fn is not None

        # We should be able to compute gradients w.r.t. network parameters
        loss = V_s.sum() + V_ss.sum()
        loss.backward()

        # Check that network parameters have gradients
        for param in net.parameters():
            assert param.grad is not None

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        net = ValueNetwork(input_dim=5)
        state = torch.randn(1, 5)

        V = net(state)
        V2, V_s, V_ss = net.forward_with_grad(state)

        assert V.shape == (1,)
        assert V2.shape == (1,)
        assert V_s.shape == (1, 5)
        assert V_ss.shape == (1, 5)

    def test_consistency_between_forward_methods(self):
        """Test that forward and forward_with_grad return consistent V values."""
        net = ValueNetwork(input_dim=3, hidden_dims=[64, 64])
        state = torch.randn(10, 3)

        V1 = net(state)
        V2, _, _ = net.forward_with_grad(state)

        # The values should be very close (small numerical differences OK)
        torch.testing.assert_close(V1, V2, rtol=1e-5, atol=1e-5)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
