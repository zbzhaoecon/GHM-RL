"""
Unit tests for control/base.py - ControlSpec class.
"""

import torch
from macro_rl.control.base import ControlSpec


def test_control_spec_initialization():
    """Test ControlSpec initialization and validation."""
    spec = ControlSpec(
        dim=2,
        lower=torch.tensor([0.0, -1.0]),
        upper=torch.tensor([1.0, 1.0]),
        names=("a", "b"),
        is_singular=(False, False),
    )

    assert spec.dim == 2
    assert len(spec.names) == 2
    assert spec.lower.shape == (2,)
    assert spec.upper.shape == (2,)


def test_control_spec_clip():
    """Test action clipping."""
    spec = ControlSpec(
        dim=2,
        lower=torch.tensor([0.0, -1.0]),
        upper=torch.tensor([1.0, 1.0]),
        names=("a", "b"),
        is_singular=(False, False),
    )

    # Test clipping with out-of-bounds actions
    action = torch.tensor([[1.5, -2.0], [0.5, 0.5]])
    clipped = spec.clip(action)

    assert clipped.shape == action.shape
    # First sample should be clipped
    assert torch.allclose(clipped[0, 0], torch.tensor(1.0))
    assert torch.allclose(clipped[0, 1], torch.tensor(-1.0))
    # Second sample should be unchanged
    assert torch.allclose(clipped[1, 0], torch.tensor(0.5))
    assert torch.allclose(clipped[1, 1], torch.tensor(0.5))


def test_control_spec_normalize_denormalize():
    """Test normalization and denormalization round-trip."""
    spec = ControlSpec(
        dim=2,
        lower=torch.tensor([0.0, -1.0]),
        upper=torch.tensor([1.0, 1.0]),
        names=("a", "b"),
        is_singular=(False, False),
    )

    action = torch.tensor([[0.5, 0.0], [1.0, -1.0]])

    # Normalize
    norm = spec.normalize(action)
    assert norm.shape == action.shape

    # Check normalization is correct
    # For dim 0: [0, 1] -> [0, 1], so 0.5 -> 0.5
    assert torch.allclose(norm[0, 0], torch.tensor(0.5))
    # For dim 1: [-1, 1] -> [0, 1], so 0.0 -> 0.5
    assert torch.allclose(norm[0, 1], torch.tensor(0.5))

    # Denormalize
    recon = spec.denormalize(norm)
    assert torch.allclose(action, recon, atol=1e-6)


def test_control_spec_sample_uniform():
    """Test uniform sampling from action space."""
    spec = ControlSpec(
        dim=2,
        lower=torch.tensor([0.0, -1.0]),
        upper=torch.tensor([1.0, 1.0]),
        names=("a", "b"),
        is_singular=(False, False),
    )

    n_samples = 100
    samples = spec.sample_uniform(n_samples)

    assert samples.shape == (n_samples, 2)

    # Check bounds
    assert torch.all(samples[:, 0] >= 0.0)
    assert torch.all(samples[:, 0] <= 1.0)
    assert torch.all(samples[:, 1] >= -1.0)
    assert torch.all(samples[:, 1] <= 1.0)


def test_control_spec_clip_batched():
    """Test clipping with different batch shapes."""
    spec = ControlSpec(
        dim=2,
        lower=torch.tensor([0.0, -1.0]),
        upper=torch.tensor([1.0, 1.0]),
        names=("a", "b"),
        is_singular=(False, False),
    )

    # Test 3D tensor (batch, time, dim)
    action = torch.tensor([
        [[1.5, -2.0], [0.5, 0.5]],
        [[0.2, 0.8], [-0.5, -1.5]]
    ])  # shape: (2, 2, 2)

    clipped = spec.clip(action)
    assert clipped.shape == action.shape

    # Check bounds are enforced
    assert torch.all(clipped[..., 0] >= 0.0)
    assert torch.all(clipped[..., 0] <= 1.0)
    assert torch.all(clipped[..., 1] >= -1.0)
    assert torch.all(clipped[..., 1] <= 1.0)
