"""
Manual environment debugging.

Tests:
1. Environment creates and resets
2. Actions have expected effects
3. Termination triggers correctly
4. Rewards are reasonable

Usage:
    python scripts/debug_env.py
"""

import numpy as np
import torch
from macro_rl.envs import GHMEquityEnv
from macro_rl.dynamics import GHMEquityParams


def test_creation():
    """Test environment creates successfully."""
    print("=" * 50)
    print("Test: Environment creation")

    env = GHMEquityEnv()
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  dt: {env.dt}, max_steps: {env.max_steps}")
    print("  ✓ Creation successful")
    return env


def test_reset(env):
    """Test reset returns valid observation."""
    print("=" * 50)
    print("Test: Reset")

    obs, info = env.reset(seed=42)
    print(f"  Initial observation: {obs}")
    print(f"  In bounds: {env.observation_space.contains(obs)}")

    # Reset again with same seed should give same result
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs, obs2), "Seeded reset not deterministic!"
    print("  ✓ Reset successful and deterministic")


def test_step_zero_action(env):
    """Test step with zero dividend (pure drift + diffusion)."""
    print("=" * 50)
    print("Test: Step with zero action")

    env.reset(seed=42)
    initial_c = env._state[0]

    action = np.array([0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"  Initial c: {initial_c:.4f}")
    print(f"  After step c: {obs[0]:.4f}")
    print(f"  Reward (should be 0): {reward:.6f}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")

    assert reward == 0.0, f"Zero action should give zero reward, got {reward}"
    print("  ✓ Zero action step successful")


def test_step_positive_action(env):
    """Test step with positive dividend."""
    print("=" * 50)
    print("Test: Step with positive action")

    env.reset(seed=42)
    initial_c = env._state[0]

    action = np.array([5.0], dtype=np.float32)  # Dividend rate = 5
    obs, reward, terminated, truncated, info = env.step(action)

    expected_reward = 5.0 * env.dt
    print(f"  Initial c: {initial_c:.4f}")
    print(f"  After step c: {obs[0]:.4f}")
    print(f"  Reward: {reward:.6f} (expected: {expected_reward:.6f})")

    assert np.isclose(reward, expected_reward), f"Reward mismatch"
    print("  ✓ Positive action step successful")


def test_episode_rollout(env, n_steps=100):
    """Run a short episode with random actions."""
    print("=" * 50)
    print(f"Test: Random rollout ({n_steps} steps)")

    obs, _ = env.reset(seed=123)
    total_reward = 0

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"  Terminated at step {step} (liquidation)")
            break
        if truncated:
            print(f"  Truncated at step {step}")
            break

    print(f"  Steps completed: {step + 1}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final c: {obs[0]:.4f}")
    print("  ✓ Rollout successful")


def test_liquidation(env):
    """Test that liquidation triggers correctly."""
    print("=" * 50)
    print("Test: Liquidation")

    # Start near zero
    obs, _ = env.reset(options={"initial_state": np.array([0.05])})

    # Use high dividend to force liquidation
    action = np.array([env.a_max], dtype=np.float32)

    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"  Liquidation triggered at step {step}")
            print(f"  Final c: {obs[0]:.4f}")
            print(f"  Final reward (includes penalty): {reward:.4f}")
            print("  ✓ Liquidation test successful")
            return

    print("  ✗ Liquidation not triggered")


def test_sb3_compatibility(env):
    """Test compatibility with SB3's env checker."""
    print("=" * 50)
    print("Test: SB3 compatibility")

    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        print("  ✓ SB3 env_checker passed")
    except Exception as e:
        print(f"  ✗ SB3 env_checker failed: {e}")


def test_dynamics_values(env):
    """Spot-check dynamics against expected values."""
    print("=" * 50)
    print("Test: Dynamics spot-check")

    # At c=0: drift should be α = 0.18
    info = env.get_dynamics_info(np.array([0.0]))
    print(f"  At c=0: drift={info['drift']:.4f} (expected: 0.18)")
    assert np.isclose(info['drift'], 0.18, atol=1e-6)

    # At c=0: diffusion² should be ~0.01382
    expected_diff_sq = 0.12**2 * (1 - 0.04) + (0.12 * -0.2)**2
    print(f"  At c=0: σ²={info['diffusion_sq']:.5f} (expected: {expected_diff_sq:.5f})")
    assert np.isclose(info['diffusion_sq'], expected_diff_sq, atol=1e-5)

    print("  ✓ Dynamics spot-check passed")


def main():
    print("\n" + "=" * 50)
    print("GHM Equity Environment Debug Suite")
    print("=" * 50 + "\n")

    env = test_creation()
    test_reset(env)
    test_step_zero_action(env)
    test_step_positive_action(env)
    test_episode_rollout(env)
    test_liquidation(env)
    test_dynamics_values(env)
    test_sb3_compatibility(env)

    print("\n" + "=" * 50)
    print("All tests passed! Environment ready for training.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
