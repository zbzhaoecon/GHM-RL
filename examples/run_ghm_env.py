"""
Example script demonstrating GHM equity environment usage.

This script shows how to:
1. Create the environment
2. Run episodes with random/fixed policies
3. Visualize trajectories
4. Test basic functionality

Phase 3 implementation.
"""

import numpy as np
import gymnasium as gym
from macro_rl.envs import GHMEquityEnv
from macro_rl.dynamics import GHMEquityParams


def run_random_policy(n_episodes=5, max_steps=200, seed=42):
    """Run episodes with random policy."""
    print("=" * 60)
    print("Running GHM Equity Environment with Random Policy")
    print("=" * 60)

    env = GHMEquityEnv(seed=seed, max_steps=max_steps)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}:")
        print(f"  Initial cash: c = {obs[0]:.4f}")

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                reason = "liquidated" if terminated else "truncated"
                print(f"  Ended after {steps} steps ({reason})")
                print(f"  Final cash: c = {obs[0]:.4f}")
                print(f"  Total reward: {episode_reward:.4f}")
                break

        if not (terminated or truncated):
            print(f"  Completed {steps} steps")
            print(f"  Final cash: c = {obs[0]:.4f}")
            print(f"  Total reward: {episode_reward:.4f}")


def run_fixed_policy(dividend_rate=1.0, n_episodes=3, max_steps=200, seed=42):
    """Run episodes with fixed dividend policy."""
    print("\n" + "=" * 60)
    print(f"Running GHM Equity Environment with Fixed Policy (a={dividend_rate})")
    print("=" * 60)

    env = GHMEquityEnv(seed=seed, max_steps=max_steps)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        trajectory = [obs[0]]

        print(f"\nEpisode {episode + 1}:")
        print(f"  Initial cash: c = {obs[0]:.4f}")

        for step in range(max_steps):
            action = np.array([dividend_rate])
            obs, reward, terminated, truncated, info = env.step(action)

            trajectory.append(obs[0])
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                reason = "liquidated" if terminated else "truncated"
                print(f"  Ended after {steps} steps ({reason})")
                print(f"  Final cash: c = {obs[0]:.4f}")
                print(f"  Total reward: {episode_reward:.4f}")
                print(f"  Average cash: {np.mean(trajectory):.4f}")
                break


def demonstrate_gym_make():
    """Demonstrate using gym.make() interface."""
    print("\n" + "=" * 60)
    print("Demonstrating gym.make() Interface")
    print("=" * 60)

    # Create environment via registration
    env = gym.make("GHMEquity-v0")

    print(f"Environment: {env.unwrapped.__class__.__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run one episode
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation: {obs}")

    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Episode completed after {step + 1} steps")
    print(f"Total reward: {total_reward:.4f}")

    env.close()


def test_determinism():
    """Test that same seed gives same results."""
    print("\n" + "=" * 60)
    print("Testing Determinism")
    print("=" * 60)

    seed = 123
    action_sequence = [1.0, 0.5, 2.0, 0.1, 1.5]

    # Run 1
    env1 = GHMEquityEnv(seed=seed)
    obs1, _ = env1.reset(seed=seed)
    trajectory1 = [obs1[0]]

    for a in action_sequence:
        obs, _, _, _, _ = env1.step(np.array([a]))
        trajectory1.append(obs[0])

    # Run 2
    env2 = GHMEquityEnv(seed=seed)
    obs2, _ = env2.reset(seed=seed)
    trajectory2 = [obs2[0]]

    for a in action_sequence:
        obs, _, _, _, _ = env2.step(np.array([a]))
        trajectory2.append(obs[0])

    # Check
    trajectories_match = np.allclose(trajectory1, trajectory2, atol=1e-10)
    print(f"Same seed produces identical trajectories: {trajectories_match}")

    if trajectories_match:
        print("✓ Determinism test passed!")
    else:
        print("✗ Determinism test failed!")
        print(f"Trajectory 1: {trajectory1}")
        print(f"Trajectory 2: {trajectory2}")


def show_environment_parameters():
    """Display environment parameters."""
    print("\n" + "=" * 60)
    print("Environment Parameters")
    print("=" * 60)

    # Default parameters
    env = GHMEquityEnv()
    params = env.dynamics.params

    print("\nGHM Model Parameters:")
    for key, value in params.items():
        print(f"  {key:12s} = {value:.4f}")

    print(f"\nEnvironment Settings:")
    print(f"  dt           = {env.dt:.4f}")
    print(f"  max_steps    = {env.max_steps}")
    print(f"  a_max        = {env.a_max:.4f}")
    print(f"  liquidation  = {env.liquidation_penalty:.4f}")

    print(f"\nDiscount Factor (for SAC/PPO):")
    gamma = env.get_expected_discount_factor()
    print(f"  γ = exp(-ρ*dt) = {gamma:.6f}")
    print(f"  Recommended: γ ≈ 0.99 - 0.999")


if __name__ == "__main__":
    # Show parameters
    show_environment_parameters()

    # Test determinism
    test_determinism()

    # Demonstrate gym.make()
    demonstrate_gym_make()

    # Run with random policy
    run_random_policy(n_episodes=3, max_steps=100)

    # Run with fixed policies
    run_fixed_policy(dividend_rate=0.5, n_episodes=2, max_steps=150)
    run_fixed_policy(dividend_rate=5.0, n_episodes=2, max_steps=150)

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
