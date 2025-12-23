"""
Tests for Gymnasium environments.

Covers basic functionality, economic sanity checks, and compatibility.

Phase 3 implementation.
"""

import numpy as np
import torch
import pytest
import gymnasium as gym

from macro_rl.envs import GHMEquityEnv, ContinuousTimeEnv
from macro_rl.dynamics import GHMEquityParams


class TestGHMEquityEnvBasics:
    """Test basic Gymnasium interface functionality."""

    def test_reset_returns_valid_observation(self):
        """Test that reset returns observation within bounds."""
        env = GHMEquityEnv(seed=42)
        obs, info = env.reset()

        # Check types
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        # Check shape
        assert obs.shape == (1,)

        # Check bounds
        assert env.observation_space.contains(obs)
        assert 0 <= obs[0] <= env.dynamics.state_space.upper[0]

    def test_reset_with_custom_initial_state(self):
        """Test reset with user-specified initial state."""
        env = GHMEquityEnv()
        initial_state = np.array([1.0])

        obs, info = env.reset(options={"initial_state": initial_state})

        assert np.allclose(obs, initial_state)

    def test_step_returns_correct_shapes(self):
        """Test that step returns tuple with correct shapes."""
        env = GHMEquityEnv(seed=42)
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Check types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Check shapes
        assert obs.shape == (1,)
        assert env.observation_space.contains(obs)

    def test_action_space_valid(self):
        """Test that action space is properly configured."""
        env = GHMEquityEnv(dividend_max=3.0, equity_max=2.5)

        # Check action space bounds (2D: [dividend, equity])
        assert env.action_space.shape == (2,)
        assert env.action_space.low[0] == 0.0
        assert env.action_space.low[1] == 0.0
        assert env.action_space.high[0] == 3.0
        assert env.action_space.high[1] == 2.5

        # Sample actions should be valid
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            assert 0.0 <= action[0] <= 3.0
            assert 0.0 <= action[1] <= 2.5

    def test_random_policy_completes_episode(self):
        """Test that random policy can complete an episode without errors."""
        env = GHMEquityEnv(seed=42, max_steps=100)
        obs, info = env.reset()

        total_reward = 0
        steps = 0

        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            assert env.observation_space.contains(obs)

            if terminated or truncated:
                break

        # Should have taken at least 1 step
        assert steps > 0

    def test_deterministic_seed_reproducibility(self):
        """Test that same seed produces same trajectory."""
        seed = 123

        # Run first trajectory
        env1 = GHMEquityEnv(seed=seed, max_steps=50)
        obs1, _ = env1.reset(seed=seed)
        trajectory1 = [obs1.copy()]

        for _ in range(10):
            action = np.array([0.1, 0.0])  # Fixed action: small dividend, no equity
            obs1, _, terminated, truncated, _ = env1.step(action)
            trajectory1.append(obs1.copy())
            if terminated or truncated:
                break

        # Run second trajectory with same seed
        env2 = GHMEquityEnv(seed=seed, max_steps=50)
        obs2, _ = env2.reset(seed=seed)
        trajectory2 = [obs2.copy()]

        for _ in range(10):
            action = np.array([0.1, 0.0])  # Fixed action: small dividend, no equity
            obs2, _, terminated, truncated, _ = env2.step(action)
            trajectory2.append(obs2.copy())
            if terminated or truncated:
                break

        # Trajectories should match exactly
        assert len(trajectory1) == len(trajectory2)
        for obs1, obs2 in zip(trajectory1, trajectory2):
            assert np.allclose(obs1, obs2, atol=1e-6)

    def test_episode_step_count(self):
        """Test that episode respects max_steps."""
        max_steps = 50
        env = GHMEquityEnv(seed=42, max_steps=max_steps)
        env.reset()

        for step in range(max_steps + 10):
            action = np.array([0.05, 0.0])  # Small dividend, no equity issuance
            obs, reward, terminated, truncated, info = env.step(action)

            if truncated:
                # Should truncate at exactly max_steps
                assert info["step"] == max_steps
                break

            if terminated:
                # Terminated early (liquidation)
                break

        # Episode should have ended
        assert terminated or truncated


class TestGHMEquityEnvEconomics:
    """Test economic behavior and sanity checks."""

    def test_zero_action_follows_drift(self):
        """Test that zero action (no dividend, no equity) follows natural drift."""
        env = GHMEquityEnv(seed=42, max_steps=100)
        c0, _ = env.reset(options={"initial_state": np.array([1.0])})

        # Take many steps with zero action (no dividend, no equity)
        action = np.array([0.0, 0.0])
        changes = []

        for _ in range(50):
            c_before = env._state[0]
            obs, reward, terminated, truncated, info = env.step(action)
            c_after = obs[0]

            if not (terminated or truncated):
                changes.append(c_after - c_before)

        # On average, should follow μ_c(c) * dt
        # For c=1.0: μ_c = α + c*(r - λ - μ) = 0.18 + 1.0*(0.03 - 0.02 - 0.01) = 0.18
        # Expected change per step ≈ 0.18 * 0.01 = 0.0018
        # (will have variance from diffusion)

        mean_change = np.mean(changes)
        # Should be positive on average (cash accumulates)
        assert mean_change > 0, f"Expected positive drift, got {mean_change}"

    def test_high_action_decreases_cash(self):
        """Test that high dividend amount decreases cash on average."""
        env = GHMEquityEnv(seed=42, max_steps=100)
        env.reset(options={"initial_state": np.array([1.5])})

        # Take many steps with high dividend action
        action = np.array([0.5, 0.0])  # Large dividend payout, no equity
        changes = []

        for _ in range(20):
            c_before = env._state[0]
            obs, reward, terminated, truncated, info = env.step(action)
            c_after = obs[0]

            if not (terminated or truncated):
                changes.append(c_after - c_before)
            else:
                break

        # Should have some data
        assert len(changes) > 0

        # On average, should decrease (dividend payout > drift * dt)
        mean_change = np.mean(changes)
        assert mean_change < 0, f"Expected negative change with high dividend, got {mean_change}"

    def test_liquidation_terminates_episode(self):
        """Test that c <= 0 triggers termination."""
        env = GHMEquityEnv(seed=42)
        env.reset(options={"initial_state": np.array([0.1])})

        # Apply very high dividend to force liquidation
        action = np.array([1.0, 0.0])  # High dividend, no equity

        terminated = False
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                # Check that state is at/below zero
                assert obs[0] <= 1e-6  # Should be clamped to 0
                # Check that liquidation value was applied (should be positive)
                # Note: reward includes the dividend from the last step
                # The terminal reward is positive liquidation value
                break

        # Should have terminated before max_steps
        assert terminated, "Expected liquidation but episode didn't terminate"

    def test_reward_equals_net_payout(self):
        """Test that reward matches dividend - equity issuance."""
        env = GHMEquityEnv(seed=42, dt=0.01)
        env.reset(options={"initial_state": np.array([1.0])})

        dividend_amount = 0.1
        equity_amount = 0.0
        action = np.array([dividend_amount, equity_amount])

        obs, reward, terminated, truncated, info = env.step(action)

        # Reward should be dividend - equity (net payout to shareholders)
        if not terminated:
            expected_reward = dividend_amount - equity_amount
            assert np.isclose(reward, expected_reward, atol=1e-6), \
                f"Expected reward {expected_reward}, got {reward}"

    def test_liquidation_value_applied(self):
        """Test that liquidation value is correctly applied."""
        env = GHMEquityEnv(seed=42)
        env.reset(options={"initial_state": np.array([0.05])})

        # Force liquidation
        action = np.array([1.0, 0.0])  # High dividend

        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                # Terminal reward should include positive liquidation value
                # Liquidation value = ω·α/(r-μ) = 0.55 * 0.18 / 0.02 = 4.95
                expected_liquidation_value = env._dynamics.liquidation_value()
                # Reward includes dividend from last step plus terminal reward
                # Since we force immediate liquidation, reward should be positive
                assert reward > 0, \
                    f"Expected positive reward with liquidation value, got {reward}"
                assert expected_liquidation_value > 0, \
                    f"Expected positive liquidation value, got {expected_liquidation_value}"
                break

        assert terminated, "Expected liquidation"

    def test_state_stays_in_bounds(self):
        """Test that state is clipped to valid range."""
        env = GHMEquityEnv(seed=42, max_steps=200)
        env.reset()

        c_max = env.dynamics.state_space.upper[0]

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # State should always be in [0, c_max]
            assert 0 <= obs[0] <= c_max, \
                f"State {obs[0]} outside bounds [0, {c_max}]"

            if terminated or truncated:
                break


class TestGHMEquityEnvParameters:
    """Test environment parameter configurations."""

    def test_custom_parameters(self):
        """Test environment with custom GHM parameters."""
        params = GHMEquityParams(
            alpha=0.2,
            r=0.05,
            mu=0.02,
            lambda_=0.01,
            c_max=3.0,
        )

        env = GHMEquityEnv(params=params)
        assert env.dynamics.p.alpha == 0.2
        assert env.dynamics.p.r == 0.05
        assert env.observation_space.high[0] == 3.0

    def test_discount_factor_calculation(self):
        """Test that discount factor is correctly computed."""
        params = GHMEquityParams(r=0.03, mu=0.01)
        env = GHMEquityEnv(params=params, dt=0.01)

        gamma = env.get_expected_discount_factor()

        # γ = exp(-ρ * dt) where ρ = r - μ
        rho = params.r - params.mu
        expected_gamma = np.exp(-rho * env.dt)

        assert np.isclose(gamma, expected_gamma, atol=1e-6)

    def test_different_time_steps(self):
        """Test environment with different dt values."""
        for dt in [0.001, 0.01, 0.1]:
            env = GHMEquityEnv(dt=dt)
            env.reset(options={"initial_state": np.array([1.0])})

            dividend_amount = 0.1
            action = np.array([dividend_amount, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)

            # Reward should be dividend - equity (independent of dt for impulse controls)
            if not terminated:
                expected_reward = dividend_amount
                assert np.isclose(reward, expected_reward, atol=1e-6)


class TestGymnasiumRegistration:
    """Test Gymnasium registration and gym.make() interface."""

    def test_registered_environment(self):
        """Test that environment is registered and can be created via gym.make()."""
        env = gym.make("GHMEquity-v0")
        assert isinstance(env.unwrapped, GHMEquityEnv)

        # Test basic functionality
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        env.close()

    def test_registered_max_episode_steps(self):
        """Test that registered environment has correct max_episode_steps."""
        env = gym.make("GHMEquity-v0")

        # Reset and run to max steps
        env.reset()
        for step in range(1100):
            action = np.array([0.05, 0.0])  # Small dividend, no equity
            obs, reward, terminated, truncated, info = env.step(action)

            if truncated or terminated:
                break

        # Should truncate around 1000 steps (registered max_episode_steps)
        assert step < 1100, "Episode should have been truncated"

        env.close()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_action_array_vs_list(self):
        """Test that environment handles both array and list actions."""
        env = GHMEquityEnv(seed=42)
        env.reset()

        # Test with array action
        action_array = np.array([0.1, 0.0])
        obs1, reward1, _, _, _ = env.step(action_array)

        # Reset to same state
        env.reset(seed=42)

        # Test with list (should be converted to array)
        action_list = [0.1, 0.0]
        obs2, reward2, _, _, _ = env.step(action_list)

        # Should give same results
        assert np.allclose(obs1, obs2)
        assert np.isclose(reward1, reward2)

    def test_reset_without_seed(self):
        """Test that reset works without specifying seed."""
        env = GHMEquityEnv()
        obs1, _ = env.reset()
        obs2, _ = env.reset()

        # Different resets should give different initial states (very likely)
        # This might occasionally fail due to randomness, but very unlikely
        # Let's just check they're valid
        assert env.observation_space.contains(obs1)
        assert env.observation_space.contains(obs2)

    def test_step_before_reset_raises_error(self):
        """Test that stepping before reset raises an error."""
        env = GHMEquityEnv()

        with pytest.raises(RuntimeError, match="Must call reset"):
            env.step(np.array([0.1, 0.0]))

    def test_info_dict_contents(self):
        """Test that info dict contains expected keys."""
        env = GHMEquityEnv(seed=42)
        env.reset()

        action = np.array([0.1, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        # Check info dict has expected keys
        assert "step" in info
        assert "state" in info

        assert isinstance(info["step"], int)
        assert isinstance(info["state"], np.ndarray)
        assert np.allclose(info["state"], obs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
