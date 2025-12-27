#!/usr/bin/env python3
"""
Quick diagnostic to compare AC and MC reward function parameters.

Usage:
    python scripts/compare_solvers.py

This script simulates how both solvers set up their reward functions
and prints the key parameters to identify any discrepancies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# ============================================================================
# 1. Check what AC uses (via train_with_config.py with YAML config)
# ============================================================================
print("=" * 80)
print("ACTOR-CRITIC SETUP (via train_with_config.py)")
print("=" * 80)

from macro_rl.config import load_config
from macro_rl.config.setup_utils import setup_from_config

config_path = "configs/actor_critic_time_augmented_config.yaml"
print(f"Loading config from: {config_path}")

config_manager = load_config(config_path)
config = config_manager.config

# Get dynamics params from config
print(f"\nConfig dynamics parameters:")
print(f"  alpha = {config.dynamics.alpha}")
print(f"  omega = {config.dynamics.omega}")
print(f"  r = {config.dynamics.r}")
print(f"  mu = {config.dynamics.mu}")

print(f"\nConfig reward parameters:")
print(f"  discount_rate = {config.reward.discount_rate} (null means use r-mu)")
print(f"  liquidation_rate = {config.reward.liquidation_rate}")
print(f"  liquidation_flow = {config.reward.liquidation_flow}")
print(f"  issuance_cost = {config.reward.issuance_cost}")
print(f"  fixed_cost = {getattr(config.reward, 'fixed_cost', 0.0)}")

print(f"\nConfig training parameters:")
print(f"  use_time_augmented = {config.training.use_time_augmented}")
print(f"  use_sparse_rewards = {config.training.use_sparse_rewards}")

# Setup actual components
dynamics, control_spec, reward_fn, policy, baseline, simulator, device = setup_from_config(config_manager)

print(f"\nActual reward function (GHMRewardFunction) parameters:")
print(f"  discount_rate = {reward_fn.discount_rate_value}")
print(f"  issuance_cost = {reward_fn.issuance_cost}")
print(f"  liquidation_value = {reward_fn.liquidation_value}")
print(f"  fixed_cost = {reward_fn.fixed_cost}")

print(f"\nSimulator sparse rewards: {simulator.use_sparse_rewards}")
print(f"Dynamics type: {type(dynamics).__name__}")
print(f"State dimension: {dynamics.state_space.dim}")

# ============================================================================
# 2. Check what MC standalone script uses
# ============================================================================
print("\n" + "=" * 80)
print("MONTE CARLO SETUP (via train_monte_carlo_ghm_time_augmented.py)")
print("=" * 80)

from macro_rl.dynamics.ghm_equity import GHMEquityTimeAugmentedDynamics, GHMEquityParams
from macro_rl.rewards.ghm_rewards import GHMRewardFunction as GHMRewardFunctionMC
from macro_rl.simulation.trajectory import TrajectorySimulator

# MC script uses GHMEquityParams() defaults
params = GHMEquityParams()
print(f"\nGHMEquityParams defaults:")
print(f"  alpha = {params.alpha}")
print(f"  omega = {params.omega}")
print(f"  r = {params.r}")
print(f"  mu = {params.mu}")
print(f"  lambda_ = {params.lambda_}")

# MC script config.liquidation_flow defaults to None, so uses params.alpha
liquidation_flow_mc = params.alpha  # This is the default behavior

print(f"\nMC reward function setup (from script lines 801-806):")
print(f"  discount_rate = params.r - params.mu = {params.r - params.mu}")
print(f"  issuance_cost = params.lambda_ = {params.lambda_} (but ignored, hardcoded to 1.0)")
print(f"  liquidation_rate = params.omega = {params.omega}")
print(f"  liquidation_flow = {liquidation_flow_mc} (config.liquidation_flow or params.alpha)")

reward_fn_mc = GHMRewardFunctionMC(
    discount_rate=params.r - params.mu,
    issuance_cost=params.lambda_,
    liquidation_rate=params.omega,
    liquidation_flow=liquidation_flow_mc,
)

print(f"\nActual MC reward function parameters:")
print(f"  discount_rate = {reward_fn_mc.discount_rate_value}")
print(f"  issuance_cost = {reward_fn_mc.issuance_cost}")
print(f"  liquidation_value = {reward_fn_mc.liquidation_value}")
print(f"  fixed_cost = {reward_fn_mc.fixed_cost}")

# MC simulator (doesn't pass use_sparse_rewards, so defaults to False)
print(f"\nMC simulator sparse rewards: False (default, not passed)")

# ============================================================================
# 3. Compare the two
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

ac_lv = reward_fn.liquidation_value
mc_lv = reward_fn_mc.liquidation_value

print(f"\nLiquidation Value:")
print(f"  AC: {ac_lv:.4f}")
print(f"  MC: {mc_lv:.4f}")
print(f"  Difference: {abs(ac_lv - mc_lv):.6f}")

print(f"\nDiscount Rate:")
print(f"  AC: {reward_fn.discount_rate_value:.4f}")
print(f"  MC: {reward_fn_mc.discount_rate_value:.4f}")

print(f"\nIssuance Cost (internal, should be 1.0 for both):")
print(f"  AC: {reward_fn.issuance_cost:.4f}")
print(f"  MC: {reward_fn_mc.issuance_cost:.4f}")

print(f"\nFixed Cost:")
print(f"  AC: {reward_fn.fixed_cost:.6f}")
print(f"  MC: {reward_fn_mc.fixed_cost:.6f}")

print(f"\nSparse Rewards:")
print(f"  AC: {simulator.use_sparse_rewards}")
print(f"  MC: False (default)")

# ============================================================================
# 4. Test terminal reward computation
# ============================================================================
print("\n" + "=" * 80)
print("TERMINAL REWARD TEST")
print("=" * 80)

# Create dummy states
test_states = torch.tensor([[1.0, 5.0], [0.5, 3.0], [0.0, 0.0]])  # (c, τ)
test_terminated = torch.tensor([0.0, 0.0, 1.0])  # Only last one terminated

print(f"\nTest states (c, τ):")
for i, (c, tau) in enumerate(test_states):
    print(f"  State {i}: c={c.item():.2f}, τ={tau.item():.2f}, terminated={test_terminated[i].item():.0f}")

ac_terminal = reward_fn.terminal_reward(test_states, test_terminated)
mc_terminal = reward_fn_mc.terminal_reward(test_states, test_terminated)

print(f"\nTerminal rewards:")
print(f"  AC: {ac_terminal.tolist()}")
print(f"  MC: {mc_terminal.tolist()}")

# ============================================================================
# 5. Test a simple trajectory return computation
# ============================================================================
print("\n" + "=" * 80)
print("TRAJECTORY RETURN SIMULATION")
print("=" * 80)

# Simulate a simple trajectory with zero actions (no dividends, no equity)
n_steps = 100
dt = 0.1
batch_size = 10

# Create dummy trajectory data
rewards = torch.zeros(batch_size, n_steps)  # Zero step rewards
masks = torch.ones(batch_size, n_steps)  # All steps active
terminal_rewards_ac = torch.full((batch_size,), reward_fn.liquidation_value)
terminal_rewards_mc = torch.full((batch_size,), reward_fn_mc.liquidation_value)

# Compute returns like _compute_returns does
discount_rate = reward_fn.discount_rate_value
returns_ac = torch.zeros(batch_size)
returns_mc = torch.zeros(batch_size)

for t in range(n_steps):
    discount = torch.exp(torch.tensor(-discount_rate * t * dt))
    returns_ac = returns_ac + discount * rewards[:, t] * masks[:, t]
    returns_mc = returns_mc + discount * rewards[:, t] * masks[:, t]

# Add terminal reward
termination_times = masks.sum(dim=1)
terminal_discount = torch.exp(-discount_rate * termination_times * dt)
returns_ac = returns_ac + terminal_discount * terminal_rewards_ac
returns_mc = returns_mc + terminal_discount * terminal_rewards_mc

print(f"\nWith zero step rewards (100 steps, dt=0.1, all active):")
print(f"  Termination time: {termination_times[0].item()}")
print(f"  Terminal discount: {terminal_discount[0].item():.4f}")
print(f"  AC terminal contribution: {(terminal_discount[0] * terminal_rewards_ac[0]).item():.4f}")
print(f"  MC terminal contribution: {(terminal_discount[0] * terminal_rewards_mc[0]).item():.4f}")
print(f"  AC total return: {returns_ac[0].item():.4f}")
print(f"  MC total return: {returns_mc[0].item():.4f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if abs(ac_lv - mc_lv) > 0.01:
    print(f"\n⚠️  LIQUIDATION VALUE MISMATCH!")
    print(f"   AC: {ac_lv:.4f}, MC: {mc_lv:.4f}")
    print(f"   This explains the ~{abs(ac_lv - mc_lv):.2f} difference in returns!")
else:
    print(f"\n✓ Liquidation values match: {ac_lv:.4f}")
    print("\nIf returns still differ, the issue is likely in:")
    print("  1. The learned policies (one is better than the other)")
    print("  2. Sparse vs dense reward computation (unlikely to cause ~4.5 diff)")
    print("  3. Different initial state distributions")
