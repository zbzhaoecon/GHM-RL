"""
Quick test to verify GPU usage during actual training.

This runs a very short training job and monitors GPU usage.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from macro_rl.dynamics.ghm_equity import GHMEquityTimeAugmentedDynamics, GHMEquityParams
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.networks.policy import GaussianPolicy
from macro_rl.networks.value import ValueNetwork
from macro_rl.simulation.trajectory import TrajectorySimulator
from macro_rl.solvers.monte_carlo import MonteCarloPolicyGradient

print("=" * 80)
print("GPU Training Test")
print("=" * 80)

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# Setup components
print("\nSetting up model...")
params = GHMEquityParams()
dynamics = GHMEquityTimeAugmentedDynamics(params, T=10.0)
control_spec = GHMControlSpec()
reward_fn = GHMRewardFunction(
    discount_rate=params.r - params.mu,
    issuance_cost=params.lambda_,
    liquidation_rate=params.omega,
    liquidation_flow=params.alpha,
)

# Create networks on GPU
policy = GaussianPolicy(
    input_dim=2,  # (c, tau)
    output_dim=2,  # (dividend, equity)
    hidden_dims=[64, 64],
    action_bounds=(control_spec.lower, control_spec.upper),
).to(device)

baseline = ValueNetwork(
    input_dim=2,
    hidden_dims=[64, 64],
).to(device)

print(f"✓ Policy on device: {next(policy.parameters()).device}")
print(f"✓ Baseline on device: {next(baseline.parameters()).device}")

if device.type == 'cuda':
    print(f"GPU memory after model creation: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# Setup simulator
simulator = TrajectorySimulator(
    dynamics=dynamics,
    control_spec=control_spec,
    reward_fn=reward_fn,
    dt=0.01,
    T=10.0,
)

# Create solver with policy adapter
from scripts.train_monte_carlo_ghm_time_augmented import PolicyAdapter
policy_adapted = PolicyAdapter(policy)

solver = MonteCarloPolicyGradient(
    policy=policy_adapted,
    simulator=simulator,
    baseline=baseline,
    n_trajectories=100,  # Small for testing
    lr_policy=3e-4,
    lr_baseline=1e-3,
)

# Store dynamics
solver.dynamics = dynamics

print("\nRunning 3 training steps...")
for step in range(1, 4):
    metrics = solver.train_step()

    print(f"\nStep {step}:")
    print(f"  Return: {metrics['return/mean']:.3f}")
    print(f"  Policy Loss: {metrics['loss/policy']:.4f}")

    if device.type == 'cuda':
        gpu_mem = torch.cuda.memory_allocated(0) / 1024**2
        gpu_cached = torch.cuda.memory_reserved(0) / 1024**2
        print(f"  GPU memory allocated: {gpu_mem:.1f} MB")
        print(f"  GPU memory cached: {gpu_cached:.1f} MB")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)

if device.type == 'cuda':
    print(f"\nFinal GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print("\nIf you see GPU memory increasing above, GPU is being used correctly.")
    print("If GPU memory stays at 0, there's an issue with device placement.")
else:
    print("\nWARNING: Test ran on CPU, not GPU!")
