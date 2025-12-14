"""
Diagnostic script to identify sources of validation failures.

This script helps determine whether validation issues are due to:
1. The learned policy itself (fundamental problem)
2. Noisy value estimation method
3. Inappropriate numerical derivative parameters
4. Wrong training hyperparameters (especially gamma)

Usage:
    python scripts/diagnose_validation.py --model path/to/model
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import SAC

from macro_rl.envs import GHMEquityEnv


def compare_value_estimation_methods(model: SAC, c_grid: np.ndarray, env: GHMEquityEnv):
    """
    Compare different value estimation methods to identify noise sources.

    Returns dict with:
    - critic_q1, critic_q2: Values from two Q-networks
    - rollout_mean, rollout_std: Monte Carlo estimates
    - noise_level: Quantification of estimation noise
    """
    n_rollouts = 20
    n_grid = len(c_grid)

    # Method 1: Critic network
    print("  Estimating from critic network...")
    q1_vals = np.zeros(n_grid)
    q2_vals = np.zeros(n_grid)

    with torch.no_grad():
        for i, c in enumerate(c_grid):
            obs = torch.tensor([[c]], dtype=torch.float32)
            action = model.actor(obs)
            q1 = model.critic(obs, action)[0].item()
            q2 = model.critic(obs, action)[1].item()
            q1_vals[i] = q1
            q2_vals[i] = q2

    critic_mean = (q1_vals + q2_vals) / 2
    critic_std = np.abs(q1_vals - q2_vals) / 2

    # Method 2: Rollouts (sample a few points)
    print("  Estimating from rollouts (20 episodes at 20 points)...")
    sample_indices = np.linspace(0, n_grid-1, 20, dtype=int)
    rollout_estimates = np.zeros((len(sample_indices), n_rollouts))

    gamma = env.get_expected_discount_factor()

    for i, idx in enumerate(sample_indices):
        c = c_grid[idx]
        for ep in range(n_rollouts):
            obs = np.array([c], dtype=np.float32)
            env._state = obs
            env._step_count = 0

            total_return = 0.0
            discount = 1.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_return += discount * reward
                discount *= gamma

            rollout_estimates[i, ep] = total_return

    rollout_mean_samples = rollout_estimates.mean(axis=1)
    rollout_std_samples = rollout_estimates.std(axis=1)

    # Noise analysis
    print("\n  Noise Analysis:")
    print(f"    Critic disagreement (mean): {critic_std.mean():.4f}")
    print(f"    Critic disagreement (max):  {critic_std.max():.4f}")
    print(f"    Rollout std (mean): {rollout_std_samples.mean():.4f}")
    print(f"    Rollout std (max):  {rollout_std_samples.max():.4f}")

    # Compare critic vs rollout at sample points
    critic_at_samples = critic_mean[sample_indices]
    diff = np.abs(critic_at_samples - rollout_mean_samples)
    print(f"    Critic vs Rollout difference: {diff.mean():.4f} ± {diff.std():.4f}")

    return {
        'c_grid': c_grid,
        'critic_q1': q1_vals,
        'critic_q2': q2_vals,
        'critic_mean': critic_mean,
        'critic_std': critic_std,
        'sample_indices': sample_indices,
        'rollout_mean': rollout_mean_samples,
        'rollout_std': rollout_std_samples,
    }


def check_training_config(model: SAC, env: GHMEquityEnv):
    """
    Check if model was trained with correct hyperparameters.
    """
    print("\n2. TRAINING CONFIGURATION CHECK")
    print("-" * 50)

    # Check gamma
    model_gamma = model.gamma
    expected_gamma = env.get_expected_discount_factor()
    gamma_error = abs(model_gamma - expected_gamma)

    print(f"  Discount factor (γ):")
    print(f"    Model's gamma:    {model_gamma:.6f}")
    print(f"    Expected gamma:   {expected_gamma:.6f}")
    print(f"    Error:            {gamma_error:.6f}")

    if gamma_error < 0.0001:
        print(f"    Status: ✓ CORRECT")
    elif gamma_error < 0.01:
        print(f"    Status: ⚠ MINOR MISMATCH")
    else:
        print(f"    Status: ✗ WRONG (This is likely causing validation failures!)")
        print(f"    Impact: Agent optimized wrong objective function")

    # Check other relevant params
    print(f"\n  Other hyperparameters:")
    print(f"    Learning rate:    {model.learning_rate}")
    print(f"    Buffer size:      {model.buffer_size}")
    print(f"    Batch size:       {model.batch_size}")
    print(f"    Tau (soft update): {model.tau}")

    return {
        'model_gamma': model_gamma,
        'expected_gamma': expected_gamma,
        'gamma_error': gamma_error,
        'gamma_ok': gamma_error < 0.01,
    }


def analyze_policy_quality(model: SAC, c_grid: np.ndarray):
    """
    Analyze the learned policy to identify issues.
    """
    print("\n3. POLICY QUALITY ANALYSIS")
    print("-" * 50)

    policy = np.zeros_like(c_grid)
    for i, c in enumerate(c_grid):
        obs = np.array([c], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        policy[i] = action[0]

    # Compute statistics
    policy_mean = policy.mean()
    policy_std = policy.std()
    policy_min = policy.min()
    policy_max = policy.max()

    print(f"  Policy statistics:")
    print(f"    Mean:   {policy_mean:.4f}")
    print(f"    Std:    {policy_std:.4f}")
    print(f"    Range:  [{policy_min:.4f}, {policy_max:.4f}]")

    # Check for threshold behavior
    policy_grad = np.gradient(policy)
    max_grad_idx = np.argmax(policy_grad)
    max_grad = policy_grad[max_grad_idx]
    c_at_max_grad = c_grid[max_grad_idx]

    print(f"\n  Threshold detection:")
    print(f"    Max policy gradient: {max_grad:.4f} at c = {c_at_max_grad:.4f}")

    # Check if policy is roughly constant then jumps
    low_action_mask = policy < np.median(policy)
    high_action_mask = policy >= np.median(policy)

    if low_action_mask.sum() > 0 and high_action_mask.sum() > 0:
        mean_low = policy[low_action_mask].mean()
        mean_high = policy[high_action_mask].mean()
        ratio = mean_high / mean_low if mean_low > 0.01 else np.inf

        print(f"    Mean action (low half):  {mean_low:.4f}")
        print(f"    Mean action (high half): {mean_high:.4f}")
        print(f"    Ratio: {ratio:.1f}x")

        if ratio > 8:
            print(f"    Status: ✓ STRONG threshold behavior")
        elif ratio > 3:
            print(f"    Status: ✓ MODERATE threshold behavior")
        else:
            print(f"    Status: ✗ WEAK threshold behavior")
    else:
        print(f"    Status: ⚠ Cannot assess threshold behavior")

    return {
        'policy': policy,
        'policy_grad': policy_grad,
        'max_grad_c': c_at_max_grad,
    }


def test_smoothing_parameters(c_grid, V_raw):
    """
    Test different smoothing parameters to find optimal settings.
    """
    print("\n4. SMOOTHING PARAMETER SENSITIVITY")
    print("-" * 50)

    from scipy.ndimage import gaussian_filter1d

    sigmas = [1.0, 2.0, 3.0, 5.0, 7.0]
    print(f"  Testing sigma values: {sigmas}")

    dc = c_grid[1] - c_grid[0]

    print(f"\n  Results (first derivative smoothness):")
    print(f"  {'Sigma':<8} {'Std(F\\')':<12} {'Range(F\\')':<15} {'Oscillations':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*15} {'-'*12}")

    best_sigma = 2.0
    min_oscillation = np.inf

    for sigma in sigmas:
        V_smooth = gaussian_filter1d(V_raw, sigma=sigma, mode='nearest')
        V_prime = np.gradient(V_smooth, dc)

        std_deriv = V_prime.std()
        range_deriv = V_prime.max() - V_prime.min()

        # Count oscillations (sign changes in second derivative of first derivative)
        V_prime_grad = np.gradient(V_prime)
        oscillations = np.sum(np.diff(np.sign(V_prime_grad)) != 0)

        print(f"  {sigma:<8.1f} {std_deriv:<12.4f} {range_deriv:<15.4f} {oscillations:<12d}")

        if oscillations < min_oscillation:
            min_oscillation = oscillations
            best_sigma = sigma

    print(f"\n  Recommendation: Use sigma = {best_sigma:.1f} for this dataset")

    return {'best_sigma': best_sigma}


def create_diagnostic_plots(
    results_dict: dict,
    config_dict: dict,
    policy_dict: dict,
    output_path: Path
):
    """Create diagnostic visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    c_grid = results_dict['c_grid']

    # 1. Value estimation comparison
    ax = axes[0, 0]
    ax.plot(c_grid, results_dict['critic_mean'], 'b-', linewidth=2, label='Critic (mean of Q1, Q2)')
    ax.fill_between(c_grid,
                     results_dict['critic_mean'] - results_dict['critic_std'],
                     results_dict['critic_mean'] + results_dict['critic_std'],
                     alpha=0.3, color='b', label='Critic disagreement')

    # Add rollout samples
    sample_idx = results_dict['sample_indices']
    ax.errorbar(c_grid[sample_idx], results_dict['rollout_mean'],
                yerr=results_dict['rollout_std'],
                fmt='ro', capsize=5, alpha=0.7, label='Rollout estimates (20 eps)')

    ax.set_xlabel('Cash Ratio (c)', fontsize=11)
    ax.set_ylabel('Value Estimate', fontsize=11)
    ax.set_title('Value Estimation: Critic vs Rollouts', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Critic disagreement
    ax = axes[0, 1]
    ax.plot(c_grid, results_dict['critic_q1'], 'r-', linewidth=1.5, alpha=0.7, label='Q1(c, π(c))')
    ax.plot(c_grid, results_dict['critic_q2'], 'b-', linewidth=1.5, alpha=0.7, label='Q2(c, π(c))')
    ax.fill_between(c_grid,
                     results_dict['critic_q1'],
                     results_dict['critic_q2'],
                     alpha=0.2, color='gray', label='Disagreement region')

    ax.set_xlabel('Cash Ratio (c)', fontsize=11)
    ax.set_ylabel('Q-value', fontsize=11)
    ax.set_title('Critic Networks Disagreement', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Policy and gradient
    ax = axes[1, 0]
    ax.plot(c_grid, policy_dict['policy'], 'g-', linewidth=2, label='Policy a(c)')
    ax.axvline(policy_dict['max_grad_c'], color='r', linestyle='--', alpha=0.7,
              label=f"Max gradient at c={policy_dict['max_grad_c']:.3f}")
    ax.set_xlabel('Cash Ratio (c)', fontsize=11)
    ax.set_ylabel('Dividend Rate (a)', fontsize=11)
    ax.set_title('Learned Policy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Training config summary
    ax = axes[1, 1]
    ax.axis('off')

    gamma_status = "✓ CORRECT" if config_dict['gamma_ok'] else "✗ WRONG"
    gamma_color = 'green' if config_dict['gamma_ok'] else 'red'

    text = f"""
TRAINING CONFIGURATION CHECK

Discount Factor:
  Model gamma:    {config_dict['model_gamma']:.6f}
  Expected:       {config_dict['expected_gamma']:.6f}
  Error:          {config_dict['gamma_error']:.6f}
  Status:         {gamma_status}

"""

    if not config_dict['gamma_ok']:
        text += """
⚠ CRITICAL ISSUE DETECTED ⚠

The model was trained with incorrect gamma!
This causes the agent to optimize the wrong
objective function, leading to validation
failures.

SOLUTION: Retrain with correct gamma from
environment.get_expected_discount_factor()
"""

    ax.text(0.1, 0.5, text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / 'diagnostic_plots.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved diagnostic plots to {output_path / 'diagnostic_plots.png'}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose validation issues")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--n-grid", type=int, default=100, help="Grid points (smaller for speed)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Setup
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path(str(model_path) + '.zip')
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {args.model}")

    output_dir = Path(args.output) if args.output else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GHM VALIDATION DIAGNOSTICS")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}\n")

    # Load
    print("Loading model and environment...")
    model = SAC.load(model_path)
    env = GHMEquityEnv(dt=0.01, max_steps=1000, a_max=10.0, liquidation_penalty=5.0)

    # Create grid
    c_min = 0.01
    c_max = env.dynamics.state_space.upper.numpy()[0]
    c_grid = np.linspace(c_min, c_max, args.n_grid)

    # Run diagnostics
    print("\n" + "=" * 70)
    print("1. VALUE ESTIMATION METHOD COMPARISON")
    print("-" * 50)
    results_dict = compare_value_estimation_methods(model, c_grid, env)

    config_dict = check_training_config(model, env)

    policy_dict = analyze_policy_quality(model, c_grid)

    smoothing_dict = test_smoothing_parameters(c_grid, results_dict['critic_mean'])

    # Create diagnostic plots
    print("\n" + "=" * 70)
    print("Creating diagnostic plots...")
    create_diagnostic_plots(results_dict, config_dict, policy_dict, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    if not config_dict['gamma_ok']:
        print("\n⚠ CRITICAL ISSUE: Incorrect discount factor")
        print(f"  The model was trained with gamma={config_dict['model_gamma']:.6f}")
        print(f"  but should use gamma={config_dict['expected_gamma']:.6f}")
        print("\n  ACTION REQUIRED: Retrain the model with correct gamma")
        print("  This is the likely root cause of validation failures.\n")
    else:
        print("\n✓ Discount factor is correct")

        print("\nPossible causes of validation issues:")
        print("  1. Value estimation noise - try using critic network + stronger smoothing")
        print(f"     Recommended: --use-critic --sigma-value {smoothing_dict['best_sigma']:.1f}")
        print("  2. Insufficient training - check if rewards have plateaued in tensorboard")
        print("  3. Numerical derivatives are inherently noisy for RL solutions")
        print("  4. RL solutions don't exactly satisfy HJB (they're approximate)")

    print("\nNext steps:")
    print("  1. Review diagnostic_plots.png for visual insights")
    print("  2. Run enhanced validation:")
    print(f"     python scripts/validate_improved.py --model {args.model} \\")
    print(f"            --sigma-value {smoothing_dict['best_sigma']:.1f} --sigma-deriv1 {smoothing_dict['best_sigma']-0.5:.1f}")
    print("  3. If gamma is wrong, retrain with corrected code")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
