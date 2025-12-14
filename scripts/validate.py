"""
Validate learned GHM solution against analytical properties.

This script checks that the learned value function and policy satisfy:
1. Smooth pasting: F'(c*) = 1
2. Super-contact: F''(c*) = 0
3. HJB equation in continuation region
4. Monotonicity: F'(c) > 0
5. Concavity: F''(c) < 0
6. Policy threshold behavior

Usage:
    python scripts/validate.py --model models/ghm_equity/final_model
    python scripts/validate.py --model models/ghm_equity/final_model --n-episodes 50 --n-grid 200
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from stable_baselines3 import SAC

from macro_rl.envs import GHMEquityEnv
from macro_rl.dynamics import GHMEquityParams


def estimate_value_function(
    model: SAC,
    env: GHMEquityEnv,
    n_episodes: int = 50,
    n_grid: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate value function by running episodes and averaging returns.

    Args:
        model: Trained SAC model
        env: GHM environment
        n_episodes: Number of episodes to average over
        n_grid: Number of grid points for value estimation

    Returns:
        c_grid: State grid points
        V_mean: Mean value at each grid point
        V_std: Standard deviation at each grid point
    """
    # Create state grid
    c_min = 0.01
    c_max = env.dynamics.state_space.upper.numpy()[0]
    c_grid = np.linspace(c_min, c_max, n_grid)

    # Estimate value at each grid point
    V_estimates = np.zeros((n_grid, n_episodes))

    print(f"\nEstimating value function on {n_grid} grid points...")
    print(f"Running {n_episodes} episodes per point...")

    for i, c in enumerate(c_grid):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_grid}")

        for ep in range(n_episodes):
            # Start from this state
            obs = np.array([c], dtype=np.float32)
            env._state = obs
            env._step_count = 0

            # Rollout and accumulate discounted rewards
            total_return = 0.0
            discount = 1.0
            gamma = env.get_expected_discount_factor()

            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_return += discount * reward
                discount *= gamma

            V_estimates[i, ep] = total_return

    V_mean = V_estimates.mean(axis=1)
    V_std = V_estimates.std(axis=1)

    return c_grid, V_mean, V_std


def compute_numerical_derivatives(c_grid: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute numerical first and second derivatives using central differences.

    Args:
        c_grid: State grid points (n,)
        V: Value function values (n,)

    Returns:
        V_prime: First derivative (n,)
        V_double_prime: Second derivative (n,)
    """
    dc = c_grid[1] - c_grid[0]

    # First derivative: central difference
    V_prime = np.gradient(V, dc)

    # Second derivative: central difference of first derivative
    V_double_prime = np.gradient(V_prime, dc)

    return V_prime, V_double_prime


def detect_threshold(c_grid: np.ndarray, policy: np.ndarray, V_prime: np.ndarray) -> float:
    """
    Detect payout threshold c* where policy sharply increases.

    Uses combination of:
    1. Where V'(c) is closest to 1
    2. Where policy starts increasing rapidly

    Args:
        c_grid: State grid
        policy: Policy values
        V_prime: First derivative of value function

    Returns:
        c_star: Detected threshold
    """
    # Method 1: Where V'(c) closest to 1
    idx_smooth_pasting = np.argmin(np.abs(V_prime - 1.0))

    # Method 2: Where policy exceeds threshold (e.g., 0.5)
    high_action_mask = policy > 0.5
    if high_action_mask.any():
        idx_policy = np.where(high_action_mask)[0][0]
    else:
        idx_policy = len(c_grid) - 1

    # Average the two methods
    idx = (idx_smooth_pasting + idx_policy) // 2
    c_star = c_grid[idx]

    return c_star


def compute_hjb_residual(
    c_grid: np.ndarray,
    V: np.ndarray,
    V_prime: np.ndarray,
    V_double_prime: np.ndarray,
    params: GHMEquityParams
) -> np.ndarray:
    """
    Compute HJB equation residual in continuation region.

    HJB: (r - μ) F(c) = μ_c(c) F'(c) + 0.5 σ_c²(c) F''(c)

    Residual = |(r - μ) F - μ_c F' - 0.5 σ² F''|

    Args:
        c_grid: State grid
        V: Value function
        V_prime: First derivative
        V_double_prime: Second derivative
        params: GHM parameters

    Returns:
        residual: HJB residual at each grid point
    """
    # Compute drift μ_c(c)
    drift = params.alpha + c_grid * (params.r - params.lambda_ - params.mu)

    # Compute diffusion squared σ_c²(c)
    linear_term = params.rho * params.sigma_X - c_grid * params.sigma_A
    diffusion_sq = params.sigma_X**2 * (1 - params.rho**2) + linear_term**2

    # Discount rate
    rho = params.r - params.mu

    # HJB residual
    lhs = rho * V
    rhs = drift * V_prime + 0.5 * diffusion_sq * V_double_prime
    residual = np.abs(lhs - rhs)

    return residual


def evaluate_policy(
    model: SAC,
    c_grid: np.ndarray
) -> np.ndarray:
    """
    Evaluate policy at grid points.

    Args:
        model: Trained SAC model
        c_grid: State grid

    Returns:
        policy: Action values at each grid point
    """
    policy = np.zeros_like(c_grid)

    for i, c in enumerate(c_grid):
        obs = np.array([c], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        policy[i] = action[0]

    return policy


def check_validation_criteria(
    c_grid: np.ndarray,
    V: np.ndarray,
    V_prime: np.ndarray,
    V_double_prime: np.ndarray,
    policy: np.ndarray,
    hjb_residual: np.ndarray,
    c_star: float,
    params: GHMEquityParams
) -> dict:
    """
    Check all validation criteria and return results.

    Returns:
        results: Dictionary with all validation metrics and pass/fail status
    """
    results = {}

    # Find index of c*
    idx_star = np.argmin(np.abs(c_grid - c_star))

    # 1. Smooth pasting: F'(c*) = 1
    V_prime_at_cstar = V_prime[idx_star]
    smooth_pasting_error = np.abs(V_prime_at_cstar - 1.0)
    smooth_pasting_pass = smooth_pasting_error < 0.2

    results['smooth_pasting'] = {
        'value': V_prime_at_cstar,
        'error': smooth_pasting_error,
        'pass': smooth_pasting_pass
    }

    # 2. Super-contact: F''(c*) = 0
    V_double_prime_at_cstar = V_double_prime[idx_star]
    super_contact_error = np.abs(V_double_prime_at_cstar)
    super_contact_pass = super_contact_error < 1.0

    results['super_contact'] = {
        'value': V_double_prime_at_cstar,
        'error': super_contact_error,
        'pass': super_contact_pass
    }

    # 3. HJB residual (only check before c*)
    idx_continuation = c_grid < c_star
    if idx_continuation.any():
        hjb_mean = hjb_residual[idx_continuation].mean()
        hjb_max = hjb_residual[idx_continuation].max()
        hjb_pass = hjb_mean < 0.5
    else:
        hjb_mean = hjb_residual.mean()
        hjb_max = hjb_residual.max()
        hjb_pass = False

    results['hjb_residual'] = {
        'mean': hjb_mean,
        'max': hjb_max,
        'pass': hjb_pass
    }

    # 4. Monotonicity: F'(c) > 0 everywhere
    monotonic = np.all(V_prime > -0.01)  # Allow small numerical errors

    results['monotonicity'] = {
        'pass': bool(monotonic),
        'min_derivative': V_prime.min()
    }

    # 5. Concavity: F''(c) < 0 in continuation region
    if idx_continuation.any():
        concave = np.all(V_double_prime[idx_continuation] < 0.1)  # Allow small errors
        min_second_deriv = V_double_prime[idx_continuation].min()
    else:
        concave = False
        min_second_deriv = V_double_prime.min()

    results['concavity'] = {
        'pass': bool(concave),
        'min_second_derivative': min_second_deriv
    }

    # 6. Policy threshold behavior
    below_threshold = c_grid < c_star
    above_threshold = c_grid >= c_star

    if below_threshold.any() and above_threshold.any():
        mean_action_below = policy[below_threshold].mean()
        mean_action_above = policy[above_threshold].mean()

        if mean_action_below > 0:
            action_ratio = mean_action_above / mean_action_below
        else:
            action_ratio = np.inf if mean_action_above > 0 else 1.0

        threshold_behavior_pass = action_ratio > 5.0
    else:
        mean_action_below = 0.0
        mean_action_above = 0.0
        action_ratio = 1.0
        threshold_behavior_pass = False

    results['policy_threshold'] = {
        'mean_below': mean_action_below,
        'mean_above': mean_action_above,
        'ratio': action_ratio,
        'pass': threshold_behavior_pass
    }

    # Overall pass
    all_pass = (
        smooth_pasting_pass and
        super_contact_pass and
        hjb_pass and
        monotonic and
        concave and
        threshold_behavior_pass
    )

    results['overall_pass'] = all_pass
    results['threshold'] = c_star

    return results


def print_validation_results(results: dict):
    """Print validation results in a nice format."""
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    # Threshold
    print(f"\n1. THRESHOLD DETECTION")
    print(f"   c* = {results['threshold']:.4f}")

    # Smooth pasting
    sp = results['smooth_pasting']
    status = "✓ PASS" if sp['pass'] else "✗ FAIL"
    print(f"\n2. SMOOTH PASTING CONDITION: F'(c*) = 1")
    print(f"   F'(c*) = {sp['value']:.4f}")
    print(f"   Error  = {sp['error']:.4f}")
    print(f"   Status: {status}")

    # Super-contact
    sc = results['super_contact']
    status = "✓ PASS" if sc['pass'] else "✗ FAIL"
    print(f"\n3. SUPER-CONTACT CONDITION: F''(c*) = 0")
    print(f"   F''(c*) = {sc['value']:.4f}")
    print(f"   Error   = {sc['error']:.4f}")
    print(f"   Status: {status}")

    # HJB residual
    hjb = results['hjb_residual']
    status = "✓ PASS" if hjb['pass'] else "✗ FAIL"
    print(f"\n4. HJB RESIDUAL")
    print(f"   Mean |residual| = {hjb['mean']:.4f}")
    print(f"   Max  |residual| = {hjb['max']:.4f}")
    print(f"   Status: {status}")

    # Monotonicity
    mono = results['monotonicity']
    status = "✓ YES" if mono['pass'] else "✗ NO"
    print(f"\n5. VALUE FUNCTION PROPERTIES")
    print(f"   Monotonic (F' > 0): {status}")
    if not mono['pass']:
        print(f"   Min derivative: {mono['min_derivative']:.4f}")

    # Concavity
    conc = results['concavity']
    status = "✓ YES" if conc['pass'] else "✗ NO"
    print(f"   Concave (F'' < 0):  {status}")
    if not conc['pass']:
        print(f"   Min second derivative: {conc['min_second_derivative']:.4f}")

    # Policy threshold
    pt = results['policy_threshold']
    status = "✓ PASS" if pt['pass'] else "✗ FAIL"
    print(f"\n6. POLICY PROPERTIES")
    print(f"   Mean action below c*: {pt['mean_below']:.4f}")
    print(f"   Mean action above c*: {pt['mean_above']:.4f}")
    print(f"   Ratio: {pt['ratio']:.1f}x")
    print(f"   Status: {status}")

    # Overall
    print("\n" + "=" * 60)
    if results['overall_pass']:
        print("OVERALL ASSESSMENT: ✓ Solution appears VALID")
    else:
        print("OVERALL ASSESSMENT: ⚠ Solution may have ISSUES")
        print("\nFailed criteria:")
        if not sp['pass']:
            print("  - Smooth pasting condition not satisfied")
        if not sc['pass']:
            print("  - Super-contact condition not satisfied")
        if not hjb['pass']:
            print("  - HJB residual too large")
        if not mono['pass']:
            print("  - Value function not monotonic")
        if not conc['pass']:
            print("  - Value function not concave in continuation region")
        if not pt['pass']:
            print("  - Policy doesn't show clear threshold behavior")

        print("\nConsider:")
        print("  - Training for more timesteps")
        print("  - Adjusting hyperparameters (gamma, liquidation_penalty)")
        print("  - Checking environment setup")
    print("=" * 60 + "\n")


def create_validation_plots(
    c_grid: np.ndarray,
    V: np.ndarray,
    V_std: np.ndarray,
    V_prime: np.ndarray,
    V_double_prime: np.ndarray,
    policy: np.ndarray,
    hjb_residual: np.ndarray,
    c_star: float,
    output_path: Path
):
    """Create comprehensive validation plots."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Policy
    ax = axes[0, 0]
    ax.plot(c_grid, policy, 'b-', linewidth=2)
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, label=f'c* = {c_star:.3f}')
    ax.set_xlabel('Cash Ratio (c)')
    ax.set_ylabel('Dividend Rate (a)')
    ax.set_title('Policy Function a(c)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Value function
    ax = axes[0, 1]
    ax.plot(c_grid, V, 'b-', linewidth=2, label='V(c)')
    ax.fill_between(c_grid, V - V_std, V + V_std, alpha=0.2, label='±1 std')
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, label=f'c* = {c_star:.3f}')
    ax.set_xlabel('Cash Ratio (c)')
    ax.set_ylabel('Value F(c)')
    ax.set_title('Value Function F(c)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. First derivative
    ax = axes[1, 0]
    ax.plot(c_grid, V_prime, 'b-', linewidth=2, label="F'(c)")
    ax.axhline(1.0, color='g', linestyle='--', alpha=0.7, label='Target = 1')
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, label=f'c* = {c_star:.3f}')
    ax.set_xlabel('Cash Ratio (c)')
    ax.set_ylabel("F'(c)")
    ax.set_title("First Derivative F'(c) - Smooth Pasting")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Second derivative
    ax = axes[1, 1]
    ax.plot(c_grid, V_double_prime, 'b-', linewidth=2, label="F''(c)")
    ax.axhline(0.0, color='g', linestyle='--', alpha=0.7, label='Target = 0')
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, label=f'c* = {c_star:.3f}')
    ax.set_xlabel('Cash Ratio (c)')
    ax.set_ylabel("F''(c)")
    ax.set_title("Second Derivative F''(c) - Super-Contact")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5. HJB residual
    ax = axes[2, 0]
    ax.plot(c_grid, hjb_residual, 'b-', linewidth=2)
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, label=f'c* = {c_star:.3f}')
    ax.set_xlabel('Cash Ratio (c)')
    ax.set_ylabel('|Residual|')
    ax.set_title('HJB Equation Residual')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    # 6. Combined view (matching paper figures)
    ax = axes[2, 1]
    ax2 = ax.twinx()

    l1 = ax.plot(c_grid, V, 'b-', linewidth=2, label='Value F(c)')
    l2 = ax2.plot(c_grid, policy, 'g-', linewidth=2, label='Policy a(c)')
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Cash Ratio (c)')
    ax.set_ylabel('Value F(c)', color='b')
    ax2.set_ylabel('Policy a(c)', color='g')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    ax.set_title('Value and Policy Functions')
    ax.grid(True, alpha=0.3)

    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / 'validation_plots.png', dpi=150, bbox_inches='tight')
    print(f"Saved validation plots to {output_path / 'validation_plots.png'}")

    # Create separate combined plot matching paper style
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()

    l1 = ax.plot(c_grid, V, 'b-', linewidth=2.5, label='Value F(c)')
    l2 = ax2.plot(c_grid, policy, 'g-', linewidth=2.5, label='Policy a(c)')
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, linewidth=2, label=f'c* = {c_star:.3f}')

    ax.set_xlabel('Cash Ratio (c)', fontsize=12)
    ax.set_ylabel('Value F(c)', color='b', fontsize=12)
    ax2.set_ylabel('Policy a(c)', color='g', fontsize=12)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    ax.set_title('GHM Equity Model: Learned Solution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Combine legends
    lines = l1 + l2 + [plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2)]
    labels = ['Value F(c)', 'Policy a(c)', f'Threshold c* = {c_star:.3f}']
    ax.legend(lines, labels, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / 'value_and_policy.png', dpi=150, bbox_inches='tight')
    print(f"Saved combined plot to {output_path / 'value_and_policy.png'}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Validate learned GHM solution")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--n-episodes", type=int, default=50, help="Episodes for value estimation")
    parser.add_argument("--n-grid", type=int, default=200, help="Grid points for evaluation")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: model dir)")
    args = parser.parse_args()

    # Setup paths
    model_path = Path(args.model)
    if not model_path.exists():
        # Try adding .zip extension
        model_path = Path(str(model_path) + '.zip')
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {args.model}")

    output_dir = Path(args.output) if args.output else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GHM SOLUTION VALIDATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Episodes per grid point: {args.n_episodes}")
    print(f"Grid points: {args.n_grid}")

    # Load model
    print("\nLoading model...")
    model = SAC.load(model_path)

    # Create environment
    env = GHMEquityEnv(
        dt=0.01,
        max_steps=1000,
        a_max=10.0,
        liquidation_penalty=5.0,
    )
    params = env._dynamics.p

    # Estimate value function
    c_grid, V_mean, V_std = estimate_value_function(
        model, env, args.n_episodes, args.n_grid
    )

    # Evaluate policy
    print("\nEvaluating policy...")
    policy = evaluate_policy(model, c_grid)

    # Compute derivatives
    print("Computing numerical derivatives...")
    V_prime, V_double_prime = compute_numerical_derivatives(c_grid, V_mean)

    # Detect threshold
    print("Detecting payout threshold...")
    c_star = detect_threshold(c_grid, policy, V_prime)
    print(f"Detected c* = {c_star:.4f}")

    # Compute HJB residual
    print("Computing HJB residual...")
    hjb_residual = compute_hjb_residual(
        c_grid, V_mean, V_prime, V_double_prime, params
    )

    # Check validation criteria
    print("Checking validation criteria...")
    results = check_validation_criteria(
        c_grid, V_mean, V_prime, V_double_prime, policy,
        hjb_residual, c_star, params
    )

    # Print results
    print_validation_results(results)

    # Create plots
    print("Creating validation plots...")
    create_validation_plots(
        c_grid, V_mean, V_std, V_prime, V_double_prime,
        policy, hjb_residual, c_star, output_dir
    )

    # Save numerical data
    print("Saving numerical data...")
    np.savez(
        output_dir / 'validation_data.npz',
        c_grid=c_grid,
        V_mean=V_mean,
        V_std=V_std,
        V_prime=V_prime,
        V_double_prime=V_double_prime,
        policy=policy,
        hjb_residual=hjb_residual,
        c_star=c_star,
        results=results
    )
    print(f"Saved data to {output_dir / 'validation_data.npz'}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    # Exit with appropriate code
    exit_code = 0 if results['overall_pass'] else 1
    return exit_code


if __name__ == "__main__":
    exit(main())
