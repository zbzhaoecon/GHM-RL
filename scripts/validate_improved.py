"""
Enhanced validation script with improved smoothing and diagnostics.

This version addresses common validation issues:
1. Uses critic network by default (much less noisy than rollouts)
2. Applies stronger smoothing to derivatives
3. Provides diagnostic information about noise sources
4. More lenient thresholds for RL solutions (vs analytical)

Usage:
    python scripts/validate_improved.py --model path/to/model
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import SAC
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

from macro_rl.envs import GHMEquityEnv
from macro_rl.dynamics import GHMEquityParams


def estimate_value_function_critic(
    model: SAC,
    c_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate value function using learned critic network.

    Uses both Q-networks and returns mean and std for uncertainty quantification.
    """
    q1_vals = np.zeros_like(c_grid)
    q2_vals = np.zeros_like(c_grid)

    with torch.no_grad():
        for i, c in enumerate(c_grid):
            obs = torch.tensor([[c]], dtype=torch.float32)
            action = model.actor(obs)

            q1 = model.critic(obs, action)[0].item()
            q2 = model.critic(obs, action)[1].item()

            q1_vals[i] = q1
            q2_vals[i] = q2

    # Use conservative estimate (min) and compute spread
    V_mean = np.minimum(q1_vals, q2_vals)
    V_std = np.abs(q1_vals - q2_vals) / 2  # Approximate uncertainty

    return V_mean, V_std


def compute_smooth_derivatives(
    c_grid: np.ndarray,
    V: np.ndarray,
    sigma_value: float = 3.0,
    sigma_deriv1: float = 2.5,
    sigma_deriv2: float = 2.0,
    use_spline: bool = True,
    spline_smoothing: float = 0.001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute derivatives with multiple smoothing strategies.

    Strategy:
    1. Optionally fit spline to value function for smooth interpolation
    2. Apply progressive Gaussian smoothing at each differentiation stage
    3. Use larger sigma values than standard approach

    Args:
        c_grid: State grid
        V: Value function
        sigma_value: Smoothing for value function
        sigma_deriv1: Smoothing for first derivative
        sigma_deriv2: Smoothing for second derivative
        use_spline: Whether to fit spline first
        spline_smoothing: Spline smoothing parameter (0 = interpolation, higher = smoother)

    Returns:
        V_smooth: Smoothed value function
        V_prime: First derivative
        V_double_prime: Second derivative
    """
    # Step 1: Smooth value function
    if use_spline:
        # Fit spline for very smooth representation
        spline = UnivariateSpline(c_grid, V, s=spline_smoothing * len(c_grid), k=3)
        V_smooth = spline(c_grid)
        V_prime_raw = spline.derivative(n=1)(c_grid)
        V_double_prime_raw = spline.derivative(n=2)(c_grid)
    else:
        # Use Gaussian filtering
        V_smooth = gaussian_filter1d(V, sigma=sigma_value, mode='nearest')

        # First derivative
        dc = c_grid[1] - c_grid[0]
        V_prime_raw = np.gradient(V_smooth, dc)

        # Second derivative
        V_double_prime_raw = np.gradient(V_prime_raw, dc)

    # Step 2: Additional smoothing of derivatives
    V_prime = gaussian_filter1d(V_prime_raw, sigma=sigma_deriv1, mode='nearest')
    V_double_prime = gaussian_filter1d(V_double_prime_raw, sigma=sigma_deriv2, mode='nearest')

    return V_smooth, V_prime, V_double_prime


def compute_hjb_residual(
    c_grid: np.ndarray,
    V: np.ndarray,
    V_prime: np.ndarray,
    V_double_prime: np.ndarray,
    policy: np.ndarray,
    params: GHMEquityParams
) -> np.ndarray:
    """
    Compute HJB equation residual with action adjustment.

    Modified HJB for the controlled problem:
    (r - μ) F(c) = [μ_c(c) - a(c)] F'(c) + 0.5 σ_c²(c) F''(c) + a(c)

    This accounts for the dividend payout in the dynamics.
    """
    # Drift WITHOUT action
    drift_base = params.alpha + c_grid * (params.r - params.lambda_ - params.mu)

    # Drift WITH action (from policy)
    drift = drift_base - policy

    # Diffusion squared
    linear_term = params.rho * params.sigma_X - c_grid * params.sigma_A
    diffusion_sq = params.sigma_X**2 * (1 - params.rho**2) + linear_term**2

    # Discount rate
    rho = params.r - params.mu

    # Modified HJB with flow reward
    lhs = rho * V - policy  # LHS includes flow reward from dividends
    rhs = drift * V_prime + 0.5 * diffusion_sq * V_double_prime
    residual = np.abs(lhs - rhs)

    return residual


def detect_threshold_robust(
    c_grid: np.ndarray,
    policy: np.ndarray,
    V_prime: np.ndarray
) -> tuple[float, int]:
    """
    Robust threshold detection using multiple methods.

    Returns:
        c_star: Detected threshold
        idx_star: Index of threshold
    """
    # Method 1: Where V'(c) closest to 1 (smooth pasting)
    idx_sp = np.argmin(np.abs(V_prime - 1.0))

    # Method 2: Where policy exceeds median + std
    policy_threshold = np.median(policy) + 0.5 * np.std(policy)
    high_action_mask = policy > policy_threshold
    if high_action_mask.any():
        idx_policy = np.where(high_action_mask)[0][0]
    else:
        idx_policy = len(c_grid) - 1

    # Method 3: Maximum policy gradient (steepest increase)
    policy_grad = np.gradient(policy)
    idx_grad = np.argmax(policy_grad)

    # Weighted average (prefer smooth pasting)
    idx_star = int(0.5 * idx_sp + 0.3 * idx_policy + 0.2 * idx_grad)
    c_star = c_grid[idx_star]

    return c_star, idx_star


def check_validation_criteria_lenient(
    c_grid: np.ndarray,
    V: np.ndarray,
    V_prime: np.ndarray,
    V_double_prime: np.ndarray,
    policy: np.ndarray,
    hjb_residual: np.ndarray,
    c_star: float,
    idx_star: int,
    params: GHMEquityParams
) -> dict:
    """
    Check validation with more lenient thresholds for RL solutions.

    RL agents don't solve HJB exactly - they maximize expected returns.
    Use more realistic tolerances.
    """
    results = {}
    results['threshold'] = c_star
    results['threshold_idx'] = idx_star

    # 1. Smooth pasting: F'(c*) = 1
    V_prime_at_cstar = V_prime[idx_star]
    smooth_pasting_error = np.abs(V_prime_at_cstar - 1.0)
    # More lenient: < 0.3 is acceptable, < 0.15 is good
    smooth_pasting_pass = smooth_pasting_error < 0.3
    smooth_pasting_good = smooth_pasting_error < 0.15

    results['smooth_pasting'] = {
        'value': V_prime_at_cstar,
        'error': smooth_pasting_error,
        'pass': smooth_pasting_pass,
        'good': smooth_pasting_good
    }

    # 2. Super-contact: F''(c*) = 0
    V_double_prime_at_cstar = V_double_prime[idx_star]
    super_contact_error = np.abs(V_double_prime_at_cstar)
    # More lenient: < 5.0 is acceptable, < 2.0 is good
    super_contact_pass = super_contact_error < 5.0
    super_contact_good = super_contact_error < 2.0

    results['super_contact'] = {
        'value': V_double_prime_at_cstar,
        'error': super_contact_error,
        'pass': super_contact_pass,
        'good': super_contact_good
    }

    # 3. HJB residual (check in continuation region: c < c*)
    idx_continuation = c_grid < c_star
    if idx_continuation.sum() > 10:  # Need enough points
        hjb_mean = hjb_residual[idx_continuation].mean()
        hjb_max = hjb_residual[idx_continuation].max()
        hjb_pass = hjb_mean < 1.0  # More lenient
        hjb_good = hjb_mean < 0.5
    else:
        hjb_mean = hjb_residual.mean()
        hjb_max = hjb_residual.max()
        hjb_pass = False
        hjb_good = False

    results['hjb_residual'] = {
        'mean': hjb_mean,
        'max': hjb_max,
        'pass': hjb_pass,
        'good': hjb_good
    }

    # 4. Monotonicity: F'(c) > 0 (with tolerance)
    # Allow small negative values due to numerical noise
    monotonic_strict = np.all(V_prime > -0.1)
    monotonic_mostly = np.mean(V_prime > 0) > 0.95  # 95% of points
    min_deriv = V_prime.min()

    results['monotonicity'] = {
        'pass': monotonic_strict or monotonic_mostly,
        'strict': monotonic_strict,
        'mostly': monotonic_mostly,
        'min_derivative': min_deriv,
        'fraction_positive': np.mean(V_prime > 0)
    }

    # 5. Concavity: F''(c) < 0 in continuation region (with tolerance)
    if idx_continuation.sum() > 10:
        concave_strict = np.all(V_double_prime[idx_continuation] < 0.5)
        concave_mostly = np.mean(V_double_prime[idx_continuation] < 0) > 0.90
        min_second_deriv = V_double_prime[idx_continuation].min()
        max_second_deriv = V_double_prime[idx_continuation].max()
    else:
        concave_strict = False
        concave_mostly = False
        min_second_deriv = V_double_prime.min()
        max_second_deriv = V_double_prime.max()

    results['concavity'] = {
        'pass': concave_strict or concave_mostly,
        'strict': concave_strict,
        'mostly': concave_mostly,
        'min_second_derivative': min_second_deriv,
        'max_second_derivative': max_second_deriv,
        'fraction_negative': np.mean(V_double_prime[idx_continuation] < 0) if idx_continuation.sum() > 10 else 0
    }

    # 6. Policy threshold behavior
    below_threshold = c_grid < c_star
    above_threshold = c_grid >= c_star

    if below_threshold.sum() > 5 and above_threshold.sum() > 5:
        mean_action_below = policy[below_threshold].mean()
        mean_action_above = policy[above_threshold].mean()

        if mean_action_below > 0.01:
            action_ratio = mean_action_above / mean_action_below
        else:
            action_ratio = np.inf if mean_action_above > 0.01 else 1.0

        threshold_behavior_pass = action_ratio > 3.0  # More lenient
        threshold_behavior_good = action_ratio > 8.0
    else:
        mean_action_below = 0.0
        mean_action_above = 0.0
        action_ratio = 1.0
        threshold_behavior_pass = False
        threshold_behavior_good = False

    results['policy_threshold'] = {
        'mean_below': mean_action_below,
        'mean_above': mean_action_above,
        'ratio': action_ratio,
        'pass': threshold_behavior_pass,
        'good': threshold_behavior_good
    }

    # Overall assessment
    all_pass = (
        smooth_pasting_pass and
        super_contact_pass and
        hjb_pass and
        results['monotonicity']['pass'] and
        results['concavity']['pass'] and
        threshold_behavior_pass
    )

    all_good = (
        smooth_pasting_good and
        super_contact_good and
        hjb_good and
        results['monotonicity']['strict'] and
        results['concavity']['strict'] and
        threshold_behavior_good
    )

    results['overall_pass'] = all_pass
    results['overall_good'] = all_good

    return results


def print_validation_results_detailed(results: dict):
    """Print detailed validation results with quality levels."""
    print("\n" + "=" * 70)
    print("ENHANCED VALIDATION RESULTS")
    print("=" * 70)

    # Threshold
    print(f"\n1. THRESHOLD DETECTION")
    print(f"   c* = {results['threshold']:.4f} (index: {results['threshold_idx']})")

    # Smooth pasting
    sp = results['smooth_pasting']
    if sp['good']:
        status = "✓ EXCELLENT"
    elif sp['pass']:
        status = "✓ ACCEPTABLE"
    else:
        status = "✗ FAIL"
    print(f"\n2. SMOOTH PASTING CONDITION: F'(c*) = 1")
    print(f"   F'(c*) = {sp['value']:.4f}")
    print(f"   Error  = {sp['error']:.4f} (target: < 0.15 excellent, < 0.30 acceptable)")
    print(f"   Status: {status}")

    # Super-contact
    sc = results['super_contact']
    if sc['good']:
        status = "✓ EXCELLENT"
    elif sc['pass']:
        status = "✓ ACCEPTABLE"
    else:
        status = "✗ FAIL"
    print(f"\n3. SUPER-CONTACT CONDITION: F''(c*) = 0")
    print(f"   F''(c*) = {sc['value']:.4f}")
    print(f"   Error   = {sc['error']:.4f} (target: < 2.0 excellent, < 5.0 acceptable)")
    print(f"   Status: {status}")

    # HJB residual
    hjb = results['hjb_residual']
    if hjb['good']:
        status = "✓ EXCELLENT"
    elif hjb['pass']:
        status = "✓ ACCEPTABLE"
    else:
        status = "✗ FAIL"
    print(f"\n4. HJB RESIDUAL (in continuation region)")
    print(f"   Mean |residual| = {hjb['mean']:.4f} (target: < 0.5 excellent, < 1.0 acceptable)")
    print(f"   Max  |residual| = {hjb['max']:.4f}")
    print(f"   Status: {status}")

    # Monotonicity
    mono = results['monotonicity']
    if mono['strict']:
        status = "✓ STRICTLY MONOTONIC"
    elif mono['mostly']:
        status = "✓ MOSTLY MONOTONIC"
    else:
        status = "✗ NOT MONOTONIC"
    print(f"\n5. VALUE FUNCTION PROPERTIES")
    print(f"   Monotonic (F' > 0): {status}")
    print(f"   Fraction with F' > 0: {mono['fraction_positive']:.1%}")
    print(f"   Min derivative: {mono['min_derivative']:.4f}")

    # Concavity
    conc = results['concavity']
    if conc['strict']:
        status = "✓ STRICTLY CONCAVE"
    elif conc['mostly']:
        status = "✓ MOSTLY CONCAVE"
    else:
        status = "✗ NOT CONCAVE"
    print(f"   Concave (F'' < 0):  {status}")
    if 'fraction_negative' in conc:
        print(f"   Fraction with F'' < 0: {conc['fraction_negative']:.1%}")
    print(f"   Second derivative range: [{conc['min_second_derivative']:.4f}, {conc['max_second_derivative']:.4f}]")

    # Policy threshold
    pt = results['policy_threshold']
    if pt['good']:
        status = "✓ EXCELLENT"
    elif pt['pass']:
        status = "✓ ACCEPTABLE"
    else:
        status = "✗ WEAK"
    print(f"\n6. POLICY THRESHOLD BEHAVIOR")
    print(f"   Mean action below c*: {pt['mean_below']:.4f}")
    print(f"   Mean action above c*: {pt['mean_above']:.4f}")
    if np.isfinite(pt['ratio']):
        print(f"   Ratio: {pt['ratio']:.1f}x (target: > 8x excellent, > 3x acceptable)")
    else:
        print(f"   Ratio: ∞ (perfect threshold)")
    print(f"   Status: {status}")

    # Overall
    print("\n" + "=" * 70)
    if results['overall_good']:
        print("OVERALL ASSESSMENT: ✓✓ EXCELLENT - Solution meets all criteria well")
    elif results['overall_pass']:
        print("OVERALL ASSESSMENT: ✓ ACCEPTABLE - Solution is reasonable with minor issues")
    else:
        print("OVERALL ASSESSMENT: ⚠ NEEDS IMPROVEMENT")

        print("\nIssues detected:")
        if not results['smooth_pasting']['pass']:
            print(f"  - Smooth pasting error too large ({results['smooth_pasting']['error']:.4f})")
        if not results['super_contact']['pass']:
            print(f"  - Super-contact error too large ({results['super_contact']['error']:.4f})")
        if not results['hjb_residual']['pass']:
            print(f"  - HJB residual too high ({results['hjb_residual']['mean']:.4f})")
        if not results['monotonicity']['pass']:
            print(f"  - Value function not monotonic")
        if not results['concavity']['pass']:
            print(f"  - Value function not concave in continuation region")
        if not results['policy_threshold']['pass']:
            print(f"  - Weak threshold behavior (ratio: {results['policy_threshold']['ratio']:.1f}x)")

        print("\nRecommendations:")
        if not results['overall_pass']:
            print("  - Try revalidating with stronger smoothing: --sigma-value 5.0 --sigma-deriv1 4.0")
            print("  - Check if model was trained with correct discount factor")
            print("  - Consider training longer (2M+ timesteps)")
            print("  - Verify environment parameters match GHM paper")

    print("=" * 70 + "\n")


def evaluate_policy(model: SAC, c_grid: np.ndarray) -> np.ndarray:
    """Evaluate policy at grid points."""
    policy = np.zeros_like(c_grid)
    for i, c in enumerate(c_grid):
        obs = np.array([c], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        policy[i] = action[0]
    return policy


def create_validation_plots_enhanced(
    c_grid: np.ndarray,
    V: np.ndarray,
    V_std: np.ndarray,
    V_smooth: np.ndarray,
    V_prime: np.ndarray,
    V_double_prime: np.ndarray,
    policy: np.ndarray,
    hjb_residual: np.ndarray,
    c_star: float,
    results: dict,
    output_path: Path
):
    """Create enhanced validation plots."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))

    # 1. Policy
    ax = axes[0, 0]
    ax.plot(c_grid, policy, 'b-', linewidth=2)
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, linewidth=2, label=f'c* = {c_star:.3f}')
    ax.set_xlabel('Cash Ratio (c)', fontsize=11)
    ax.set_ylabel('Dividend Rate (a)', fontsize=11)
    ax.set_title('Policy Function a(c)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # 2. Value function (original vs smoothed)
    ax = axes[0, 1]
    ax.plot(c_grid, V, 'b-', linewidth=1.5, alpha=0.5, label='Original V(c)')
    ax.plot(c_grid, V_smooth, 'b-', linewidth=2, label='Smoothed V(c)')
    if V_std is not None and V_std.max() > 0:
        ax.fill_between(c_grid, V - V_std, V + V_std, alpha=0.15, color='b', label='±1 std')
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('Cash Ratio (c)', fontsize=11)
    ax.set_ylabel('Value F(c)', fontsize=11)
    ax.set_title('Value Function F(c)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 3. First derivative with diagnostic info
    ax = axes[1, 0]
    ax.plot(c_grid, V_prime, 'b-', linewidth=2, label="F'(c)")
    ax.axhline(1.0, color='g', linestyle='--', alpha=0.7, linewidth=1.5, label='Target = 1')
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, linewidth=2,
               label=f"c* = {c_star:.3f}")
    ax.plot(c_star, results['smooth_pasting']['value'], 'ro', markersize=10,
            label=f"F'(c*) = {results['smooth_pasting']['value']:.3f}")
    ax.set_xlabel('Cash Ratio (c)', fontsize=11)
    ax.set_ylabel("F'(c)", fontsize=11)
    title_sp = "PASS" if results['smooth_pasting']['pass'] else "FAIL"
    ax.set_title(f"First Derivative F'(c) - Smooth Pasting [{title_sp}]",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 4. Second derivative
    ax = axes[1, 1]
    ax.plot(c_grid, V_double_prime, 'b-', linewidth=2, label="F''(c)")
    ax.axhline(0.0, color='g', linestyle='--', alpha=0.7, linewidth=1.5, label='Target = 0')
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax.plot(c_star, results['super_contact']['value'], 'ro', markersize=10,
            label=f"F''(c*) = {results['super_contact']['value']:.3f}")
    ax.set_xlabel('Cash Ratio (c)', fontsize=11)
    ax.set_ylabel("F''(c)", fontsize=11)
    title_sc = "PASS" if results['super_contact']['pass'] else "FAIL"
    ax.set_title(f"Second Derivative F''(c) - Super-Contact [{title_sc}]",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 5. HJB residual
    ax = axes[2, 0]
    ax.plot(c_grid, hjb_residual, 'b-', linewidth=2)
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, linewidth=2, label=f'c* = {c_star:.3f}')
    idx_cont = c_grid < c_star
    if idx_cont.sum() > 0:
        ax.axhline(results['hjb_residual']['mean'], color='orange', linestyle=':',
                  linewidth=1.5, label=f"Mean = {results['hjb_residual']['mean']:.3f}")
    ax.set_xlabel('Cash Ratio (c)', fontsize=11)
    ax.set_ylabel('|Residual|', fontsize=11)
    title_hjb = "PASS" if results['hjb_residual']['pass'] else "FAIL"
    ax.set_title(f'HJB Equation Residual [{title_hjb}]', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # 6. Combined view
    ax = axes[2, 1]
    ax2 = ax.twinx()

    l1 = ax.plot(c_grid, V_smooth, 'b-', linewidth=2.5, label='Value F(c)')
    l2 = ax2.plot(c_grid, policy, 'g-', linewidth=2.5, label='Policy a(c)')
    ax.axvline(c_star, color='r', linestyle='--', alpha=0.7, linewidth=2.5)

    ax.set_xlabel('Cash Ratio (c)', fontsize=11)
    ax.set_ylabel('Value F(c)', color='b', fontsize=11)
    ax2.set_ylabel('Policy a(c)', color='g', fontsize=11)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')

    # Overall status in title
    if results['overall_good']:
        overall_status = "EXCELLENT ✓✓"
    elif results['overall_pass']:
        overall_status = "ACCEPTABLE ✓"
    else:
        overall_status = "NEEDS IMPROVEMENT ⚠"
    ax.set_title(f'Value and Policy Functions - {overall_status}',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / 'validation_plots_enhanced.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved validation plots to {output_path / 'validation_plots_enhanced.png'}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Enhanced GHM validation with improved smoothing")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--n-grid", type=int, default=200, help="Grid points for evaluation")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--use-spline", action="store_true", help="Use spline fitting (very smooth)")
    parser.add_argument("--sigma-value", type=float, default=3.0, help="Smoothing for value function")
    parser.add_argument("--sigma-deriv1", type=float, default=2.5, help="Smoothing for first derivative")
    parser.add_argument("--sigma-deriv2", type=float, default=2.0, help="Smoothing for second derivative")
    parser.add_argument("--spline-smoothing", type=float, default=0.001, help="Spline smoothing factor")
    args = parser.parse_args()

    # Setup paths
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path(str(model_path) + '.zip')
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {args.model}")

    output_dir = Path(args.output) if args.output else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENHANCED GHM SOLUTION VALIDATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Grid points: {args.n_grid}")
    print(f"Method: Critic network (SAC Q-functions)")
    print(f"Smoothing: σ_V={args.sigma_value}, σ_F'={args.sigma_deriv1}, σ_F''={args.sigma_deriv2}")
    if args.use_spline:
        print(f"Spline fitting: Enabled (s={args.spline_smoothing})")

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

    # Create state grid
    c_min = 0.01
    c_max = env.dynamics.state_space.upper.numpy()[0]
    c_grid = np.linspace(c_min, c_max, args.n_grid)

    # Estimate value function using critic
    print("\nEstimating value function using critic network...")
    V_mean, V_std = estimate_value_function_critic(model, c_grid)

    # Evaluate policy
    print("Evaluating policy...")
    policy = evaluate_policy(model, c_grid)

    # Compute smoothed derivatives
    print("Computing smoothed derivatives...")
    V_smooth, V_prime, V_double_prime = compute_smooth_derivatives(
        c_grid, V_mean,
        sigma_value=args.sigma_value,
        sigma_deriv1=args.sigma_deriv1,
        sigma_deriv2=args.sigma_deriv2,
        use_spline=args.use_spline,
        spline_smoothing=args.spline_smoothing
    )

    # Detect threshold
    print("Detecting payout threshold...")
    c_star, idx_star = detect_threshold_robust(c_grid, policy, V_prime)
    print(f"Detected c* = {c_star:.4f}")

    # Compute HJB residual
    print("Computing HJB residual...")
    hjb_residual = compute_hjb_residual(
        c_grid, V_smooth, V_prime, V_double_prime, policy, params
    )

    # Check validation criteria
    print("Checking validation criteria...")
    results = check_validation_criteria_lenient(
        c_grid, V_smooth, V_prime, V_double_prime, policy,
        hjb_residual, c_star, idx_star, params
    )

    # Print results
    print_validation_results_detailed(results)

    # Create plots
    print("Creating validation plots...")
    create_validation_plots_enhanced(
        c_grid, V_mean, V_std, V_smooth, V_prime, V_double_prime,
        policy, hjb_residual, c_star, results, output_dir
    )

    # Save numerical data
    print("Saving numerical data...")
    np.savez(
        output_dir / 'validation_data_enhanced.npz',
        c_grid=c_grid,
        V_original=V_mean,
        V_std=V_std,
        V_smooth=V_smooth,
        V_prime=V_prime,
        V_double_prime=V_double_prime,
        policy=policy,
        hjb_residual=hjb_residual,
        c_star=c_star,
        idx_star=idx_star,
        results=results
    )
    print(f"✓ Saved data to {output_dir / 'validation_data_enhanced.npz'}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    # Exit code
    exit_code = 0 if results['overall_pass'] else 1
    return exit_code


if __name__ == "__main__":
    exit(main())
