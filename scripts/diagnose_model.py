"""
Comprehensive diagnostic script to verify GHM model components.

Runs a series of tests to identify what's going wrong with training.

Experiments:
1. Parameter verification: Check config matches theoretical values
2. Drift/diffusion verification: Ensure dynamics are correct
3. Reward function verification: Test step and terminal rewards
4. Theoretical benchmark: Compute value under no-control policy
5. Monte Carlo sanity check: Simple MC estimation of value
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import math

# Import components
from macro_rl.dynamics.ghm_equity import GHMEquityParams, GHMEquityTimeAugmentedDynamics
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.simulation.trajectory import TrajectorySimulator


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_1_parameter_verification():
    """Verify model parameters match theoretical values."""
    print_header("TEST 1: Parameter Verification")

    # Theoretical values from Décamps et al (2017)
    theory = {
        'alpha': 0.18,      # Mean cash flow rate
        'mu': 0.01,         # Growth rate
        'r': 0.03,          # Interest rate
        'lambda_': 0.02,    # Carry cost (λ)
        'sigma_A': 0.25,    # Permanent shock volatility
        'sigma_X': 0.12,    # Transitory shock volatility
        'rho': -0.2,        # Correlation
        'p': 1.06,          # Proportional issuance cost
        'phi': 0.002,       # Fixed issuance cost
        'omega': 0.55,      # Recovery rate at bankruptcy
    }

    # Derived values
    theory['discount_rate'] = theory['r'] - theory['mu']  # 0.02
    theory['liquidation_value'] = theory['omega'] * theory['alpha'] / theory['discount_rate']

    # Create model objects
    params = GHMEquityParams()

    print("Parameter comparison:")
    print("-" * 50)
    print(f"{'Parameter':<15} {'Theory':<12} {'Model':<12} {'Match'}")
    print("-" * 50)

    all_match = True
    checks = [
        ('alpha', theory['alpha'], params.alpha),
        ('mu', theory['mu'], params.mu),
        ('r', theory['r'], params.r),
        ('lambda', theory['lambda_'], params.lambda_),
        ('sigma_A', theory['sigma_A'], params.sigma_A),
        ('sigma_X', theory['sigma_X'], params.sigma_X),
        ('rho', theory['rho'], params.rho),
        ('p', theory['p'], params.p),
        ('phi', theory['phi'], params.phi),
        ('omega', theory['omega'], params.omega),
    ]

    for name, expected, actual in checks:
        match = abs(expected - actual) < 1e-6
        status = "✓" if match else "✗"
        print(f"{name:<15} {expected:<12.4f} {actual:<12.4f} {status}")
        if not match:
            all_match = False

    # Check derived values
    print("\nDerived values:")
    print("-" * 50)
    discount_rate = params.r - params.mu
    liquidation_value = params.omega * params.alpha / discount_rate
    print(f"Discount rate (r-μ):     Expected {theory['discount_rate']:.4f}, Got {discount_rate:.4f}")
    print(f"Liquidation value (ωα/(r-μ)): Expected {theory['liquidation_value']:.4f}, Got {liquidation_value:.4f}")

    return all_match


def test_2_drift_verification():
    """Verify drift formula matches theory."""
    print_header("TEST 2: Drift Formula Verification")

    params = GHMEquityParams()
    dynamics = GHMEquityTimeAugmentedDynamics(params, T=10.0)

    # Test points
    test_cases = [
        {'c': 1.0, 'tau': 5.0, 'a_L': 0.0, 'a_E': 0.0},  # No action
        {'c': 0.5, 'tau': 3.0, 'a_L': 0.1, 'a_E': 0.0},  # Dividend only
        {'c': 0.2, 'tau': 8.0, 'a_L': 0.0, 'a_E': 0.5},  # Equity only
        {'c': 1.5, 'tau': 1.0, 'a_L': 0.2, 'a_E': 0.3},  # Both actions
    ]

    print("Drift verification (dc/dt = α + c(r-λ-μ) - a_L + a_E/p - φ·𝟙(a_E>0)):")
    print("-" * 70)
    all_correct = True

    for case in test_cases:
        c, tau, a_L, a_E = case['c'], case['tau'], case['a_L'], case['a_E']

        # Create tensors
        state = torch.tensor([[c, tau]], dtype=torch.float32)
        action = torch.tensor([[a_L, a_E]], dtype=torch.float32)

        # Compute via model
        drift_model = dynamics.drift(state, action)
        drift_c = drift_model[0, 0].item()
        drift_tau = drift_model[0, 1].item()

        # Compute theoretical
        is_issuing = 1.0 if a_E > 1e-6 else 0.0
        drift_c_theory = (params.alpha
                         + c * (params.r - params.lambda_ - params.mu)
                         - a_L
                         + a_E / params.p
                         - params.phi * is_issuing)
        drift_tau_theory = -1.0  # Time decreases

        match_c = abs(drift_c - drift_c_theory) < 1e-6
        match_tau = abs(drift_tau - drift_tau_theory) < 1e-6

        status = "✓" if (match_c and match_tau) else "✗"
        print(f"c={c:.1f}, τ={tau:.1f}, a_L={a_L:.1f}, a_E={a_E:.1f}")
        print(f"  Drift c:   Theory={drift_c_theory:.6f}, Model={drift_c:.6f} {status if match_c else '✗'}")
        print(f"  Drift τ:   Theory={drift_tau_theory:.6f}, Model={drift_tau:.6f} {status if match_tau else '✗'}")

        if not (match_c and match_tau):
            all_correct = False

    return all_correct


def test_3_diffusion_verification():
    """Verify diffusion formula matches theory."""
    print_header("TEST 3: Diffusion Formula Verification")

    params = GHMEquityParams()
    dynamics = GHMEquityTimeAugmentedDynamics(params, T=10.0)

    # Test at different cash levels
    test_c_values = [0.1, 0.5, 1.0, 1.5, 2.0]

    print("Diffusion verification (σ_c² = σ_X²(1-ρ²) + (ρσ_X - cσ_A)²):")
    print("-" * 60)
    all_correct = True

    for c in test_c_values:
        state = torch.tensor([[c, 5.0]], dtype=torch.float32)  # tau doesn't affect diffusion

        # Model computation
        diff_model = dynamics.diffusion(state)
        sigma_c = diff_model[0, 0].item()
        sigma_tau = diff_model[0, 1].item()

        # Theory: σ_c = sqrt(σ_X²(1-ρ²) + (ρσ_X - cσ_A)²)
        sigma_X = params.sigma_X
        sigma_A = params.sigma_A
        rho = params.rho

        term1 = sigma_X**2 * (1 - rho**2)
        term2 = (rho * sigma_X - c * sigma_A)**2
        sigma_c_theory = math.sqrt(term1 + term2)
        sigma_tau_theory = 0.0  # Time has no diffusion

        match_c = abs(sigma_c - sigma_c_theory) < 1e-5
        match_tau = abs(sigma_tau - sigma_tau_theory) < 1e-5

        status = "✓" if (match_c and match_tau) else "✗"
        print(f"c={c:.1f}: Theory σ_c={sigma_c_theory:.6f}, Model σ_c={sigma_c:.6f} {status}")

        if not (match_c and match_tau):
            all_correct = False

    return all_correct


def test_4_reward_verification():
    """Verify reward function computation."""
    print_header("TEST 4: Reward Function Verification")

    # Create reward function with correct parameters
    reward_fn = GHMRewardFunction(
        discount_rate=0.02,
        issuance_cost=1.0,  # Deprecated but kept
        liquidation_rate=0.55,
        liquidation_flow=0.18,
        fixed_cost=0.002,
    )

    dt = 0.1

    print(f"Reward function parameters:")
    print(f"  discount_rate: {reward_fn.discount_rate_value}")
    print(f"  issuance_cost: {reward_fn.issuance_cost}")
    print(f"  liquidation_value: {reward_fn.liquidation_value}")
    print(f"  fixed_cost: {reward_fn.fixed_cost}")
    print()

    # Test step rewards
    test_cases = [
        {'a_L': 0.0, 'a_E': 0.0, 'desc': 'No action'},
        {'a_L': 1.0, 'a_E': 0.0, 'desc': 'Dividend only'},
        {'a_L': 0.0, 'a_E': 0.5, 'desc': 'Equity only'},
        {'a_L': 0.5, 'a_E': 0.2, 'desc': 'Both actions'},
    ]

    print("Step reward verification (r = (a_L - a_E - φ·𝟙(a_E>0)) × dt):")
    print("-" * 60)
    all_correct = True

    for case in test_cases:
        a_L, a_E = case['a_L'], case['a_E']

        state = torch.tensor([[1.0, 5.0]], dtype=torch.float32)
        action = torch.tensor([[a_L, a_E]], dtype=torch.float32)

        reward = reward_fn.step_reward(state, action, state, dt).item()

        # Theory: r = (a_L - 1.0 * a_E - φ·𝟙(a_E>0)) × dt
        is_issuing = 1.0 if a_E > 1e-6 else 0.0
        reward_theory = (a_L - 1.0 * a_E - 0.002 * is_issuing) * dt

        match = abs(reward - reward_theory) < 1e-8
        status = "✓" if match else "✗"
        print(f"{case['desc']:<20} a_L={a_L:.1f}, a_E={a_E:.1f}")
        print(f"  Theory: {reward_theory:.8f}, Model: {reward:.8f} {status}")

        if not match:
            all_correct = False

    # Test terminal reward
    print("\nTerminal reward verification:")
    print("-" * 60)

    state = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # Bankrupt
    terminated = torch.tensor([True])
    terminal = reward_fn.terminal_reward(state, terminated).item()

    expected_terminal = 0.55 * 0.18 / 0.02  # ωα/(r-μ) = 4.95
    match = abs(terminal - expected_terminal) < 1e-5
    status = "✓" if match else "✗"
    print(f"Liquidation value: Expected={expected_terminal:.4f}, Got={terminal:.4f} {status}")

    if not match:
        all_correct = False

    return all_correct


def test_5_theoretical_benchmark():
    """Compute theoretical value function benchmark."""
    print_header("TEST 5: Theoretical Value Function Benchmark")

    params = GHMEquityParams()

    # Key quantities
    rho = params.r - params.mu  # Discount rate = 0.02
    alpha = params.alpha  # 0.18
    omega = params.omega  # 0.55
    liquidation_value = omega * alpha / rho  # 4.95

    print("Key theoretical values:")
    print(f"  α (cash flow rate):      {alpha:.4f}")
    print(f"  ρ (discount rate r-μ):   {rho:.4f}")
    print(f"  ω (recovery rate):       {omega:.4f}")
    print(f"  Liquidation value:       {liquidation_value:.4f}")
    print()

    # Value of just collecting cash flows (no dividend, no equity)
    # Under no action, cash evolves with drift = α + c(r-λ-μ)
    # At c=0, drift = α = 0.18 > 0, so positive drift
    # Value = E[∫ e^(-ρt) a_L dt] where a_L = dividend policy

    # Simple benchmark: Value of perpetual cash flow α at rate ρ
    perpetuity_value = alpha / rho
    print(f"Perpetuity value (α/ρ):    {perpetuity_value:.4f}")

    # Expected value range:
    # - Minimum: Liquidation immediately = 4.95
    # - Maximum: Optimal policy with infinite horizon ≈ α/ρ + adjustments
    print(f"\nExpected value range:")
    print(f"  Lower bound (liquidation): {liquidation_value:.4f}")
    print(f"  Upper bound estimate:      ~{perpetuity_value:.4f} (perpetuity)")
    print(f"\nIf trained values are << 4.95, something is wrong!")

    return True


def test_6_simple_monte_carlo():
    """Run simple Monte Carlo to estimate value."""
    print_header("TEST 6: Simple Monte Carlo Estimation")

    params = GHMEquityParams()
    dynamics = GHMEquityTimeAugmentedDynamics(params, T=10.0)

    reward_fn = GHMRewardFunction(
        discount_rate=0.02,
        issuance_cost=1.0,
        liquidation_rate=0.55,
        liquidation_flow=0.18,
        fixed_cost=0.002,
    )

    control_spec = GHMControlSpec(
        a_L_max=20.0,
        a_E_max=4.0,
        issuance_threshold=0.05,
        issuance_cost=0.002,
    )

    # Simulation parameters
    n_trajectories = 1000
    dt = 0.1
    max_steps = 100  # T=10.0
    discount_rate = 0.02

    # Initial states: c ~ Uniform[0.1, 2.0], τ = 10.0
    torch.manual_seed(42)
    c_init = torch.rand(n_trajectories) * 1.9 + 0.1
    tau_init = torch.full((n_trajectories,), 10.0)
    states = torch.stack([c_init, tau_init], dim=1)

    # Run trajectories with NO ACTION policy (a_L = 0, a_E = 0)
    print("Running Monte Carlo with NO ACTION policy (a_L=0, a_E=0)...")

    returns_no_action = torch.zeros(n_trajectories)
    current_states = states.clone()

    for t in range(max_steps):
        # No action
        actions = torch.zeros(n_trajectories, 2)

        # Check termination
        c = current_states[:, 0]
        tau = current_states[:, 1]
        bankrupt = c <= 0
        horizon_end = tau <= 0

        if bankrupt.all() or horizon_end.all():
            break

        # Compute reward (should be 0 for no action)
        rewards = reward_fn.step_reward(current_states, actions, current_states, dt)

        # Discount
        discount = math.exp(-discount_rate * t * dt)
        returns_no_action += discount * rewards

        # Step dynamics (simple Euler-Maruyama)
        drift = dynamics.drift(current_states, actions)
        diffusion = dynamics.diffusion(current_states)
        noise = torch.randn_like(current_states)
        next_states = current_states + drift * dt + diffusion * math.sqrt(dt) * noise

        # Enforce bounds
        next_states[:, 0] = torch.clamp(next_states[:, 0], min=0)
        next_states[:, 1] = torch.clamp(next_states[:, 1], min=0)

        current_states = next_states

    # Add terminal value
    terminal_discount = math.exp(-discount_rate * max_steps * dt)
    terminal_rewards = reward_fn.terminal_reward(current_states, current_states[:, 0] <= 0)
    returns_no_action += terminal_discount * terminal_rewards

    print(f"\nNo-action policy results:")
    print(f"  Mean return:  {returns_no_action.mean().item():.4f}")
    print(f"  Std return:   {returns_no_action.std().item():.4f}")
    print(f"  Min return:   {returns_no_action.min().item():.4f}")
    print(f"  Max return:   {returns_no_action.max().item():.4f}")

    # Now run with GREEDY DIVIDEND policy (pay out all cash as dividend)
    print("\nRunning Monte Carlo with GREEDY DIVIDEND policy (a_L=c/dt, a_E=0)...")

    returns_greedy = torch.zeros(n_trajectories)
    current_states = states.clone()

    for t in range(max_steps):
        c = current_states[:, 0]
        tau = current_states[:, 1]

        # Greedy dividend: pay out all cash
        a_L = c / dt  # Rate such that dividend × dt = c
        a_E = torch.zeros_like(a_L)
        actions = torch.stack([a_L, a_E], dim=1)

        # Compute reward
        rewards = reward_fn.step_reward(current_states, actions, current_states, dt)

        # Discount
        discount = math.exp(-discount_rate * t * dt)
        returns_greedy += discount * rewards

        # After paying all dividends, c should become ~0 (plus noise)
        drift = dynamics.drift(current_states, actions)
        diffusion = dynamics.diffusion(current_states)
        noise = torch.randn_like(current_states)
        next_states = current_states + drift * dt + diffusion * math.sqrt(dt) * noise

        next_states[:, 0] = torch.clamp(next_states[:, 0], min=0)
        next_states[:, 1] = torch.clamp(next_states[:, 1], min=0)

        current_states = next_states

        # Most trajectories should go bankrupt quickly
        if (current_states[:, 0] <= 0).all():
            break

    # Add terminal value
    terminal_rewards = reward_fn.terminal_reward(current_states, current_states[:, 0] <= 0)
    returns_greedy += math.exp(-discount_rate * max_steps * dt) * terminal_rewards

    print(f"\nGreedy dividend policy results:")
    print(f"  Mean return:  {returns_greedy.mean().item():.4f}")
    print(f"  Std return:   {returns_greedy.std().item():.4f}")
    print(f"  Min return:   {returns_greedy.min().item():.4f}")
    print(f"  Max return:   {returns_greedy.max().item():.4f}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"  Liquidation value (theoretical minimum): 4.95")
    print(f"  No-action policy value:                  {returns_no_action.mean().item():.4f}")
    print(f"  Greedy dividend policy value:            {returns_greedy.mean().item():.4f}")

    if returns_no_action.mean().item() > 4.0:
        print("\n✓ No-action value is reasonable (> 4.0)")
    else:
        print("\n✗ No-action value is too low (should be > 4.0)")

    return True


def main():
    print("\n" + "="*70)
    print("  GHM MODEL DIAGNOSTIC TESTS")
    print("="*70)

    results = []

    # Run all tests
    results.append(("Parameter Verification", test_1_parameter_verification()))
    results.append(("Drift Verification", test_2_drift_verification()))
    results.append(("Diffusion Verification", test_3_diffusion_verification()))
    results.append(("Reward Verification", test_4_reward_verification()))
    results.append(("Theoretical Benchmark", test_5_theoretical_benchmark()))
    results.append(("Monte Carlo Estimation", test_6_simple_monte_carlo()))

    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    print(f"{'Test':<30} {'Result'}")
    print("-" * 40)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:<30} {status}")

    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("All diagnostics PASSED!")
    else:
        print("Some diagnostics FAILED - see above for details")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
