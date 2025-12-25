"""
Test to analyze the impact of liquidation value on policy learning.

This test runs training with different liquidation values to see
how it affects the learned policy and bankruptcy rate.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from macro_rl.dynamics.ghm_equity import GHMEquityParams


def analyze_liquidation_values():
    """
    Analyze different liquidation value settings.
    """

    print("=" * 80)
    print("ANALYSIS: Liquidation Value Impact")
    print("=" * 80)

    # Default parameters
    base_params = GHMEquityParams()

    print("\nDefault parameters:")
    print(f"  omega (recovery rate): {base_params.omega}")
    print(f"  alpha (cash flow): {base_params.alpha}")
    print(f"  r (discount rate): {base_params.r}")
    print(f"  mu (drift): {base_params.mu}")

    # Compute default liquidation value
    default_liq = base_params.liquidation_value
    print(f"\n  => Liquidation value: {default_liq:.4f}")

    # Analyze what this means
    print("\n" + "=" * 80)
    print("ECONOMIC INTERPRETATION")
    print("=" * 80)

    # Sustainable dividend rate (no growth, no issuance)
    # At steady state: dc/dt = alpha - a_L = 0
    # So: a_L = alpha
    sustainable_dividend_rate = base_params.alpha

    # Present value of sustainable dividends (perpetuity)
    # PV = a_L / r
    sustainable_pv = sustainable_dividend_rate / base_params.r

    print(f"\nSustainable dividend analysis:")
    print(f"  Sustainable dividend rate: {sustainable_dividend_rate:.4f}")
    print(f"  PV of perpetual dividends (a_L/r): {sustainable_pv:.4f}")

    print(f"\nLiquidation value: {default_liq:.4f}")
    print(f"Ratio (liquidation / sustainable PV): {default_liq / sustainable_pv:.4f}")

    print("\n⚠️  PROBLEM:")
    if default_liq > sustainable_pv * 0.5:
        print(f"  Liquidation value ({default_liq:.4f}) is too high!")
        print(f"  It's {default_liq / sustainable_pv * 100:.1f}% of the sustainable firm value.")
        print(f"  This makes bankruptcy attractive compared to long-term operations.")

    # Test different liquidation values
    print("\n" + "=" * 80)
    print("RECOMMENDED LIQUIDATION VALUES")
    print("=" * 80)

    test_values = [0.0, 0.1, 0.5, 1.0, 2.0, default_liq]

    print(f"\n{'Liquidation Value':<20} {'vs Sustainable PV':<20} {'Recommendation':<30}")
    print("-" * 70)

    for liq_val in test_values:
        ratio = liq_val / sustainable_pv if sustainable_pv > 0 else 0

        if liq_val < 0.01:
            rec = "✅ BEST: No bankruptcy incentive"
        elif liq_val < sustainable_pv * 0.1:
            rec = "✅ GOOD: Small recovery"
        elif liq_val < sustainable_pv * 0.3:
            rec = "⚠️  OK: Moderate recovery"
        else:
            rec = "❌ BAD: Too high, incentivizes bankruptcy"

        print(f"{liq_val:<20.4f} {ratio:<20.2%} {rec:<30}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("\nFor realistic firm dynamics:")
    print("  1. Set liquidation_value = 0.0 (firms get nothing in bankruptcy)")
    print("     - Most conservative")
    print("     - Removes bankruptcy incentive")
    print("     - Realistic for equity holders (they're last in line)")
    print()
    print("  2. Or set liquidation_value = 0.1 to 0.5 (small recovery)")
    print("     - More realistic for some scenarios")
    print("     - Still discourages bankruptcy")
    print()
    print("  3. NEVER use the current default (4.95)")
    print("     - This is 55% of sustainable firm value")
    print("     - Makes bankruptcy optimal!")

    print("\n" + "=" * 80)
    print("HOW TO FIX")
    print("=" * 80)
    print("\nEdit: macro_rl/dynamics/ghm_equity.py")
    print()
    print("Change line 52 from:")
    print("  self.liquidation_value = self.omega * self.alpha / (self.r - self.mu)")
    print()
    print("To:")
    print("  self.liquidation_value = 0.0  # No recovery in bankruptcy")
    print()
    print("Or:")
    print("  self.liquidation_value = 0.1  # Small fixed recovery")

    return default_liq, sustainable_pv


if __name__ == "__main__":
    default_liq, sustainable_pv = analyze_liquidation_values()

    if default_liq > sustainable_pv * 0.3:
        print("\n⚠️  WARNING: Current liquidation value is too high!")
        print("This will cause the agent to learn bankruptcy-seeking policies.")
        sys.exit(1)
    else:
        print("\n✅ Liquidation value is reasonable.")
        sys.exit(0)
