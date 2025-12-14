"""
Simple validation script for envs module structure.

This script validates the module structure and imports without
requiring full dependency installation.

Phase 3 implementation.
"""

import sys
import os

def validate_structure():
    """Validate that all files exist."""
    print("Validating envs module structure...")

    required_files = [
        "macro_rl/envs/__init__.py",
        "macro_rl/envs/base.py",
        "macro_rl/envs/ghm_equity_env.py",
        "macro_rl/envs/README.md",
        "tests/test_envs.py",
        "examples/run_ghm_env.py",
    ]

    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (MISSING)")
            all_exist = False

    return all_exist


def check_imports():
    """Check that imports work (if dependencies available)."""
    print("\nChecking imports...")

    try:
        import gymnasium
        print("  ✓ gymnasium")
    except ImportError:
        print("  ✗ gymnasium (not installed)")
        return False

    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy (not installed)")
        return False

    try:
        import torch
        print("  ✓ torch")
    except ImportError:
        print("  ⚠ torch (not installed - required for full testing)")
        print("    Tests will fail without torch")
        return False

    return True


def validate_code_structure():
    """Validate code structure without importing."""
    print("\nValidating code structure...")

    # Check base.py
    with open("macro_rl/envs/base.py") as f:
        base_content = f.read()
        checks = [
            ("ContinuousTimeEnv class", "class ContinuousTimeEnv"),
            ("reset method", "def reset"),
            ("step method", "def step"),
            ("_apply_action_and_evolve", "def _apply_action_and_evolve"),
            ("_get_terminated", "def _get_terminated"),
        ]

        for name, pattern in checks:
            if pattern in base_content:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name} (MISSING)")

    # Check ghm_equity_env.py
    with open("macro_rl/envs/ghm_equity_env.py") as f:
        ghm_content = f.read()
        checks = [
            ("GHMEquityEnv class", "class GHMEquityEnv"),
            ("_sample_initial_state", "def _sample_initial_state"),
            ("_apply_action_and_evolve", "def _apply_action_and_evolve"),
            ("_get_terminated", "def _get_terminated"),
            ("_get_terminal_reward", "def _get_terminal_reward"),
        ]

        for name, pattern in checks:
            if pattern in ghm_content:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name} (MISSING)")

    # Check __init__.py
    with open("macro_rl/envs/__init__.py") as f:
        init_content = f.read()
        checks = [
            ("ContinuousTimeEnv export", "ContinuousTimeEnv"),
            ("GHMEquityEnv export", "GHMEquityEnv"),
            ("Gymnasium registration", "register"),
        ]

        for name, pattern in checks:
            if pattern in init_content:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name} (MISSING)")


def main():
    """Run all validations."""
    print("=" * 60)
    print("Phase 3: Environments Module Validation")
    print("=" * 60 + "\n")

    structure_ok = validate_structure()
    deps_ok = check_imports()
    validate_code_structure()

    print("\n" + "=" * 60)
    if structure_ok:
        print("✓ Module structure is complete")
    else:
        print("✗ Module structure has issues")

    if deps_ok:
        print("✓ All dependencies are installed")
        print("\nTo run tests:")
        print("  pytest tests/test_envs.py -v")
        print("\nTo run examples:")
        print("  python examples/run_ghm_env.py")
    else:
        print("⚠ Some dependencies are missing")
        print("\nTo install dependencies:")
        print("  pip install torch gymnasium")
        print("\nFor RL training (optional):")
        print("  pip install stable-baselines3")

    print("=" * 60)


if __name__ == "__main__":
    main()
