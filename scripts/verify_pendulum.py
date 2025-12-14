"""
Verify Pendulum training results.

This script checks:
1. Model file exists and can be loaded
2. Evaluation shows improved performance
3. TensorBoard logs were created

Usage:
    python scripts/verify_pendulum.py --model models/pendulum_debug/final_model
"""

import argparse
from pathlib import Path
import numpy as np


def check_model_exists(model_path: Path) -> bool:
    """Check if model file exists."""
    print("=" * 50)
    print("Test 1: Model File Exists")
    print("=" * 50)

    zip_path = Path(str(model_path) + ".zip")
    exists = zip_path.exists()

    if exists:
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model found: {zip_path}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Model not found: {zip_path}")
        return False


def check_model_loadable(model_path: Path) -> bool:
    """Check if model can be loaded."""
    print("\n" + "=" * 50)
    print("Test 2: Model Can Be Loaded")
    print("=" * 50)

    try:
        from stable_baselines3 import SAC
        model = SAC.load(model_path)
        print("✓ Model loaded successfully")
        print(f"  Policy type: {type(model.policy).__name__}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def check_logs_exist(output_dir: Path) -> bool:
    """Check if training logs exist."""
    print("\n" + "=" * 50)
    print("Test 3: Logs Exist")
    print("=" * 50)

    logs_dir = output_dir / "logs"
    tb_dir = output_dir / "tensorboard"

    logs_exist = logs_dir.exists()
    tb_exist = tb_dir.exists()

    print(f"{'✓' if logs_exist else '✗'} Evaluation logs: {logs_dir}")
    print(f"{'✓' if tb_exist else '✗'} TensorBoard logs: {tb_dir}")

    if tb_exist:
        tb_files = list(tb_dir.rglob("events.out.tfevents.*"))
        print(f"  TensorBoard events: {len(tb_files)} files")

    return logs_exist and tb_exist


def evaluate_model(model_path: Path, n_episodes: int = 10) -> bool:
    """Evaluate the trained model."""
    print("\n" + "=" * 50)
    print("Test 4: Model Performance")
    print("=" * 50)

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_util import make_vec_env

        # Load model and environment
        model = SAC.load(model_path)
        env = make_vec_env("Pendulum-v1", n_envs=1)

        # Run episodes
        rewards = []
        for ep in range(n_episodes):
            obs = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < 200:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                steps += 1

            rewards.append(total_reward)

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        print(f"  Episodes: {n_episodes}")
        print(f"  Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")

        # Pendulum-v1 baseline: random ~-1500, good policy ~-200
        if mean_reward > -500:
            print("✓ Performance is GOOD (reward > -500)")
            success = True
        elif mean_reward > -1000:
            print("⚠ Performance is FAIR (reward > -1000)")
            success = True
        else:
            print("✗ Performance is POOR (reward < -1000)")
            success = False

        return success

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify Pendulum training")
    parser.add_argument("--model", type=str, default="models/pendulum_debug/final_model")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = model_path.parent

    print("\n" + "=" * 50)
    print("PENDULUM TRAINING VERIFICATION")
    print("=" * 50 + "\n")

    # Run all checks
    results = []
    results.append(("Model exists", check_model_exists(model_path)))
    results.append(("Model loadable", check_model_loadable(model_path)))
    results.append(("Logs exist", check_logs_exist(output_dir)))
    results.append(("Performance", evaluate_model(model_path, args.episodes)))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Infrastructure validated!")
        print("\nNext steps:")
        print("  1. Run: python scripts/debug_env.py")
        print("  2. Run: python scripts/train_ghm.py")
    else:
        print("✗ SOME CHECKS FAILED - Review above for issues")
    print("=" * 50 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
