"""
Efficient hyperparameter search with resource-aware configuration.

This is an optimized version for systems with sufficient GPU/RAM to run
many trials in parallel (e.g., 10-12 workers).

Key improvements:
- Smart grid search focusing on most impactful hyperparameters
- Automatic resource-based worker count recommendation
- Focused search spaces for faster iteration
"""

import sys
import os

# Import the main search infrastructure
sys.path.insert(0, os.path.dirname(__file__))
from hyperparameter_search import (
    SearchSpace, SearchConfig, HyperparameterSearch, main as original_main
)


class EfficientSearchSpace(SearchSpace):
    """
    Efficient search space focused on most impactful hyperparameters.

    Strategy:
    1. Focus on learning rates (highest impact)
    2. Limited regularization values
    3. Fixed network architecture (for speed)
    """

    def __post_init__(self):
        """Define focused search space."""
        # Learning rates - MOST IMPORTANT
        self.lr_policy = [1e-4, 3e-4, 5e-4, 1e-3]
        self.lr_baseline = [3e-4, 1e-3, 3e-3]

        # Regularization - IMPORTANT
        self.entropy_weight = [0.01, 0.05, 0.1]
        self.action_reg_weight = [0.0, 0.01]
        self.max_grad_norm = [0.5, 1.0]

        # Training parameters
        self.n_trajectories = [500]  # Fixed for consistent comparison

        # Network architecture - FIXED for efficiency
        self.policy_hidden = [(64, 64)]  # Single best architecture
        self.value_hidden = [(64, 64)]


class FocusedSearchSpace(SearchSpace):
    """
    Very focused search - only learning rates and key regularization.

    Total configurations: 4 × 3 × 2 = 24 trials
    With 10 workers: ~2.4 batches, very fast!
    """

    def __post_init__(self):
        # Only tune the most critical hyperparameters
        self.lr_policy = [1e-4, 3e-4, 5e-4, 1e-3]
        self.lr_baseline = [3e-4, 1e-3, 3e-3]
        self.entropy_weight = [0.05, 0.1]

        # Everything else fixed
        self.action_reg_weight = [0.01]
        self.max_grad_norm = [0.5]
        self.n_trajectories = [500]
        self.policy_hidden = [(64, 64)]
        self.value_hidden = [(64, 64)]


class ComprehensiveSearchSpace(SearchSpace):
    """
    Comprehensive but still efficient search.

    Total configurations: 4 × 3 × 3 × 2 × 2 × 2 × 3 = 864 trials
    With 10 workers: ~86 batches
    With n_iterations=3000: ~3-4 hours total
    """

    def __post_init__(self):
        # Learning rates
        self.lr_policy = [1e-4, 3e-4, 5e-4, 1e-3]
        self.lr_baseline = [3e-4, 1e-3, 3e-3]

        # Regularization
        self.entropy_weight = [0.01, 0.05, 0.1]
        self.action_reg_weight = [0.0, 0.01]
        self.max_grad_norm = [0.5, 1.0]

        # Training
        self.n_trajectories = [250, 500]

        # Network architecture - limited options
        self.policy_hidden = [(32, 32), (64, 64), (128, 128)]
        self.value_hidden = [(32, 32), (64, 64), (128, 128)]


def estimate_resources(search_space: SearchSpace, n_iterations: int, n_workers: int):
    """Estimate time and resources for a search."""
    if search_space.__class__.__name__ == "FocusedSearchSpace":
        total_configs = 24
    elif search_space.__class__.__name__ == "EfficientSearchSpace":
        total_configs = 4 * 3 * 3 * 2 * 2  # 144
    elif search_space.__class__.__name__ == "ComprehensiveSearchSpace":
        total_configs = 4 * 3 * 3 * 2 * 2 * 2 * 3  # 864
    else:
        # Calculate from actual search space
        from itertools import product
        total_configs = len(list(product(
            search_space.lr_policy,
            search_space.lr_baseline,
            search_space.entropy_weight,
            search_space.action_reg_weight,
            search_space.max_grad_norm,
            search_space.n_trajectories,
            search_space.policy_hidden,
            search_space.value_hidden,
        )))

    # Rough estimates
    seconds_per_trial = n_iterations * 0.5  # ~0.5 seconds per iteration with GPU
    batches = (total_configs + n_workers - 1) // n_workers
    total_seconds = batches * seconds_per_trial

    hours = total_seconds / 3600

    print("\n" + "=" * 80)
    print("RESOURCE ESTIMATION")
    print("=" * 80)
    print(f"Total configurations: {total_configs}")
    print(f"Parallel workers: {n_workers}")
    print(f"Batches: {batches}")
    print(f"Iterations per trial: {n_iterations}")
    print(f"\nEstimated time per trial: {seconds_per_trial / 60:.1f} minutes")
    print(f"Estimated total time: {hours:.1f} hours")
    print(f"\nResource usage (peak):")
    print(f"  GPU RAM: ~{n_workers * 1.2:.1f} GB")
    print(f"  System RAM: ~{n_workers * 4:.1f} GB")
    print("=" * 80 + "\n")


def main():
    """Main entry point with resource-aware recommendations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Efficient hyperparameter search with resource optimization"
    )

    # Search strategy
    parser.add_argument("--strategy", type=str, default="focused",
                       choices=["focused", "efficient", "comprehensive"],
                       help="Search strategy: focused (24 configs), efficient (144 configs), comprehensive (864 configs)")

    # Use parent parser for other arguments
    parser.add_argument("--n_iterations", type=int, default=5000,
                       help="Training iterations per trial")
    parser.add_argument("--n_workers", type=int, default=10,
                       help="Number of parallel workers (recommended: 10 for your resources)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda recommended)")
    parser.add_argument("--results_dir", type=str, default="results/hyperparam_search_efficient",
                       help="Results directory")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--seed_base", type=int, default=123)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--ckpt_freq", type=int, default=5000)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # Select search space based on strategy
    if args.strategy == "focused":
        search_space = FocusedSearchSpace()
        print("Using FOCUSED search space (24 configurations)")
        print("  - Focus: Learning rates + entropy weight")
        print("  - Fast iteration, good for initial exploration")
    elif args.strategy == "efficient":
        search_space = EfficientSearchSpace()
        print("Using EFFICIENT search space (144 configurations)")
        print("  - Focus: Learning rates + regularization")
        print("  - Balanced speed and coverage")
    else:
        search_space = ComprehensiveSearchSpace()
        print("Using COMPREHENSIVE search space (864 configurations)")
        print("  - All hyperparameters + network architectures")
        print("  - Thorough but slower")

    # Create config
    config = SearchConfig(
        search_type="grid",  # Always grid for efficient search
        n_trials=20,  # Not used for grid search
        n_iterations=args.n_iterations,
        n_workers=args.n_workers,
        dt=args.dt,
        T=args.T,
        seed_base=args.seed_base,
        device=args.device,
        results_dir=args.results_dir,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        ckpt_freq=args.ckpt_freq,
        resume=args.resume,
    )

    # Show resource estimates
    estimate_resources(search_space, args.n_iterations, args.n_workers)

    # Confirm before proceeding
    response = input("Proceed with search? [y/N]: ")
    if response.lower() != 'y':
        print("Search cancelled.")
        return

    # Run search
    search = HyperparameterSearch(config, search_space)
    search.run()


if __name__ == "__main__":
    main()
