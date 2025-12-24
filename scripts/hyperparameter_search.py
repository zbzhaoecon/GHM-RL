"""
Hyperparameter search script for Monte Carlo Policy Gradient on GHM Model.

This script performs hyperparameter search by running multiple training jobs
with different hyperparameter configurations and tracking the best results.

Supports:
- Random search: Randomly sample hyperparameter combinations
- Grid search: Exhaustive search over discrete values
- Multiple search strategies for efficient exploration

Usage:
    # Random search with 20 trials
    python scripts/hyperparameter_search.py --search_type random --n_trials 20 --n_iterations 5000

    # Grid search over specific hyperparameters
    python scripts/hyperparameter_search.py --search_type grid --n_iterations 5000

    # Resume a previous search
    python scripts/hyperparameter_search.py --resume results/hyperparam_search/search_results.json

    # Parallel execution with 4 workers
    python scripts/hyperparameter_search.py --search_type random --n_trials 20 --n_workers 4
"""

import argparse
import json
import os
import sys
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import itertools


@dataclass
class SearchSpace:
    """Define the hyperparameter search space."""

    # Learning rates
    lr_policy: List[float] = None
    lr_baseline: List[float] = None

    # Regularization
    entropy_weight: List[float] = None
    action_reg_weight: List[float] = None
    max_grad_norm: List[float] = None

    # Training parameters
    n_trajectories: List[int] = None

    # Network architecture
    policy_hidden: List[tuple] = None
    value_hidden: List[tuple] = None

    def __post_init__(self):
        """Set default search space if not provided."""
        if self.lr_policy is None:
            self.lr_policy = [1e-4, 3e-4, 1e-3]
        if self.lr_baseline is None:
            self.lr_baseline = [3e-4, 1e-3, 3e-3]
        if self.entropy_weight is None:
            self.entropy_weight = [0.01, 0.05, 0.1]
        if self.action_reg_weight is None:
            self.action_reg_weight = [0.0, 0.01, 0.05]
        if self.max_grad_norm is None:
            self.max_grad_norm = [0.5, 1.0, 2.0]
        if self.n_trajectories is None:
            self.n_trajectories = [250, 500, 1000]
        if self.policy_hidden is None:
            self.policy_hidden = [(32, 32), (64, 64), (128, 128), (64, 64, 64)]
        if self.value_hidden is None:
            self.value_hidden = [(32, 32), (64, 64), (128, 128), (64, 64, 64)]

    def sample_random(self) -> Dict[str, Any]:
        """Sample a random hyperparameter configuration."""
        # Use random.choice from Python's random module for lists/tuples
        import random

        config = {
            'lr_policy': float(np.random.choice(self.lr_policy)),
            'lr_baseline': float(np.random.choice(self.lr_baseline)),
            'entropy_weight': float(np.random.choice(self.entropy_weight)),
            'action_reg_weight': float(np.random.choice(self.action_reg_weight)),
            'max_grad_norm': float(np.random.choice(self.max_grad_norm)),
            'n_trajectories': int(np.random.choice(self.n_trajectories)),
            'policy_hidden': random.choice(self.policy_hidden),
            'value_hidden': random.choice(self.value_hidden),
        }
        return config

    def grid_configs(self) -> List[Dict[str, Any]]:
        """Generate all grid search configurations."""
        # For grid search, we'll use a smaller subset to avoid combinatorial explosion
        # Users can customize this by editing the search space

        keys = ['lr_policy', 'lr_baseline', 'entropy_weight', 'action_reg_weight',
                'max_grad_norm', 'n_trajectories', 'policy_hidden', 'value_hidden']

        # Get all combinations
        values = [getattr(self, key) for key in keys]
        all_combinations = list(itertools.product(*values))

        configs = []
        for combo in all_combinations:
            config = {key: val for key, val in zip(keys, combo)}
            # Convert numpy int to Python int
            if 'n_trajectories' in config:
                config['n_trajectories'] = int(config['n_trajectories'])
            configs.append(config)

        return configs


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""
    search_type: str = "random"  # "random" or "grid"
    n_trials: int = 20  # Number of trials for random search
    n_iterations: int = 5000  # Training iterations per trial
    n_workers: int = 1  # Number of parallel workers

    # Fixed training parameters (not searched)
    dt: float = 0.01
    T: float = 10.0
    seed_base: int = 123  # Base seed (will increment for each trial)
    device: str = "cpu"

    # Logging
    results_dir: str = "results/hyperparam_search"
    log_freq: int = 100
    eval_freq: int = 500
    ckpt_freq: int = 5000

    # Resume
    resume: Optional[str] = None


class HyperparameterSearch:
    """Hyperparameter search orchestrator."""

    def __init__(self, config: SearchConfig, search_space: SearchSpace):
        self.config = config
        self.search_space = search_space

        # Create results directory
        os.makedirs(config.results_dir, exist_ok=True)

        # Results storage
        self.results = []
        self.best_config = None
        self.best_return = -float('inf')

        # Results file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_file = os.path.join(
            config.results_dir,
            f"search_results_{timestamp}.json"
        )

        # If resuming, load previous results
        if config.resume:
            self.load_results(config.resume)

    def run(self):
        """Run hyperparameter search."""
        print("=" * 80)
        print("Hyperparameter Search for Monte Carlo Policy Gradient")
        print("=" * 80)
        print(f"Search type: {self.config.search_type}")
        print(f"Results directory: {self.config.results_dir}")
        print(f"Training iterations per trial: {self.config.n_iterations}")
        print("=" * 80)

        # Generate configurations to search
        if self.config.search_type == "random":
            configs = self.generate_random_configs()
        elif self.config.search_type == "grid":
            configs = self.search_space.grid_configs()
        else:
            raise ValueError(f"Unknown search_type: {self.config.search_type}")

        print(f"\nTotal configurations to evaluate: {len(configs)}")
        print(f"Number of workers: {self.config.n_workers}")
        print()

        # Run trials
        for trial_idx, hyperparam_config in enumerate(configs):
            trial_num = len(self.results) + 1

            print(f"\n{'=' * 80}")
            print(f"Trial {trial_num}/{len(configs)}")
            print(f"{'=' * 80}")
            print("Hyperparameters:")
            for key, value in hyperparam_config.items():
                print(f"  {key}: {value}")
            print()

            # Run training with this configuration
            result = self.run_trial(trial_num, hyperparam_config)

            # Store results
            self.results.append({
                'trial_num': trial_num,
                'hyperparameters': hyperparam_config,
                'metrics': result,
            })

            # Update best configuration
            if result['best_return'] > self.best_return:
                self.best_return = result['best_return']
                self.best_config = hyperparam_config.copy()
                print(f"\n🏆 New best configuration found!")
                print(f"   Best return: {self.best_return:.4f}")

            # Save results after each trial
            self.save_results()

            print(f"\n{'=' * 80}")
            print(f"Trial {trial_num} complete")
            print(f"Best return so far: {self.best_return:.4f}")
            print(f"{'=' * 80}\n")

        # Print final summary
        self.print_summary()

    def generate_random_configs(self) -> List[Dict[str, Any]]:
        """Generate random hyperparameter configurations."""
        configs = []
        for _ in range(self.config.n_trials):
            config = self.search_space.sample_random()
            configs.append(config)
        return configs

    def run_trial(self, trial_num: int, hyperparam_config: Dict[str, Any]) -> Dict[str, float]:
        """Run a single training trial with given hyperparameters."""
        # Create trial-specific directories
        trial_dir = os.path.join(self.config.results_dir, f"trial_{trial_num:03d}")
        log_dir = os.path.join(trial_dir, "logs")
        ckpt_dir = os.path.join(trial_dir, "checkpoints")

        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Build command to run training script
        cmd = [
            sys.executable,  # Use same Python interpreter
            "scripts/train_monte_carlo_ghm_time_augmented.py",
            "--n_iterations", str(self.config.n_iterations),
            "--dt", str(self.config.dt),
            "--T", str(self.config.T),
            "--seed", str(self.config.seed_base + trial_num),
            "--device", str(self.config.device),
            "--log_dir", log_dir,
            "--log_freq", str(self.config.log_freq),
            "--eval_freq", str(self.config.eval_freq),
            "--ckpt_freq", str(self.config.ckpt_freq),
            "--ckpt_dir", ckpt_dir,
        ]

        # Add hyperparameters to command
        cmd.extend(["--lr_policy", str(hyperparam_config['lr_policy'])])
        cmd.extend(["--lr_baseline", str(hyperparam_config['lr_baseline'])])
        cmd.extend(["--entropy_weight", str(hyperparam_config['entropy_weight'])])
        cmd.extend(["--action_reg_weight", str(hyperparam_config['action_reg_weight'])])
        cmd.extend(["--max_grad_norm", str(hyperparam_config['max_grad_norm'])])
        cmd.extend(["--n_trajectories", str(hyperparam_config['n_trajectories'])])

        # Add network architecture
        policy_hidden = hyperparam_config['policy_hidden']
        value_hidden = hyperparam_config['value_hidden']
        cmd.extend(["--policy_hidden"] + [str(x) for x in policy_hidden])
        cmd.extend(["--value_hidden"] + [str(x) for x in value_hidden])

        # Save hyperparameter config to trial directory
        config_path = os.path.join(trial_dir, "hyperparameters.json")
        with open(config_path, 'w') as f:
            # Convert tuples to lists for JSON serialization
            config_serializable = {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in hyperparam_config.items()
            }
            json.dump(config_serializable, f, indent=2)

        # Run training
        print(f"Running command:")
        print(" ".join(cmd))
        print()

        stdout_log = os.path.join(trial_dir, "stdout.log")
        stderr_log = os.path.join(trial_dir, "stderr.log")

        with open(stdout_log, 'w') as fout, open(stderr_log, 'w') as ferr:
            process = subprocess.run(
                cmd,
                stdout=fout,
                stderr=ferr,
                cwd=os.getcwd(),
            )

        # Check if training succeeded
        if process.returncode != 0:
            print(f"⚠️  Training failed with return code {process.returncode}")
            print(f"   Check logs: {stderr_log}")
            return {
                'best_return': -float('inf'),
                'final_return': -float('inf'),
                'success': False,
            }

        # Extract best return from checkpoint
        best_model_path = os.path.join(ckpt_dir, "best_model.pt")
        final_model_path = os.path.join(ckpt_dir, "final_model.pt")

        best_return = -float('inf')
        final_return = -float('inf')

        try:
            import torch
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location='cpu')
                best_return = checkpoint.get('best_return', -float('inf'))

            if os.path.exists(final_model_path):
                checkpoint = torch.load(final_model_path, map_location='cpu')
                final_return = checkpoint.get('best_return', -float('inf'))
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")

        return {
            'best_return': best_return,
            'final_return': final_return,
            'success': True,
            'stdout_log': stdout_log,
            'stderr_log': stderr_log,
            'checkpoint_dir': ckpt_dir,
        }

    def save_results(self):
        """Save search results to JSON file."""
        results_data = {
            'search_config': asdict(self.config),
            'search_space': asdict(self.search_space),
            'best_config': self.best_config,
            'best_return': self.best_return,
            'results': self.results,
            'timestamp': datetime.now().isoformat(),
        }

        # Convert tuples to lists for JSON serialization
        def convert_tuples(obj):
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tuples(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                return obj

        results_data = convert_tuples(results_data)

        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\n💾 Results saved to: {self.results_file}")

    def load_results(self, results_path: str):
        """Load previous search results."""
        print(f"Loading previous results from: {results_path}")
        with open(results_path, 'r') as f:
            results_data = json.load(f)

        self.results = results_data.get('results', [])
        self.best_config = results_data.get('best_config')
        self.best_return = results_data.get('best_return', -float('inf'))

        print(f"Loaded {len(self.results)} previous trials")
        print(f"Best return so far: {self.best_return:.4f}")

    def print_summary(self):
        """Print summary of search results."""
        print("\n" + "=" * 80)
        print("HYPERPARAMETER SEARCH SUMMARY")
        print("=" * 80)
        print(f"Total trials: {len(self.results)}")
        print(f"Best return: {self.best_return:.4f}")
        print("\nBest hyperparameters:")
        for key, value in self.best_config.items():
            print(f"  {key}: {value}")

        # Print top 5 configurations
        print("\n" + "-" * 80)
        print("Top 5 Configurations:")
        print("-" * 80)

        sorted_results = sorted(
            self.results,
            key=lambda x: x['metrics'].get('best_return', -float('inf')),
            reverse=True
        )

        for i, result in enumerate(sorted_results[:5]):
            print(f"\n{i+1}. Trial {result['trial_num']}")
            print(f"   Return: {result['metrics'].get('best_return', -float('inf')):.4f}")
            print(f"   Hyperparameters:")
            for key, value in result['hyperparameters'].items():
                print(f"     {key}: {value}")

        print("\n" + "=" * 80)
        print(f"Results saved to: {self.results_file}")
        print("=" * 80)


def parse_args() -> SearchConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for Monte Carlo Policy Gradient"
    )

    # Search configuration
    parser.add_argument("--search_type", type=str, default="random",
                       choices=["random", "grid"],
                       help="Type of search: random or grid")
    parser.add_argument("--n_trials", type=int, default=20,
                       help="Number of trials for random search")
    parser.add_argument("--n_iterations", type=int, default=5000,
                       help="Training iterations per trial")
    parser.add_argument("--n_workers", type=int, default=1,
                       help="Number of parallel workers (not yet implemented)")

    # Fixed training parameters
    parser.add_argument("--dt", type=float, default=0.01,
                       help="Time discretization")
    parser.add_argument("--T", type=float, default=10.0,
                       help="Episode horizon")
    parser.add_argument("--seed_base", type=int, default=123,
                       help="Base random seed")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cuda/cpu)")

    # Logging
    parser.add_argument("--results_dir", type=str, default="results/hyperparam_search",
                       help="Directory to save results")
    parser.add_argument("--log_freq", type=int, default=100,
                       help="Logging frequency")
    parser.add_argument("--eval_freq", type=int, default=500,
                       help="Evaluation frequency")
    parser.add_argument("--ckpt_freq", type=int, default=5000,
                       help="Checkpoint frequency")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to previous results to resume from")

    args = parser.parse_args()

    config = SearchConfig(
        search_type=args.search_type,
        n_trials=args.n_trials,
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

    return config


def main():
    """Main entry point."""
    # Parse configuration
    config = parse_args()

    # Create search space (using defaults)
    search_space = SearchSpace()

    # Create and run search
    search = HyperparameterSearch(config, search_space)
    search.run()


if __name__ == "__main__":
    main()
