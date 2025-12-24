# Hyperparameter Search for Monte Carlo Policy Gradient

This document explains how to use the hyperparameter search script to find optimal hyperparameters for training the Monte Carlo Policy Gradient solver on the GHM equity management model.

## Overview

The `hyperparameter_search.py` script automates the process of training multiple models with different hyperparameter configurations to identify the best performing combination. It supports two search strategies:

- **Random Search**: Randomly samples hyperparameter combinations from a predefined search space
- **Grid Search**: Exhaustively evaluates all combinations in a discrete grid

## Quick Start

### Random Search (Recommended)

Run 20 trials with random hyperparameter combinations:

```bash
python scripts/hyperparameter_search.py --search_type random --n_trials 20 --n_iterations 5000
```

### Grid Search

Run exhaustive search over all hyperparameter combinations:

```bash
python scripts/hyperparameter_search.py --search_type grid --n_iterations 5000
```

**⚠️ Warning**: Grid search can result in a very large number of trials. With the default search space, this could be hundreds or thousands of configurations.

## Command Line Arguments

### Search Configuration

- `--search_type`: Type of search (`random` or `grid`). Default: `random`
- `--n_trials`: Number of trials for random search. Default: `20`
- `--n_iterations`: Training iterations per trial. Default: `5000`
- `--n_workers`: Number of parallel workers (not yet implemented). Default: `1`

### Training Parameters

- `--dt`: Time discretization. Default: `0.01`
- `--T`: Episode horizon. Default: `10.0`
- `--seed_base`: Base random seed (increments for each trial). Default: `123`
- `--device`: Device to use (`cpu` or `cuda`). Default: `cpu`

### Logging and Results

- `--results_dir`: Directory to save search results. Default: `results/hyperparam_search`
- `--log_freq`: Logging frequency (iterations). Default: `100`
- `--eval_freq`: Evaluation frequency (iterations). Default: `500`
- `--ckpt_freq`: Checkpoint save frequency (iterations). Default: `5000`

### Resume Search

- `--resume`: Path to previous results JSON to continue a search

## Search Space

The default search space covers the following hyperparameters:

### Learning Rates
- **lr_policy**: `[1e-4, 3e-4, 1e-3]`
- **lr_baseline**: `[3e-4, 1e-3, 3e-3]`

### Regularization
- **entropy_weight**: `[0.01, 0.05, 0.1]`
- **action_reg_weight**: `[0.0, 0.01, 0.05]`
- **max_grad_norm**: `[0.5, 1.0, 2.0]`

### Training Parameters
- **n_trajectories**: `[250, 500, 1000]`

### Network Architecture
- **policy_hidden**: `[(32, 32), (64, 64), (128, 128), (64, 64, 64)]`
- **value_hidden**: `[(32, 32), (64, 64), (128, 128), (64, 64, 64)]`

To customize the search space, edit the `SearchSpace.__post_init__()` method in the script.

## Output

### Directory Structure

The script creates the following directory structure:

```
results/hyperparam_search/
├── search_results_TIMESTAMP.json  # Summary of all trials and best config
├── trial_001/
│   ├── hyperparameters.json       # Hyperparameters for this trial
│   ├── stdout.log                 # Training output
│   ├── stderr.log                 # Error output
│   ├── logs/                      # TensorBoard logs
│   └── checkpoints/               # Model checkpoints
│       ├── best_model.pt
│       └── final_model.pt
├── trial_002/
│   └── ...
└── ...
```

### Results JSON

The `search_results_TIMESTAMP.json` file contains:

- **search_config**: Configuration used for the search
- **search_space**: Hyperparameter search space
- **best_config**: Best hyperparameter configuration found
- **best_return**: Best return achieved
- **results**: List of all trials with their hyperparameters and metrics

## Examples

### Example 1: Quick Random Search

Run a quick search with 10 trials and shorter training:

```bash
python scripts/hyperparameter_search.py \
    --search_type random \
    --n_trials 10 \
    --n_iterations 3000 \
    --device cuda
```

### Example 2: Focused Search on Learning Rates

To focus only on learning rates, modify the `SearchSpace` class to set other hyperparameters to single values:

```python
# In SearchSpace.__post_init__()
self.lr_policy = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
self.lr_baseline = [3e-4, 1e-3, 3e-3, 1e-2]
self.entropy_weight = [0.05]  # Fixed
self.action_reg_weight = [0.01]  # Fixed
# ... etc
```

### Example 3: Resume a Search

If a search was interrupted, resume from the last saved results:

```bash
python scripts/hyperparameter_search.py \
    --resume results/hyperparam_search/search_results_20231215-143022.json \
    --n_trials 30
```

### Example 4: Grid Search with Reduced Space

For a manageable grid search, create a custom search space with fewer options:

```python
# Minimal grid search
search_space = SearchSpace()
search_space.lr_policy = [3e-4, 1e-3]
search_space.lr_baseline = [1e-3, 3e-3]
search_space.entropy_weight = [0.05]
search_space.action_reg_weight = [0.01]
search_space.max_grad_norm = [0.5]
search_space.n_trajectories = [500]
search_space.policy_hidden = [(64, 64)]
search_space.value_hidden = [(64, 64)]
```

This would result in 2 × 2 = 4 trials.

## Analyzing Results

### View Best Configuration

The script prints a summary at the end showing:
- Best hyperparameter configuration
- Top 5 configurations by performance
- Path to detailed results JSON

### Load Results Programmatically

```python
import json

with open('results/hyperparam_search/search_results_TIMESTAMP.json', 'r') as f:
    results = json.load(f)

print("Best hyperparameters:")
print(results['best_config'])

print(f"\nBest return: {results['best_return']:.4f}")

# Analyze all trials
for trial in results['results']:
    trial_num = trial['trial_num']
    best_return = trial['metrics']['best_return']
    print(f"Trial {trial_num}: {best_return:.4f}")
```

### Visualize with TensorBoard

Each trial saves TensorBoard logs. Compare multiple trials:

```bash
tensorboard --logdir results/hyperparam_search/
```

## Tips for Effective Search

1. **Start with Random Search**: Random search is often more efficient than grid search, especially with many hyperparameters.

2. **Use Shorter Training Initially**: Run a first pass with `--n_iterations 3000` to quickly identify promising regions, then do a refined search with longer training.

3. **Monitor Progress**: Check the `search_results_*.json` file periodically to see if the search is finding good configurations.

4. **Focus on Important Hyperparameters**: Learning rates and regularization weights typically have the largest impact.

5. **Check for Variance**: If results vary widely across seeds, increase `n_iterations` or average across multiple seeds.

6. **Use GPU if Available**: Add `--device cuda` to significantly speed up training.

## Troubleshooting

### Out of Memory
- Reduce `n_trajectories`
- Reduce network size (smaller `policy_hidden` and `value_hidden`)
- Use smaller batch sizes

### Training Instability
- Search for lower learning rates
- Increase `max_grad_norm` for more aggressive clipping
- Increase `entropy_weight` for more exploration

### Poor Performance Across All Configs
- Increase `n_iterations` - the model may need more training
- Check that the base training script works correctly
- Verify reward function and dynamics are correctly specified

## Next Steps

After finding the best hyperparameters:

1. **Verify Results**: Run the best configuration multiple times with different seeds to ensure consistency

2. **Longer Training**: Train with the best hyperparameters for more iterations to see final performance

3. **Fine-tune**: Do a focused search around the best configuration

Example verification run:

```bash
python scripts/train_monte_carlo_ghm_time_augmented.py \
    --lr_policy 3e-4 \
    --lr_baseline 1e-3 \
    --entropy_weight 0.05 \
    --action_reg_weight 0.01 \
    --max_grad_norm 0.5 \
    --n_trajectories 500 \
    --policy_hidden 64 64 \
    --value_hidden 64 64 \
    --n_iterations 20000 \
    --seed 999
```
