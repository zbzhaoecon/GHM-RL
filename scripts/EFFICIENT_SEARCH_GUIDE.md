# Efficient Hyperparameter Search Guide

Based on your resources (50GB RAM, 15GB GPU), you can run **10 parallel workers** efficiently!

## Quick Start

### 1. Focused Search (FASTEST - Recommended First)
**24 configurations, ~30 minutes with 10 workers**

```bash
python scripts/hyperparameter_search_efficient.py \
    --strategy focused \
    --n_iterations 3000 \
    --n_workers 10 \
    --device cuda
```

**What it searches:**
- Learning rates: 4 × 3 = 12 combinations
- Entropy weight: 2 values
- **Total: 24 trials**

### 2. Efficient Search (BALANCED)
**144 configurations, ~3-4 hours with 10 workers**

```bash
python scripts/hyperparameter_search_efficient.py \
    --strategy efficient \
    --n_iterations 5000 \
    --n_workers 10 \
    --device cuda
```

**What it searches:**
- Learning rates: 4 × 3
- Regularization: 3 × 2 × 2
- **Total: 144 trials**

### 3. Comprehensive Search (THOROUGH)
**864 configurations, ~12-15 hours with 10 workers**

```bash
python scripts/hyperparameter_search_efficient.py \
    --strategy comprehensive \
    --n_iterations 5000 \
    --n_workers 10 \
    --device cuda
```

**What it searches:**
- Everything including network architectures
- **Total: 864 trials**

## Resource Utilization

With `--n_workers 10`:
- **GPU RAM**: ~12 GB (10 × 1.2 GB)
- **System RAM**: ~40 GB (10 × 4 GB)
- **Margin**: 3 GB GPU, 10 GB RAM (safe!)

## Recommended Strategy

### Stage 1: Quick Exploration (30 min)
```bash
python scripts/hyperparameter_search_efficient.py \
    --strategy focused \
    --n_iterations 3000 \
    --n_workers 10 \
    --device cuda
```

**Look at results:**
```bash
# View best configuration
cat results/hyperparam_search_efficient/search_results_*.json | grep -A 5 "best_config"
```

### Stage 2: Refined Search (3-4 hours)
Based on Stage 1 results, run efficient or comprehensive search:

```bash
python scripts/hyperparameter_search_efficient.py \
    --strategy efficient \
    --n_iterations 5000 \
    --n_workers 10 \
    --device cuda
```

### Stage 3: Final Training (if needed)
Train with best hyperparameters for more iterations:

```bash
python scripts/train_monte_carlo_ghm_time_augmented.py \
    --n_iterations 20000 \
    --lr_policy <best_value> \
    --lr_baseline <best_value> \
    --entropy_weight <best_value> \
    --n_workers 1 \
    --device cuda
```

## Monitoring Progress

### Watch GPU Usage
```bash
# In another terminal/cell
watch -n 1 nvidia-smi
```

You should see:
- 10 Python processes using GPU
- GPU memory: ~10-12 GB

### Check Progress
```bash
# Number of completed trials
ls results/hyperparam_search_efficient/trial_* -d | wc -l

# Best result so far
python -c "
import json
with open('results/hyperparam_search_efficient/search_results_<TAB>.json') as f:
    data = json.load(f)
    print(f'Best return: {data[\"best_return\"]:.4f}')
    print('Best config:', data['best_config'])
"
```

## Customizing Search Spaces

Edit `scripts/hyperparameter_search_efficient.py` to customize:

```python
class MySearchSpace(SearchSpace):
    def __post_init__(self):
        # Your custom search space
        self.lr_policy = [1e-4, 5e-4, 1e-3]  # 3 values
        self.lr_baseline = [1e-3, 3e-3]       # 2 values
        # ... etc
        # Total configs = product of all list lengths
```

## Troubleshooting

### GPU Out of Memory
```bash
# Reduce workers to 8
python scripts/hyperparameter_search_efficient.py \
    --strategy focused \
    --n_workers 8 \
    --device cuda
```

### Too Slow
```bash
# Reduce iterations for faster testing
python scripts/hyperparameter_search_efficient.py \
    --strategy focused \
    --n_iterations 2000 \
    --n_workers 10 \
    --device cuda
```

### Want Even Faster
```bash
# Use only 2 learning rate values for ultra-fast test
# Edit FocusedSearchSpace in the script:
# self.lr_policy = [3e-4, 1e-3]  # Just 2 values
# Total: 2 × 2 × 2 = 8 configurations!
```

## Expected Performance

| Strategy       | Configs | Time (n_workers=10) | Time (n_workers=2) |
|----------------|---------|---------------------|---------------------|
| Focused        | 24      | ~30 min             | ~2.5 hours          |
| Efficient      | 144     | ~3 hours            | ~15 hours           |
| Comprehensive  | 864     | ~12 hours           | ~3.5 days           |

**With 10 workers, you're ~5x faster than using 2 workers!**

## Tips

1. **Start Small**: Always start with `--strategy focused` to get quick results

2. **Monitor Resources**: Keep an eye on `nvidia-smi` and resource panel

3. **Iterate**: Use results from focused search to inform efficient/comprehensive search

4. **Save Results**: Results are saved incrementally, so you can Ctrl+C anytime

5. **Compare**: Use TensorBoard to compare all trials:
   ```bash
   tensorboard --logdir results/hyperparam_search_efficient/
   ```
