# GHM-RL Scripts

This directory contains scripts for training and validating GHM equity models using reinforcement learning.

## Training

Train a SAC agent on the GHM equity model:

```bash
# Basic training (100k timesteps)
python scripts/train_ghm.py

# Extended training with custom parameters
python scripts/train_ghm.py --timesteps 500000 --output models/ghm_equity --n-envs 4
```

**Parameters:**
- `--timesteps`: Total training steps (default: 100000)
- `--output`: Output directory for models (default: models/ghm_equity)
- `--n-envs`: Number of parallel environments (default: 4)
- `--seed`: Random seed (default: 42)
- `--eval-freq`: Evaluation frequency (default: 5000)

**Outputs:**
- `final_model.zip`: Trained SAC model
- `best/best_model.zip`: Best model during training (by evaluation reward)
- `checkpoints/`: Periodic checkpoints
- `tensorboard/`: Training logs

**Monitor training:**
```bash
tensorboard --logdir models/ghm_equity/tensorboard
```

---

## Validation

Validate that the learned solution satisfies analytical properties:

```bash
# Validate a trained model
python scripts/validate.py --model models/ghm_equity/final_model

# High-quality validation (more episodes, finer grid)
python scripts/validate.py --model models/ghm_equity/final_model --n-episodes 100 --n-grid 300
```

**Parameters:**
- `--model`: Path to trained model (required)
- `--n-episodes`: Episodes per grid point for value estimation (default: 50)
- `--n-grid`: Number of grid points (default: 200)
- `--output`: Output directory (default: same as model)

**Outputs:**
- Console output with PASS/FAIL for each criterion
- `validation_plots.png`: Six-panel diagnostic plots
- `value_and_policy.png`: Combined plot matching paper figures
- `validation_data.npz`: Raw numerical data

**Validation criteria:**
1. ✓ Smooth pasting: F'(c*) ≈ 1
2. ✓ Super-contact: F''(c*) ≈ 0
3. ✓ HJB residual small
4. ✓ Monotonicity: F'(c) > 0
5. ✓ Concavity: F''(c) < 0 for c < c*
6. ✓ Policy threshold behavior

See [docs/VALIDATION.md](../docs/VALIDATION.md) for detailed explanation of validation criteria and how to interpret results.

---

## Evaluation

Evaluate a trained model and visualize trajectories:

```bash
python scripts/evaluate.py --model models/ghm_equity/final_model --n-episodes 10
```

**Parameters:**
- `--model`: Path to trained model (required)
- `--n-episodes`: Number of episodes to run (default: 10)

**Outputs:**
- Console output with episode statistics
- Trajectory plots showing cash evolution and policy decisions

---

## Debugging

Debug environment dynamics and verify implementation:

```bash
python scripts/debug_env.py
```

This script:
- Checks environment step mechanics
- Verifies reward calculation
- Tests termination conditions
- Displays sample trajectories

---

## Quick Start Workflow

1. **Train a model:**
   ```bash
   python scripts/train_ghm.py --timesteps 500000
   ```

2. **Monitor progress:**
   ```bash
   tensorboard --logdir models/ghm_equity/tensorboard
   ```

3. **Validate solution:**
   ```bash
   python scripts/validate.py --model models/ghm_equity/final_model
   ```

4. **Check for PASS/FAIL** in console output and inspect plots

5. **If validation fails**, consider:
   - Training longer (1M+ timesteps)
   - Adjusting hyperparameters in `train_ghm.py`
   - See troubleshooting in [docs/VALIDATION.md](../docs/VALIDATION.md)

---

## Expected Results

For well-trained models, you should see:

- **Threshold c***: Around 0.6-0.7
- **Value function**: Concave below c*, linear above
- **Policy**: Near-zero for c < c*, jumps at c*, high for c > c*
- **All validation checks**: PASS

Compare with Figures 1-2 in `GHM_v2.pdf` for reference.

---

## Advanced Usage

### Custom Parameters

Edit `train_ghm.py` to change:
- Environment parameters (dt, max_steps, liquidation_penalty)
- SAC hyperparameters (learning_rate, gamma, buffer_size)
- GHM model parameters (in `macro_rl/dynamics/ghm_equity.py`)

### Loading Validation Data

```python
import numpy as np

# Load validation results
data = np.load('models/ghm_equity/validation_data.npz', allow_pickle=True)
c_grid = data['c_grid']
V_mean = data['V_mean']
policy = data['policy']
results = data['results'].item()

print(f"Threshold: {results['threshold']:.4f}")
print(f"Smooth pasting error: {results['smooth_pasting']['error']:.4f}")
```

### Comparing Multiple Models

```bash
# Train with different parameters
python scripts/train_ghm.py --timesteps 500000 --output models/run1
python scripts/train_ghm.py --timesteps 1000000 --output models/run2

# Validate both
python scripts/validate.py --model models/run1/final_model
python scripts/validate.py --model models/run2/final_model

# Compare validation_plots.png and metrics
```

---

## References

- **docs/VALIDATION.md**: Detailed validation methodology
- **GHM_v2.pdf**: Original paper with analytical solutions
- **macro_rl/**: Core implementation of dynamics and environments
