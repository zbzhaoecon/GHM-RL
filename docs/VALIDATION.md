# Validation Guide for GHM Solutions

## Overview

After training, we need to verify the learned solution is correct. This document explains:
1. What analytical properties the solution must satisfy
2. How to check these properties
3. How to interpret the validation results

---

## Analytical Properties of the GHM Solution

The optimal value function F(c) and policy a(c) must satisfy several conditions:

### 1. Smooth Pasting Condition

At the payout threshold c*, the value function's slope equals 1:

$$F'(c^*) = 1$$

**Interpretation**: At c*, the marginal value of an extra dollar of cash equals the marginal value of paying it out as dividends.

**How we check**: Compute numerical derivative of V(c) and evaluate at the detected threshold.

**Tolerance**: |F'(c*) - 1| < 0.1 is good, < 0.2 is acceptable.

### 2. Super-Contact Condition

At c*, the second derivative is zero:

$$F''(c^*) = 0$$

**Interpretation**: The value function transitions smoothly from concave (below c*) to linear (above c*).

**How we check**: Compute second numerical derivative at threshold.

**Tolerance**: |F''(c*)| < 0.5 is good, < 1.0 is acceptable.

### 3. HJB Equation (Continuation Region)

In the continuation region (c < c*), the value function satisfies:

$$(r - \mu) F(c) = \mu_c(c) F'(c) + \frac{1}{2}\sigma_c^2(c) F''(c)$$

**Interpretation**: The value equals the expected present value of future cash flows.

**How we check**: Compute residual: `(r-μ)F - μ_c F' - 0.5 σ² F''`. Should be ~0.

**Tolerance**: Mean |residual| < 0.5 is good.

### 4. Monotonicity

The value function is increasing:

$$F'(c) > 0 \quad \forall c \in [0, c^*]$$

**Interpretation**: More cash is always better (before hitting the payout threshold).

### 5. Concavity

The value function is concave in the continuation region:

$$F''(c) < 0 \quad \forall c \in [0, c^*)$$

**Interpretation**: Diminishing marginal value of cash due to precautionary motive.

### 6. Policy Threshold Behavior

The optimal policy is:
- **Retain**: a(c) ≈ 0 for c < c* (keep all cash)
- **Payout**: a(c) → large for c ≥ c* (pay excess to shareholders)

**How we check**: Ratio of mean action above vs below c* should be >> 1.

---

## Using the Validation Script

```bash
python scripts/validate.py --model models/ghm_equity/final_model --n-episodes 50
```

### Output

The script produces:

1. **Console output**: Summary of all validation metrics with PASS/FAIL

2. **validation_plots.png**: Six-panel figure showing:
   - Policy a(c)
   - Value function F(c) with confidence bands
   - First derivative F'(c) with target line at 1
   - Second derivative F''(c) with target line at 0
   - HJB residual
   - Combined value and policy plot

3. **value_and_policy.png**: Combined plot matching paper figures

4. **validation_data.npz**: Raw numerical data for further analysis

---

## Interpreting Results

### Good Solution

```
VALIDATION RESULTS
==================================================

1. THRESHOLD DETECTION
   c* = 0.6500

2. SMOOTH PASTING CONDITION: F'(c*) = 1
   F'(c*) = 1.0234
   Error  = 0.0234
   Status: ✓ PASS

3. SUPER-CONTACT CONDITION: F''(c*) = 0
   F''(c*) = -0.1523
   Error   = 0.1523
   Status: ✓ PASS

4. HJB RESIDUAL
   Mean |residual| = 0.0821
   Max  |residual| = 0.3456
   Status: ✓ PASS

5. VALUE FUNCTION PROPERTIES
   Monotonic (F' > 0): ✓ YES
   Concave (F'' < 0):  ✓ YES

6. POLICY PROPERTIES
   Mean action below c*: 0.1234
   Mean action above c*: 5.6789
   Ratio: 46.0x
   Status: ✓ PASS

OVERALL ASSESSMENT: ✓ Solution appears VALID
```

### Problematic Solution

```
VALIDATION RESULTS
==================================================

1. THRESHOLD DETECTION
   c* = 0.4200

2. SMOOTH PASTING CONDITION: F'(c*) = 1
   F'(c*) = 0.5432
   Error  = 0.4568
   Status: ✗ FAIL

...

OVERALL ASSESSMENT: ⚠ Solution may have ISSUES:
  - Smooth pasting condition not satisfied
  - Policy doesn't show clear threshold behavior

Consider:
  - Training for more timesteps
  - Adjusting hyperparameters (gamma, liquidation_penalty)
```

---

## Common Issues and Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No clear threshold | Undertrained | Train longer (500k+ steps) |
| F'(c*) ≠ 1 | Wrong gamma or reward scale | Adjust gamma closer to 1.0 |
| Large HJB residual | Value function noisy | More episodes for estimation |
| Non-monotonic | Early training | Train longer |
| c* too low | Liquidation penalty too high | Reduce penalty |
| c* too high | Liquidation penalty too low | Increase penalty |

---

## Expected Values from Paper

From GHM_v2.pdf (Figures 1-2):

| Parameter Set | Expected c* | Value Shape |
|---------------|-------------|-------------|
| Low issuance cost | ~0.6-0.7 | Concave, levels off at c* |
| High issuance cost | ~0.8-0.9 | More concave |

The policy should show:
- Near-zero action for c < c*
- Steep increase around c*
- High action for c > c*

---

## Advanced Validation

### Compare with Finite Difference Solution

If you have a reference FDM solution:

```python
# Load reference
fdm_values = np.load("reference_fdm_solution.npy")

# Compare
error = np.abs(learned_values - fdm_values)
print(f"Mean error vs FDM: {error.mean():.4f}")
print(f"Max error vs FDM: {error.max():.4f}")
```

### Check Value Function Analytically at c*

At c = c*, the value function should equal:

$$F(c^*) = \frac{\alpha + c^*(r - \lambda - \mu)}{r - \mu}$$

This is because at c*, the firm is in steady-state payout mode.

```python
# From GHM parameters
alpha = 0.18
r = 0.03
mu = 0.01
lambda_ = 0.02
rho_eff = r - mu  # 0.02

# Expected value at c*
c_star = 0.65  # detected
expected_F_cstar = (alpha + c_star * (r - lambda_ - mu)) / rho_eff
print(f"Expected F(c*) = {expected_F_cstar:.4f}")
```

---

## Visualization Tips

### Comparing with Paper Figures

The learned value function should visually match Figures 1-2 of GHM_v2.pdf:
- Shape: Concave below c*, linear above
- Threshold: Marked by dashed vertical line
- Smooth: No kinks or discontinuities

### Checking Policy Quality

A good policy plot shows:
- Flat near zero for c < c*
- Sharp transition at c*
- High plateau for c > c*

A poor policy shows:
- Gradual increase (no clear threshold)
- Noisy/fluctuating values
- Non-monotonic behavior

---

## Implementation Details

### Value Function Estimation

The validation script estimates the value function by:

1. Creating a grid of states from c_min to c_max
2. For each grid point, running multiple episodes starting from that state
3. Computing discounted returns using the learned policy
4. Averaging across episodes to get V(c) with confidence bands

This Monte Carlo estimation can be noisy, so we use:
- Multiple episodes (default: 50) per grid point
- Fine grid (default: 200 points)
- Standard deviation bands to visualize uncertainty

### Numerical Derivatives

We use central finite differences:

```python
# First derivative
V'(c) ≈ (V(c+h) - V(c-h)) / (2h)

# Second derivative
V''(c) ≈ (V(c+h) - 2V(c) + V(c-h)) / h²
```

where h is the grid spacing.

### Threshold Detection

We detect c* using two methods and average:

1. **Smooth pasting**: Point where |F'(c) - 1| is minimized
2. **Policy jump**: First point where a(c) > threshold (0.5)

This robust approach handles noisy estimates.

---

## Troubleshooting

### "Model not found" error

Make sure the model path is correct. The script tries both with and without `.zip` extension.

### Validation takes too long

Reduce `--n-episodes` (faster but noisier) or `--n-grid` (coarser resolution).

### All tests fail

Your model may be undertrained. Try:
1. Train for more timesteps (500k-1M)
2. Check tensorboard logs for learning curves
3. Verify environment parameters match the paper

### HJB residual very large

This usually means:
- Value function is noisy (increase `--n-episodes`)
- Model hasn't converged (train longer)
- Numerical derivatives are unstable (check grid spacing)

---

## References

- **GHM_v2.pdf**: Original paper with analytical solution properties
- **scripts/validate.py**: Implementation of all validation checks
- **scripts/train_ghm.py**: Training script that produces models to validate
