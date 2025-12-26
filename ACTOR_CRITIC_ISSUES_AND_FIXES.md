# Actor-Critic Training Issues and Fixes

## Summary

This document outlines the critical issues found in the actor-critic training setup and provides comprehensive fixes.

## Critical Issues Identified

### 1. **Missing Required Config Parameters** (CRITICAL)

**Problem:** The config file `configs/time_augmented_sparse_config.yaml` was missing essential actor-critic parameters:
- `network.hidden_dims` - Required for ActorCritic initialization
- `network.shared_layers` - Required for ActorCritic initialization
- `solver.critic_loss` - Required for critic training
- `solver.actor_loss` - Required for actor training
- `solver.hjb_weight` - Required for HJB regularization
- `solver.use_parallel` - Required for parallel simulation
- `solver.n_workers` - Required for worker configuration

**Impact:** When `solver_type=actor_critic`, the code would crash trying to access `config.network.hidden_dims` (line 106 in `macro_rl/config/setup_utils.py`).

**Fix:** Added all missing parameters to the config file with proper defaults.

---

### 2. **Incompatible Distribution Type** (CRITICAL)

**Problem:** Config specifies `distribution_type: beta`, but Beta distribution has issues with reparameterization gradients needed for pathwise actor-critic training.

**Evidence:**
- ActorCritic uses `actor_loss: pathwise` which requires differentiable reparameterization
- Beta distribution's reparameterization in `macro_rl/distributions.py` may not provide stable gradients
- TanhNormal is the recommended distribution for pathwise gradients

**Impact:** Unstable gradients, policy divergence, and training failures.

**Fix:** Changed `distribution_type: tanh_normal` in the new actor-critic config file.

---

### 3. **Action Bounds Visualization Issue** (HIGH PRIORITY)

**Problem:** Visualization shows equity issuance values of 6-10, but action bounds specify `equity_max: 0.5`.

**Possible Causes:**
1. **Scaling Bug:** The policy network may be outputting values in the wrong scale
2. **Visualization Bug:** The visualization may be plotting raw network outputs instead of bounded actions
3. **Action Clipping Not Applied:** The `clip()` function may not be called during evaluation

**Investigation Needed:**
```python
# In macro_rl/visualization/__init__.py line 61:
actions_mean, _ = policy.sample(states, deterministic=True)
```

This should return bounded actions. Need to verify:
- ActorCritic.sample() properly bounds outputs
- GaussianPolicy.sample() applies action_bounds correctly
- Visualization is not accidentally plotting normalized values

**Recommended Fix:** Add explicit clipping in visualization:
```python
from macro_rl.control.ghm_control import GHMControlSpec
control_spec = GHMControlSpec(...)
actions_mean = control_spec.clip(actions_mean)
```

---

### 4. **Config Mismatch** (MEDIUM PRIORITY)

**Problem:** The original config has `solver_type: monte_carlo` but user was running with `solver_type: actor_critic` (based on training output).

**Impact:** Confusion and potential for using wrong hyperparameters.

**Fix:** Created separate dedicated config file `configs/actor_critic_time_augmented_config.yaml` with all actor-critic parameters properly set.

---

### 5. **Missing Hyperparameter Documentation** (LOW PRIORITY)

**Problem:** No documentation on which hyperparameters to tune for actor-critic training.

**Impact:** Users don't know how to debug training issues or improve performance.

**Fix:** Added comprehensive hyperparameter tuning guide in new config file.

---

## Files Modified

1. **configs/time_augmented_sparse_config.yaml**
   - Added `network.hidden_dims` and `network.shared_layers`
   - Added all `solver.*` actor-critic parameters
   - Added warning about Beta distribution

2. **configs/actor_critic_time_augmented_config.yaml** (NEW)
   - Complete actor-critic configuration
   - Properly sets `solver_type: actor_critic`
   - Uses `tanh_normal` distribution
   - Includes hyperparameter tuning guide
   - Sets `n_iterations: 5000` to match user's training run

## Recommended Actions

### Immediate (Required)

1. **Use the new config file:**
   ```bash
   python scripts/train_with_config.py --config configs/actor_critic_time_augmented_config.yaml
   ```

2. **Verify action bounds** after a few iterations:
   - Check that equity issuance values are in [0, 0.5]
   - Check that dividend values are in [0, 10]
   - If still seeing values > 0.5 for equity, this indicates a deeper bug in the policy network

### Short-term (Recommended)

3. **Add action clipping to visualization** (macro_rl/visualization/__init__.py):
   ```python
   def compute_policy_value_time_augmented(policy, baseline, dynamics, control_spec, n_points=100):
       # ... existing code ...
       with torch.no_grad():
           actions_mean, _ = policy.sample(states, deterministic=True)
           actions_mean = control_spec.clip(actions_mean)  # ADD THIS LINE
           actions_mean = actions_mean.cpu().numpy()
   ```

4. **Debug action bounds** by adding logging to `ActorCritic.sample()`:
   ```python
   def sample(self, state, deterministic=False):
       feat = self._features(state)
       action, log_prob = self.actor.sample(feat, deterministic=deterministic)
       # ADD LOGGING
       print(f"Action before clip: min={action.min()}, max={action.max()}")
       print(f"Action bounds: {self.action_bounds}")
       return action, log_prob
   ```

### Long-term (Nice to have)

5. **Add input validation** to catch missing config parameters early
6. **Add unit tests** for action bounding in all policy types
7. **Add automated checks** in training loop to verify actions are within bounds

## Hyperparameter Tuning Guide

### For Stable Training

If training is unstable (policy divergence, NaN losses):

1. **Decrease learning rate:** `lr: 0.0001` (from 0.0003)
2. **Increase gradient clipping:** `max_grad_norm: 0.5` (from 1.0)
3. **Increase HJB weight:** `hjb_weight: 0.5` (from 0.1) - adds more model-based regularization
4. **Decrease entropy weight:** `entropy_weight: 0.001` (from 0.01) - reduces exploration

### For Faster Learning

If training is too slow:

1. **Increase learning rate:** `lr: 0.001` (from 0.0003)
2. **Decrease HJB weight:** `hjb_weight: 0.01` (from 0.1) - more data-driven
3. **Increase trajectories:** `n_trajectories: 4096` (from 2048) - better gradient estimates

### For Better Exploration

If policy converges to suboptimal solution:

1. **Increase entropy weight:** `entropy_weight: 0.05` (from 0.01)
2. **Use separate actor/critic networks:** `shared_layers: 0` (from 1)

## Expected Behavior

After applying these fixes, you should see:

1. ✅ Training starts without crashes
2. ✅ Equity issuance values stay within [0, 0.5]
3. ✅ Dividend values stay within [0, 10]
4. ✅ Losses decrease smoothly over iterations
5. ✅ No NaN or Inf values in losses
6. ✅ Policy gradually improves (returns increase)

## Debugging Checklist

If training still fails:

- [ ] Verify config has `solver_type: actor_critic`
- [ ] Verify config has `network.hidden_dims` and `network.shared_layers`
- [ ] Verify config has `solver.critic_loss` and `solver.actor_loss`
- [ ] Verify `distribution_type: tanh_normal` (not beta)
- [ ] Check action bounds in visualization plots
- [ ] Check for NaN/Inf in losses (tensorboard or logs)
- [ ] Verify gradient norms are reasonable (< 10)
- [ ] Check that returns are improving over time

## Additional Resources

- **Actor-Critic Theory:** See `macro_rl/solvers/actor_critic.py` docstring
- **Pathwise Gradients:** See `macro_rl/simulation/differentiable.py` docstring
- **HJB Equation:** See `ModelBasedActorCritic.compute_hjb_residual()` line 152
- **Action Distributions:** See `macro_rl/distributions.py`

## Questions?

If you encounter issues not covered here, check:
1. TensorBoard logs for loss curves
2. Visualization plots for action bounds
3. Training logs for error messages
4. Git history for recent changes
