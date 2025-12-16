# Solvers Module

## Purpose

Implement model-based RL algorithms that leverage known dynamics to solve continuous-time optimal control problems.

## Why Model-Based?

Model-free RL (PPO, SAC) doesn't use the fact that we know:
```
dc = μ(c)dt + σ(c)dW  ← Exact formula!
```

Model-based methods exploit this to:
1. **Simulate freely**: No environment interaction needed
2. **Explore completely**: Sample any initial state
3. **Reduce variance**: Use many trajectories or exact gradients
4. **Validate rigorously**: Check against HJB equation

## Three Main Approaches

### 1. Monte Carlo Policy Gradient (`monte_carlo.py`)

**Idea**: REINFORCE but with unlimited free simulation.

**Algorithm**:
```
for iteration:
    Sample 1000 initial states
    Simulate 1000 trajectories using known dynamics
    Compute returns
    Estimate gradient: ∇J ≈ E[∇log π(a|s) · (R - baseline)]
    Update policy
```

**Pros**:
- Conceptually simple
- Works with any policy

**Cons**:
- High variance (REINFORCE)
- Slower convergence

**When to use**: Baseline comparison, debugging

### 2. Pathwise Gradient (`pathwise.py`)

**Idea**: Make simulation differentiable, backprop through trajectories.

**Algorithm**:
```
for iteration:
    Sample initial states and noise ε
    Simulate trajectories (all operations differentiable)
        a = μ_θ(s) + σ_θ(s)·ε  ← Reparameterization
        s_next = s + drift·dt + diffusion·√dt·ε
    Compute returns (differentiable)
    loss = -mean(returns)
    loss.backward()  ← Exact gradient through chain rule!
    optimizer.step()
```

**Pros**:
- **Much lower variance** than REINFORCE
- Faster convergence
- Theoretically grounded

**Cons**:
- Requires differentiable policy (Gaussian, deterministic)
- Slightly more complex implementation

**When to use**: **RECOMMENDED** - Best for most cases

### 3. Deep Galerkin Method (`deep_galerkin.py`)

**Idea**: Don't simulate at all. Directly solve the HJB PDE.

**Algorithm**:
```
for iteration:
    Sample random points in state space
    Compute HJB residual:
        residual = (r-μ)V - max_a[r(s,a) + μ(s,a)·∇V + ½σ²(s)·∇²V]
    Minimize ||residual||²
    Update value network
```

Extract policy from V via FOC.

**Pros**:
- No simulation needed
- Directly enforces optimality
- Mesh-free

**Cons**:
- Requires Hessian computation
- Boundary conditions tricky
- May struggle with kinks

**When to use**: Advanced, for comparison with trajectory-based methods

## Comparison

| Method | Gradient Type | Variance | Speed | Complexity |
|--------|---------------|----------|-------|------------|
| Monte Carlo | REINFORCE | High | Slow | Low |
| Pathwise | Exact | Low | Fast | Medium |
| Deep Galerkin | PDE residual | Low | Fast | High |

## Implementation Priority

1. **Phase 1**: Pathwise Gradient (most practical)
2. **Phase 2**: Monte Carlo (for comparison)
3. **Phase 3**: Deep Galerkin (advanced)

## TODO for Implementation

### Monte Carlo (`monte_carlo.py`)
- [ ] Implement `_estimate_policy_gradient()`
- [ ] Implement `_update_baseline()`
- [ ] Implement training loop with logging
- [ ] Test with dummy policy

### Pathwise (`pathwise.py`)
- [ ] Implement `_compute_loss()` with differentiable simulation
- [ ] Implement `_update()` training step
- [ ] Verify gradients flow correctly
- [ ] Compare variance with Monte Carlo

### Deep Galerkin (`deep_galerkin.py`)
- [ ] Implement `_hjb_residual()` with autograd
- [ ] Implement `_boundary_loss()`
- [ ] Implement `_extract_policy()` from value function
- [ ] Test on toy problem with known solution

## Testing Strategy

Create `tests/solvers/`:

### `test_monte_carlo.py`
- Test with linear policy on simple problem
- Verify gradient estimates (check with finite differences)
- Test baseline variance reduction

### `test_pathwise.py`
- **Critical**: Verify gradient flow
- Compare gradients with finite differences
- Compare variance with Monte Carlo
- Test convergence on GHM

### `test_deep_galerkin.py`
- Test HJB residual computation
- Test boundary conditions
- Compare learned value with analytical (if available)

## Integration Flow

```
dynamics + control + reward
            ↓
       simulator
            ↓
       solvers
      ↙    ↓    ↘
    MC   PW   DGM
      ↘    ↓    ↙
        policy
            ↓
      validation
```

## Common Pitfalls

1. **Pathwise**: Forgetting to detach noise (must be fixed, not learnable)
2. **Monte Carlo**: Not using baseline (very high variance)
3. **Deep Galerkin**: Not enforcing boundary conditions properly
4. **All**: Not logging metrics (can't diagnose issues)

## Future Extensions

- [ ] Add trust region methods (TRPO-style)
- [ ] Add experience replay for efficiency
- [ ] Add adaptive sampling (focus on high-error regions)
- [ ] Add multi-objective optimization

## References

- Pathwise gradients: "Simple Statistical Gradient-Following Algorithms for Connectionist RL" (Williams 1992)
- Reparameterization trick: "Auto-Encoding Variational Bayes" (Kingma & Welling 2013)
- Deep Galerkin: "Solving high-dimensional PDEs using deep learning" (Sirignano & Spiliopoulos 2018)
