# Values Module

## Purpose

Provide value function representations for continuous-time control, with support for automatic differentiation needed for HJB equation solvers.

## Key Components

### 1. ValueNetwork (`neural.py`)

**Purpose**: Neural network value function with autodiff support.

**Critical Feature**: Must support gradient and Hessian computation for HJB:
```
(r-μ)V = max_a [r(s,a) + μ(s,a)·∇V + ½σ²(s)·∇²V]
                           ^^^         ^^^^
                         Need V_c     Need V_cc
```

**Usage**:
```python
value_net = ValueNetwork(state_dim=1, hidden_dims=[64, 64])
V = value_net(state)
V_c = value_net.gradient(state)  # For HJB drift term
V_cc = value_net.hessian(state)  # For HJB diffusion term
```

## TODO for Implementation

- [ ] Implement ValueNetwork with tanh activations (smooth)
- [ ] Implement gradient() using torch.autograd.grad
- [ ] Implement hessian() using double autograd
- [ ] Test gradient/Hessian computation
- [ ] Add proper initialization

## Testing

- Test gradient computation vs finite differences
- Test Hessian computation vs finite differences
- Test batch dimensions
