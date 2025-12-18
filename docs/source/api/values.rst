Values Module
=============

The values module provides value function representations for reinforcement learning.

Base Classes
------------

.. automodule:: macro_rl.values.base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`ValueFunction` abstract base class defines the interface for all value functions:

.. math::

   V^\pi(s) = \mathbb{E}^\pi\left[\int_0^\tau e^{-\rho t} r(X_t, a_t) dt + e^{-\rho \tau} g(X_\tau) \mid X_0 = s\right]

Key Methods
~~~~~~~~~~~

* :meth:`~macro_rl.values.base.ValueFunction.__call__`: Evaluate value at state
* :meth:`~macro_rl.values.base.ValueFunction.gradient`: Compute :math:`\nabla_s V(s)`
* :meth:`~macro_rl.values.base.ValueFunction.hessian`: Compute :math:`\nabla_s^2 V(s)`

Neural Value Functions
----------------------

.. automodule:: macro_rl.values.neural
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Neural network-based value function implementations.

The :class:`ValueNetwork` wraps a PyTorch network as a value function:

Example
~~~~~~~

.. code-block:: python

   from macro_rl.values.neural import ValueNetwork
   from macro_rl.networks.value import ValueNetwork as ValueNet
   import torch

   # Create neural value function
   net = ValueNet(state_dim=1, hidden_dims=[128, 128])
   value_fn = ValueNetwork(network=net)

   # Evaluate value
   state = torch.tensor([[5.0]])
   value = value_fn(state)
   print(f"Value: {value}")

   # Compute gradient
   grad = value_fn.gradient(state)
   print(f"Gradient: {grad}")

   # Compute Hessian
   hess = value_fn.hessian(state)
   print(f"Hessian: {hess}")

Analytical Value Functions
---------------------------

.. automodule:: macro_rl.values.analytical
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Wrapper for analytical value functions when closed-form solutions exist.

The :class:`AnalyticalValue` class wraps a callable that computes the value analytically:

Example
~~~~~~~

.. code-block:: python

   from macro_rl.values.analytical import AnalyticalValue
   import torch

   # Define analytical solution (e.g., for GBM)
   def gbm_value(state, params):
       # V(x) = x^β / (ρ - μ)
       # Simplified example
       return state ** 2 / 0.05

   value_fn = AnalyticalValue(value_function=gbm_value)

   # Evaluate
   state = torch.tensor([[1.0]])
   value = value_fn(state)
   print(f"Analytical value: {value}")

HJB Optimality Condition
~~~~~~~~~~~~~~~~~~~~~~~~~

For optimal value functions, the Hamilton-Jacobi-Bellman (HJB) equation must be satisfied:

.. math::

   \rho V(s) = \max_a \left\{ r(s, a) + \mathcal{L}^a V(s) \right\}

where :math:`\mathcal{L}^a` is the infinitesimal generator:

.. math::

   \mathcal{L}^a V(s) = \mu(s, a)^\top \nabla_s V(s) + \frac{1}{2} \text{tr}(\sigma(s) \sigma(s)^\top \nabla_s^2 V(s))

The value function module provides tools to:

1. Compute value approximations via neural networks
2. Compute gradients and Hessians for HJB validation
3. Compare learned values with analytical solutions

See Also
~~~~~~~~

* :mod:`macro_rl.validation.hjb_residual`: For HJB residual computation
* :mod:`macro_rl.validation.analytical_comparison`: For comparing learned vs analytical solutions
