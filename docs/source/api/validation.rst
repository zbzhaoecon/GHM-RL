Validation Module
=================

The validation module provides tools for validating learned solutions against theoretical optimality conditions.

HJB Residual Validation
-----------------------

.. automodule:: macro_rl.validation.hjb_residual
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`HJBValidator` computes the Hamilton-Jacobi-Bellman residual to check if a value function satisfies the optimality condition.

HJB Equation
~~~~~~~~~~~~

The HJB equation for the optimal value function :math:`V^*` is:

.. math::

   \rho V^*(s) = \max_a \left\{ r(s, a) + \mathcal{L}^a V^*(s) \right\}

where :math:`\mathcal{L}^a` is the infinitesimal generator:

.. math::

   \mathcal{L}^a V(s) = \mu(s, a)^\top \nabla_s V(s) + \frac{1}{2} \text{tr}(\sigma(s) \sigma(s)^\top \nabla_s^2 V(s))

HJB Residual
~~~~~~~~~~~~

The HJB residual measures how well a candidate value function satisfies the optimality condition:

.. math::

   \text{Residual}(s) = \left| \rho V(s) - \max_a [r(s, a) + \mathcal{L}^a V(s)] \right|

A small residual indicates the value function is close to optimal.

Key Methods
~~~~~~~~~~~

* :meth:`~macro_rl.validation.hjb_residual.HJBValidator.compute_residual`: Compute HJB residual at given states
* :meth:`~macro_rl.validation.hjb_residual.HJBValidator.validate`: Compute residual statistics over state space
* :meth:`~macro_rl.validation.hjb_residual.HJBValidator.plot_residual`: Visualize residual distribution

Example
~~~~~~~

.. code-block:: python

   from macro_rl.validation.hjb_residual import HJBValidator
   from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
   from macro_rl.control.ghm_control import GHMControlSpec
   from macro_rl.rewards.ghm_rewards import GHMRewardFunction
   from macro_rl.networks.value import ValueNetwork
   import torch

   # Setup
   params = GHMEquityParams()
   dynamics = GHMEquityDynamics(params)
   control_spec = GHMControlSpec(params)
   reward_fn = GHMRewardFunction(params, control_spec)

   # Trained value network
   value_net = ValueNetwork(state_dim=1, hidden_dims=[128, 128])
   # ... (assume trained)

   # Create validator
   validator = HJBValidator(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       value_network=value_net,
   )

   # Sample states for validation
   test_states = dynamics.state_space.sample_uniform(1000)

   # Compute HJB residual
   residuals = validator.compute_residual(test_states)

   print(f"Mean residual: {residuals.mean():.6f}")
   print(f"Max residual: {residuals.max():.6f}")
   print(f"Std residual: {residuals.std():.6f}")

   # Validation statistics
   stats = validator.validate(n_samples=1000)
   print(f"Validation results: {stats}")

   # Plot residual
   validator.plot_residual(test_states, residuals, save_path="hjb_residual.png")

Boundary Condition Validation
------------------------------

.. automodule:: macro_rl.validation.boundary_conditions
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`BoundaryValidator` checks if boundary conditions are satisfied.

Common Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Dirichlet**: :math:`V(s_{\text{boundary}}) = g(s_{\text{boundary}})`
2. **Neumann**: :math:`\nabla_s V(s_{\text{boundary}}) = h(s_{\text{boundary}})`
3. **Robin**: :math:`\alpha V + \beta \nabla_s V = f`

For the GHM model:

* **Lower boundary** (:math:`c = 0`): :math:`V(0) = 0` (bankruptcy)
* **Upper boundary** (:math:`c = c_{\max}`): Free boundary or reflection

Example
~~~~~~~

.. code-block:: python

   from macro_rl.validation.boundary_conditions import BoundaryValidator

   validator = BoundaryValidator(
       dynamics=dynamics,
       value_network=value_net,
       boundary_conditions={
           'lower': {'type': 'dirichlet', 'value': 0.0},
           'upper': {'type': 'neumann', 'value': 1.0},
       }
   )

   # Validate boundaries
   boundary_errors = validator.validate()
   print(f"Lower boundary error: {boundary_errors['lower']:.6f}")
   print(f"Upper boundary error: {boundary_errors['upper']:.6f}")

Analytical Comparison
---------------------

.. automodule:: macro_rl.validation.analytical_comparison
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`AnalyticalComparator` compares learned solutions with analytical solutions when available.

Key Metrics
~~~~~~~~~~~

1. **Value Error**: :math:`\|V_{\text{learned}} - V_{\text{analytical}}\|`
2. **Policy Error**: :math:`\|\pi_{\text{learned}} - \pi_{\text{analytical}}\|`
3. **Relative Error**: :math:`\frac{\|V_{\text{learned}} - V_{\text{analytical}}\|}{\|V_{\text{analytical}}\|}`

Example
~~~~~~~

.. code-block:: python

   from macro_rl.validation.analytical_comparison import AnalyticalComparator

   # Assume we have analytical solution
   def analytical_value(state):
       # Closed-form solution (if available)
       return state ** 2 / (2 * params.r)

   def analytical_policy(state):
       # Optimal policy (if available)
       return torch.tensor([params.alpha * params.mu, 0.0])

   comparator = AnalyticalComparator(
       learned_value=value_net,
       learned_policy=policy_net,
       analytical_value=analytical_value,
       analytical_policy=analytical_policy,
   )

   # Sample states
   test_states = dynamics.state_space.sample_uniform(1000)

   # Compare values
   value_errors = comparator.compare_values(test_states)
   print(f"Mean value error: {value_errors['mean']:.6f}")
   print(f"Max value error: {value_errors['max']:.6f}")
   print(f"Relative error: {value_errors['relative']:.6f}")

   # Compare policies
   policy_errors = comparator.compare_policies(test_states)
   print(f"Mean policy error: {policy_errors['mean']:.6f}")

   # Visualize comparison
   comparator.plot_comparison(test_states, save_path="comparison.png")

Validation Workflow
-------------------

Recommended validation workflow after training:

1. **HJB Residual Check**

   .. code-block:: python

      hjb_validator = HJBValidator(dynamics, control_spec, reward_fn, value_net)
      hjb_stats = hjb_validator.validate(n_samples=5000)

      if hjb_stats['mean_residual'] > 1e-3:
          print("Warning: Large HJB residual detected")

2. **Boundary Condition Check**

   .. code-block:: python

      boundary_validator = BoundaryValidator(dynamics, value_net, boundary_conditions)
      boundary_errors = boundary_validator.validate()

      if any(err > 1e-2 for err in boundary_errors.values()):
          print("Warning: Boundary conditions not satisfied")

3. **Analytical Comparison** (if available)

   .. code-block:: python

      comparator = AnalyticalComparator(value_net, policy_net, analytical_value, analytical_policy)
      comparison = comparator.compare_values(test_states)

      print(f"Relative error vs analytical: {comparison['relative']:.2%}")

4. **Monte Carlo Validation**

   .. code-block:: python

      # Simulate trajectories and compare returns
      from macro_rl.simulation.trajectory import TrajectorySimulator

      simulator = TrajectorySimulator(dynamics, control_spec, reward_fn, n_steps=1000, dt=0.01)
      trajectories = simulator.rollout(policy_net, initial_states, n_trajectories=1000)

      predicted_values = value_net(initial_states)
      actual_returns = trajectories.returns

      value_error = (predicted_values - actual_returns).abs().mean()
      print(f"Monte Carlo value error: {value_error:.4f}")

Visualization Tools
-------------------

The validation module provides visualization utilities:

.. code-block:: python

   import matplotlib.pyplot as plt
   from macro_rl.validation import plot_value_function, plot_policy, plot_hjb_residual

   # Plot learned value function
   plot_value_function(
       value_net,
       state_space=dynamics.state_space,
       save_path="value_function.png"
   )

   # Plot learned policy
   plot_policy(
       policy_net,
       state_space=dynamics.state_space,
       control_spec=control_spec,
       save_path="policy.png"
   )

   # Plot HJB residual heatmap
   plot_hjb_residual(
       hjb_validator,
       state_space=dynamics.state_space,
       save_path="hjb_residual.png"
   )

See Also
~~~~~~~~

* :mod:`macro_rl.networks.value`: Value networks with gradient computation
* :mod:`macro_rl.solvers.deep_galerkin`: Direct HJB residual minimization
* :mod:`macro_rl.numerics.differentiation`: Automatic differentiation utilities
