Simulation Module
=================

The simulation module provides tools for simulating stochastic differential equations (SDEs) and generating trajectories for reinforcement learning.

SDE Integration
---------------

.. automodule:: macro_rl.simulation.sde
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`SDEIntegrator` class provides numerical integration schemes for SDEs:

Euler-Maruyama Scheme
~~~~~~~~~~~~~~~~~~~~~

.. math::

   X_{t+\Delta t} = X_t + \mu(X_t, a_t) \Delta t + \sigma(X_t) \sqrt{\Delta t} \, Z_t

where :math:`Z_t \sim \mathcal{N}(0, I)`.

Milstein Scheme (Higher-Order)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   X_{t+\Delta t} = X_t + \mu(X_t, a_t) \Delta t + \sigma(X_t) \sqrt{\Delta t} \, Z_t + \frac{1}{2} \sigma(X_t) \sigma'(X_t) (\Delta t Z_t^2 - \Delta t)

Example
~~~~~~~

.. code-block:: python

   from macro_rl.simulation.sde import SDEIntegrator
   from macro_rl.dynamics.test_models import GBMDynamics
   import torch

   dynamics = GBMDynamics(mu=0.1, sigma=0.2)
   integrator = SDEIntegrator(dynamics, dt=0.01, method='euler')

   # Simulate one step
   state = torch.tensor([[1.0]])
   action = torch.tensor([[0.0]])
   noise = torch.randn(1, 1)
   next_state = integrator.step(state, action, noise)

Trajectory Simulation
---------------------

.. automodule:: macro_rl.simulation.trajectory
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`TrajectorySimulator` generates full trajectories using a policy:

Key Classes
~~~~~~~~~~~

* :class:`~macro_rl.simulation.trajectory.TrajectorySimulator`: Main simulator class
* :class:`~macro_rl.simulation.trajectory.TrajectoryBatch`: Container for trajectory data

Trajectory Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`TrajectoryBatch` contains:

* ``states``: State sequences :math:`(N, T+1, d_x)`
* ``actions``: Action sequences :math:`(N, T, d_a)`
* ``rewards``: Flow rewards :math:`(N, T)`
* ``terminal_rewards``: Terminal rewards :math:`(N,)`
* ``masks``: Continuation masks :math:`(N, T)` (0 if terminated, 1 otherwise)
* ``returns``: Discounted returns :math:`(N,)`
* ``lengths``: Episode lengths :math:`(N,)`

Example
~~~~~~~

.. code-block:: python

   from macro_rl.simulation.trajectory import TrajectorySimulator
   from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
   from macro_rl.control.ghm_control import GHMControlSpec
   from macro_rl.rewards.ghm_rewards import GHMRewardFunction
   from macro_rl.networks.actor_critic import ActorCritic
   import torch

   # Setup
   params = GHMEquityParams()
   dynamics = GHMEquityDynamics(params)
   control_spec = GHMControlSpec(params)
   reward_fn = GHMRewardFunction(params, control_spec)
   policy = ActorCritic(state_dim=1, action_dim=2, hidden_dims=[128, 128])

   # Create simulator
   simulator = TrajectorySimulator(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       n_steps=100,
       dt=0.01,
   )

   # Generate trajectories
   initial_states = torch.tensor([[5.0], [6.0], [7.0]])
   trajectories = simulator.rollout(policy, initial_states)

   print(f"Returns: {trajectories.returns}")
   print(f"Episode lengths: {trajectories.lengths}")

Differentiable Simulation
--------------------------

.. automodule:: macro_rl.simulation.differentiable
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`DifferentiableSimulator` enables end-to-end backpropagation through trajectories for pathwise gradient estimation.

Key Features
~~~~~~~~~~~~

* **Fully differentiable**: All operations support PyTorch autograd
* **Soft termination**: Uses sigmoid masking instead of hard stops
* **Pathwise gradients**: Compute :math:`\nabla_\theta \mathbb{E}[R(\tau)]` via reparameterization

Reparameterization Trick
~~~~~~~~~~~~~~~~~~~~~~~~~

Actions are sampled using:

.. math::

   a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

This allows gradients to flow through the policy parameters :math:`\theta`.

Example
~~~~~~~

.. code-block:: python

   from macro_rl.simulation.differentiable import DifferentiableSimulator
   import torch

   # Create differentiable simulator
   simulator = DifferentiableSimulator(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       n_steps=100,
       dt=0.01,
   )

   # Generate trajectories with gradients
   initial_states = torch.tensor([[5.0]], requires_grad=True)
   noise = torch.randn(1, 100, 2)  # Pre-sampled noise

   returns, trajectory = simulator.simulate(
       policy=policy,
       initial_states=initial_states,
       noise=noise,
       return_trajectory=True,
   )

   # Backpropagate
   loss = -returns.mean()
   loss.backward()

   # Gradients are available
   print(f"Policy gradient norm: {sum(p.grad.norm() for p in policy.parameters())}")
