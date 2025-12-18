Solvers Module
==============

The solvers module provides model-based reinforcement learning algorithms for continuous-time optimal control problems.

Base Classes
------------

.. automodule:: macro_rl.solvers.base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`Solver` abstract base class defines the interface for all solvers:

* :meth:`~macro_rl.solvers.base.Solver.train`: Train the policy/value function
* :meth:`~macro_rl.solvers.base.Solver.train_step`: Single training iteration
* :meth:`~macro_rl.solvers.base.Solver.evaluate`: Evaluate learned policy

The :class:`SolverResult` dataclass contains solver outputs:

* ``policy``: Trained policy
* ``value_function``: Trained value function (if applicable)
* ``diagnostics``: Training metrics and diagnostics

Model-Based Actor-Critic
-------------------------

.. automodule:: macro_rl.solvers.actor_critic
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`ModelBasedActorCritic` solver jointly learns policy and value function using multiple loss components.

Algorithm Overview
~~~~~~~~~~~~~~~~~~

1. **Generate Trajectories**: Sample initial states and simulate trajectories using current policy
2. **Critic Update**: Train value network using combination of:

   * Monte Carlo returns: :math:`\mathcal{L}_{\text{MC}} = \|V(s_0) - R\|^2`
   * TD bootstrapping: :math:`\mathcal{L}_{\text{TD}} = \|V(s_t) - (r_t + \gamma V(s_{t+1}))\|^2`
   * HJB residual: :math:`\mathcal{L}_{\text{HJB}} = \|\rho V - \max_a [r + \mathcal{L}^a V]\|^2`

3. **Actor Update**: Improve policy using:

   * Pathwise gradients: :math:`\nabla_\theta \mathbb{E}[R(\tau)] = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) R]`
   * REINFORCE: :math:`\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) A^\pi(s, a)]`

Loss Configurations
~~~~~~~~~~~~~~~~~~~

The solver supports flexible loss configuration via the ``critic_loss_type`` parameter:

* ``"mc"``: Monte Carlo returns only
* ``"td"``: TD bootstrapping only
* ``"hjb"``: HJB residual only
* ``"mc+hjb"``: Combination of MC and HJB (recommended)

Parameters
~~~~~~~~~~

Key initialization parameters:

* ``dynamics``: Continuous-time dynamics model
* ``control_spec``: Control specification with bounds
* ``reward_function``: Reward function
* ``actor_critic``: Joint actor-critic network
* ``n_trajectories``: Number of trajectories per iteration
* ``n_steps``: Time steps per trajectory
* ``dt``: Time discretization
* ``actor_lr``: Actor learning rate
* ``critic_lr``: Critic learning rate
* ``critic_loss_type``: Type of critic loss (see above)
* ``hjb_weight``: Weight for HJB residual term
* ``td_lambda``: TD(λ) parameter for eligibility traces

Example
~~~~~~~

.. code-block:: python

   from macro_rl.solvers.actor_critic import ModelBasedActorCritic
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

   # Create network
   network = ActorCritic(
       state_dim=1,
       action_dim=2,
       hidden_dims=[128, 128],
       shared_trunk=True,
   )

   # Initialize solver
   solver = ModelBasedActorCritic(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       actor_critic=network,
       n_trajectories=32,
       n_steps=100,
       dt=0.01,
       actor_lr=1e-4,
       critic_lr=1e-3,
       critic_loss_type="mc+hjb",
       hjb_weight=0.1,
   )

   # Train
   history = solver.train(
       n_iterations=1000,
       log_freq=100,
   )

   # Evaluate
   eval_results = solver.evaluate(n_episodes=100)
   print(f"Mean return: {eval_results['mean_return']:.2f}")

Training Diagnostics
~~~~~~~~~~~~~~~~~~~~

The ``train`` method returns a dictionary with training history:

* ``actor_loss``: Actor loss values
* ``critic_loss``: Critic loss values
* ``mc_loss``: Monte Carlo loss component
* ``hjb_residual``: HJB residual values
* ``mean_return``: Mean return on training trajectories
* ``policy_entropy``: Policy entropy
* ``value_std``: Standard deviation of value predictions

Pathwise Gradient Solver
-------------------------

.. automodule:: macro_rl.solvers.pathwise
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`PathwiseGradient` solver uses differentiable simulation for end-to-end gradient estimation.

Algorithm Overview
~~~~~~~~~~~~~~~~~~

1. **Differentiable Rollout**: Simulate trajectories with reparameterization trick
2. **Compute Returns**: Calculate total discounted return :math:`R = \sum_{t=0}^T \gamma^t r_t`
3. **Backpropagate**: Compute :math:`\nabla_\theta \mathbb{E}[R]` via PyTorch autograd
4. **Update Policy**: Gradient ascent on expected return

Advantages
~~~~~~~~~~

* **Low variance**: Direct gradient through deterministic simulation
* **No baseline needed**: Gradients are unbiased without control variates
* **Model exploitation**: Leverages known dynamics for efficiency

Example
~~~~~~~

.. code-block:: python

   from macro_rl.solvers.pathwise import PathwiseGradient
   from macro_rl.networks.policy import GaussianPolicy

   # Create policy
   policy = GaussianPolicy(state_dim=1, action_dim=2, hidden_dims=[128, 128])

   # Initialize solver
   solver = PathwiseGradient(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       policy=policy,
       n_trajectories=64,
       n_steps=100,
       dt=0.01,
       learning_rate=1e-3,
   )

   # Train
   history = solver.train(n_iterations=5000)

Monte Carlo Policy Gradient
----------------------------

.. automodule:: macro_rl.solvers.monte_carlo
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`MonteCarloPolicyGradient` solver implements REINFORCE with baseline.

Algorithm Overview
~~~~~~~~~~~~~~~~~~

1. **Generate Trajectories**: Simulate using current policy
2. **Compute Returns**: :math:`R_t = \sum_{k=t}^T \gamma^{k-t} r_k`
3. **Compute Advantages**: :math:`A_t = R_t - V(s_t)` (with optional baseline)
4. **Policy Gradient**: :math:`\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) A_t]`
5. **Update**: Gradient ascent

Example
~~~~~~~

.. code-block:: python

   from macro_rl.solvers.monte_carlo import MonteCarloPolicyGradient

   solver = MonteCarloPolicyGradient(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       policy=policy,
       baseline=value_network,  # Optional baseline
       n_trajectories=128,
       n_steps=100,
       dt=0.01,
       learning_rate=1e-3,
   )

   history = solver.train(n_iterations=10000)

Deep Galerkin Method
--------------------

.. automodule:: macro_rl.solvers.deep_galerkin
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`DeepGalerkinMethod` solver directly minimizes the HJB residual to find the optimal value function.

Algorithm Overview
~~~~~~~~~~~~~~~~~~

1. **Sample Points**: Sample interior and boundary points from state space
2. **Compute HJB Residual**:

   .. math::

      \text{Residual}(s) = \left|\rho V(s) - \max_a [r(s,a) + \mathcal{L}^a V(s)]\right|

3. **Compute Boundary Error**: Check boundary conditions (e.g., :math:`V(0) = 0`)
4. **Minimize Loss**: :math:`\mathcal{L} = \mathbb{E}[\text{Residual}^2] + \lambda \mathbb{E}[\text{Boundary Error}^2]`
5. **Extract Policy**: Compute :math:`\pi^*(s) = \arg\max_a [r(s,a) + \mathcal{L}^a V(s)]`

Example
~~~~~~~

.. code-block:: python

   from macro_rl.solvers.deep_galerkin import DeepGalerkinMethod
   from macro_rl.networks.value import ValueNetwork

   # Create value network
   value_net = ValueNetwork(state_dim=1, hidden_dims=[256, 256, 128])

   solver = DeepGalerkinMethod(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       value_network=value_net,
       n_interior=1000,
       n_boundary=100,
       learning_rate=1e-3,
       boundary_weight=10.0,
   )

   history = solver.train(n_iterations=20000)

   # Extract policy from learned value function
   policy = solver.extract_policy()

Comparison of Solvers
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Solver
     - Gradient Type
     - Variance
     - Sample Efficiency
     - Computational Cost
   * - Actor-Critic
     - Mixed
     - Medium
     - High
     - Medium
   * - Pathwise Gradient
     - Pathwise
     - Low
     - High
     - Low
   * - Monte Carlo
     - Score Function
     - High
     - Low
     - Low
   * - Deep Galerkin
     - PDE Residual
     - N/A
     - Very High
     - High

**Recommendations:**

* **General use**: :class:`ModelBasedActorCritic` with ``critic_loss_type="mc+hjb"``
* **Low variance gradients**: :class:`PathwiseGradient`
* **Direct PDE solving**: :class:`DeepGalerkinMethod`
* **Baseline comparison**: :class:`MonteCarloPolicyGradient`
