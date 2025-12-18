Getting Started
===============

Installation
------------

Requirements
~~~~~~~~~~~~

* Python >= 3.8
* PyTorch >= 1.12.0
* NumPy >= 1.21.0
* Gymnasium >= 0.26.0 (optional, for RL environments)

Install from Source
~~~~~~~~~~~~~~~~~~~

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/yourusername/GHM-RL.git
   cd GHM-RL
   pip install -e .

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For visualization and analysis:

.. code-block:: bash

   pip install matplotlib seaborn pandas

For RL library integration:

.. code-block:: bash

   # Stable-Baselines3
   pip install stable-baselines3

   # RLlib
   pip install ray[rllib]

Quick Start
-----------

Basic Training Example
~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example of training an agent on the GHM equity model:

.. code-block:: python

   import torch
   from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
   from macro_rl.control.ghm_control import GHMControlSpec
   from macro_rl.rewards.ghm_rewards import GHMRewardFunction
   from macro_rl.networks.actor_critic import ActorCritic
   from macro_rl.solvers.actor_critic import ModelBasedActorCritic

   # 1. Define model parameters
   params = GHMEquityParams(
       alpha=0.5,      # Profitability
       mu=0.1,         # Drift
       r=0.05,         # Risk-free rate
       sigma_A=0.2,    # Asset volatility
       sigma_X=0.3,    # Idiosyncratic volatility
       rho=0.7,        # Correlation
       lambda_=0.1,    # Equity issuance cost
       tau=0.35,       # Tax rate
       phi=0.1,        # Bankruptcy cost
   )

   # 2. Initialize dynamics, control spec, and reward function
   dynamics = GHMEquityDynamics(params)
   control_spec = GHMControlSpec(params)
   reward_fn = GHMRewardFunction(params, control_spec)

   # 3. Create actor-critic network
   network = ActorCritic(
       state_dim=1,
       action_dim=2,
       hidden_dims=[128, 128],
       activation='tanh',
       shared_trunk=True,
   )

   # 4. Initialize solver
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

   # 5. Train
   print("Training...")
   history = solver.train(
       n_iterations=1000,
       log_freq=100,
   )

   # 6. Evaluate
   print("Evaluating...")
   eval_results = solver.evaluate(n_episodes=100)
   print(f"Mean return: {eval_results['mean_return']:.2f}")
   print(f"Std return: {eval_results['std_return']:.2f}")

   # 7. Save model
   torch.save(network.state_dict(), "ghm_model.pt")

Using Gymnasium Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides Gymnasium-compatible environments:

.. code-block:: python

   from macro_rl.envs.ghm_equity_env import GHMEquityEnv
   from macro_rl.dynamics.ghm_equity import GHMEquityParams
   import numpy as np

   # Create environment
   params = GHMEquityParams()
   env = GHMEquityEnv(
       params=params,
       dt=0.01,
       max_steps=1000,
       initial_state=5.0,
   )

   # Standard Gymnasium interface
   state, info = env.reset()
   done = False
   total_reward = 0

   while not done:
       # Random policy
       action = env.action_space.sample()

       next_state, reward, terminated, truncated, info = env.step(action)

       total_reward += reward
       done = terminated or truncated
       state = next_state

   print(f"Total reward: {total_reward:.2f}")

Key Concepts
------------

Continuous-Time Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework uses continuous-time stochastic differential equations (SDEs):

.. math::

   dX_t = \mu(X_t, a_t) dt + \sigma(X_t) dW_t

where:

* :math:`X_t`: State (e.g., cash holdings)
* :math:`a_t`: Control/action (e.g., dividend, issuance)
* :math:`\mu`: Drift function
* :math:`\sigma`: Diffusion function
* :math:`W_t`: Brownian motion

Objective Function
~~~~~~~~~~~~~~~~~~

The agent maximizes expected discounted utility:

.. math::

   V(x_0) = \max_{\pi} \mathbb{E}^\pi\left[\int_0^\tau e^{-\rho t} r(X_t, a_t) dt + e^{-\rho \tau} g(X_\tau)\right]

where:

* :math:`\pi`: Policy
* :math:`r`: Flow reward
* :math:`g`: Terminal reward
* :math:`\rho`: Discount rate
* :math:`\tau`: Stopping time

Model-Based Learning
~~~~~~~~~~~~~~~~~~~~

The framework exploits known dynamics for efficiency:

1. **Sample initial states** from state space
2. **Simulate trajectories** using known dynamics
3. **Update policy/value** using gradients
4. **Validate** with HJB residual

This approach is more sample-efficient than model-free RL.

HJB Equation
~~~~~~~~~~~~

Optimal value functions satisfy the Hamilton-Jacobi-Bellman equation:

.. math::

   \rho V(x) = \max_a \left\{ r(x, a) + \mu(x, a)^\top \nabla_x V + \frac{1}{2} \text{tr}(\sigma\sigma^\top \nabla_x^2 V) \right\}

The framework provides tools to:

* Minimize HJB residual directly (Deep Galerkin Method)
* Use HJB residual as auxiliary loss (Actor-Critic)
* Validate learned solutions

Next Steps
----------

* :doc:`tutorials/index`: Step-by-step tutorials
* :doc:`concepts`: Detailed conceptual overview
* :doc:`examples`: Complete examples
* :doc:`api/index`: API reference

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: Training is unstable

**Solution**:

* Reduce learning rates (``actor_lr``, ``critic_lr``)
* Increase batch size (``n_trajectories``)
* Enable gradient clipping
* Use smaller time step (``dt``)

**Issue**: HJB residual is large

**Solution**:

* Increase network capacity (``hidden_dims``)
* Train longer (``n_iterations``)
* Adjust HJB weight (``hjb_weight``)
* Check boundary conditions

**Issue**: Out of memory

**Solution**:

* Reduce batch size (``n_trajectories``)
* Reduce time steps (``n_steps``)
* Use smaller network (``hidden_dims``)
* Enable gradient checkpointing

GPU Usage
~~~~~~~~~

To use GPU acceleration:

.. code-block:: python

   import torch

   # Check if CUDA is available
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")

   # Move model to GPU
   network = network.to(device)

   # Solver automatically uses device of network
   solver = ModelBasedActorCritic(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       actor_critic=network,
       # ... other parameters
   )

Reproducibility
~~~~~~~~~~~~~~~

For reproducible results:

.. code-block:: python

   import torch
   import numpy as np
   import random

   def set_seed(seed=42):
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       np.random.seed(seed)
       random.seed(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False

   set_seed(42)

Getting Help
------------

* **Documentation**: https://ghm-rl.readthedocs.io
* **GitHub Issues**: https://github.com/yourusername/GHM-RL/issues
* **Email**: your.email@example.com

Citation
--------

If you use this framework in your research, please cite:

.. code-block:: bibtex

   @software{ghm_rl_2025,
     title = {MacroRL: Model-Based Reinforcement Learning for Continuous-Time Macro Finance},
     author = {Zibo Zhao},
     year = {2025},
     url = {https://github.com/zbzhaoecon/GHM-RL},
   }
