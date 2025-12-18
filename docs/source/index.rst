.. GHM-RL documentation master file

Welcome to GHM-RL Documentation
================================

**GHM-RL** is a model-based reinforcement learning framework for solving continuous-time optimal control problems in corporate finance, with a focus on the Gârleanu-Hackbarth-Morellec (GHM) equity management model.

The framework provides:

* **Continuous-time dynamics** with known drift and diffusion functions
* **Differentiable simulation** for pathwise gradient estimation
* **Model-based RL algorithms** including Actor-Critic, Pathwise Gradient, and Deep Galerkin Method
* **Neural network policies and value functions** with automatic differentiation
* **Gymnasium environment** wrappers for standard RL interfaces
* **Validation tools** for HJB residual checking and analytical comparison

Quick Start
-----------

.. code-block:: python

   import torch
   from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
   from macro_rl.solvers.actor_critic import ModelBasedActorCritic
   from macro_rl.networks.actor_critic import ActorCritic
   from macro_rl.control.ghm_control import GHMControlSpec
   from macro_rl.rewards.ghm_rewards import GHMRewardFunction

   # Define model parameters
   params = GHMEquityParams(
       alpha=0.5,    # Profitability
       mu=0.1,       # Drift
       r=0.05,       # Risk-free rate
       sigma_A=0.2,  # Asset volatility
       sigma_X=0.3,  # Idiosyncratic volatility
       rho=0.7,      # Correlation
       lambda_=0.1,  # Equity issuance cost
       tau=0.35,     # Tax rate
       phi=0.1,      # Bankruptcy cost
   )

   # Initialize dynamics, control spec, and reward function
   dynamics = GHMEquityDynamics(params)
   control_spec = GHMControlSpec(params)
   reward_fn = GHMRewardFunction(params, control_spec)

   # Create actor-critic network
   network = ActorCritic(
       state_dim=1,
       action_dim=2,
       hidden_dims=[128, 128],
   )

   # Initialize and train solver
   solver = ModelBasedActorCritic(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       actor_critic=network,
       n_trajectories=32,
       n_steps=100,
       dt=0.01,
   )

   # Train the model
   history = solver.train(n_iterations=1000)

Key Features
------------

Model-Based Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

The framework exploits known continuous-time dynamics to achieve superior sample efficiency:

* **Unlimited trajectory generation** at any initial state
* **Pathwise gradient estimation** through differentiable simulation
* **HJB residual minimization** for direct PDE solving
* **Monte Carlo methods** with model-generated rollouts

Continuous-Time Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Native support for stochastic differential equations (SDEs):

.. math::

   dX_t = \mu(X_t, a_t) dt + \sigma(X_t) dW_t

where :math:`X_t` is the state, :math:`a_t` is the control, and :math:`W_t` is a Brownian motion.

Neural Function Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch-based neural networks with automatic differentiation:

* **Gaussian policies** with state-dependent mean and variance
* **Value networks** with gradient computation for HJB validation
* **Actor-Critic architectures** with optional shared trunk
* **Flexible loss functions** (Monte Carlo, TD, HJB residual)

GHM Equity Model
~~~~~~~~~~~~~~~~

Complete implementation of the 1D GHM equity management problem:

.. math::

   V(c) = \max_{a_L, a_E} \mathbb{E}\left[\int_0^{\tau} e^{-\rho t} (a_{L,t} - (1+\lambda)a_{E,t}) dt + e^{-\rho \tau} L(c_\tau)\right]

subject to:

.. math::

   dc_t = [\alpha \mu - a_{L,t} + a_{E,t}] dt + \alpha (\sigma_A dW^A_t + \sigma_X dW^X_t)

with controls:

* :math:`a_L \in [0, c]`: Dividend payout (continuous)
* :math:`a_E \geq 0`: Equity issuance (impulse control)
* :math:`\lambda`: Proportional issuance cost

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   tutorials/index
   concepts
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/dynamics
   api/simulation
   api/control
   api/rewards
   api/policies
   api/networks
   api/values
   api/solvers
   api/envs
   api/validation
   api/numerics
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
