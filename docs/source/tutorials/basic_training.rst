Basic Training Tutorial
=======================

This tutorial walks through training a model-based actor-critic agent on the GHM equity management problem.

Step 1: Import Required Modules
--------------------------------

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt

   from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
   from macro_rl.control.ghm_control import GHMControlSpec
   from macro_rl.rewards.ghm_rewards import GHMRewardFunction
   from macro_rl.networks.actor_critic import ActorCritic
   from macro_rl.solvers.actor_critic import ModelBasedActorCritic

Step 2: Define Model Parameters
--------------------------------

Set up the economic parameters for the GHM model:

.. code-block:: python

   params = GHMEquityParams(
       alpha=0.5,        # Profitability: higher = more cash flow
       mu=0.1,           # Drift rate: expected return
       r=0.05,           # Risk-free rate: discount factor
       sigma_A=0.2,      # Asset volatility: market risk
       sigma_X=0.3,      # Idiosyncratic volatility: firm-specific risk
       rho=0.7,          # Correlation: between market and firm shocks
       lambda_=0.1,      # Issuance cost: 10% of amount raised
       tau=0.35,         # Corporate tax rate
       phi=0.1,          # Bankruptcy cost: 10% of liquidation value
       c_max=10.0,       # Maximum cash level
       c_barrier=0.0,    # Bankruptcy threshold
   )

   print("Model Parameters:")
   print(f"  Profitability (alpha): {params.alpha}")
   print(f"  Expected return (mu): {params.mu}")
   print(f"  Risk-free rate (r): {params.r}")
   print(f"  Total volatility: {np.sqrt(params.sigma_A**2 + params.sigma_X**2):.3f}")

Step 3: Initialize Components
------------------------------

Create dynamics, control specification, and reward function:

.. code-block:: python

   # Dynamics: continuous-time SDE
   dynamics = GHMEquityDynamics(params)
   print(f"\nState space: [{dynamics.state_space.lower_bounds[0]:.1f}, "
         f"{dynamics.state_space.upper_bounds[0]:.1f}]")

   # Control specification: dividend + equity issuance
   control_spec = GHMControlSpec(params)
   print(f"Action dimension: {control_spec.action_dim}")
   print(f"Action bounds: [{control_spec.lower_bounds}, {control_spec.upper_bounds}]")

   # Reward function: dividends - issuance costs
   reward_fn = GHMRewardFunction(params, control_spec)

Step 4: Create Neural Network
------------------------------

Design the actor-critic architecture:

.. code-block:: python

   network = ActorCritic(
       state_dim=1,                      # Cash level
       action_dim=2,                     # Dividend + issuance
       hidden_dims=[128, 128, 64],       # 3-layer MLP
       activation='tanh',                # Activation function
       shared_trunk=True,                # Share features between actor and critic
       log_std_bounds=(-5, 2),           # Constrain policy variance
   )

   print(f"\nNetwork architecture:")
   print(f"  Input dimension: {network.state_dim}")
   print(f"  Hidden layers: {network.hidden_dims}")
   print(f"  Output dimensions: {network.action_dim} (policy), 1 (value)")
   print(f"  Total parameters: {sum(p.numel() for p in network.parameters()):,}")

Step 5: Initialize Solver
--------------------------

Configure the model-based actor-critic solver:

.. code-block:: python

   solver = ModelBasedActorCritic(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       actor_critic=network,

       # Simulation settings
       n_trajectories=32,                # Batch size
       n_steps=100,                      # Steps per trajectory
       dt=0.01,                          # Time discretization

       # Optimization settings
       actor_lr=1e-4,                    # Actor learning rate
       critic_lr=1e-3,                   # Critic learning rate (higher)

       # Loss configuration
       critic_loss_type="mc+hjb",        # Combine MC and HJB losses
       hjb_weight=0.1,                   # Weight for HJB residual
       td_lambda=0.95,                   # TD(λ) parameter

       # Regularization
       entropy_coef=0.01,                # Encourage exploration
       max_grad_norm=1.0,                # Gradient clipping
   )

   print(f"\nSolver configuration:")
   print(f"  Trajectories per iteration: {solver.n_trajectories}")
   print(f"  Time horizon: {solver.n_steps * solver.dt:.1f}s")
   print(f"  Critic loss: {solver.critic_loss_type}")

Step 6: Train the Agent
------------------------

Run the training loop with periodic logging:

.. code-block:: python

   print("\nStarting training...")

   history = solver.train(
       n_iterations=1000,               # Total training iterations
       log_freq=100,                    # Print every 100 iterations
   )

   print("\nTraining complete!")

**Expected output:**

.. code-block:: text

   Iteration 100: actor_loss=-12.34, critic_loss=2.56, mean_return=15.78
   Iteration 200: actor_loss=-18.92, critic_loss=1.23, mean_return=22.45
   Iteration 300: actor_loss=-23.11, critic_loss=0.87, mean_return=26.33
   ...
   Iteration 1000: actor_loss=-28.56, critic_loss=0.21, mean_return=30.12

Step 7: Evaluate Performance
-----------------------------

Test the learned policy on new episodes:

.. code-block:: python

   print("\nEvaluating learned policy...")

   eval_results = solver.evaluate(n_episodes=100)

   print(f"Mean return: {eval_results['mean_return']:.2f}")
   print(f"Std return: {eval_results['std_return']:.2f}")
   print(f"Min return: {eval_results['min_return']:.2f}")
   print(f"Max return: {eval_results['max_return']:.2f}")
   print(f"Mean episode length: {eval_results['mean_length']:.1f}")

Step 8: Visualize Training Progress
------------------------------------

Plot training metrics:

.. code-block:: python

   fig, axes = plt.subplots(2, 2, figsize=(12, 8))

   # Plot 1: Returns over time
   axes[0, 0].plot(history['mean_return'])
   axes[0, 0].set_xlabel('Iteration')
   axes[0, 0].set_ylabel('Mean Return')
   axes[0, 0].set_title('Training Returns')
   axes[0, 0].grid(True)

   # Plot 2: Actor loss
   axes[0, 1].plot(history['actor_loss'])
   axes[0, 1].set_xlabel('Iteration')
   axes[0, 1].set_ylabel('Actor Loss')
   axes[0, 1].set_title('Actor Loss')
   axes[0, 1].grid(True)

   # Plot 3: Critic loss
   axes[1, 0].plot(history['critic_loss'])
   axes[1, 0].set_xlabel('Iteration')
   axes[1, 0].set_ylabel('Critic Loss')
   axes[1, 0].set_title('Critic Loss')
   axes[1, 0].set_yscale('log')
   axes[1, 0].grid(True)

   # Plot 4: HJB residual
   if 'hjb_residual' in history:
       axes[1, 1].plot(history['hjb_residual'])
       axes[1, 1].set_xlabel('Iteration')
       axes[1, 1].set_ylabel('HJB Residual')
       axes[1, 1].set_title('HJB Residual')
       axes[1, 1].set_yscale('log')
       axes[1, 1].grid(True)

   plt.tight_layout()
   plt.savefig('training_progress.png', dpi=150)
   plt.show()

Step 9: Visualize Learned Policy
---------------------------------

Plot policy and value function across state space:

.. code-block:: python

   # Create grid over state space
   c_grid = torch.linspace(0.1, 10.0, 100).unsqueeze(1)

   # Evaluate policy and value
   with torch.no_grad():
       mean_actions, values = network(c_grid)
       dividends = mean_actions[:, 0].numpy()
       issuances = mean_actions[:, 1].numpy()
       values = values.squeeze().numpy()

   c_grid_np = c_grid.squeeze().numpy()

   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   # Plot 1: Dividend policy
   axes[0].plot(c_grid_np, dividends)
   axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
   axes[0].set_xlabel('Cash Level')
   axes[0].set_ylabel('Dividend Rate')
   axes[0].set_title('Dividend Policy')
   axes[0].grid(True)

   # Plot 2: Issuance policy
   axes[1].plot(c_grid_np, issuances)
   axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
   axes[1].set_xlabel('Cash Level')
   axes[1].set_ylabel('Issuance Rate')
   axes[1].set_title('Equity Issuance Policy')
   axes[1].grid(True)

   # Plot 3: Value function
   axes[2].plot(c_grid_np, values)
   axes[2].set_xlabel('Cash Level')
   axes[2].set_ylabel('Value')
   axes[2].set_title('Value Function')
   axes[2].grid(True)

   plt.tight_layout()
   plt.savefig('learned_policy.png', dpi=150)
   plt.show()

Step 10: Save Trained Model
----------------------------

Save the network weights for later use:

.. code-block:: python

   # Save model
   torch.save({
       'network_state_dict': network.state_dict(),
       'params': params,
       'eval_results': eval_results,
       'history': history,
   }, 'ghm_trained_model.pt')

   print("Model saved to 'ghm_trained_model.pt'")

   # Load model later
   checkpoint = torch.load('ghm_trained_model.pt')
   network.load_state_dict(checkpoint['network_state_dict'])
   print("Model loaded successfully")

Complete Script
---------------

Here's the complete training script:

.. code-block:: python

   import torch
   from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
   from macro_rl.control.ghm_control import GHMControlSpec
   from macro_rl.rewards.ghm_rewards import GHMRewardFunction
   from macro_rl.networks.actor_critic import ActorCritic
   from macro_rl.solvers.actor_critic import ModelBasedActorCritic

   # Setup
   params = GHMEquityParams()
   dynamics = GHMEquityDynamics(params)
   control_spec = GHMControlSpec(params)
   reward_fn = GHMRewardFunction(params, control_spec)

   # Network
   network = ActorCritic(state_dim=1, action_dim=2, hidden_dims=[128, 128])

   # Solver
   solver = ModelBasedActorCritic(
       dynamics, control_spec, reward_fn, network,
       n_trajectories=32, n_steps=100, dt=0.01,
       actor_lr=1e-4, critic_lr=1e-3,
       critic_loss_type="mc+hjb", hjb_weight=0.1,
   )

   # Train
   history = solver.train(n_iterations=1000, log_freq=100)

   # Evaluate
   results = solver.evaluate(n_episodes=100)
   print(f"Mean return: {results['mean_return']:.2f}")

   # Save
   torch.save(network.state_dict(), 'model.pt')

Next Steps
----------

* :doc:`custom_dynamics`: Implement custom dynamics
* :doc:`validation`: Validate learned solutions
* :doc:`advanced_solvers`: Try other algorithms
