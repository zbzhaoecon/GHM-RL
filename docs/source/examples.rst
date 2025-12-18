Examples
========

Complete examples demonstrating various use cases of the GHM-RL framework.

Training GHM Equity Model
--------------------------

Basic training with actor-critic:

.. literalinclude:: ../../scripts/train_actor_critic_ghm_model1.py
   :language: python
   :caption: scripts/train_actor_critic_ghm_model1.py

Using Gymnasium Environment
----------------------------

Standard RL interface:

.. code-block:: python

   from macro_rl.envs.ghm_equity_env import GHMEquityEnv
   from macro_rl.dynamics.ghm_equity import GHMEquityParams
   import numpy as np

   # Create environment
   params = GHMEquityParams()
   env = GHMEquityEnv(params=params, dt=0.01, max_steps=1000)

   # Run episodes
   for episode in range(10):
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

       print(f"Episode {episode + 1}: Return = {total_reward:.2f}")

Custom Dynamics Model
---------------------

Implementing a custom model:

.. code-block:: python

   from macro_rl.dynamics.base import ContinuousTimeDynamics
   from macro_rl.core.state_space import StateSpace
   import torch

   class MyCustomDynamics(ContinuousTimeDynamics):
       """Custom stochastic volatility model"""

       def __init__(self, mu=0.1, kappa=2.0, theta=0.04, sigma=0.3):
           self.mu = mu
           self.kappa = kappa
           self.theta = theta
           self.sigma = sigma

           # State: [price, volatility]
           self._state_space = StateSpace(
               lower_bounds=torch.tensor([0.1, 0.01]),
               upper_bounds=torch.tensor([100.0, 1.0]),
               state_names=['price', 'volatility']
           )

       @property
       def state_space(self):
           return self._state_space

       def drift(self, state, action=None):
           """
           dS = μ S dt
           dV = κ(θ - V) dt
           """
           price = state[:, 0:1]
           volatility = state[:, 1:2]

           drift_price = self.mu * price
           drift_vol = self.kappa * (self.theta - volatility)

           return torch.cat([drift_price, drift_vol], dim=1)

       def diffusion(self, state):
           """
           dS: √V S dW1
           dV: σ √V dW2
           """
           price = state[:, 0:1]
           volatility = state[:, 1:2]

           diff_price = torch.sqrt(volatility) * price
           diff_vol = self.sigma * torch.sqrt(volatility)

           return torch.cat([diff_price, diff_vol], dim=1)

       def discount_rate(self):
           return 0.05

   # Use custom dynamics
   dynamics = MyCustomDynamics()
   print(f"State space: {dynamics.state_space}")

Integration with Stable-Baselines3
-----------------------------------

Using SB3 with Gymnasium environment:

.. code-block:: python

   from stable_baselines3 import PPO
   from stable_baselines3.common.vec_env import DummyVecEnv
   from stable_baselines3.common.evaluation import evaluate_policy
   from macro_rl.envs.ghm_equity_env import GHMEquityEnv
   from macro_rl.dynamics.ghm_equity import GHMEquityParams

   # Create environment
   def make_env():
       params = GHMEquityParams()
       return GHMEquityEnv(params=params)

   env = DummyVecEnv([make_env])

   # Train PPO
   model = PPO(
       "MlpPolicy",
       env,
       learning_rate=3e-4,
       n_steps=2048,
       batch_size=64,
       verbose=1,
   )

   print("Training PPO...")
   model.learn(total_timesteps=100000)

   # Evaluate
   mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
   print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

   # Save model
   model.save("ppo_ghm_equity")

HJB Validation
--------------

Validating learned solutions:

.. code-block:: python

   from macro_rl.validation.hjb_residual import HJBValidator
   from macro_rl.networks.value import ValueNetwork
   import torch

   # Assume we have trained network
   value_net = ValueNetwork(state_dim=1, hidden_dims=[128, 128])
   # ... (load trained weights)

   # Create validator
   validator = HJBValidator(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       value_network=value_net,
   )

   # Sample test states
   test_states = dynamics.state_space.sample_uniform(1000)

   # Compute HJB residual
   residuals = validator.compute_residual(test_states)

   print(f"HJB Residual Statistics:")
   print(f"  Mean: {residuals.mean():.6f}")
   print(f"  Std: {residuals.std():.6f}")
   print(f"  Max: {residuals.max():.6f}")
   print(f"  Median: {residuals.median():.6f}")

   # Plot residual distribution
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 4))

   plt.subplot(1, 2, 1)
   plt.scatter(test_states.squeeze().numpy(), residuals.numpy(), alpha=0.5)
   plt.xlabel('Cash Level')
   plt.ylabel('HJB Residual')
   plt.title('HJB Residual vs State')
   plt.grid(True)

   plt.subplot(1, 2, 2)
   plt.hist(residuals.numpy(), bins=50, density=True)
   plt.xlabel('HJB Residual')
   plt.ylabel('Density')
   plt.title('Residual Distribution')
   plt.grid(True)

   plt.tight_layout()
   plt.savefig('hjb_validation.png')
   plt.show()

Pathwise Gradient Training
---------------------------

Using differentiable simulation:

.. code-block:: python

   from macro_rl.solvers.pathwise import PathwiseGradient
   from macro_rl.networks.policy import GaussianPolicy

   # Create policy
   policy = GaussianPolicy(
       state_dim=1,
       action_dim=2,
       hidden_dims=[128, 128],
   )

   # Create solver
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

   # Train with low-variance gradients
   history = solver.train(
       n_iterations=5000,
       log_freq=500,
   )

   print(f"Final mean return: {history['mean_return'][-1]:.2f}")

Deep Galerkin Method
--------------------

Direct PDE solving:

.. code-block:: python

   from macro_rl.solvers.deep_galerkin import DeepGalerkinMethod
   from macro_rl.networks.value import ValueNetwork
   from macro_rl.numerics.sampling import sobol_sample

   # Create value network
   value_net = ValueNetwork(
       state_dim=1,
       hidden_dims=[256, 256, 128],
   )

   # Create solver
   solver = DeepGalerkinMethod(
       dynamics=dynamics,
       control_spec=control_spec,
       reward_function=reward_fn,
       value_network=value_net,
       n_interior=1000,
       n_boundary=100,
       learning_rate=1e-3,
       boundary_weight=10.0,
       sampler='sobol',  # Low-discrepancy sampling
   )

   # Minimize HJB residual
   history = solver.train(
       n_iterations=20000,
       log_freq=1000,
   )

   # Extract optimal policy
   policy = solver.extract_policy()

   print("Training complete. Policy extracted from value function.")

Batch Processing
----------------

Training multiple models in parallel:

.. code-block:: python

   import torch.multiprocessing as mp
   from itertools import product

   def train_single_model(config):
       """Train a single model with given config"""
       alpha, lambda_ = config

       params = GHMEquityParams(alpha=alpha, lambda_=lambda_)
       dynamics = GHMEquityDynamics(params)
       control_spec = GHMControlSpec(params)
       reward_fn = GHMRewardFunction(params, control_spec)

       network = ActorCritic(state_dim=1, action_dim=2, hidden_dims=[128, 128])

       solver = ModelBasedActorCritic(
           dynamics, control_spec, reward_fn, network,
           n_trajectories=32, n_steps=100, dt=0.01,
       )

       history = solver.train(n_iterations=1000)
       results = solver.evaluate(n_episodes=100)

       return {
           'alpha': alpha,
           'lambda': lambda_,
           'mean_return': results['mean_return'],
           'history': history,
       }

   if __name__ == '__main__':
       # Grid of parameters
       alphas = [0.3, 0.5, 0.7]
       lambdas = [0.05, 0.1, 0.15]
       configs = list(product(alphas, lambdas))

       # Train in parallel
       with mp.Pool(processes=4) as pool:
           results = pool.map(train_single_model, configs)

       # Analyze results
       for result in results:
           print(f"α={result['alpha']:.1f}, λ={result['lambda']:.2f}: "
                 f"Return={result['mean_return']:.2f}")

Hyperparameter Tuning
----------------------

Using Optuna for hyperparameter optimization:

.. code-block:: python

   import optuna

   def objective(trial):
       """Objective function for hyperparameter tuning"""

       # Suggest hyperparameters
       hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
       n_layers = trial.suggest_int('n_layers', 2, 4)
       actor_lr = trial.suggest_float('actor_lr', 1e-5, 1e-3, log=True)
       critic_lr = trial.suggest_float('critic_lr', 1e-4, 1e-2, log=True)
       hjb_weight = trial.suggest_float('hjb_weight', 0.01, 1.0, log=True)

       # Create network
       hidden_dims = [hidden_size] * n_layers
       network = ActorCritic(state_dim=1, action_dim=2, hidden_dims=hidden_dims)

       # Create solver
       solver = ModelBasedActorCritic(
           dynamics, control_spec, reward_fn, network,
           n_trajectories=32, n_steps=100, dt=0.01,
           actor_lr=actor_lr,
           critic_lr=critic_lr,
           hjb_weight=hjb_weight,
       )

       # Train
       history = solver.train(n_iterations=500)

       # Evaluate
       results = solver.evaluate(n_episodes=50)

       return results['mean_return']

   # Run optimization
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=50)

   print(f"Best hyperparameters: {study.best_params}")
   print(f"Best return: {study.best_value:.2f}")

See Also
--------

* :doc:`getting_started`: Quick start guide
* :doc:`tutorials/index`: Step-by-step tutorials
* :doc:`api/index`: API reference
