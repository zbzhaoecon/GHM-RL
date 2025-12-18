Environments Module
===================

The environments module provides Gymnasium-compatible wrappers for continuous-time models.

Base Classes
------------

.. automodule:: macro_rl.envs.base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`ContinuousTimeEnv` abstract base class extends Gymnasium's ``Env`` interface for continuous-time models:

Gymnasium Interface
~~~~~~~~~~~~~~~~~~~

* :meth:`~macro_rl.envs.base.ContinuousTimeEnv.reset`: Reset environment to initial state
* :meth:`~macro_rl.envs.base.ContinuousTimeEnv.step`: Take action and advance time by ``dt``
* :meth:`~macro_rl.envs.base.ContinuousTimeEnv.render`: Visualize current state (optional)
* :meth:`~macro_rl.envs.base.ContinuousTimeEnv.close`: Clean up resources

Additional Methods
~~~~~~~~~~~~~~~~~~

* :meth:`~macro_rl.envs.base.ContinuousTimeEnv.seed`: Set random seed
* :meth:`~macro_rl.envs.base.ContinuousTimeEnv.get_state`: Get current state
* :meth:`~macro_rl.envs.base.ContinuousTimeEnv.set_state`: Set current state

Spaces
~~~~~~

The environment defines:

* ``observation_space``: Gymnasium Box space for states
* ``action_space``: Gymnasium Box space for actions

Example Base Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import gymnasium as gym
   from macro_rl.envs.base import ContinuousTimeEnv

   # All continuous-time environments follow Gymnasium interface
   env = ...  # Some ContinuousTimeEnv subclass

   # Standard Gymnasium loop
   state, info = env.reset()
   done = False
   total_reward = 0

   while not done:
       action = policy(state)
       next_state, reward, terminated, truncated, info = env.step(action)
       done = terminated or truncated
       total_reward += reward
       state = next_state

   print(f"Total reward: {total_reward}")

GHM Equity Environment
----------------------

.. automodule:: macro_rl.envs.ghm_equity_env
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`GHMEquityEnv` provides a Gymnasium environment for the 1D GHM equity management problem.

State Space
~~~~~~~~~~~

* **State**: Cash holdings :math:`c \in [c_{\text{barrier}}, c_{\text{max}}]`
* **Observation Space**: ``Box(low=c_barrier, high=c_max, shape=(1,))``

Action Space
~~~~~~~~~~~~

* **Action**: :math:`(a_L, a_E)` where

  * :math:`a_L \in [0, c]`: Dividend payout
  * :math:`a_E \geq 0`: Equity issuance

* **Action Space**: ``Box(low=[0, 0], high=[c_max, a_E_max], shape=(2,))``

Dynamics
~~~~~~~~

.. math::

   dc_t = [\alpha \mu - a_{L,t} + a_{E,t}] dt + \alpha (\sigma_A dW^A_t + \sigma_X dW^X_t)

Rewards
~~~~~~~

* **Flow Reward**: :math:`r_t = a_{L,t} - (1 + \lambda) a_{E,t}`
* **Terminal Reward**: :math:`g(c_\tau) = (1 - \phi) c_\tau`

Termination
~~~~~~~~~~~

Episode terminates when:

1. **Bankruptcy**: :math:`c_t \leq c_{\text{barrier}}`
2. **Max steps**: Time horizon reached

Parameters
~~~~~~~~~~

The environment is initialized with:

* ``params``: :class:`~macro_rl.dynamics.ghm_equity.GHMEquityParams`
* ``dt``: Time discretization (default: 0.01)
* ``max_steps``: Maximum steps per episode (default: 1000)
* ``initial_state``: Initial cash level or distribution

Example
~~~~~~~

.. code-block:: python

   from macro_rl.envs.ghm_equity_env import GHMEquityEnv
   from macro_rl.dynamics.ghm_equity import GHMEquityParams
   import numpy as np

   # Create environment
   params = GHMEquityParams(
       alpha=0.5,
       mu=0.1,
       r=0.05,
       sigma_A=0.2,
       sigma_X=0.3,
       rho=0.7,
       lambda_=0.1,
   )

   env = GHMEquityEnv(
       params=params,
       dt=0.01,
       max_steps=1000,
       initial_state=5.0,  # Start with cash = 5.0
   )

   # Run episode
   state, info = env.reset()
   done = False
   total_reward = 0
   steps = 0

   while not done:
       # Random policy (for demonstration)
       dividend = np.random.uniform(0, min(state[0], 1.0))
       issuance = np.random.uniform(0, 0.5)
       action = np.array([dividend, issuance])

       next_state, reward, terminated, truncated, info = env.step(action)

       total_reward += reward
       steps += 1
       done = terminated or truncated
       state = next_state

   print(f"Episode finished after {steps} steps")
   print(f"Total reward: {total_reward:.2f}")
   print(f"Final cash: {state[0]:.2f}")
   print(f"Bankruptcy: {info.get('bankruptcy', False)}")

Integration with RL Libraries
------------------------------

The Gymnasium interface allows seamless integration with standard RL libraries:

Stable-Baselines3 Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from stable_baselines3 import PPO
   from macro_rl.envs.ghm_equity_env import GHMEquityEnv
   from macro_rl.dynamics.ghm_equity import GHMEquityParams

   # Create vectorized environments
   def make_env():
       return GHMEquityEnv(params=GHMEquityParams())

   env = make_env()

   # Train PPO
   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=100000)

   # Evaluate
   mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
   print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

RLlib Example
~~~~~~~~~~~~~

.. code-block:: python

   from ray.rllib.algorithms.ppo import PPOConfig
   from ray.tune.registry import register_env

   def env_creator(env_config):
       return GHMEquityEnv(params=GHMEquityParams(**env_config))

   register_env("ghm_equity", env_creator)

   config = (
       PPOConfig()
       .environment("ghm_equity")
       .training(train_batch_size=4000)
   )

   algo = config.build()
   for i in range(100):
       result = algo.train()
       print(f"Iteration {i}: reward = {result['episode_reward_mean']:.2f}")

Custom Environments
-------------------

To create a custom continuous-time environment:

1. Subclass :class:`ContinuousTimeEnv`
2. Implement :meth:`_setup_spaces`
3. Implement :meth:`_compute_drift`
4. Implement :meth:`_compute_diffusion`
5. Implement :meth:`_compute_reward`
6. Implement :meth:`_check_termination`

Example Template
~~~~~~~~~~~~~~~~

.. code-block:: python

   from macro_rl.envs.base import ContinuousTimeEnv
   import gymnasium as gym
   import torch

   class MyCustomEnv(ContinuousTimeEnv):
       def __init__(self, params, dt=0.01, max_steps=1000):
           self.params = params
           super().__init__(dt=dt, max_steps=max_steps)

       def _setup_spaces(self):
           self.observation_space = gym.spaces.Box(
               low=0.0, high=10.0, shape=(1,), dtype=np.float32
           )
           self.action_space = gym.spaces.Box(
               low=-1.0, high=1.0, shape=(1,), dtype=np.float32
           )

       def _compute_drift(self, state, action):
           # Implement drift μ(x, a)
           return self.params.mu * state + action

       def _compute_diffusion(self, state):
           # Implement diffusion σ(x)
           return self.params.sigma * torch.ones_like(state)

       def _compute_reward(self, state, action, next_state):
           # Implement reward function
           return action[0] - 0.5 * action[0]**2

       def _check_termination(self, state):
           # Check if episode should terminate
           return state[0] <= 0 or state[0] >= 10.0
