Rewards Module
==============

The rewards module defines objective functions for reinforcement learning, including flow rewards and terminal values.

Base Classes
------------

.. automodule:: macro_rl.rewards.base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`RewardFunction` abstract base class defines the interface for all reward functions:

.. math::

   J = \mathbb{E}\left[\int_0^\tau e^{-\rho t} r(X_t, a_t) dt + e^{-\rho \tau} g(X_\tau)\right]

Key Methods
~~~~~~~~~~~

* :meth:`~macro_rl.rewards.base.RewardFunction.step_reward`: Flow reward :math:`r(x, a)`
* :meth:`~macro_rl.rewards.base.RewardFunction.terminal_reward`: Terminal value :math:`g(x)`

GHM Reward Functions
--------------------

.. automodule:: macro_rl.rewards.ghm_rewards
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`GHMRewardFunction` implements the objective for the equity management problem:

Flow Reward
~~~~~~~~~~~

.. math::

   r(c, a_L, a_E) = a_L - (1 + \lambda) a_E

The firm receives dividends :math:`a_L` but pays issuance cost :math:`(1 + \lambda) a_E`.

Terminal Value
~~~~~~~~~~~~~~

.. math::

   g(c_\tau) = (1 - \phi) c_\tau

At bankruptcy (:math:`c_\tau = 0`), the firm receives liquidation value with cost :math:`\phi`.

Example
~~~~~~~

.. code-block:: python

   from macro_rl.rewards.ghm_rewards import GHMRewardFunction
   from macro_rl.control.ghm_control import GHMControlSpec
   from macro_rl.dynamics.ghm_equity import GHMEquityParams
   import torch

   params = GHMEquityParams(lambda_=0.1, phi=0.1)
   control_spec = GHMControlSpec(params)
   reward_fn = GHMRewardFunction(params, control_spec)

   # Compute flow reward
   state = torch.tensor([[5.0]])
   action = torch.tensor([[0.5, 0.2]])  # Dividend = 0.5, Issuance = 0.2

   reward = reward_fn.step_reward(state, action)
   print(f"Flow reward: {reward}")  # 0.5 - 1.1 * 0.2 = 0.28

   # Terminal value at bankruptcy
   terminal_state = torch.tensor([[0.0]])
   terminal_value = reward_fn.terminal_reward(terminal_state)
   print(f"Terminal value: {terminal_value}")  # 0.0 (total loss)

   # Terminal value with remaining cash
   terminal_state = torch.tensor([[2.0]])
   terminal_value = reward_fn.terminal_reward(terminal_state)
   print(f"Terminal value: {terminal_value}")  # 0.9 * 2.0 = 1.8

GHM Reward with Penalty
~~~~~~~~~~~~~~~~~~~~~~~

The :class:`GHMRewardWithPenalty` extends the base reward with additional penalties:

.. math::

   r(c, a_L, a_E) = a_L - (1 + \lambda) a_E - \alpha_{\text{bankruptcy}} \mathbb{1}_{c < c_{\min}} - \alpha_{\text{variance}} \text{Var}(a)

Example
~~~~~~~

.. code-block:: python

   from macro_rl.rewards.ghm_rewards import GHMRewardWithPenalty

   reward_fn_penalty = GHMRewardWithPenalty(
       params,
       control_spec,
       bankruptcy_penalty=10.0,
       variance_penalty=0.1,
   )

   # Penalty for approaching bankruptcy
   state = torch.tensor([[0.1]])  # Low cash
   action = torch.tensor([[0.05, 0.0]])

   reward = reward_fn_penalty.step_reward(state, action)
   # Includes bankruptcy proximity penalty

Terminal Value Specifications
------------------------------

.. automodule:: macro_rl.rewards.terminal
   :members:
   :undoc-members:
   :show-inheritance:

Utility functions and classes for terminal value specifications:

* :func:`~macro_rl.rewards.terminal.liquidation_value`: Compute liquidation value
* :func:`~macro_rl.rewards.terminal.continuation_value`: Analytical continuation value (if available)

Example
~~~~~~~

.. code-block:: python

   from macro_rl.rewards.terminal import liquidation_value
   import torch

   # Liquidation with 10% cost
   cash = torch.tensor([[5.0], [0.0], [10.0]])
   liquidation_cost = 0.1

   terminal_values = liquidation_value(cash, liquidation_cost)
   print(terminal_values)  # [4.5, 0.0, 9.0]
