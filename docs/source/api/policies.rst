Policies Module
===============

The policies module defines policy representations for reinforcement learning agents.

Base Classes
------------

.. automodule:: macro_rl.policies.base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`Policy` abstract base class defines the interface for all policies:

* :meth:`~macro_rl.policies.base.Policy.act`: Sample action given state
* :meth:`~macro_rl.policies.base.Policy.act_deterministic`: Deterministic action (e.g., mean)
* :meth:`~macro_rl.policies.base.Policy.log_prob`: Log probability of action given state

Neural Policies
---------------

.. automodule:: macro_rl.policies.neural
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Neural network-based policy implementations:

Gaussian Policy
~~~~~~~~~~~~~~~

The :class:`GaussianPolicy` implements a stochastic policy with Gaussian distribution:

.. math::

   a \sim \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))

where :math:`\mu_\theta` and :math:`\Sigma_\theta` are neural networks.

**Reparameterization Trick:**

.. math::

   a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

This enables pathwise gradient estimation.

Deterministic Policy
~~~~~~~~~~~~~~~~~~~~

The :class:`DeterministicPolicy` implements a deterministic policy:

.. math::

   a = \mu_\theta(s)

Used in DDPG-style algorithms.

Example
~~~~~~~

.. code-block:: python

   from macro_rl.policies.neural import GaussianPolicy
   import torch

   # Create Gaussian policy
   policy = GaussianPolicy(
       state_dim=1,
       action_dim=2,
       hidden_dims=[128, 128],
       activation='tanh',
   )

   # Sample action
   state = torch.tensor([[5.0]])
   action = policy.act(state)
   print(f"Sampled action: {action}")

   # Get mean action (deterministic)
   mean_action = policy.act_deterministic(state)
   print(f"Mean action: {mean_action}")

   # Compute log probability
   log_prob = policy.log_prob(state, action)
   print(f"Log probability: {log_prob}")

Barrier Policies
----------------

.. automodule:: macro_rl.policies.barrier
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Threshold-based policies for benchmarking:

* :class:`~macro_rl.policies.barrier.BarrierPolicy`: Simple threshold policy
* :class:`~macro_rl.policies.barrier.TwoBarrierPolicy`: Two-threshold policy

Example
~~~~~~~

.. code-block:: python

   from macro_rl.policies.barrier import BarrierPolicy
   import torch

   # Barrier policy: pay dividend if cash > barrier
   policy = BarrierPolicy(
       barrier=5.0,
       action_below=torch.tensor([0.0, 1.0]),  # Issuance below barrier
       action_above=torch.tensor([1.0, 0.0]),  # Dividend above barrier
   )

   # Test policy
   state_below = torch.tensor([[3.0]])
   state_above = torch.tensor([[7.0]])

   action_below = policy.act(state_below)
   action_above = policy.act(state_above)

   print(f"Action below barrier: {action_below}")  # [0.0, 1.0]
   print(f"Action above barrier: {action_above}")  # [1.0, 0.0]

Tabular Policies
----------------

.. automodule:: macro_rl.policies.tabular
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Grid-based policies for finite state spaces:

* :class:`~macro_rl.policies.tabular.TabularPolicy`: Lookup table policy
* :class:`~macro_rl.policies.tabular.GridPolicy`: Interpolated grid policy

Example
~~~~~~~

.. code-block:: python

   from macro_rl.policies.tabular import GridPolicy
   import torch

   # Create 10x10 grid over state space [0, 10]
   policy = GridPolicy(
       state_bounds=torch.tensor([[0.0], [10.0]]),
       action_dim=2,
       grid_size=10,
   )

   # Set policy values at grid points
   # policy.table has shape (10, 2)
   policy.table = torch.rand(10, 2)

   # Query policy (uses interpolation)
   state = torch.tensor([[5.5]])
   action = policy.act(state)
   print(f"Interpolated action: {action}")
