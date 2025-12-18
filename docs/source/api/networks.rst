Networks Module
===============

The networks module provides PyTorch neural network architectures for policies and value functions.

Policy Networks
---------------

.. automodule:: macro_rl.networks.policy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`GaussianPolicy` implements a fully functional Gaussian policy network with reparameterization:

Architecture
~~~~~~~~~~~~

.. math::

   \text{Input: } s \in \mathbb{R}^{d_s} \\
   \text{Hidden: } h = \text{MLP}(s) \\
   \text{Mean: } \mu = W_\mu h + b_\mu \\
   \text{Log-std: } \log\sigma = W_\sigma h + b_\sigma \\
   \text{Action: } a = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

Key Methods
~~~~~~~~~~~

* :meth:`~macro_rl.networks.policy.GaussianPolicy.forward`: Compute mean and log-std
* :meth:`~macro_rl.networks.policy.GaussianPolicy.sample`: Sample action with reparameterization
* :meth:`~macro_rl.networks.policy.GaussianPolicy.sample_with_noise`: Sample with pre-specified noise
* :meth:`~macro_rl.networks.policy.GaussianPolicy.log_prob`: Compute log probability
* :meth:`~macro_rl.networks.policy.GaussianPolicy.entropy`: Compute policy entropy

Example
~~~~~~~

.. code-block:: python

   from macro_rl.networks.policy import GaussianPolicy
   import torch

   # Create policy network
   policy = GaussianPolicy(
       state_dim=1,
       action_dim=2,
       hidden_dims=[128, 128, 64],
       activation='tanh',
       log_std_bounds=(-5, 2),
   )

   # Forward pass
   state = torch.tensor([[5.0]])
   mean, log_std = policy.forward(state)
   print(f"Mean: {mean}, Log-std: {log_std}")

   # Sample action
   action, log_prob = policy.sample(state)
   print(f"Action: {action}, Log-prob: {log_prob}")

   # Deterministic action
   action_det = policy.mean_action(state)
   print(f"Deterministic action: {action_det}")

   # Entropy
   entropy = policy.entropy(state)
   print(f"Entropy: {entropy}")

Value Networks
--------------

.. automodule:: macro_rl.networks.value
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`ValueNetwork` implements a value function with gradient computation for HJB validation:

Architecture
~~~~~~~~~~~~

.. math::

   V_\phi(s) = W_{\text{out}} \cdot \text{MLP}_\phi(s) + b_{\text{out}}

Key Methods
~~~~~~~~~~~

* :meth:`~macro_rl.networks.value.ValueNetwork.forward`: Compute value :math:`V(s)`
* :meth:`~macro_rl.networks.value.ValueNetwork.forward_with_grad`: Compute value and gradients :math:`(V, \nabla_s V, \nabla_s^2 V)`

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

The ``forward_with_grad`` method uses PyTorch autograd to compute:

.. math::

   V(s), \quad \nabla_s V(s), \quad \nabla_s^2 V(s)

These gradients are needed for HJB residual computation:

.. math::

   \rho V(s) = \max_a \left\{ r(s, a) + \mu(s, a)^\top \nabla_s V + \frac{1}{2} \text{tr}(\sigma\sigma^\top \nabla_s^2 V) \right\}

Example
~~~~~~~

.. code-block:: python

   from macro_rl.networks.value import ValueNetwork
   import torch

   # Create value network
   value_net = ValueNetwork(
       state_dim=1,
       hidden_dims=[128, 128, 64],
       activation='tanh',
   )

   # Forward pass
   state = torch.tensor([[5.0]], requires_grad=True)
   value = value_net(state)
   print(f"Value: {value}")

   # Compute with gradients
   value, grad, hess = value_net.forward_with_grad(state)
   print(f"V(s) = {value}")
   print(f"∇V(s) = {grad}")
   print(f"∇²V(s) = {hess}")

Actor-Critic Networks
---------------------

.. automodule:: macro_rl.networks.actor_critic
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`ActorCritic` combines policy and value networks with optional shared trunk:

Architecture Options
~~~~~~~~~~~~~~~~~~~~

1. **Separate Networks:**

   .. math::

      \pi_\theta(s) = \text{PolicyNet}_\theta(s) \\
      V_\phi(s) = \text{ValueNet}_\phi(s)

2. **Shared Trunk:**

   .. math::

      h = \text{MLP}_{\text{trunk}}(s) \\
      \pi_\theta(s) = \text{PolicyHead}_\theta(h) \\
      V_\phi(s) = \text{ValueHead}_\phi(h)

Key Methods
~~~~~~~~~~~

* :meth:`~macro_rl.networks.actor_critic.ActorCritic.forward`: Return (mean_action, value)
* :meth:`~macro_rl.networks.actor_critic.ActorCritic.act`: Sample action from policy
* :meth:`~macro_rl.networks.actor_critic.ActorCritic.evaluate`: Compute value only
* :meth:`~macro_rl.networks.actor_critic.ActorCritic.evaluate_actions`: Compute (value, log_prob, entropy)
* :meth:`~macro_rl.networks.actor_critic.ActorCritic.sample_with_noise`: Sample with pre-specified noise
* :meth:`~macro_rl.networks.actor_critic.ActorCritic.evaluate_with_grad`: Compute value with gradients for HJB

Example
~~~~~~~

.. code-block:: python

   from macro_rl.networks.actor_critic import ActorCritic
   import torch

   # Create actor-critic with shared trunk
   ac = ActorCritic(
       state_dim=1,
       action_dim=2,
       hidden_dims=[128, 128],
       activation='tanh',
       shared_trunk=True,
   )

   # Forward pass
   state = torch.tensor([[5.0]])
   mean_action, value = ac.forward(state)
   print(f"Mean action: {mean_action}, Value: {value}")

   # Sample action
   action = ac.act(state, deterministic=False)
   print(f"Sampled action: {action}")

   # Evaluate action
   value, log_prob, entropy = ac.evaluate_actions(state, action)
   print(f"Value: {value}, Log-prob: {log_prob}, Entropy: {entropy}")

   # HJB gradients
   value, grad, hess = ac.evaluate_with_grad(state)
   print(f"Value gradients: {grad}, Hessian: {hess}")

MLP Building Block
~~~~~~~~~~~~~~~~~~

The :class:`MLP` class provides a simple multi-layer perceptron:

.. code-block:: python

   from macro_rl.networks.actor_critic import MLP

   mlp = MLP(
       input_dim=10,
       hidden_dims=[128, 64],
       output_dim=5,
       activation='relu',
   )

   x = torch.randn(32, 10)
   y = mlp(x)  # Shape: (32, 5)
