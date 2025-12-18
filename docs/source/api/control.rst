Control Module
==============

The control module defines control specifications with bounds, normalization, and feasibility constraints.

Base Classes
------------

.. automodule:: macro_rl.control.base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`ControlSpec` abstract base class defines the interface for all control specifications:

* Action bounds and dimensions
* Normalization and denormalization
* Feasibility masking
* Action clipping

Key Methods
~~~~~~~~~~~

* :meth:`~macro_rl.control.base.ControlSpec.normalize_action`: Normalize actions to [0, 1] or [-1, 1]
* :meth:`~macro_rl.control.base.ControlSpec.denormalize_action`: Denormalize to original bounds
* :meth:`~macro_rl.control.base.ControlSpec.clip_action`: Clip actions to feasible region
* :meth:`~macro_rl.control.base.ControlSpec.get_action_mask`: Compute state-dependent feasibility mask

GHM Control Specification
--------------------------

.. automodule:: macro_rl.control.ghm_control
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`GHMControlSpec` class implements two-control specification for the GHM equity model:

Control Variables
~~~~~~~~~~~~~~~~~

1. **Dividend payout** :math:`a_L \in [0, c]`: Continuous flow control
2. **Equity issuance** :math:`a_E \geq 0`: Impulse control with cost :math:`\lambda`

State-Dependent Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Dividend constraint: :math:`a_L \leq c` (cannot pay more than cash on hand)
* Non-negativity: :math:`a_L, a_E \geq 0`
* Upper bounds: :math:`a_L \leq c`, :math:`a_E \leq a_{E,\max}`

Example
~~~~~~~

.. code-block:: python

   from macro_rl.control.ghm_control import GHMControlSpec
   from macro_rl.dynamics.ghm_equity import GHMEquityParams
   import torch

   params = GHMEquityParams()
   control_spec = GHMControlSpec(params)

   # Check action dimensions
   print(f"Action dimension: {control_spec.action_dim}")  # 2

   # Get feasible action mask
   state = torch.tensor([[5.0]])  # Cash = 5.0
   action = torch.tensor([[6.0, 1.0]])  # Invalid: dividend > cash

   mask = control_spec.get_action_mask(state, action)
   print(f"Feasible: {mask}")  # [False, True]

   # Clip to feasible region
   clipped = control_spec.clip_action(state, action)
   print(f"Clipped action: {clipped}")  # [[5.0, 1.0]]

GHM Control with Barrier
~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`GHMControlSpecWithBarrier` extends the base specification with forced recapitalization at a barrier level:

.. math::

   \text{If } c_t < c_{\text{barrier}}: \quad a_{E,t} = c_{\text{barrier}} - c_t

Example
~~~~~~~

.. code-block:: python

   from macro_rl.control.ghm_control import GHMControlSpecWithBarrier

   # Control spec with barrier at c = 0.5
   control_spec_barrier = GHMControlSpecWithBarrier(
       params,
       barrier_level=0.5,
   )

   # Below barrier: forced issuance
   state = torch.tensor([[0.3]])
   action = torch.tensor([[0.0, 0.0]])

   # Clips to enforce barrier
   clipped = control_spec_barrier.clip_action(state, action)
   print(f"Forced issuance: {clipped}")  # [[0.0, 0.2]]

Action Masking
--------------

.. automodule:: macro_rl.control.masking
   :members:
   :undoc-members:
   :show-inheritance:

Utility functions for computing and applying action masks:

* :func:`~macro_rl.control.masking.compute_feasibility_mask`: Compute binary feasibility mask
* :func:`~macro_rl.control.masking.apply_mask`: Apply mask to actions or probabilities
* :func:`~macro_rl.control.masking.masked_softmax`: Softmax with masked-out actions

Example
~~~~~~~

.. code-block:: python

   from macro_rl.control.masking import compute_feasibility_mask, apply_mask
   import torch

   # Binary mask: [True, False, True]
   mask = torch.tensor([True, False, True])

   # Action logits
   logits = torch.tensor([1.0, 2.0, 0.5])

   # Apply mask (sets infeasible to -inf)
   masked_logits = apply_mask(logits, mask)
   print(masked_logits)  # [1.0, -inf, 0.5]

   # Softmax only over feasible actions
   probs = torch.softmax(masked_logits, dim=-1)
   print(probs)  # [0.622, 0.0, 0.378]
