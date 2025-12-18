Core Module
===========

The core module provides foundational abstractions for state space representation and parameter management.

State Space
-----------

.. automodule:: macro_rl.core.state_space
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`StateSpace` class represents the state space with bounds, dimension, and utilities for sampling, normalization, and clipping.

Key Methods
~~~~~~~~~~~

* :meth:`~macro_rl.core.state_space.StateSpace.contains`: Check if states are within bounds
* :meth:`~macro_rl.core.state_space.StateSpace.sample_uniform`: Sample uniformly from state space
* :meth:`~macro_rl.core.state_space.StateSpace.normalize`: Normalize states to [0, 1]
* :meth:`~macro_rl.core.state_space.StateSpace.denormalize`: Denormalize from [0, 1] to original bounds
* :meth:`~macro_rl.core.state_space.StateSpace.clip`: Clip states to stay within bounds

Example
~~~~~~~

.. code-block:: python

   import torch
   from macro_rl.core.state_space import StateSpace

   # Define 1D state space for cash holdings
   state_space = StateSpace(
       lower_bounds=torch.tensor([0.0]),
       upper_bounds=torch.tensor([10.0]),
       state_names=['cash']
   )

   # Sample 100 random initial states
   initial_states = state_space.sample_uniform(100)

   # Check if states are valid
   is_valid = state_space.contains(initial_states)

   # Normalize states to [0, 1]
   normalized = state_space.normalize(initial_states)

Parameters
----------

.. automodule:: macro_rl.core.params
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
~~~~~~~~~~~~~~~~~

* :func:`~macro_rl.core.params.validate_params`: Validate parameter dataclasses
* :func:`~macro_rl.core.params.params_to_dict`: Convert parameter dataclasses to dictionaries

Example
~~~~~~~

.. code-block:: python

   from dataclasses import dataclass
   from macro_rl.core.params import validate_params, params_to_dict

   @dataclass
   class ModelParams:
       alpha: float
       mu: float
       sigma: float

   params = ModelParams(alpha=0.5, mu=0.1, sigma=0.2)
   validate_params(params)  # Validates all fields are numeric
   param_dict = params_to_dict(params)  # Convert to dict
