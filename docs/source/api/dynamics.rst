Dynamics Module
===============

The dynamics module provides implementations of continuous-time stochastic dynamical systems, with a focus on the GHM equity management model.

Base Classes
------------

.. automodule:: macro_rl.dynamics.base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`ContinuousTimeDynamics` abstract base class defines the interface for all continuous-time models:

.. math::

   dX_t = \mu(X_t, a_t) dt + \sigma(X_t) dW_t

Key Methods
~~~~~~~~~~~

* :meth:`~macro_rl.dynamics.base.ContinuousTimeDynamics.drift`: Drift function :math:`\mu(x, a)`
* :meth:`~macro_rl.dynamics.base.ContinuousTimeDynamics.diffusion`: Diffusion function :math:`\sigma(x)`
* :meth:`~macro_rl.dynamics.base.ContinuousTimeDynamics.diffusion_squared`: Squared diffusion :math:`\sigma^2(x)`
* :meth:`~macro_rl.dynamics.base.ContinuousTimeDynamics.discount_rate`: Discount rate :math:`\rho`
* :meth:`~macro_rl.dynamics.base.ContinuousTimeDynamics.sample_interior`: Sample interior points
* :meth:`~macro_rl.dynamics.base.ContinuousTimeDynamics.sample_boundary`: Sample boundary points

GHM Equity Dynamics
-------------------

.. automodule:: macro_rl.dynamics.ghm_equity
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`GHMEquityDynamics` class implements the 1D GHM equity management model with cash dynamics:

.. math::

   dc_t = [\alpha \mu - a_{L,t} + a_{E,t}] dt + \alpha (\sigma_A dW^A_t + \sigma_X dW^X_t)

where:

* :math:`c_t`: Cash holdings
* :math:`a_{L,t}`: Dividend payout rate
* :math:`a_{E,t}`: Equity issuance rate
* :math:`\alpha`: Profitability parameter
* :math:`\mu`: Expected return
* :math:`\sigma_A, \sigma_X`: Asset and idiosyncratic volatilities
* :math:`\rho`: Correlation between shocks

Parameters
~~~~~~~~~~

The :class:`GHMEquityParams` dataclass contains all model parameters:

* ``alpha``: Profitability
* ``mu``: Drift rate
* ``r``: Risk-free rate
* ``sigma_A``: Asset volatility
* ``sigma_X``: Idiosyncratic volatility
* ``rho``: Correlation
* ``lambda_``: Equity issuance cost
* ``tau``: Corporate tax rate
* ``phi``: Bankruptcy cost
* ``c_max``: Maximum cash level
* ``c_barrier``: Bankruptcy barrier

Example
~~~~~~~

.. code-block:: python

   from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
   import torch

   # Define model parameters
   params = GHMEquityParams(
       alpha=0.5,
       mu=0.1,
       r=0.05,
       sigma_A=0.2,
       sigma_X=0.3,
       rho=0.7,
       lambda_=0.1,
       tau=0.35,
       phi=0.1,
   )

   # Initialize dynamics
   dynamics = GHMEquityDynamics(params)

   # Compute drift and diffusion
   state = torch.tensor([[5.0]])  # Cash = 5.0
   action = torch.tensor([[0.5, 0.0]])  # Dividend = 0.5, Issuance = 0
   drift = dynamics.drift(state, action)
   diffusion = dynamics.diffusion(state)

   print(f"Drift: {drift}")
   print(f"Diffusion: {diffusion}")

Test Models
-----------

.. automodule:: macro_rl.dynamics.test_models
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The test models module provides simple dynamics for testing and validation:

Geometric Brownian Motion (GBM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   dX_t = \mu X_t dt + \sigma X_t dW_t

Ornstein-Uhlenbeck (OU) Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   dX_t = \theta(\bar{X} - X_t) dt + \sigma dW_t

Example
~~~~~~~

.. code-block:: python

   from macro_rl.dynamics.test_models import GBMDynamics, OUDynamics
   import torch

   # GBM with drift 0.1 and volatility 0.2
   gbm = GBMDynamics(mu=0.1, sigma=0.2, x_min=0.1, x_max=10.0)
   state = torch.tensor([[1.0]])
   drift = gbm.drift(state)  # Returns mu * state
   diffusion = gbm.diffusion(state)  # Returns sigma * state

   # OU process with mean reversion
   ou = OUDynamics(theta=0.5, x_bar=1.0, sigma=0.2)
   drift = ou.drift(state)  # Returns theta * (x_bar - state)
   diffusion = ou.diffusion(state)  # Returns sigma
