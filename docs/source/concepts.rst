Core Concepts
=============

This page provides a detailed conceptual overview of the GHM-RL framework.

Continuous-Time Control Problems
---------------------------------

State Dynamics
~~~~~~~~~~~~~~

The framework models systems with continuous-time stochastic dynamics:

.. math::

   dX_t = \mu(X_t, a_t) dt + \sigma(X_t) dW_t

where:

* :math:`X_t \in \mathbb{R}^{d_x}`: State vector at time :math:`t`
* :math:`a_t \in \mathcal{A}(X_t)`: Control/action from feasible set
* :math:`\mu: \mathbb{R}^{d_x} \times \mathcal{A} \to \mathbb{R}^{d_x}`: Drift function
* :math:`\sigma: \mathbb{R}^{d_x} \to \mathbb{R}^{d_x \times d_w}`: Diffusion function
* :math:`W_t \in \mathbb{R}^{d_w}`: Standard Brownian motion

**Example (GHM Equity Model)**:

.. math::

   dc_t = [\alpha \mu - a_{L,t} + a_{E,t}] dt + \alpha (\sigma_A dW^A_t + \sigma_X dW^X_t)

where :math:`c_t` is cash, :math:`a_{L,t}` is dividend, :math:`a_{E,t}` is equity issuance.

Objective Function
~~~~~~~~~~~~~~~~~~

The agent seeks to maximize expected discounted utility:

.. math::

   V(x_0) = \max_{\pi \in \Pi} \mathbb{E}^\pi\left[\int_0^\tau e^{-\rho t} r(X_t, a_t) dt + e^{-\rho \tau} g(X_\tau) \mid X_0 = x_0\right]

where:

* :math:`\pi`: Policy (decision rule) from policy class :math:`\Pi`
* :math:`r: \mathbb{R}^{d_x} \times \mathcal{A} \to \mathbb{R}`: Flow reward
* :math:`g: \mathbb{R}^{d_x} \to \mathbb{R}`: Terminal reward
* :math:`\rho > 0`: Discount rate
* :math:`\tau`: Stopping time (bankruptcy, max time, etc.)

**Example (GHM Equity Model)**:

.. math::

   V(c) = \max \mathbb{E}\left[\int_0^\tau e^{-\rho t} (a_{L,t} - (1+\lambda)a_{E,t}) dt + e^{-\rho \tau} (1-\phi)c_\tau\right]

Bellman Optimality
~~~~~~~~~~~~~~~~~~

The optimal value function satisfies the **Hamilton-Jacobi-Bellman (HJB) equation**:

.. math::

   \rho V(x) = \max_{a \in \mathcal{A}(x)} \left\{ r(x, a) + \mathcal{L}^a V(x) \right\}

where :math:`\mathcal{L}^a` is the infinitesimal generator:

.. math::

   \mathcal{L}^a V(x) = \mu(x, a)^\top \nabla_x V(x) + \frac{1}{2} \text{tr}\left(\sigma(x) \sigma(x)^\top \nabla_x^2 V(x)\right)

The optimal policy is:

.. math::

   \pi^*(x) = \arg\max_{a \in \mathcal{A}(x)} \left\{ r(x, a) + \mathcal{L}^a V^*(x) \right\}

Model-Based Reinforcement Learning
-----------------------------------

Exploiting Known Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework assumes dynamics :math:`(\mu, \sigma)` are **known analytically**. This enables:

1. **Unlimited trajectory generation** at any initial state
2. **Differentiable simulation** for low-variance gradients
3. **Direct PDE solving** via HJB residual minimization
4. **Sample efficiency** far beyond model-free RL

Comparison: Model-Based vs Model-Free
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Model-Based (GHM-RL)
     - Model-Free (e.g., PPO)
   * - Dynamics
     - Known analytically
     - Unknown, learned from data
   * - Sample Source
     - Simulated anywhere
     - Real environment only
   * - Sample Efficiency
     - Very high
     - Low to medium
   * - Gradient Variance
     - Low (pathwise)
     - High (score function)
   * - Applicability
     - Finance, physics, control
     - Games, robotics, general

Neural Function Approximation
------------------------------

Policy Representation
~~~~~~~~~~~~~~~~~~~~~

**Gaussian Policy**: Stochastic policy with state-dependent mean and variance:

.. math::

   \pi_\theta(a \mid x) = \mathcal{N}(a \mid \mu_\theta(x), \Sigma_\theta(x))

where :math:`\mu_\theta, \Sigma_\theta` are neural networks.

**Reparameterization Trick**: Sample via

.. math::

   a = \mu_\theta(x) + \sigma_\theta(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

This makes sampling differentiable w.r.t. :math:`\theta`.

Value Function Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Neural Value Network**: Approximate :math:`V(x) \approx V_\phi(x)` with neural network.

**Gradient Computation**: For HJB validation, compute:

.. math::

   V_\phi(x), \quad \nabla_x V_\phi(x), \quad \nabla_x^2 V_\phi(x)

using PyTorch autograd.

Actor-Critic Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Separate Networks**:

.. math::

   \text{Actor: } \pi_\theta(x) \quad \text{Critic: } V_\phi(x)

**Shared Trunk**:

.. math::

   h = \text{MLP}_{\text{trunk}}(x) \\
   \pi_\theta(x) = \text{PolicyHead}(h) \\
   V_\phi(x) = \text{ValueHead}(h)

Shared trunk reduces parameters but couples gradients.

Learning Algorithms
-------------------

Model-Based Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~

**Algorithm**:

1. Sample initial states :math:`\{x_0^{(i)}\}_{i=1}^N`
2. Simulate trajectories using current policy
3. Update critic with loss:

   .. math::

      \mathcal{L}_{\text{critic}} = \mathcal{L}_{\text{MC}} + \alpha \mathcal{L}_{\text{HJB}}

   where:

   * :math:`\mathcal{L}_{\text{MC}} = \|V_\phi(x_0) - R\|^2` (Monte Carlo)
   * :math:`\mathcal{L}_{\text{HJB}} = \|\rho V_\phi - \max_a [r + \mathcal{L}^a V_\phi]\|^2` (HJB residual)

4. Update actor with policy gradient:

   .. math::

      \nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid x) A^\pi(x, a)]

5. Repeat until convergence

**Advantages**:

* Combines Monte Carlo (unbiased) with HJB (PDE constraint)
* Flexible loss configuration
* Stable training

Pathwise Gradient Method
~~~~~~~~~~~~~~~~~~~~~~~~~

**Algorithm**:

1. Sample initial states and noise :math:`\{\epsilon_t\}`
2. Differentiable simulation:

   .. math::

      x_{t+1} = x_t + \mu(x_t, a_t) \Delta t + \sigma(x_t) \sqrt{\Delta t} \epsilon_t

   with :math:`a_t = \pi_\theta(x_t, \epsilon_t)`

3. Compute return :math:`R = \sum_t \gamma^t r(x_t, a_t)`
4. Backpropagate :math:`\nabla_\theta R` through entire trajectory
5. Update :math:`\theta \leftarrow \theta + \alpha \nabla_\theta R`

**Advantages**:

* Low variance (no score function)
* Direct gradient through dynamics
* No baseline needed

**Limitations**:

* Requires differentiable dynamics
* Can have high memory cost (long trajectories)

Deep Galerkin Method
~~~~~~~~~~~~~~~~~~~~~

**Algorithm**:

1. Sample collocation points :math:`\{x_i\}` from state space
2. Compute HJB residual:

   .. math::

      \text{Residual}(x_i) = \left|\rho V_\phi(x_i) - \max_a [r(x_i, a) + \mathcal{L}^a V_\phi(x_i)]\right|

3. Compute boundary error:

   .. math::

      \text{Boundary}(x_j) = |V_\phi(x_j) - g(x_j)|

4. Minimize loss:

   .. math::

      \mathcal{L} = \frac{1}{N_{\text{int}}} \sum_i \text{Residual}(x_i)^2 + \lambda \frac{1}{N_{\text{bnd}}} \sum_j \text{Boundary}(x_j)^2

5. Extract policy: :math:`\pi(x) = \arg\max_a [r(x, a) + \mathcal{L}^a V_\phi(x)]`

**Advantages**:

* Direct PDE solving
* No simulation needed
* Guarantees HJB satisfaction

**Limitations**:

* Requires computing gradients and Hessians
* Sampling strategy matters (Sobol, LHS)
* Extracting policy requires optimization per state

Monte Carlo Policy Gradient (REINFORCE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Algorithm**:

1. Simulate trajectories using current policy
2. Compute returns :math:`R_t = \sum_{k=t}^T \gamma^{k-t} r_k`
3. Compute advantages :math:`A_t = R_t - V_\phi(x_t)`
4. Policy gradient:

   .. math::

      \nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid x_t) A_t\right]

5. Update policy and baseline

**Advantages**:

* Simple and general
* Unbiased gradient estimates
* Works with discrete and continuous actions

**Limitations**:

* High variance
* Requires baseline for stability
* Sample inefficient

Simulation and Integration
---------------------------

SDE Numerical Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Euler-Maruyama Scheme**:

.. math::

   X_{t+\Delta t} = X_t + \mu(X_t, a_t) \Delta t + \sigma(X_t) \sqrt{\Delta t} Z_t, \quad Z_t \sim \mathcal{N}(0, I)

Strong convergence: :math:`\mathcal{O}(\sqrt{\Delta t})`

**Milstein Scheme** (higher order):

.. math::

   X_{t+\Delta t} = X_t + \mu \Delta t + \sigma \sqrt{\Delta t} Z_t + \frac{1}{2} \sigma \sigma' (\Delta t Z_t^2 - \Delta t)

Strong convergence: :math:`\mathcal{O}(\Delta t)`

Differentiable Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Key requirements for pathwise gradients:

1. **Reparameterization**: Actions sampled via :math:`a = \mu_\theta(x) + \sigma_\theta(x) \odot \epsilon`
2. **Fixed noise**: Pre-sample :math:`\{\epsilon_t\}` before simulation
3. **Soft termination**: Use sigmoid masks instead of hard stops:

   .. math::

      m_t = \sigma\left(\frac{x_t - x_{\text{barrier}}}{\delta}\right)

4. **PyTorch autograd**: All operations use differentiable PyTorch ops

Trajectory Generation
~~~~~~~~~~~~~~~~~~~~~

**TrajectoryBatch** contains:

* ``states``: :math:`\{x_t\}_{t=0}^T`
* ``actions``: :math:`\{a_t\}_{t=0}^{T-1}`
* ``rewards``: :math:`\{r_t\}_{t=0}^{T-1}`
* ``masks``: :math:`\{m_t\}_{t=0}^{T-1}` (0 if terminated)
* ``returns``: :math:`R = \sum_t \gamma^t m_t r_t`

Validation and Diagnostics
---------------------------

HJB Residual Check
~~~~~~~~~~~~~~~~~~

Compute residual at test points:

.. math::

   \text{Residual}(x) = \left|\rho V(x) - \max_a [r(x, a) + \mathcal{L}^a V(x)]\right|

**Interpretation**:

* Small residual (:math:`< 10^{-3}`): Solution is approximately optimal
* Large residual (:math:`> 10^{-1}`): Poor approximation, more training needed
* Spatially varying residual: Check boundary conditions and network capacity

Boundary Conditions
~~~~~~~~~~~~~~~~~~~

Check terminal conditions:

* **Dirichlet**: :math:`V(x_{\text{boundary}}) = g(x_{\text{boundary}})`
* **Neumann**: :math:`\nabla_x V(x_{\text{boundary}}) = h(x_{\text{boundary}})`

Analytical Comparison
~~~~~~~~~~~~~~~~~~~~~

If analytical solution :math:`V^*` exists:

* **Value error**: :math:`\|V_\phi - V^*\|`
* **Policy error**: :math:`\|\pi_\theta - \pi^*\|`
* **Relative error**: :math:`\|V_\phi - V^*\| / \|V^*\|`

Monte Carlo Validation
~~~~~~~~~~~~~~~~~~~~~~~

Compare predicted values with simulated returns:

.. math::

   \text{Error} = |V_\phi(x_0) - \mathbb{E}[R \mid x_0]|

Large discrepancy indicates:

* Value network underfitting
* Policy not matching value function
* Insufficient training

GHM Equity Management Model
----------------------------

Problem Formulation
~~~~~~~~~~~~~~~~~~~

Firm with stochastic cash flow:

.. math::

   dc_t = [\alpha \mu - a_{L,t} + a_{E,t}] dt + \alpha (\sigma_A dW^A_t + \sigma_X dW^X_t)

**Controls**:

* :math:`a_{L,t} \in [0, c_t]`: Dividend payout (continuous)
* :math:`a_{E,t} \geq 0`: Equity issuance (impulse control)

**Objective**:

.. math::

   \max \mathbb{E}\left[\int_0^\tau e^{-\rho t} (a_{L,t} - (1+\lambda)a_{E,t}) dt + e^{-\rho \tau} (1-\phi)c_\tau\right]

**Termination**: :math:`\tau = \inf\{t : c_t \leq 0\}` (bankruptcy)

Economic Interpretation
~~~~~~~~~~~~~~~~~~~~~~~

* **Dividend**: Shareholders receive :math:`a_L` per unit time
* **Issuance cost**: Raising capital costs :math:`\lambda` (proportional)
* **Bankruptcy**: Firm liquidates with cost :math:`\phi`
* **Trade-off**: Pay dividends vs. save cash vs. issue equity

Optimal Policy Structure
~~~~~~~~~~~~~~~~~~~~~~~~

The optimal policy typically has form:

* **Dividend barrier**: Pay dividends when :math:`c > c^*_L`
* **Issuance barrier**: Issue equity when :math:`c < c^*_E`
* **Inaction region**: :math:`c \in [c^*_E, c^*_L]`, no action

**Learning goal**: Neural policy should discover these barriers.

See Also
--------

* :doc:`getting_started`: Installation and quick start
* :doc:`tutorials/index`: Step-by-step tutorials
* :doc:`api/index`: API reference
* :doc:`examples`: Complete examples
