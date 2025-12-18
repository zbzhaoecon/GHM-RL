Numerics Module
===============

The numerics module provides numerical computation utilities for differentiation, integration, and sampling.

Differentiation
---------------

.. automodule:: macro_rl.numerics.differentiation
   :members:
   :undoc-members:
   :show-inheritance:

Automatic differentiation utilities using PyTorch autograd.

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

.. function:: gradient(f: Callable, x: Tensor) -> Tensor

   Compute gradient :math:`\nabla_x f(x)` using automatic differentiation.

   :param f: Scalar-valued function :math:`f: \mathbb{R}^n \to \mathbb{R}`
   :param x: Input tensor of shape ``(batch_size, n)``
   :return: Gradient tensor of shape ``(batch_size, n)``

Example
"""""""

.. code-block:: python

   from macro_rl.numerics.differentiation import gradient
   import torch

   # Define function f(x) = x^2
   def f(x):
       return (x ** 2).sum(dim=-1)

   x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
   grad = gradient(f, x)  # Returns [[2.0, 4.0], [6.0, 8.0]]

Hessian Computation
~~~~~~~~~~~~~~~~~~~

.. function:: hessian_diagonal(f: Callable, x: Tensor) -> Tensor

   Compute diagonal of Hessian :math:`\text{diag}(\nabla^2_x f(x))`.

   :param f: Scalar-valued function
   :param x: Input tensor of shape ``(batch_size, n)``
   :return: Hessian diagonal of shape ``(batch_size, n)``

.. function:: hessian_matrix(f: Callable, x: Tensor) -> Tensor

   Compute full Hessian matrix :math:`\nabla^2_x f(x)`.

   :param f: Scalar-valued function
   :param x: Input tensor of shape ``(batch_size, n)``
   :return: Hessian matrix of shape ``(batch_size, n, n)``

.. function:: mixed_partial(f: Callable, x: Tensor, i: int, j: int) -> Tensor

   Compute mixed partial derivative :math:`\frac{\partial^2 f}{\partial x_i \partial x_j}`.

   :param f: Scalar-valued function
   :param x: Input tensor
   :param i: First dimension index
   :param j: Second dimension index
   :return: Mixed partial derivative

Example
"""""""

.. code-block:: python

   from macro_rl.numerics.differentiation import hessian_diagonal, hessian_matrix
   import torch

   # f(x) = x1^2 + x2^2
   def f(x):
       return (x ** 2).sum(dim=-1)

   x = torch.tensor([[1.0, 2.0]], requires_grad=True)

   # Diagonal: [2.0, 2.0]
   hess_diag = hessian_diagonal(f, x)

   # Full matrix: [[2, 0], [0, 2]]
   hess_full = hessian_matrix(f, x)

HJB Operator
~~~~~~~~~~~~

The differentiation module is particularly useful for computing the HJB operator:

.. code-block:: python

   from macro_rl.numerics.differentiation import gradient, hessian_diagonal

   def compute_hjb_operator(value_net, state, action, dynamics):
       """Compute L^a V(s) = μ^T ∇V + (1/2) tr(σσ^T ∇²V)"""

       # Value and gradients
       def value_fn(s):
           return value_net(s).squeeze()

       # First derivative: ∇V
       grad_v = gradient(value_fn, state)

       # Second derivative: ∇²V (diagonal)
       hess_v = hessian_diagonal(value_fn, state)

       # Drift and diffusion
       drift = dynamics.drift(state, action)
       diffusion_sq = dynamics.diffusion_squared(state)

       # L^a V = μ^T ∇V + (1/2) σ² ∇²V
       operator = (drift * grad_v).sum(dim=-1) + 0.5 * (diffusion_sq * hess_v).sum(dim=-1)

       return operator

Integration
-----------

.. automodule:: macro_rl.numerics.integration
   :members:
   :undoc-members:
   :show-inheritance:

Numerical integration utilities for computing expectations and integrals.

Quadrature Methods
~~~~~~~~~~~~~~~~~~

.. function:: trapezoidal_rule(f: Callable, a: float, b: float, n: int) -> float

   Trapezoidal rule for numerical integration.

   :param f: Function to integrate
   :param a: Lower bound
   :param b: Upper bound
   :param n: Number of points
   :return: Approximate integral

.. function:: simpson_rule(f: Callable, a: float, b: float, n: int) -> float

   Simpson's rule for numerical integration.

   :param f: Function to integrate
   :param a: Lower bound
   :param b: Upper bound
   :param n: Number of points (must be even)
   :return: Approximate integral

Monte Carlo Integration
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: monte_carlo_integrate(f: Callable, sampler: Callable, n_samples: int) -> Tensor

   Monte Carlo integration: :math:`\mathbb{E}[f(X)] \approx \frac{1}{N} \sum_{i=1}^N f(X_i)`.

   :param f: Function to integrate
   :param sampler: Function that returns samples
   :param n_samples: Number of Monte Carlo samples
   :return: Approximate expectation

Example
~~~~~~~

.. code-block:: python

   from macro_rl.numerics.integration import monte_carlo_integrate
   import torch

   # Compute E[X^2] where X ~ N(0, 1)
   def f(x):
       return x ** 2

   def sampler(n):
       return torch.randn(n, 1)

   result = monte_carlo_integrate(f, sampler, n_samples=10000)
   print(f"E[X^2] ≈ {result:.4f}")  # Should be close to 1.0

Sampling
--------

.. automodule:: macro_rl.numerics.sampling
   :members:
   :undoc-members:
   :show-inheritance:

Sampling strategies for state space exploration and collocation methods.

Uniform Sampling
~~~~~~~~~~~~~~~~

.. function:: uniform_sample(n_points: int, lower: Tensor, upper: Tensor, seed: Optional[int] = None) -> Tensor

   Uniform random sampling from hyperrectangle.

   :param n_points: Number of samples
   :param lower: Lower bounds
   :param upper: Upper bounds
   :param seed: Random seed
   :return: Sampled points of shape ``(n_points, dim)``

Example
~~~~~~~

.. code-block:: python

   from macro_rl.numerics.sampling import uniform_sample
   import torch

   # Sample 1000 points from [0, 10]
   samples = uniform_sample(
       n_points=1000,
       lower=torch.tensor([0.0]),
       upper=torch.tensor([10.0]),
       seed=42
   )

Boundary Sampling
~~~~~~~~~~~~~~~~~

.. function:: boundary_sample(n_points: int, lower: Tensor, upper: Tensor, which: str, dim: int, seed: Optional[int] = None) -> Tensor

   Sample from boundary of state space.

   :param n_points: Number of samples
   :param lower: Lower bounds
   :param upper: Upper bounds
   :param which: 'lower' or 'upper' boundary
   :param dim: Dimension index to fix
   :param seed: Random seed
   :return: Boundary samples

Example
~~~~~~~

.. code-block:: python

   from macro_rl.numerics.sampling import boundary_sample

   # Sample from lower boundary (c = 0)
   lower_boundary = boundary_sample(
       n_points=100,
       lower=torch.tensor([0.0]),
       upper=torch.tensor([10.0]),
       which='lower',
       dim=0,
   )

Quasi-Random Sampling
~~~~~~~~~~~~~~~~~~~~~

.. function:: sobol_sample(n_points: int, lower: Tensor, upper: Tensor, seed: Optional[int] = None) -> Tensor

   Low-discrepancy Sobol sequence sampling.

   :param n_points: Number of samples
   :param lower: Lower bounds
   :param upper: Upper bounds
   :param seed: Random seed
   :return: Sobol sequence points

.. function:: latin_hypercube_sample(n_points: int, lower: Tensor, upper: Tensor, seed: Optional[int] = None) -> Tensor

   Latin hypercube sampling for space-filling design.

   :param n_points: Number of samples
   :param lower: Lower bounds
   :param upper: Upper bounds
   :param seed: Random seed
   :return: Latin hypercube samples

Example
~~~~~~~

.. code-block:: python

   from macro_rl.numerics.sampling import sobol_sample, latin_hypercube_sample

   # Sobol sequence (better coverage than uniform)
   sobol = sobol_sample(
       n_points=1000,
       lower=torch.tensor([0.0, 0.0]),
       upper=torch.tensor([10.0, 5.0]),
   )

   # Latin hypercube (stratified sampling)
   lhs = latin_hypercube_sample(
       n_points=1000,
       lower=torch.tensor([0.0]),
       upper=torch.tensor([10.0]),
   )

Grid Sampling
~~~~~~~~~~~~~

.. function:: grid_sample(n_points_per_dim: int, lower: Tensor, upper: Tensor) -> Tensor

   Regular grid sampling.

   :param n_points_per_dim: Number of points per dimension
   :param lower: Lower bounds
   :param upper: Upper bounds
   :return: Grid points of shape ``(n_points_per_dim^dim, dim)``

Example
~~~~~~~

.. code-block:: python

   from macro_rl.numerics.sampling import grid_sample

   # Create 10x10 grid
   grid = grid_sample(
       n_points_per_dim=10,
       lower=torch.tensor([0.0, 0.0]),
       upper=torch.tensor([10.0, 5.0]),
   )
   # Returns 100 points in regular grid

Mixed Sampling
~~~~~~~~~~~~~~

.. function:: mixed_sample(n_interior: int, n_boundary: int, lower: Tensor, upper: Tensor, sampler: str = 'uniform', seed: Optional[int] = None) -> Tuple[Tensor, Tensor]

   Sample both interior and boundary points.

   :param n_interior: Number of interior samples
   :param n_boundary: Number of boundary samples
   :param lower: Lower bounds
   :param upper: Upper bounds
   :param sampler: Sampling method ('uniform', 'sobol', 'lhs', 'grid')
   :param seed: Random seed
   :return: Tuple of (interior_samples, boundary_samples)

Example
~~~~~~~

.. code-block:: python

   from macro_rl.numerics.sampling import mixed_sample

   # Sample for Deep Galerkin Method
   interior, boundary = mixed_sample(
       n_interior=1000,
       n_boundary=100,
       lower=torch.tensor([0.0]),
       upper=torch.tensor([10.0]),
       sampler='sobol',
   )

   # Use for PDE loss computation
   interior_loss = compute_pde_residual(interior)
   boundary_loss = compute_boundary_error(boundary)
   total_loss = interior_loss + 10.0 * boundary_loss

Sampling Strategies Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Method
     - Coverage
     - Computational Cost
     - Best Use Case
   * - Uniform
     - Random
     - Low
     - General purpose
   * - Sobol
     - Low discrepancy
     - Low
     - Integration, PDE
   * - Latin Hypercube
     - Stratified
     - Low
     - Sensitivity analysis
   * - Grid
     - Regular
     - Medium (high-dim)
     - Visualization, 1D/2D

**Recommendations:**

* **PDE solving**: Sobol or Latin Hypercube for better coverage
* **Monte Carlo**: Uniform for unbiased estimates
* **Visualization**: Grid for regular plotting
* **High dimensions**: Avoid grid (exponential growth)
