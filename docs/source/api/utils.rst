Utilities Module
================

The utilities module provides general-purpose helper functions and tools.

Autograd Utilities
------------------

.. automodule:: macro_rl.utils.autograd
   :members:
   :undoc-members:
   :show-inheritance:

PyTorch autograd helper functions for gradient computation and manipulation.

Gradient Utilities
~~~~~~~~~~~~~~~~~~

.. function:: detach_to_numpy(tensor: Tensor) -> np.ndarray

   Safely convert PyTorch tensor to NumPy array.

   :param tensor: PyTorch tensor
   :return: NumPy array

.. function:: enable_grad(tensor: Tensor) -> Tensor

   Enable gradient tracking for tensor.

   :param tensor: Input tensor
   :return: Tensor with ``requires_grad=True``

.. function:: disable_grad(tensor: Tensor) -> Tensor

   Disable gradient tracking for tensor.

   :param tensor: Input tensor
   :return: Detached tensor

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import detach_to_numpy, enable_grad, disable_grad
   import torch

   # Tensor with gradients
   x = torch.tensor([1.0, 2.0], requires_grad=True)

   # Convert to numpy (detaches first)
   x_np = detach_to_numpy(x)

   # Disable gradients
   x_no_grad = disable_grad(x)

   # Re-enable gradients
   x_with_grad = enable_grad(x_no_grad)

Jacobian and Hessian
~~~~~~~~~~~~~~~~~~~~

.. function:: compute_jacobian(outputs: Tensor, inputs: Tensor) -> Tensor

   Compute Jacobian matrix :math:`J_{ij} = \frac{\partial y_i}{\partial x_j}`.

   :param outputs: Output tensor of shape ``(batch_size, m)``
   :param inputs: Input tensor of shape ``(batch_size, n)``
   :return: Jacobian of shape ``(batch_size, m, n)``

.. function:: compute_batch_hessian(outputs: Tensor, inputs: Tensor) -> Tensor

   Compute Hessian for batched inputs.

   :param outputs: Scalar outputs of shape ``(batch_size,)``
   :param inputs: Input tensor of shape ``(batch_size, n)``
   :return: Hessian of shape ``(batch_size, n, n)``

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import compute_jacobian, compute_batch_hessian
   import torch

   # Function: f(x) = [x1^2, x2^2]
   x = torch.tensor([[1.0, 2.0]], requires_grad=True)
   y = x ** 2

   # Jacobian: [[2*x1, 0], [0, 2*x2]]
   jac = compute_jacobian(y, x)
   print(jac)  # [[2.0, 0.0], [0.0, 4.0]]

   # Hessian of scalar function
   scalar_output = y.sum()
   hess = compute_batch_hessian(scalar_output, x)
   print(hess)  # [[2.0, 0.0], [0.0, 2.0]]

Gradient Clipping
~~~~~~~~~~~~~~~~~

.. function:: clip_grad_norm(parameters: Iterable[Tensor], max_norm: float) -> float

   Clip gradient norm of parameters.

   :param parameters: Model parameters
   :param max_norm: Maximum gradient norm
   :return: Total gradient norm before clipping

.. function:: clip_grad_value(parameters: Iterable[Tensor], clip_value: float) -> None

   Clip gradient values element-wise.

   :param parameters: Model parameters
   :param clip_value: Maximum absolute value for gradients

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import clip_grad_norm, clip_grad_value

   # After computing gradients
   loss.backward()

   # Clip gradient norm (prevents exploding gradients)
   total_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
   print(f"Gradient norm: {total_norm:.4f}")

   # Alternative: clip values
   clip_grad_value(model.parameters(), clip_value=0.5)

   # Then update
   optimizer.step()

Context Managers
~~~~~~~~~~~~~~~~

.. function:: no_grad() -> ContextManager

   Context manager for disabling gradient computation.

.. function:: enable_grad() -> ContextManager

   Context manager for enabling gradient computation.

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import no_grad, enable_grad
   import torch

   x = torch.tensor([1.0], requires_grad=True)

   # Disable gradients for inference
   with no_grad():
       y = x ** 2
       # y.requires_grad is False

   # Re-enable gradients
   with enable_grad():
       y = x ** 2
       # y.requires_grad is True

Higher-Order Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: higher_order_derivative(f: Callable, x: Tensor, order: int) -> Tensor

   Compute higher-order derivatives.

   :param f: Scalar function
   :param x: Input point
   :param order: Derivative order
   :return: Derivative tensor

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import higher_order_derivative
   import torch

   # f(x) = x^4
   def f(x):
       return (x ** 4).sum()

   x = torch.tensor([[2.0]], requires_grad=True)

   # First derivative: 4x^3
   df = higher_order_derivative(f, x, order=1)
   print(df)  # 32.0

   # Second derivative: 12x^2
   d2f = higher_order_derivative(f, x, order=2)
   print(d2f)  # 48.0

   # Third derivative: 24x
   d3f = higher_order_derivative(f, x, order=3)
   print(d3f)  # 48.0

Reparameterization Trick
~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: reparameterize(mean: Tensor, std: Tensor, noise: Optional[Tensor] = None) -> Tensor

   Sample from Gaussian using reparameterization trick.

   :param mean: Mean tensor
   :param std: Standard deviation tensor
   :param noise: Optional pre-sampled noise
   :return: Sample :math:`z = \mu + \sigma \odot \epsilon`

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import reparameterize
   import torch

   # Sample from N(1.0, 0.5^2)
   mean = torch.tensor([1.0])
   std = torch.tensor([0.5])

   # Reparameterized sampling (differentiable)
   sample = reparameterize(mean, std)

   # With pre-sampled noise (for pathwise gradients)
   noise = torch.randn_like(mean)
   sample = reparameterize(mean, std, noise=noise)

   # Gradient flows through mean and std
   loss = sample.sum()
   loss.backward()

Safe Operations
~~~~~~~~~~~~~~~

.. function:: safe_log(x: Tensor, eps: float = 1e-8) -> Tensor

   Numerically stable logarithm.

   :param x: Input tensor
   :param eps: Small constant for numerical stability
   :return: log(x + eps)

.. function:: safe_div(numerator: Tensor, denominator: Tensor, eps: float = 1e-8) -> Tensor

   Safe division avoiding division by zero.

   :param numerator: Numerator tensor
   :param denominator: Denominator tensor
   :param eps: Small constant
   :return: numerator / (denominator + eps)

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import safe_log, safe_div
   import torch

   # Safe log (avoids log(0) = -inf)
   x = torch.tensor([0.0, 1.0, 2.0])
   log_x = safe_log(x)  # [-18.42, 0.0, 0.69]

   # Safe division
   numerator = torch.tensor([1.0, 2.0])
   denominator = torch.tensor([0.0, 2.0])
   result = safe_div(numerator, denominator)  # [1e8, 1.0]

Tensor Utilities
~~~~~~~~~~~~~~~~

.. function:: broadcast_to_batch(tensor: Tensor, batch_size: int) -> Tensor

   Broadcast tensor to batch dimension.

   :param tensor: Input tensor
   :param batch_size: Target batch size
   :return: Broadcasted tensor

.. function:: flatten_batch(tensor: Tensor) -> Tensor

   Flatten all dimensions except last.

   :param tensor: Input tensor of shape ``(d1, d2, ..., dn)``
   :return: Flattened tensor of shape ``(d1*d2*...*dn-1, dn)``

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import broadcast_to_batch, flatten_batch
   import torch

   # Broadcast to batch
   x = torch.tensor([1.0, 2.0])
   x_batch = broadcast_to_batch(x, batch_size=3)
   # Shape: (3, 2)

   # Flatten batch dimensions
   x = torch.randn(2, 3, 4, 5)
   x_flat = flatten_batch(x)
   # Shape: (24, 5)

Random Seed Management
~~~~~~~~~~~~~~~~~~~~~~

.. function:: set_seed(seed: int) -> None

   Set random seed for reproducibility.

   :param seed: Random seed

.. function:: get_rng_state() -> Dict

   Get current RNG state.

   :return: Dictionary with RNG states

.. function:: set_rng_state(state: Dict) -> None

   Restore RNG state.

   :param state: Dictionary with RNG states

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import set_seed, get_rng_state, set_rng_state

   # Set seed for reproducibility
   set_seed(42)

   # Save RNG state
   state = get_rng_state()

   # ... run experiments ...

   # Restore RNG state
   set_rng_state(state)

Device Management
~~~~~~~~~~~~~~~~~

.. function:: get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device

   Get torch device (CPU/GPU).

   :param device: Device specification
   :return: torch.device object

.. function:: to_device(obj: Any, device: torch.device) -> Any

   Move object to device.

   :param obj: Tensor, module, or dict/list of tensors
   :param device: Target device
   :return: Object on target device

Example
~~~~~~~

.. code-block:: python

   from macro_rl.utils.autograd import get_device, to_device

   # Auto-detect device
   device = get_device()  # cuda if available, else cpu

   # Move to device
   model = to_device(model, device)
   data = to_device(data, device)

See Also
~~~~~~~~

* :mod:`macro_rl.numerics.differentiation`: Automatic differentiation for PDEs
* :mod:`torch.autograd`: PyTorch automatic differentiation
