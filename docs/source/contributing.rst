Contributing
============

We welcome contributions to the GHM-RL project! This guide will help you get started.

Getting Started
---------------

Fork and Clone
~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/GHM-RL.git
      cd GHM-RL

3. Add upstream remote:

   .. code-block:: bash

      git remote add upstream https://github.com/originalauthor/GHM-RL.git

Development Setup
~~~~~~~~~~~~~~~~~

Install in development mode with all dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This installs:

* Core dependencies (PyTorch, NumPy, etc.)
* Testing tools (pytest, pytest-cov)
* Linting tools (black, flake8, mypy)
* Documentation tools (Sphinx, sphinx-rtd-theme)

Code Style
----------

We follow PEP 8 style guidelines with some modifications.

Formatting
~~~~~~~~~~

Use Black for code formatting:

.. code-block:: bash

   black macro_rl tests

Configuration in ``pyproject.toml``:

.. code-block:: toml

   [tool.black]
   line-length = 100
   target-version = ['py38']

Linting
~~~~~~~

Use flake8 for linting:

.. code-block:: bash

   flake8 macro_rl tests

Type Hints
~~~~~~~~~~

Add type hints to all public functions:

.. code-block:: python

   def compute_gradient(
       f: Callable[[torch.Tensor], torch.Tensor],
       x: torch.Tensor,
   ) -> torch.Tensor:
       """
       Compute gradient of f at x.

       Args:
           f: Scalar-valued function
           x: Input tensor

       Returns:
           Gradient tensor
       """
       # Implementation
       pass

Run mypy for type checking:

.. code-block:: bash

   mypy macro_rl

Testing
-------

Writing Tests
~~~~~~~~~~~~~

All new features must include tests. Place tests in ``tests/`` directory:

.. code-block:: python

   # tests/test_dynamics.py
   import pytest
   import torch
   from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams

   def test_ghm_drift():
       """Test GHM drift computation"""
       params = GHMEquityParams()
       dynamics = GHMEquityDynamics(params)

       state = torch.tensor([[5.0]])
       action = torch.tensor([[0.5, 0.0]])

       drift = dynamics.drift(state, action)

       assert drift.shape == (1, 1)
       assert not torch.isnan(drift).any()

Running Tests
~~~~~~~~~~~~~

Run all tests:

.. code-block:: bash

   pytest

Run specific test file:

.. code-block:: bash

   pytest tests/test_dynamics.py

Run with coverage:

.. code-block:: bash

   pytest --cov=macro_rl --cov-report=html

View coverage report:

.. code-block:: bash

   open htmlcov/index.html

Test Guidelines
~~~~~~~~~~~~~~~

* Test edge cases and boundary conditions
* Test with different input shapes
* Test numerical stability (NaN, inf)
* Test gradient flow (for differentiable operations)
* Use fixtures for common setup
* Mock external dependencies

Documentation
-------------

Docstring Format
~~~~~~~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

   def train(
       self,
       n_iterations: int,
       log_freq: int = 100,
   ) -> Dict[str, List[float]]:
       """
       Train the agent for specified number of iterations.

       Args:
           n_iterations: Number of training iterations
           log_freq: Frequency of logging (in iterations)

       Returns:
           Dictionary with training history:

           * 'actor_loss': Actor loss values
           * 'critic_loss': Critic loss values
           * 'mean_return': Mean return on training trajectories

       Raises:
           ValueError: If n_iterations <= 0

       Example:
           >>> solver = ModelBasedActorCritic(...)
           >>> history = solver.train(n_iterations=1000)
           >>> print(f"Final return: {history['mean_return'][-1]}")
       """
       pass

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Build Sphinx documentation:

.. code-block:: bash

   cd docs
   make html

View locally:

.. code-block:: bash

   open build/html/index.html

Adding Examples
~~~~~~~~~~~~~~~

Add examples to ``docs/source/examples.rst`` or create new tutorial pages in ``docs/source/tutorials/``.

Pull Request Process
---------------------

1. **Create a branch** for your feature:

   .. code-block:: bash

      git checkout -b feature/my-new-feature

2. **Make your changes** with clear commit messages:

   .. code-block:: bash

      git commit -m "Add pathwise gradient solver with variance reduction"

3. **Add tests** for new functionality

4. **Update documentation** as needed

5. **Run full test suite**:

   .. code-block:: bash

      pytest
      black macro_rl tests
      flake8 macro_rl tests
      mypy macro_rl

6. **Push to your fork**:

   .. code-block:: bash

      git push origin feature/my-new-feature

7. **Create Pull Request** on GitHub with:

   * Clear description of changes
   * References to related issues
   * Screenshots (if applicable)
   * Confirmation that tests pass

Code Review
~~~~~~~~~~~

All PRs require review before merging. Reviewers will check:

* Code quality and style
* Test coverage
* Documentation completeness
* Performance implications
* Breaking changes

Areas for Contribution
----------------------

High Priority
~~~~~~~~~~~~~

* Complete pathwise gradient solver implementation
* Add more test dynamics models
* Improve HJB validation tools
* Add visualization utilities
* Performance optimizations

Medium Priority
~~~~~~~~~~~~~~~

* Multi-dimensional state spaces
* Discrete-continuous hybrid controls
* Model-free baselines for comparison
* Distributed training support
* Additional example problems

Documentation
~~~~~~~~~~~~~

* More tutorials and examples
* Conceptual guides
* Video tutorials
* Jupyter notebooks
* Benchmark results

Bug Reports
-----------

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Reproduction**: Minimal code to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:

   * Python version
   * PyTorch version
   * OS and version
   * GPU (if applicable)

Example bug report:

.. code-block:: markdown

   **Bug**: Actor-critic training crashes with NaN loss

   **Reproduction**:
   ```python
   solver = ModelBasedActorCritic(...)
   solver.train(n_iterations=1000)  # Crashes at iteration 250
   ```

   **Expected**: Training completes successfully

   **Actual**: RuntimeError: Loss is NaN at iteration 250

   **Environment**:
   - Python 3.9.7
   - PyTorch 1.12.1
   - Ubuntu 20.04
   - NVIDIA RTX 3080

Feature Requests
----------------

When requesting features:

1. **Use case**: Describe the problem you're trying to solve
2. **Proposed solution**: Suggest an approach
3. **Alternatives**: Mention alternatives you've considered
4. **Additional context**: Screenshots, references, etc.

Community Guidelines
--------------------

* Be respectful and inclusive
* Provide constructive feedback
* Help others in issues and discussions
* Follow the code of conduct
* Credit others' contributions

Contact
-------

* **GitHub Issues**: https://github.com/yourusername/GHM-RL/issues
* **Email**: your.email@example.com
* **Discussions**: https://github.com/yourusername/GHM-RL/discussions

Thank You!
----------

Thank you for contributing to GHM-RL! Your contributions help make this project better for everyone.
