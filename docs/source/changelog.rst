Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~

* Initial documentation with ReadTheDocs
* Comprehensive API reference
* Tutorials and examples

[0.1.0] - 2025-01-XX
--------------------

Added
~~~~~

* Model-based actor-critic solver with multiple loss options
* GHM equity dynamics implementation
* Gaussian policy networks with reparameterization
* Value networks with gradient computation
* Differentiable trajectory simulation
* Gymnasium environment wrapper
* State space utilities
* Control specification framework
* Reward function abstractions
* Numerical utilities (differentiation, sampling, integration)

Changed
~~~~~~~

* Refactored from single-control to two-control formulation
* Improved gradient computation efficiency
* Updated network architectures

Fixed
~~~~~

* Gradient tensor conversion bug (detach before numpy conversion)
* HJB residual computation in actor-critic
* Boundary condition handling

[0.0.1] - 2024-XX-XX
--------------------

Initial release with basic framework structure.

Legend
------

* **Added**: New features
* **Changed**: Changes in existing functionality
* **Deprecated**: Soon-to-be removed features
* **Removed**: Removed features
* **Fixed**: Bug fixes
* **Security**: Vulnerability fixes
