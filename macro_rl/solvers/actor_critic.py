"""
Model-based Actor-Critic solver.

Combines policy optimization with value function learning.
"""

import torch
from torch import Tensor
from torch.optim import Adam

from macro_rl.solvers.base import Solver, SolverResult


class ModelBasedActorCritic(Solver):
    """
    Model-based Actor-Critic.

    Combines advantages of different approaches:
        - Actor: Policy network (like Monte Carlo/Pathwise)
        - Critic: Value network (like Deep Galerkin)
        - Model: Known dynamics for simulation

    Algorithm:
        1. Simulate trajectories
        2. Update critic (value function) via TD or MC
        3. Update actor (policy) via policy gradient
        4. (Optional) Add HJB regularization to critic

    TODO: Implement model-based actor-critic
    """

    def __init__(
        self,
        policy,
        value_net,
        simulator,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
    ):
        """
        Initialize actor-critic solver.

        TODO: Implement initialization
        """
        self.policy = policy
        self.value_net = value_net
        self.simulator = simulator
        self.policy_optimizer = Adam(policy.parameters(), lr=lr_policy)
        self.value_optimizer = Adam(value_net.parameters(), lr=lr_value)

    def solve(
        self,
        dynamics,
        control_spec,
        reward_fn,
        n_iterations: int = 10000,
        **kwargs,
    ) -> SolverResult:
        """
        Solve via actor-critic.

        TODO: Implement training loop
        """
        raise NotImplementedError
