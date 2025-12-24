"""
Parallel trajectory simulation using multiprocessing.

This module provides a wrapper around TrajectorySimulator that parallelizes
trajectory rollouts across multiple CPU cores for improved performance.
"""

import multiprocessing as mp
from typing import Optional, Callable
import torch
from torch import Tensor
import numpy as np

from macro_rl.simulation.trajectory import TrajectorySimulator, TrajectoryBatch


def _rollout_worker(
    dynamics_class,
    dynamics_params,
    control_spec,
    reward_fn_class,
    reward_fn_params,
    policy_state_dict,
    policy_class,
    policy_kwargs,
    initial_states_np,
    noise_np,
    dt,
    T,
    seed,
):
    """
    Worker function for parallel trajectory rollout.

    This function is executed in a separate process and performs trajectory
    simulation on a chunk of initial states.

    Args:
        dynamics_class: Class constructor for dynamics
        dynamics_params: Parameters to initialize dynamics
        control_spec: Control specification
        reward_fn_class: Class constructor for reward function
        reward_fn_params: Parameters to initialize reward function
        policy_state_dict: Serialized policy parameters
        policy_class: Policy class constructor
        policy_kwargs: Kwargs for policy initialization
        initial_states_np: Initial states as numpy array
        noise_np: Noise as numpy array
        dt: Time step
        T: Time horizon
        seed: Random seed for this worker

    Returns:
        Dictionary of trajectory components as numpy arrays
    """
    # Set random seed for this worker
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Reconstruct dynamics, reward function, and policy
    dynamics = dynamics_class(dynamics_params)
    reward_fn = reward_fn_class(**reward_fn_params)

    # Reconstruct policy
    policy = policy_class(**policy_kwargs)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    # Convert numpy arrays to tensors
    initial_states = torch.from_numpy(initial_states_np).float()
    noise = torch.from_numpy(noise_np).float()

    # Create simulator and run rollout
    simulator = TrajectorySimulator(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        dt=dt,
        T=T,
    )

    with torch.no_grad():
        trajectories = simulator.rollout(policy, initial_states, noise)

    # Convert back to numpy for serialization
    return {
        'states': trajectories.states.cpu().numpy(),
        'actions': trajectories.actions.cpu().numpy(),
        'rewards': trajectories.rewards.cpu().numpy(),
        'masks': trajectories.masks.cpu().numpy(),
        'returns': trajectories.returns.cpu().numpy(),
        'terminal_rewards': trajectories.terminal_rewards.cpu().numpy(),
    }


class ParallelTrajectorySimulator:
    """
    Parallel trajectory simulator using multiprocessing.

    This class wraps TrajectorySimulator and distributes trajectory rollouts
    across multiple CPU cores for improved performance. It's especially useful
    when simulating large batches of trajectories.

    Example:
        >>> from macro_rl.dynamics.ghm_equity import GHMEquityDynamics, GHMEquityParams
        >>> from macro_rl.control.ghm_control import GHMControlSpec
        >>> from macro_rl.rewards.ghm_rewards import GHMRewardFunction
        >>> from macro_rl.networks.actor_critic import ActorCritic
        >>>
        >>> params = GHMEquityParams()
        >>> dynamics = GHMEquityDynamics(params)
        >>> control_spec = GHMControlSpec()
        >>> reward_fn = GHMRewardFunction(
        ...     discount_rate=params.r - params.mu,
        ...     issuance_cost=params.lambda_,
        ...     liquidation_rate=params.omega,
        ...     liquidation_flow=params.alpha,
        ... )
        >>> policy = ActorCritic(state_dim=1, action_dim=2)
        >>>
        >>> simulator = ParallelTrajectorySimulator(
        ...     dynamics=dynamics,
        ...     control_spec=control_spec,
        ...     reward_fn=reward_fn,
        ...     dt=0.01,
        ...     T=10.0,
        ...     n_workers=4,
        ... )
        >>>
        >>> initial_states = torch.rand(1000, 1) * 10.0
        >>> trajectories = simulator.rollout(policy, initial_states)
    """

    def __init__(
        self,
        dynamics,
        control_spec,
        reward_fn,
        dt: float,
        T: float,
        n_workers: Optional[int] = None,
        integrator: Optional[object] = None,
    ):
        """
        Initialize parallel trajectory simulator.

        Args:
            dynamics: Continuous-time dynamics
            control_spec: Control specification
            reward_fn: Reward function
            dt: Time step size
            T: Time horizon
            n_workers: Number of parallel workers (default: CPU count - 1)
            integrator: SDE integrator (defaults to Euler-Maruyama)
        """
        self.dynamics = dynamics
        self.control_spec = control_spec
        self.reward_fn = reward_fn
        self.dt = float(dt)
        self.T = float(T)

        # Create a single sequential simulator for fallback
        self.sequential_simulator = TrajectorySimulator(
            dynamics=dynamics,
            control_spec=control_spec,
            reward_fn=reward_fn,
            dt=dt,
            T=T,
            integrator=integrator,
        )

        # Set number of workers
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        self.n_workers = max(1, n_workers)

        # Cache dynamics and reward function parameters for serialization
        self._cache_serialization_params()

    def _cache_serialization_params(self):
        """Cache parameters needed to reconstruct dynamics and reward function."""
        from macro_rl.dynamics.ghm_equity import GHMEquityDynamics
        from macro_rl.rewards.ghm_rewards import GHMRewardFunction

        # Store dynamics class and params
        self.dynamics_class = type(self.dynamics)
        if isinstance(self.dynamics, GHMEquityDynamics):
            self.dynamics_params = self.dynamics.params
        else:
            # For other dynamics, store what we can
            self.dynamics_params = None

        # Store reward function class and params
        self.reward_fn_class = type(self.reward_fn)
        if isinstance(self.reward_fn, GHMRewardFunction):
            self.reward_fn_params = {
                'discount_rate': self.reward_fn.discount_rate,
                'issuance_cost': self.reward_fn.issuance_cost,
                'liquidation_rate': self.reward_fn.liquidation_rate,
                'liquidation_flow': self.reward_fn.liquidation_flow,
            }
        else:
            # For other reward functions, store what we can
            self.reward_fn_params = {}

    def rollout(
        self,
        policy,
        initial_states: Tensor,
        noise: Optional[Tensor] = None,
    ) -> TrajectoryBatch:
        """
        Simulate batch of trajectories in parallel.

        Args:
            policy: Policy to sample actions from
            initial_states: Initial states (batch, state_dim)
            noise: Optional pre-sampled noise (batch, n_steps, state_dim)

        Returns:
            TrajectoryBatch containing complete trajectories
        """
        # If only 1 worker or small batch, use sequential simulator
        batch_size = initial_states.shape[0]
        if self.n_workers == 1 or batch_size < self.n_workers * 2:
            return self.sequential_simulator.rollout(policy, initial_states, noise)

        # Generate noise if not provided
        if noise is None:
            state_dim = initial_states.shape[-1]
            max_steps = self.sequential_simulator.max_steps
            noise = torch.randn(batch_size, max_steps, state_dim, device=initial_states.device)

        # Split batch into chunks for each worker
        chunk_size = (batch_size + self.n_workers - 1) // self.n_workers
        chunks = []
        noise_chunks = []

        for i in range(self.n_workers):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, batch_size)
            if start_idx >= batch_size:
                break
            chunks.append(initial_states[start_idx:end_idx])
            noise_chunks.append(noise[start_idx:end_idx])

        # Prepare policy state dict
        policy_state_dict = policy.state_dict()

        # Get policy class and kwargs for reconstruction
        from macro_rl.networks.actor_critic import ActorCritic
        if isinstance(policy, ActorCritic):
            policy_class = ActorCritic
            policy_kwargs = {
                'state_dim': policy.state_dim,
                'action_dim': policy.action_dim,
                'shared_layers': policy.shared_layers,
                'hidden_dims': policy.hidden_dims,
                'action_bounds': policy.action_bounds,
            }
        else:
            # Fall back to sequential if policy type not supported
            return self.sequential_simulator.rollout(policy, initial_states, noise)

        # Prepare arguments for each worker
        worker_args = []
        for i, (chunk, noise_chunk) in enumerate(zip(chunks, noise_chunks)):
            worker_args.append((
                self.dynamics_class,
                self.dynamics_params,
                self.control_spec,
                self.reward_fn_class,
                self.reward_fn_params,
                policy_state_dict,
                policy_class,
                policy_kwargs,
                chunk.cpu().numpy(),
                noise_chunk.cpu().numpy(),
                self.dt,
                self.T,
                np.random.randint(0, 2**31 - 1),  # Random seed for this worker
            ))

        # Run rollouts in parallel
        with mp.Pool(processes=len(chunks)) as pool:
            results = pool.starmap(_rollout_worker, worker_args)

        # Concatenate results
        device = initial_states.device
        states = torch.cat([torch.from_numpy(r['states']).to(device) for r in results], dim=0)
        actions = torch.cat([torch.from_numpy(r['actions']).to(device) for r in results], dim=0)
        rewards = torch.cat([torch.from_numpy(r['rewards']).to(device) for r in results], dim=0)
        masks = torch.cat([torch.from_numpy(r['masks']).to(device) for r in results], dim=0)
        returns = torch.cat([torch.from_numpy(r['returns']).to(device) for r in results], dim=0)
        terminal_rewards = torch.cat([torch.from_numpy(r['terminal_rewards']).to(device) for r in results], dim=0)

        return TrajectoryBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            masks=masks,
            returns=returns,
            terminal_rewards=terminal_rewards,
        )
