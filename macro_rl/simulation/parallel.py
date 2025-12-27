"""
Parallel trajectory simulation using multiprocessing.

This module provides a wrapper around TrajectorySimulator that parallelizes
trajectory rollouts across multiple CPU cores for improved performance.
"""

import multiprocessing as mp
from typing import Optional
import torch
from torch import Tensor
import numpy as np


# Global worker state (initialized once per process)
_worker_simulator = None
_worker_policy = None


def _init_worker(simulator_state, policy_class_info):
    """Initialize worker process with simulator and policy template."""
    global _worker_simulator, _worker_policy
    import pickle

    # Unpickle simulator once
    _worker_simulator = pickle.loads(simulator_state)

    # Store policy class info for reconstruction
    _worker_policy = policy_class_info


def _rollout_chunk(policy_state_dict, initial_states_np, noise_np, seed):
    """
    Worker function to rollout a chunk of trajectories.

    Uses the global simulator and reconstructs policy from state dict.
    """
    global _worker_simulator, _worker_policy

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Convert numpy to tensors
    initial_states = torch.from_numpy(initial_states_np).float()
    noise = torch.from_numpy(noise_np).float()

    # Reconstruct policy from state dict
    from macro_rl.networks.actor_critic import ActorCritic

    # Extract metadata
    state_dict_copy = dict(policy_state_dict)
    state_dim = state_dict_copy.pop('_metadata_state_dim')
    action_dim = state_dict_copy.pop('_metadata_action_dim')
    hidden_dims = state_dict_copy.pop('_metadata_hidden_dims')
    shared_layers = state_dict_copy.pop('_metadata_shared_layers')
    action_bounds = state_dict_copy.pop('_metadata_action_bounds', None)

    # Create policy
    policy = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        shared_layers=shared_layers,
        action_bounds=action_bounds,
    )
    policy.load_state_dict(state_dict_copy)
    policy.eval()

    # Run rollout
    with torch.no_grad():
        trajectories = _worker_simulator.rollout(policy, initial_states, noise)

    # Convert to numpy
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

    Uses a persistent process pool to avoid repeated initialization overhead.
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
        use_sparse_rewards: bool = False,
    ):
        """Initialize parallel trajectory simulator."""
        from macro_rl.simulation.trajectory import TrajectorySimulator
        import pickle

        self.dt = float(dt)
        self.T = float(T)

        # Create sequential simulator
        self.sequential_simulator = TrajectorySimulator(
            dynamics=dynamics,
            control_spec=control_spec,
            reward_fn=reward_fn,
            dt=dt,
            T=T,
            integrator=integrator,
            use_sparse_rewards=use_sparse_rewards,
        )

        # Set number of workers
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        self.n_workers = max(1, n_workers)

        # Initialize process pool if using multiple workers
        self._pool = None
        if self.n_workers > 1:
            # Pickle simulator state once
            simulator_state = pickle.dumps(self.sequential_simulator)

            # Set spawn method for compatibility (especially on macOS)
            ctx = mp.get_context('spawn')

            # Create pool with initializer
            self._pool = ctx.Pool(
                processes=self.n_workers,
                initializer=_init_worker,
                initargs=(simulator_state, None),
            )

    def __del__(self):
        """Clean up process pool on deletion."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()

    def rollout(
        self,
        policy,
        initial_states: Tensor,
        noise: Optional[Tensor] = None,
    ):
        """Simulate batch of trajectories in parallel."""
        from macro_rl.simulation.trajectory import TrajectoryBatch

        batch_size = initial_states.shape[0]

        # Use sequential for small batches or single worker
        if self.n_workers == 1 or batch_size < self.n_workers * 4:
            return self.sequential_simulator.rollout(policy, initial_states, noise)

        # Generate noise if needed
        if noise is None:
            state_dim = initial_states.shape[-1]
            max_steps = self.sequential_simulator.max_steps
            noise = torch.randn(batch_size, max_steps, state_dim, device=initial_states.device)

        # Split into chunks
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

        # Prepare policy state dict with metadata
        policy_state_dict = policy.state_dict()
        policy_state_dict['_metadata_state_dim'] = policy.state_dim
        policy_state_dict['_metadata_action_dim'] = policy.action_dim
        policy_state_dict['_metadata_hidden_dims'] = policy.hidden_dims
        policy_state_dict['_metadata_shared_layers'] = policy.shared_layers
        policy_state_dict['_metadata_action_bounds'] = policy.action_bounds

        # Prepare worker arguments
        worker_args = [
            (
                policy_state_dict,
                chunk.cpu().numpy(),
                noise_chunk.cpu().numpy(),
                np.random.randint(0, 2**31 - 1),
            )
            for chunk, noise_chunk in zip(chunks, noise_chunks)
        ]

        # Execute in parallel
        results = self._pool.starmap(_rollout_chunk, worker_args)

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
