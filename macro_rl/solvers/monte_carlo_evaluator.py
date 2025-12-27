"""
Monte Carlo policy evaluator for GHM equity model.

This module simulates trajectories under a given policy and computes
empirical value functions. Used to validate numerical and RL solutions.
"""

import numpy as np
import torch
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo evaluation."""
    n_trajectories: int = 10000      # Number of trajectories to simulate
    n_initial_states: int = 50       # Number of initial states to evaluate
    dt: float = 0.1                  # Time step
    T: float = 10.0                  # Time horizon
    max_steps: int = 100             # Maximum steps per trajectory
    seed: Optional[int] = 42         # Random seed


class MonteCarloEvaluator:
    """
    Monte Carlo evaluator for time-augmented GHM policies.

    Simulates trajectories and computes empirical value functions.
    """

    def __init__(self, dynamics, config: MonteCarloConfig = None):
        """
        Initialize Monte Carlo evaluator.

        Args:
            dynamics: GHMEquityTimeAugmentedDynamics instance
            config: Monte Carlo configuration
        """
        self.dynamics = dynamics
        self.config = config or MonteCarloConfig()
        self.p = dynamics.p

        self.rho = self.p.r - self.p.mu  # Discount rate

        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Simulate a single trajectory under a given policy.

        Args:
            initial_state: Initial state (c, τ)
            policy_fn: Policy function mapping state to action (dividend, equity)
            verbose: Print debug info

        Returns:
            Dictionary containing trajectory data:
                - states: (n_steps+1, 2) array of states
                - actions: (n_steps, 2) array of actions
                - rewards: (n_steps,) array of rewards
                - dt_values: (n_steps,) array of time steps
                - total_return: Discounted sum of rewards
                - terminal_value: Value at termination
        """
        # Initialize
        c, tau = initial_state
        states = [initial_state.copy()]
        actions = []
        rewards = []
        dt_values = []

        t = 0.0
        step = 0

        while step < self.config.max_steps and tau > 0 and c >= 0:
            # Get action from policy
            state = np.array([c, tau], dtype=np.float32)
            action = policy_fn(state)  # (dividend, equity)
            a_L, a_E = action

            # Clip actions to valid ranges
            a_L = np.clip(a_L, 0, 20.0)
            a_E = np.clip(a_E, 0, 4.0)

            # Compute instantaneous reward
            # reward = dividend - equity_cost
            fixed_cost = self.p.phi if a_E > 1e-6 else 0.0
            reward = a_L - a_E  # Net payout to shareholders

            # Compute state transition using Euler-Maruyama
            # dc = drift * dt + diffusion * sqrt(dt) * dW
            c_tensor = torch.tensor([[c]], dtype=torch.float32)
            action_tensor = torch.tensor([[a_L, a_E]], dtype=torch.float32)
            tau_tensor = torch.tensor([[tau]], dtype=torch.float32)
            state_tensor = torch.cat([c_tensor, tau_tensor], dim=1)

            with torch.no_grad():
                drift_full = self.dynamics.drift(state_tensor, action_tensor)
                diff_full = self.dynamics.diffusion(state_tensor)

                drift_c = drift_full[0, 0].item()
                diff_c = diff_full[0, 0].item()

            # Simulate stochastic increment
            dW = np.random.normal(0, 1)
            dc = drift_c * self.config.dt + diff_c * np.sqrt(self.config.dt) * dW

            # Update state
            c_new = c + dc
            tau_new = tau - self.config.dt

            # Boundary handling
            # If c < 0, firm goes bankrupt (liquidation)
            if c_new < 0:
                c_new = 0.0
                terminal_value = self.p.liquidation_value
                states.append(np.array([c_new, tau_new]))
                actions.append(np.array([a_L, a_E]))
                rewards.append(reward * self.config.dt)
                dt_values.append(self.config.dt)
                break

            # If c > c_max, pay out excess as dividend (boundary reflection)
            if c_new > self.p.c_max:
                excess_dividend = c_new - self.p.c_max
                reward += excess_dividend
                c_new = self.p.c_max

            # Store transition
            states.append(np.array([c_new, tau_new]))
            actions.append(np.array([a_L, a_E]))
            rewards.append(reward * self.config.dt)  # Scale reward by dt
            dt_values.append(self.config.dt)

            # Update for next step
            c = c_new
            tau = tau_new
            t += self.config.dt
            step += 1

        # Terminal value
        if tau <= 0 or step >= self.config.max_steps:
            terminal_value = self.p.liquidation_value
        else:
            terminal_value = self.p.liquidation_value

        # Compute discounted return
        total_return = 0.0
        for i, (r, dt) in enumerate(zip(rewards, dt_values)):
            total_return += np.exp(-self.rho * i * dt) * r

        # Add terminal value
        total_return += np.exp(-self.rho * len(rewards) * self.config.dt) * terminal_value

        return {
            'states': np.array(states),
            'actions': np.array(actions) if actions else np.zeros((0, 2)),
            'rewards': np.array(rewards),
            'dt_values': np.array(dt_values),
            'total_return': total_return,
            'terminal_value': terminal_value,
            'n_steps': len(rewards),
        }

    def evaluate_policy(
        self,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        initial_states: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate policy over multiple initial states using Monte Carlo.

        Args:
            policy_fn: Policy function mapping state to action
            initial_states: Array of initial states (n, 2), or None to sample uniformly
            verbose: Show progress bar

        Returns:
            Dictionary containing:
                - initial_states: (n, 2) array of initial states
                - values: (n,) array of empirical values
                - std_values: (n,) array of standard errors
                - trajectories: List of trajectory dictionaries
        """
        # Generate initial states if not provided
        if initial_states is None:
            n_states = self.config.n_initial_states
            c_init = np.random.uniform(0, self.p.c_max, n_states)
            tau_init = np.full(n_states, self.config.T)
            initial_states = np.column_stack([c_init, tau_init])
        else:
            n_states = len(initial_states)

        # Evaluate each initial state
        values = np.zeros(n_states)
        std_values = np.zeros(n_states)
        all_trajectories = []

        iterator = tqdm(range(n_states), desc="Evaluating policy") if verbose else range(n_states)

        for i in iterator:
            state = initial_states[i]

            # Simulate multiple trajectories from this state
            returns = []
            trajectories_from_state = []

            for _ in range(self.config.n_trajectories // n_states):
                traj = self.simulate_trajectory(state, policy_fn)
                returns.append(traj['total_return'])
                trajectories_from_state.append(traj)

            # Compute statistics
            values[i] = np.mean(returns)
            std_values[i] = np.std(returns) / np.sqrt(len(returns))

            all_trajectories.append(trajectories_from_state)

        return {
            'initial_states': initial_states,
            'values': values,
            'std_values': std_values,
            'trajectories': all_trajectories,
        }

    def compute_realized_value_function(
        self,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        n_c: int = 50,
        n_tau: int = 50,
        n_samples_per_state: int = 100,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute realized value function on a grid using Monte Carlo.

        This is the "ground truth" value function obtained by simulating
        trajectories under the given policy.

        Args:
            policy_fn: Policy function
            n_c: Number of cash grid points
            n_tau: Number of time grid points
            n_samples_per_state: Number of trajectories per grid point
            verbose: Show progress

        Returns:
            Dictionary with:
                - c_grid: (n_c,) array
                - tau_grid: (n_tau,) array
                - V_realized: (n_c, n_tau) realized value function
                - V_std: (n_c, n_tau) standard errors
        """
        # Create grid
        c_grid = np.linspace(0, self.p.c_max, n_c)
        tau_grid = np.linspace(0.1, self.config.T, n_tau)  # Avoid tau=0

        V_realized = np.zeros((n_c, n_tau))
        V_std = np.zeros((n_c, n_tau))

        total_evals = n_c * n_tau
        iterator = tqdm(total=total_evals, desc="Computing realized value function") if verbose else None

        for i, c in enumerate(c_grid):
            for j, tau in enumerate(tau_grid):
                state = np.array([c, tau])

                # Simulate trajectories
                returns = []
                for _ in range(n_samples_per_state):
                    traj = self.simulate_trajectory(state, policy_fn, verbose=False)
                    returns.append(traj['total_return'])

                V_realized[i, j] = np.mean(returns)
                V_std[i, j] = np.std(returns) / np.sqrt(len(returns))

                if iterator:
                    iterator.update(1)

        if iterator:
            iterator.close()

        return {
            'c_grid': c_grid,
            'tau_grid': tau_grid,
            'V_realized': V_realized,
            'V_std': V_std,
        }


class NumericalPolicyWrapper:
    """Wrapper to convert VFI solution to policy function."""

    def __init__(self, vfi_solver):
        """
        Initialize wrapper.

        Args:
            vfi_solver: NumericalVFISolver instance with solved policies
        """
        self.vfi_solver = vfi_solver

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Evaluate policy at state.

        Args:
            state: (2,) array [c, tau]

        Returns:
            (2,) array [dividend, equity]
        """
        c, tau = state
        dividend, equity = self.vfi_solver.get_policy_at_state(c, tau)
        return np.array([dividend, equity])


class RLPolicyWrapper:
    """Wrapper to convert RL policy network to policy function."""

    def __init__(self, policy_network, device='cpu'):
        """
        Initialize wrapper.

        Args:
            policy_network: Trained policy network
            device: Device to run on
        """
        self.policy_network = policy_network
        self.device = device
        self.policy_network.eval()

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Evaluate policy at state.

        Args:
            state: (2,) array [c, tau]

        Returns:
            (2,) array [dividend, equity]
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.policy_network.sample(state_tensor, deterministic=True)
            action = action.cpu().numpy()[0]

        return action
