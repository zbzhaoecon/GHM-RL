"""
Model-based Actor-Critic solver.

Joint policy and value learning with known dynamics, using a single optimizer
over the entire ActorCritic module for consistent updates.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor

from macro_rl.dynamics.base import ContinuousTimeDynamics
from macro_rl.control.base import ControlSpec
from macro_rl.rewards.base import RewardFunction
from macro_rl.simulation.trajectory import TrajectorySimulator, TrajectoryBatch
from macro_rl.simulation.parallel import ParallelTrajectorySimulator
from macro_rl.simulation.differentiable import DifferentiableSimulator
from macro_rl.networks.actor_critic import ActorCritic


class ModelBasedActorCritic:
    """Joint policy and value learning with known dynamics.

    Uses a single optimizer over the entire ActorCritic module, so that
    shared trunks (if any) and separate heads are all updated together.
    """

    def __init__(
        self,
        dynamics: ContinuousTimeDynamics,
        control_spec: ControlSpec,
        reward_fn: RewardFunction,
        actor_critic: ActorCritic,
        dt: float = 0.01,
        T: float = 10.0,
        gamma: Optional[float] = None,  # If None, computed from dynamics
        # Loss configuration
        critic_loss: str = "mc+hjb",    # "mc", "td", "hjb", "mc+hjb"
        actor_loss: str = "pathwise",   # "reinforce", "pathwise"
        hjb_weight: float = 0.1,
        entropy_weight: float = 0.01,
        # Training hyperparameters
        n_trajectories: int = 100,
        lr: float = 3e-4,
        max_grad_norm: float = 0.5,
        # Parallel simulation
        use_parallel: bool = False,
        n_workers: Optional[int] = None,
        # Sparse rewards
        use_sparse_rewards: bool = True,  # Use trajectory-level returns
    ) -> None:
        self.dynamics = dynamics
        self.control_spec = control_spec
        self.reward_fn = reward_fn
        self.ac = actor_critic

        self.dt = dt
        self.T = T
        self.max_steps = int(round(T / dt))
        if gamma is None:
            # Use continuous-time discount rate from dynamics, if available
            disc_rate = getattr(dynamics, "discount_rate", lambda: 0.0)()
            self.gamma = float(np.exp(-disc_rate * dt))
        else:
            self.gamma = float(gamma)

        self.critic_loss_type = critic_loss
        self.actor_loss_type = actor_loss
        self.hjb_weight = hjb_weight
        self.entropy_weight = entropy_weight
        self.n_trajectories = n_trajectories
        self.max_grad_norm = max_grad_norm

        # Single optimizer over entire ActorCritic (actor + critic + shared)
        self.optimizer = Adam(self.ac.parameters(), lr=lr)

        # Simulators
        self.use_sparse_rewards = use_sparse_rewards
        if use_parallel:
            self.simulator = ParallelTrajectorySimulator(
                dynamics, control_spec, reward_fn, dt, T, n_workers=n_workers,
                use_sparse_rewards=use_sparse_rewards
            )
        else:
            self.simulator = TrajectorySimulator(
                dynamics, control_spec, reward_fn, dt, T,
                use_sparse_rewards=use_sparse_rewards
            )
        self.diff_simulator = DifferentiableSimulator(
            dynamics, control_spec, reward_fn, dt, T
        )

    def _sample_initial_states(self, n: int) -> Tensor:
        """
        Sample initial states from state space.

        For time-augmented dynamics, this correctly uses the dynamics'
        sample_initial_states method which fixes τ=T (time-to-horizon at start).
        Otherwise falls back to uniform sampling.
        """
        # CRITICAL FIX: Check if dynamics has custom sampling method
        # (e.g., time-augmented dynamics that fix τ=T at episode start)
        if hasattr(self.dynamics, 'sample_initial_states'):
            device = next(self.ac.parameters()).device
            return self.dynamics.sample_initial_states(n, device=device)

        # Try StateSpace API
        state_space = self.dynamics.state_space
        if hasattr(state_space, 'sample_uniform'):
            return state_space.sample_uniform(n)

        # Fallback: uniform sampling over entire state space
        import torch
        device = next(self.ac.parameters()).device
        uniform_samples = torch.rand(n, state_space.dim, device=device)
        samples = state_space.lower.to(device) + uniform_samples * (state_space.upper - state_space.lower).to(device)
        return samples

    def compute_critic_loss(
        self,
        initial_states: Tensor,
        returns: Tensor,
        trajectories: Optional[TrajectoryBatch] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute critic loss based on configured type."""
        losses: Dict[str, float] = {}
        total_loss = 0.0

        # === Monte Carlo value fit ===
        if "mc" in self.critic_loss_type:
            V_pred = self.ac.evaluate(initial_states)
            mc_loss = F.mse_loss(V_pred, returns.detach())
            losses["critic/mc"] = float(mc_loss.item())
            total_loss = total_loss + mc_loss

        # === TD(0) ===
        if "td" in self.critic_loss_type and trajectories is not None:
            s = trajectories.states[:, :-1, :].reshape(-1, initial_states.shape[-1])
            s_next = trajectories.states[:, 1:, :].reshape(-1, initial_states.shape[-1])
            r = trajectories.rewards.reshape(-1)
            masks = trajectories.masks.reshape(-1)

            V = self.ac.evaluate(s)
            with torch.no_grad():
                V_next = self.ac.evaluate(s_next)
            td_target = r + self.gamma * V_next * masks
            td_loss = F.mse_loss(V, td_target)
            losses["critic/td"] = float(td_loss.item())
            total_loss = total_loss + td_loss

        # === HJB residual ===
        if "hjb" in self.critic_loss_type:
            hjb_loss = self.compute_hjb_residual(initial_states)
            losses["critic/hjb"] = float(hjb_loss.item())
            total_loss = total_loss + self.hjb_weight * hjb_loss

        return total_loss, losses

    def compute_hjb_residual(self, states: Tensor) -> Tensor:
        """Compute mean HJB residual for value function at given states."""
        # Value and its derivatives
        V, V_s, V_ss = self.ac.evaluate_with_grad(states)

        # Policy action (deterministic mean)
        with torch.no_grad():
            action = self.ac.act(states, deterministic=True)
        action = self.control_spec.clip(action)

        # Drift and diffusion from dynamics
        drift = self.dynamics.drift(states, action)
        if hasattr(self.dynamics, "diffusion_squared"):
            sigma_sq = self.dynamics.diffusion_squared(states)
        else:
            diff = self.dynamics.diffusion(states)
            sigma_sq = diff * diff

        # Instantaneous reward rate r(s,a)/dt
        reward_rate = self.reward_fn.step_reward(
            states, action, states, self.dt
        ) / self.dt

        # Discount rate
        if hasattr(self.dynamics, "discount_rate"):
            rho = self.dynamics.discount_rate()
        else:
            rho = -np.log(self.gamma) / self.dt

        lhs = rho * V
        rhs = reward_rate + (drift * V_s).sum(dim=-1) + 0.5 * (sigma_sq * V_ss).sum(dim=-1)
        residual = lhs - rhs
        return (residual ** 2).mean()

    def compute_actor_loss(
        self,
        initial_states: Tensor,
        noise: Optional[Tensor] = None,
        trajectories: Optional[TrajectoryBatch] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute actor loss based on configured type."""
        losses: Dict[str, float] = {}

        if self.actor_loss_type == "pathwise":
            # Pathwise gradient through differentiable simulator
            if noise is None:
                noise = torch.randn(
                    initial_states.shape[0],
                    self.max_steps,
                    initial_states.shape[-1],
                    device=initial_states.device,
                )
            returns = self.diff_simulator.simulate(self.ac, initial_states, noise)
            actor_loss = -returns.mean()
            losses["actor/return"] = float(returns.mean().item())

        elif self.actor_loss_type == "reinforce":
            assert trajectories is not None, "REINFORCE needs trajectory rollouts"
            returns = trajectories.returns  # (batch,)
            values = self.ac.evaluate(initial_states).detach()
            advantages = returns - values

            # Advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Flatten states/actions over time
            # Note: states has shape (B, n_steps+1, state_dim), actions has (B, n_steps, action_dim)
            B = trajectories.states.shape[0]
            state_dim = trajectories.states.shape[-1]
            T = trajectories.actions.shape[1]  # Number of steps (transitions)
            action_dim = trajectories.actions.shape[-1]
            s_flat = trajectories.states[:, :-1, :].reshape(B * T, state_dim)
            a_flat = trajectories.actions.reshape(B * T, action_dim)

            _, log_probs_flat, _ = self.ac.evaluate_actions(
                s_flat, a_flat
            )
            log_probs = log_probs_flat.reshape(B, T)

            # Broadcast advantages over time, weighted by masks
            adv = advantages.unsqueeze(-1) * trajectories.masks
            actor_loss = -(log_probs * adv).sum(dim=-1).mean()
            losses["actor/advantage"] = float(advantages.mean().item())

        else:
            raise ValueError(f"Unknown actor_loss type: {self.actor_loss_type}")

        # Entropy regularization (optional)
        if self.entropy_weight > 0.0:
            # Compute entropy on current policy at initial states
            dummy_actions = self.ac.act(initial_states)
            _, _, entropy = self.ac.evaluate_actions(initial_states, dummy_actions)
            entropy_loss = -entropy.mean()
            actor_loss = actor_loss + self.entropy_weight * entropy_loss
            losses["actor/entropy"] = float(entropy.mean().item())

        losses["actor/loss"] = float(actor_loss.item())
        return actor_loss, losses

    def train_step(
        self,
        initial_states: Optional[Tensor] = None,
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Single actor–critic training step."""
        n = n_samples or self.n_trajectories

        # Sample initial states if not provided
        if initial_states is None:
            initial_states = self._sample_initial_states(n)
        else:
            initial_states = initial_states[:n]

        # Pre-sample noise for pathwise / differentiable simulator
        noise = torch.randn(
            n, self.max_steps, initial_states.shape[-1], device=initial_states.device
        )

        # Non-differentiable rollouts for critic + REINFORCE
        with torch.no_grad():
            trajectories = self.simulator.rollout(self.ac, initial_states)

        metrics: Dict[str, float] = {}

        # === Compute critic and actor losses ===
        critic_loss, critic_metrics = self.compute_critic_loss(
            initial_states, trajectories.returns, trajectories
        )
        actor_loss, actor_metrics = self.compute_actor_loss(
            initial_states,
            noise=noise,
            trajectories=trajectories if self.actor_loss_type != "pathwise" else None,
        )

        total_loss = critic_loss + actor_loss

        # === Single optimizer step ===
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log metrics
        metrics.update(critic_metrics)
        metrics.update(actor_metrics)
        metrics["loss/total"] = float(total_loss.item())
        metrics["return/mean"] = float(trajectories.returns.mean().item())
        metrics["return/std"] = float(trajectories.returns.std().item())
        metrics["episode_length/mean"] = float(
            trajectories.masks.sum(dim=-1).mean().item()
        )

        return metrics

    def train(
        self,
        n_iterations: int,
        log_freq: int = 100,
    ) -> Dict[str, List[float]]:
        """Run training loop for n_iterations steps."""
        history: Dict[str, List[float]] = defaultdict(list)

        for it in range(n_iterations):
            metrics = self.train_step()
            for k, v in metrics.items():
                history[k].append(v)

            if it % log_freq == 0:
                ret = metrics.get("return/mean", 0.0)
                total_l = metrics.get("loss/total", 0.0)
                print(f"[Iter {it}] return={ret:.4f}, total_loss={total_l:.4f}")

        return history

    def evaluate(self, n_episodes: int = 50) -> Dict[str, float]:
        """Evaluate current policy on fresh trajectories."""
        initial_states = self._sample_initial_states(n_episodes)
        with torch.no_grad():
            trajectories = self.simulator.rollout(self.ac, initial_states)
        return {
            "return_mean": float(trajectories.returns.mean().item()),
            "return_std": float(trajectories.returns.std().item()),
            "episode_length": float(
                trajectories.masks.sum(dim=-1).mean().item()
            ),
        }
