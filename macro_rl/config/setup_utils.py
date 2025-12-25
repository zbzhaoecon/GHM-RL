"""Utilities for setting up training components from configuration."""

import torch
import numpy as np
from typing import Tuple

from macro_rl.dynamics import GHMEquityDynamics, GHMEquityParams
from macro_rl.control.ghm_control import GHMControlSpec
from macro_rl.rewards.ghm_rewards import GHMRewardFunction
from macro_rl.networks.policy import GaussianPolicy
from macro_rl.networks.value import ValueNetwork
from macro_rl.networks.actor_critic import ActorCritic
from macro_rl.simulation.trajectory import TrajectorySimulator

from .config_manager import ConfigManager


def setup_from_config(
    config_manager: ConfigManager,
    device: torch.device = None
) -> Tuple:
    """Setup all components from configuration.

    Args:
        config_manager: ConfigManager instance
        device: Torch device (if None, use config.misc.device)

    Returns:
        Tuple of (dynamics, control_spec, reward_fn, policy, baseline, simulator, device)
    """
    config = config_manager.config

    # Set device
    if device is None:
        device = torch.device(config.misc.device)

    # Set random seeds
    if config.misc.seed is not None:
        torch.manual_seed(config.misc.seed)
        np.random.seed(config.misc.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.misc.seed)

    # Setup dynamics
    params = GHMEquityParams(
        alpha=config.dynamics.alpha,
        mu=config.dynamics.mu,
        r=config.dynamics.r,
        lambda_=config.dynamics.lambda_,
        sigma_A=config.dynamics.sigma_A,
        sigma_X=config.dynamics.sigma_X,
        rho=config.dynamics.rho,
        c_max=config.dynamics.c_max,
        p=config.dynamics.p,
        phi=config.dynamics.phi,
        omega=config.dynamics.omega,
    )
    dynamics = GHMEquityDynamics(params)

    # Setup control specification
    # Note: GHMControlSpec uses a_L_max and a_E_max, not lower/upper directly
    # The bounds are set via the parent class in __init__
    control_spec = GHMControlSpec(
        a_L_max=config.action_space.dividend_max,
        a_E_max=config.action_space.equity_max,
        issuance_threshold=config.action_space.issuance_threshold,
        issuance_cost=config.action_space.issuance_cost,
    )

    # Setup reward function
    discount_rate = config.reward.discount_rate
    if discount_rate is None:
        discount_rate = params.r - params.mu

    reward_fn = GHMRewardFunction(
        discount_rate=discount_rate,
        issuance_cost=config.reward.issuance_cost or params.lambda_,
        liquidation_rate=config.reward.liquidation_rate,
        liquidation_flow=config.reward.liquidation_flow,
    )

    # Setup policy and baseline
    state_dim = dynamics.state_space.dim
    action_dim = 2

    if config.solver.solver_type == "actor_critic":
        # Use combined actor-critic network
        actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=list(config.network.hidden_dims),
            shared_layers=config.network.shared_layers,
            action_bounds=(control_spec.lower, control_spec.upper),
        ).to(device)
        policy = actor_critic
        baseline = actor_critic  # Same network
    else:
        # Separate policy and value networks for Monte Carlo
        policy = GaussianPolicy(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=list(config.network.policy_hidden),
            action_bounds=(control_spec.lower, control_spec.upper),
            log_std_bounds=tuple(config.network.log_std_bounds),
        ).to(device)

        baseline = None
        if config.training.use_baseline:
            baseline = ValueNetwork(
                input_dim=state_dim,
                hidden_dims=list(config.network.value_hidden),
                activation=config.network.value_activation,
            ).to(device)

    # Setup simulator
    simulator = TrajectorySimulator(
        dynamics=dynamics,
        control_spec=control_spec,
        reward_fn=reward_fn,
        dt=config.training.dt,
        T=config.training.T,
    )

    return dynamics, control_spec, reward_fn, policy, baseline, simulator, device


def print_config_summary(config_manager: ConfigManager):
    """Print a summary of the configuration.

    Args:
        config_manager: ConfigManager instance
    """
    config = config_manager.config

    print("=" * 80)
    print("Configuration Summary")
    print("=" * 80)

    print("\nDynamics:")
    print(f"  alpha={config.dynamics.alpha}, mu={config.dynamics.mu}, r={config.dynamics.r}")
    print(f"  sigma_A={config.dynamics.sigma_A}, sigma_X={config.dynamics.sigma_X}, rho={config.dynamics.rho}")
    print(f"  p={config.dynamics.p}, phi={config.dynamics.phi}, omega={config.dynamics.omega}")

    print("\nAction Space:")
    print(f"  Dividend: [{config.action_space.dividend_min}, {config.action_space.dividend_max}]")
    print(f"  Equity: [{config.action_space.equity_min}, {config.action_space.equity_max}]")

    print("\nTraining:")
    print(f"  Iterations: {config.training.n_iterations}")
    print(f"  Trajectories: {config.training.n_trajectories}")
    print(f"  Horizon: T={config.training.T}, dt={config.training.dt}")
    print(f"  Learning rates: policy={config.training.lr_policy}, baseline={config.training.lr_baseline}")
    print(f"  Regularization: entropy={config.training.entropy_weight}, action={config.training.action_reg_weight}")

    print("\nNetwork:")
    if config.solver.solver_type == "actor_critic":
        print(f"  Type: Actor-Critic (shared_layers={config.network.shared_layers})")
        print(f"  Hidden dims: {config.network.hidden_dims}")
    else:
        print(f"  Type: Separate Policy/Value")
        print(f"  Policy hidden: {config.network.policy_hidden}")
        print(f"  Value hidden: {config.network.value_hidden}")

    print("\nSolver:")
    print(f"  Type: {config.solver.solver_type}")
    if config.solver.solver_type == "actor_critic":
        print(f"  Critic loss: {config.solver.critic_loss}")
        print(f"  Actor loss: {config.solver.actor_loss}")
        print(f"  HJB weight: {config.solver.hjb_weight}")

    print("\nLogging:")
    print(f"  Log dir: {config.logging.log_dir}")
    print(f"  Checkpoint dir: {config.logging.ckpt_dir}")
    print(f"  Log freq: {config.logging.log_freq}, Eval freq: {config.logging.eval_freq}")

    print("\nMisc:")
    print(f"  Seed: {config.misc.seed}")
    print(f"  Device: {config.misc.device}")
    if config.misc.experiment_name:
        print(f"  Experiment: {config.misc.experiment_name}")

    print("=" * 80)
