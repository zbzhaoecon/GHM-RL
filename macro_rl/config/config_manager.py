"""Configuration manager for GHM-RL training and simulation."""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from copy import deepcopy


@dataclass
class DynamicsConfig:
    """GHM dynamics parameters."""
    # Cash flow
    alpha: float = 0.18

    # Growth and rates
    mu: float = 0.01
    r: float = 0.03
    lambda_: float = 0.02

    # Volatility
    sigma_A: float = 0.25
    sigma_X: float = 0.12
    rho: float = -0.2

    # State bounds
    c_max: float = 2.0

    # Equity issuance costs
    p: float = 1.06
    phi: float = 0.002

    # Liquidation parameters
    omega: float = 0.55


@dataclass
class ActionSpaceConfig:
    """Action space bounds and constraints."""
    # Dividend bounds
    dividend_min: float = 0.0
    dividend_max: float = 2.0

    # Equity issuance bounds
    equity_min: float = 0.0
    equity_max: float = 2.0

    # Control spec parameters (for compatibility)
    a_L_max: float = 10.0
    a_E_max: float = 0.5
    issuance_threshold: float = 0.05
    issuance_cost: float = 0.0


@dataclass
class EnvironmentConfig:
    """Environment setup parameters."""
    dt: float = 0.01
    max_steps: int = 1000
    seed: Optional[int] = None


@dataclass
class NetworkConfig:
    """Neural network architecture parameters."""
    # Policy network
    policy_hidden: tuple = (64, 64)
    policy_activation: str = "tanh"

    # Value network
    value_hidden: tuple = (64, 64)
    value_activation: str = "tanh"

    # Actor-Critic shared architecture
    hidden_dims: tuple = (256, 256)
    shared_layers: int = 0

    # Policy-specific parameters
    log_std_bounds: tuple = (-5.0, 2.0)
    mean_output_clipping: tuple = (-10.0, 10.0)


@dataclass
class TrainingConfig:
    """Training loop parameters."""
    # Horizon
    dt: float = 0.01
    T: float = 10.0

    # Dynamics type
    use_time_augmented: bool = False  # Use time-augmented dynamics (2D state: c, τ)
    use_sparse_rewards: bool = False  # Compute trajectory return directly (reduces variance)

    # Training iterations
    n_iterations: int = 10000
    n_trajectories: int = 500

    # Optimization
    lr_policy: float = 3e-4
    lr_baseline: float = 1e-3
    lr: float = 3e-4  # Combined LR for actor-critic
    max_grad_norm: float = 0.5

    # Regularization
    advantage_normalization: bool = True
    entropy_weight: float = 0.05

    # Baseline
    use_baseline: bool = True


@dataclass
class SolverConfig:
    """Solver-specific parameters."""
    # Solver type
    solver_type: str = "monte_carlo"  # "monte_carlo" or "actor_critic"

    # Actor-Critic specific
    critic_loss: str = "mc+hjb"
    actor_loss: str = "pathwise"
    hjb_weight: float = 0.1

    # Parallel simulation
    use_parallel: bool = False
    n_workers: Optional[int] = None

    # Monte Carlo specific
    batch_size: int = 1000


@dataclass
class RewardConfig:
    """Reward function parameters."""
    # These are typically computed from dynamics, but can be overridden
    discount_rate: Optional[float] = None  # r - mu
    issuance_cost: float = 0.0
    liquidation_rate: float = 1.0
    liquidation_flow: float = 0.0


@dataclass
class LoggingConfig:
    """Logging and checkpoint parameters."""
    log_dir: str = "runs/ghm_rl"
    log_freq: int = 100
    eval_freq: int = 1000
    ckpt_freq: int = 5000
    ckpt_dir: str = "checkpoints/ghm_rl"


@dataclass
class MiscConfig:
    """Miscellaneous parameters."""
    seed: int = 123
    device: str = "cpu"
    resume: Optional[str] = None
    experiment_name: Optional[str] = None


@dataclass
class GHMConfig:
    """Complete GHM-RL configuration."""
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file (YAML or JSON based on extension)."""
        path = Path(path)
        config_dict = self.to_dict()

        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}. Use .yaml, .yml, or .json")


class ConfigManager:
    """Manager for loading and managing GHM-RL configurations."""

    def __init__(self, config: Optional[GHMConfig] = None):
        """Initialize configuration manager.

        Args:
            config: GHMConfig object. If None, uses defaults.
        """
        self.config = config or GHMConfig()

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ConfigManager":
        """Load configuration from YAML or JSON file.

        Args:
            path: Path to configuration file (.yaml, .yml, or .json)

        Returns:
            ConfigManager instance with loaded configuration
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Load file based on extension
        if path.suffix in [".yaml", ".yml"]:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}. Use .yaml, .yml, or .json")

        # Build configuration from dictionary
        config = cls._dict_to_config(config_dict)
        return cls(config)

    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> GHMConfig:
        """Convert dictionary to GHMConfig object.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            GHMConfig object
        """
        # Helper to convert tuples stored as lists back to tuples
        def convert_tuples(d: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for k, v in d.items():
                if isinstance(v, list) and k in ['policy_hidden', 'value_hidden',
                                                   'hidden_dims', 'log_std_bounds',
                                                   'mean_output_clipping']:
                    result[k] = tuple(v)
                else:
                    result[k] = v
            return result

        # Extract each section with defaults
        dynamics = DynamicsConfig(**convert_tuples(config_dict.get('dynamics', {})))
        action_space = ActionSpaceConfig(**convert_tuples(config_dict.get('action_space', {})))
        environment = EnvironmentConfig(**convert_tuples(config_dict.get('environment', {})))
        network = NetworkConfig(**convert_tuples(config_dict.get('network', {})))
        training = TrainingConfig(**convert_tuples(config_dict.get('training', {})))
        solver = SolverConfig(**convert_tuples(config_dict.get('solver', {})))
        reward = RewardConfig(**convert_tuples(config_dict.get('reward', {})))
        logging = LoggingConfig(**convert_tuples(config_dict.get('logging', {})))
        misc = MiscConfig(**convert_tuples(config_dict.get('misc', {})))

        return GHMConfig(
            dynamics=dynamics,
            action_space=action_space,
            environment=environment,
            network=network,
            training=training,
            solver=solver,
            reward=reward,
            logging=logging,
            misc=misc
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigManager":
        """Create ConfigManager from dictionary.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            ConfigManager instance
        """
        config = cls._dict_to_config(config_dict)
        return cls(config)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of updates in the format:
                     {'section.parameter': value}
                     e.g., {'training.lr_policy': 1e-3, 'dynamics.alpha': 0.2}
        """
        for key, value in updates.items():
            parts = key.split('.')
            if len(parts) != 2:
                raise ValueError(f"Invalid update key: {key}. Expected format: 'section.parameter'")

            section, param = parts
            if not hasattr(self.config, section):
                raise ValueError(f"Invalid section: {section}")

            section_obj = getattr(self.config, section)
            if not hasattr(section_obj, param):
                raise ValueError(f"Invalid parameter: {param} in section {section}")

            setattr(section_obj, param, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key in format 'section.parameter'
            default: Default value if key not found

        Returns:
            Configuration value
        """
        parts = key.split('.')
        if len(parts) != 2:
            return default

        section, param = parts
        if not hasattr(self.config, section):
            return default

        section_obj = getattr(self.config, section)
        return getattr(section_obj, param, default)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file.

        Args:
            path: Path to save configuration
        """
        self.config.save(path)

    def copy(self) -> "ConfigManager":
        """Create a deep copy of the configuration manager."""
        return ConfigManager(deepcopy(self.config))


def load_config(path: Union[str, Path]) -> ConfigManager:
    """Convenience function to load configuration from file.

    Args:
        path: Path to configuration file

    Returns:
        ConfigManager instance
    """
    return ConfigManager.from_file(path)
