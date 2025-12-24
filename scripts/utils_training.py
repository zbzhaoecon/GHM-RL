"""
Common utilities for Actor-Critic training scripts.

Provides configuration dataclass, TensorBoard setup, and checkpoint management.
"""

import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainConfig:
    """Configuration for Actor-Critic training."""

    # Core horizon / discretization
    dt: float = 0.01
    T: float = 10.0

    # Solver/optimizer
    n_iterations: int = 50_000
    n_trajectories: int = 256
    lr: float = 3e-4
    critic_loss: str = "mc+hjb"
    actor_loss: str = "pathwise"
    hjb_weight: float = 0.1
    entropy_weight: float = 0.01
    max_grad_norm: float = 0.5

    # Network
    hidden_dims: tuple = (256, 256)
    shared_layers: int = 1

    # Logging / checkpoints
    log_dir: str = "runs/ghm_model1"
    log_freq: int = 100
    eval_freq: int = 1000
    ckpt_freq: int = 5000
    ckpt_dir: str = "checkpoints/ghm_model1"

    # Parallel simulation
    use_parallel: bool = False
    n_workers: Optional[int] = None

    # Misc
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: Optional[str] = None


def create_writer(config: TrainConfig) -> SummaryWriter:
    """
    Create a TensorBoard writer with a timestamped subdirectory.

    Args:
        config: Training configuration

    Returns:
        TensorBoard SummaryWriter instance
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(config.log_dir, timestamp)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    # Log hyperparameters in TensorBoard
    writer.add_text("hparams", str(asdict(config)))
    return writer


def save_checkpoint(
    solver,
    config: TrainConfig,
    step: int,
    ckpt_name: Optional[str] = None,
    best_return: Optional[float] = None,
):
    """
    Save a training checkpoint.

    Args:
        solver: ModelBasedActorCritic solver instance
        config: Training configuration
        step: Current training step
        ckpt_name: Optional custom checkpoint name (defaults to step_{step}.pt)
        best_return: Optional best return achieved so far (for tracking)
    """
    os.makedirs(config.ckpt_dir, exist_ok=True)
    if ckpt_name is None:
        ckpt_name = f"step_{step}.pt"
    path = os.path.join(config.ckpt_dir, ckpt_name)

    checkpoint_data = {
        "step": step,
        "config": asdict(config),
        "actor_critic_state": solver.ac.state_dict(),
        "optimizer_state": solver.optimizer.state_dict(),
    }

    if best_return is not None:
        checkpoint_data["best_return"] = best_return

    torch.save(checkpoint_data, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(checkpoint_path: str, solver, config: TrainConfig):
    """
    Load a training checkpoint and resume training.

    Args:
        checkpoint_path: Path to checkpoint file
        solver: ModelBasedActorCritic solver instance
        config: Current training configuration (can override saved config)

    Returns:
        start_step: The step number to resume from
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Checkpoint] Loading from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Load model and optimizer states
    solver.ac.load_state_dict(checkpoint["actor_critic_state"])
    solver.optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_step = checkpoint["step"]
    saved_config = checkpoint.get("config", {})

    print(f"[Checkpoint] Loaded successfully!")
    print(f"  Resuming from step: {start_step}")
    print(f"  Original config: lr={saved_config.get('lr', 'N/A')}, "
          f"hidden_dims={saved_config.get('hidden_dims', 'N/A')}")
    print(f"  Current config: lr={config.lr}, hidden_dims={config.hidden_dims}")

    return start_step

