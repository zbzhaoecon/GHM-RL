"""
Abstract base class for continuous-time economic models.

This module defines the interface that all economic models must implement.
Solvers query this interface to get drift, diffusion, and discount rates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch import Tensor


@dataclass
class StateSpace:
    """
    Description of the state space for a continuous-time model.
    
    Attributes:
        dim: Number of state variables
        lower: Lower bounds for each dimension (dim,)
        upper: Upper bounds for each dimension (dim,)
        names: Human-readable names for each state variable
    
    Example:
        >>> ss = StateSpace(
        ...     dim=2,
        ...     lower=torch.tensor([0.0, 0.0]),
        ...     upper=torch.tensor([10.0, 20.0]),
        ...     names=("eta", "c")
        ... )
    """
    dim: int
    lower: Tensor
    upper: Tensor
    names: Tuple[str, ...]
    
    def __post_init__(self):
        assert len(self.lower) == self.dim
        assert len(self.upper) == self.dim
        assert len(self.names) == self.dim
        assert (self.lower < self.upper).all(), "lower must be < upper"


class ContinuousTimeDynamics(ABC):
    """
    Abstract base class for continuous-time economic models.
    
    Represents an SDE of the form:
        dX = μ(X) dt + σ(X) dW
    
    Subclasses must implement:
        - state_space: Define bounds and dimension
        - params: Return parameter dictionary
        - drift: Compute μ(x)
        - diffusion: Compute σ(x)
        - discount_rate: Return effective discount rate
    
    Example:
        >>> class MyModel(ContinuousTimeDynamics):
        ...     # implement abstract methods
        ...     pass
        >>> model = MyModel()
        >>> x = torch.randn(100, model.state_space.dim)
        >>> mu = model.drift(x)
        >>> sigma = model.diffusion(x)
    """
    
    @property
    @abstractmethod
    def state_space(self) -> StateSpace:
        """Return state space specification."""
        pass
    
    @property
    @abstractmethod
    def params(self) -> Dict[str, float]:
        """
        Return model parameters as dictionary.
        
        Used for logging, checkpointing, and reproducibility.
        """
        pass
    
    @abstractmethod
    def drift(self, x: Tensor) -> Tensor:
        """
        Drift coefficient μ(x).
        
        Args:
            x: State tensor of shape (batch, state_dim)
        
        Returns:
            Drift tensor of shape (batch, state_dim)
        """
        pass
    
    @abstractmethod
    def diffusion(self, x: Tensor) -> Tensor:
        """
        Diffusion coefficient σ(x).
        
        Args:
            x: State tensor of shape (batch, state_dim)
        
        Returns:
            Diffusion tensor of shape (batch, state_dim) for diagonal noise,
            or (batch, state_dim, noise_dim) for correlated noise.
        """
        pass
    
    @abstractmethod
    def discount_rate(self) -> float:
        """
        Effective discount rate for HJB equation.
        
        For most models this is (r - μ) where r is interest rate
        and μ is growth rate.
        """
        pass
    
    def diffusion_squared(self, x: Tensor) -> Tensor:
        """
        Squared diffusion σ(x)² for HJB equation.
        
        Default implementation squares element-wise.
        Override for efficiency or correlated noise.
        
        Args:
            x: State tensor of shape (batch, state_dim)
        
        Returns:
            Tensor of shape (batch, state_dim)
        """
        return self.diffusion(x) ** 2
    
    def sample_interior(self, n: int, device: torch.device = None) -> Tensor:
        """
        Sample n points uniformly from interior of state space.
        
        Args:
            n: Number of points to sample
            device: Target device (default: CPU)
        
        Returns:
            Tensor of shape (n, state_dim)
        """
        ss = self.state_space
        if device is None:
            device = ss.lower.device
        u = torch.rand(n, ss.dim, device=device)
        return ss.lower.to(device) + u * (ss.upper - ss.lower).to(device)
    
    def sample_boundary(
        self, 
        n: int, 
        which: str = "lower",
        dim: int = 0,
        device: torch.device = None
    ) -> Tensor:
        """
        Sample n points on domain boundary.
        
        Args:
            n: Number of points to sample
            which: "lower" or "upper" boundary
            dim: Which dimension's boundary to sample
            device: Target device
        
        Returns:
            Tensor of shape (n, state_dim) with x[:, dim] fixed at boundary
        """
        samples = self.sample_interior(n, device)
        ss = self.state_space
        
        if which == "lower":
            samples[:, dim] = ss.lower[dim]
        elif which == "upper":
            samples[:, dim] = ss.upper[dim]
        else:
            raise ValueError(f"which must be 'lower' or 'upper', got {which}")
        
        return samples
