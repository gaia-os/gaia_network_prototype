"""
Distribution module for the Gaia Network.

This module provides classes for representing and manipulating probability distributions.
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union


@dataclass
class Distribution:
    """
    Base class for probability distributions in the Gaia Network.
    """
    type: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the distribution to a dictionary for serialization."""
        return {
            "type": self.type,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Distribution':
        """Create a distribution from a dictionary."""
        return cls(
            type=data["type"],
            parameters=data["parameters"]
        )
    
    def serialize(self) -> str:
        """Serialize the distribution to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data_str: str) -> 'Distribution':
        """Deserialize a JSON string to a Distribution object."""
        data = json.loads(data_str)
        return cls.from_dict(data)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from the distribution."""
        raise NotImplementedError("Subclasses must implement sample method")
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function."""
        raise NotImplementedError("Subclasses must implement pdf method")
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        raise NotImplementedError("Subclasses must implement cdf method")


@dataclass
class NormalDistribution(Distribution):
    """
    Normal (Gaussian) distribution.
    """
    def __init__(self, mean: float, std: float):
        super().__init__(
            type="normal",
            parameters={"mean": mean, "std": std}
        )
    
    @property
    def mean(self) -> float:
        return self.parameters["mean"]
    
    @property
    def std(self) -> float:
        return self.parameters["std"]
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from the normal distribution."""
        return np.random.normal(self.mean, self.std, size=size)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Normal probability density function."""
        return (1 / (self.std * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Normal cumulative distribution function."""
        return 0.5 * (1 + np.math.erf((x - self.mean) / (self.std * np.sqrt(2))))


@dataclass
class BetaDistribution(Distribution):
    """
    Beta distribution for representing probabilities.
    """
    def __init__(self, alpha: float, beta: float):
        super().__init__(
            type="beta",
            parameters={"alpha": alpha, "beta": beta}
        )
    
    @property
    def alpha(self) -> float:
        return self.parameters["alpha"]
    
    @property
    def beta(self) -> float:
        return self.parameters["beta"]
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from the beta distribution."""
        return np.random.beta(self.alpha, self.beta, size=size)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Beta probability density function."""
        from scipy.stats import beta
        return beta.pdf(x, self.alpha, self.beta)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Beta cumulative distribution function."""
        from scipy.stats import beta
        return beta.cdf(x, self.alpha, self.beta)


@dataclass
class MarginalDistribution:
    """
    A marginal distribution with a name and associated metadata.
    """
    name: str
    distribution: Distribution
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the marginal distribution to a dictionary."""
        return {
            "name": self.name,
            "distribution": self.distribution.to_dict(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarginalDistribution':
        """Create a marginal distribution from a dictionary."""
        dist_data = data["distribution"]
        dist_type = dist_data["type"]
        
        if dist_type == "normal":
            distribution = NormalDistribution(
                mean=dist_data["parameters"]["mean"],
                std=dist_data["parameters"]["std"]
            )
        elif dist_type == "beta":
            distribution = BetaDistribution(
                alpha=dist_data["parameters"]["alpha"],
                beta=dist_data["parameters"]["beta"]
            )
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
        return cls(
            name=data["name"],
            distribution=distribution,
            metadata=data.get("metadata", {})
        )
    
    def serialize(self) -> str:
        """Serialize the marginal distribution to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data_str: str) -> 'MarginalDistribution':
        """Deserialize a JSON string to a MarginalDistribution object."""
        data = json.loads(data_str)
        return cls.from_dict(data)
