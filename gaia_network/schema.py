"""
Schema module for the Gaia Network.

This module provides classes for representing the state space schema of a node,
including latent variables, observable variables, and covariates.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set


@dataclass
class Variable:
    """
    A variable in the state space schema.
    """
    name: str
    description: str
    type: str  # e.g., "continuous", "categorical", "ordinal"
    domain: Optional[Dict[str, Any]] = None  # Domain constraints (min/max for continuous, categories for categorical)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the variable to a dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "metadata": self.metadata
        }
        if self.domain is not None:
            result["domain"] = self.domain
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Variable':
        """Create a variable from a dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            type=data["type"],
            domain=data.get("domain"),
            metadata=data.get("metadata", {})
        )


@dataclass
class Schema:
    """
    State space schema for a Gaia Network node.
    
    The schema defines the latent variables, observable variables, and covariates
    that make up the node's state space.
    """
    latents: List[Variable] = field(default_factory=list)
    observables: List[Variable] = field(default_factory=list)
    covariates: List[Variable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_latent(self, variable: Variable) -> None:
        """Add a latent variable to the schema."""
        self.latents.append(variable)
    
    def add_observable(self, variable: Variable) -> None:
        """Add an observable variable to the schema."""
        self.observables.append(variable)
    
    def add_covariate(self, variable: Variable) -> None:
        """Add a covariate to the schema."""
        self.covariates.append(variable)
    
    def get_variable_names(self) -> Dict[str, Set[str]]:
        """Get the names of all variables in the schema."""
        return {
            "latents": {var.name for var in self.latents},
            "observables": {var.name for var in self.observables},
            "covariates": {var.name for var in self.covariates}
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the schema to a dictionary."""
        return {
            "latents": [var.to_dict() for var in self.latents],
            "observables": [var.to_dict() for var in self.observables],
            "covariates": [var.to_dict() for var in self.covariates],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Schema':
        """Create a schema from a dictionary."""
        schema = cls(metadata=data.get("metadata", {}))
        
        for var_data in data.get("latents", []):
            schema.add_latent(Variable.from_dict(var_data))
        
        for var_data in data.get("observables", []):
            schema.add_observable(Variable.from_dict(var_data))
        
        for var_data in data.get("covariates", []):
            schema.add_covariate(Variable.from_dict(var_data))
        
        return schema
    
    def serialize(self) -> str:
        """Serialize the schema to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data_str: str) -> 'Schema':
        """Deserialize a JSON string to a Schema object."""
        data = json.loads(data_str)
        return cls.from_dict(data)
