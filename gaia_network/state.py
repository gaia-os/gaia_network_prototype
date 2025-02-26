"""
State module for the Gaia Network.

This module provides classes for representing the state of a node,
including its latent variables, observations, and model parameters.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Observation:
    """
    An observation in the Gaia Network.
    
    An observation is a piece of data that is used to update the state of a node.
    """
    variable_name: str
    value: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the observation to a dictionary."""
        return {
            "variable_name": self.variable_name,
            "value": self.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Observation':
        """Create an observation from a dictionary."""
        return cls(
            variable_name=data["variable_name"],
            value=data["value"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )


@dataclass
class StateCheckpoint:
    """
    A checkpoint of a node's state.
    
    A checkpoint represents the state of a node at a particular point in time.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    latent_values: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the checkpoint to a dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "latent_values": self.latent_values,
            "parameters": self.parameters,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateCheckpoint':
        """Create a checkpoint from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            latent_values=data.get("latent_values", {}),
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class State:
    """
    The state of a node in the Gaia Network.
    
    The state includes the current values of latent variables, observations,
    and model parameters, as well as a history of checkpoints.
    """
    current_checkpoint: StateCheckpoint = field(default_factory=StateCheckpoint)
    checkpoint_history: List[StateCheckpoint] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    
    def add_observation(self, observation: Observation) -> None:
        """Add an observation to the state."""
        self.observations.append(observation)
    
    def create_checkpoint(self, latent_values: Dict[str, Any] = None, 
                         parameters: Dict[str, Any] = None,
                         metadata: Dict[str, Any] = None) -> StateCheckpoint:
        """Create a new checkpoint and add it to the history."""
        # Save the current checkpoint to history
        self.checkpoint_history.append(self.current_checkpoint)
        
        # Create a new checkpoint
        new_checkpoint = StateCheckpoint(
            latent_values=latent_values or {},
            parameters=parameters or {},
            metadata=metadata or {}
        )
        
        # Update the current checkpoint
        self.current_checkpoint = new_checkpoint
        
        return new_checkpoint
    
    def get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[StateCheckpoint]:
        """Get a checkpoint by its ID."""
        if self.current_checkpoint.id == checkpoint_id:
            return self.current_checkpoint
        
        for checkpoint in self.checkpoint_history:
            if checkpoint.id == checkpoint_id:
                return checkpoint
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        return {
            "current_checkpoint": self.current_checkpoint.to_dict(),
            "checkpoint_history": [cp.to_dict() for cp in self.checkpoint_history],
            "observations": [obs.to_dict() for obs in self.observations]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """Create a state from a dictionary."""
        state = cls(
            current_checkpoint=StateCheckpoint.from_dict(data["current_checkpoint"])
        )
        
        for cp_data in data.get("checkpoint_history", []):
            state.checkpoint_history.append(StateCheckpoint.from_dict(cp_data))
        
        for obs_data in data.get("observations", []):
            state.observations.append(Observation.from_dict(obs_data))
        
        return state
    
    def serialize(self) -> str:
        """Serialize the state to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data_str: str) -> 'State':
        """Deserialize a JSON string to a State object."""
        data = json.loads(data_str)
        return cls.from_dict(data)
