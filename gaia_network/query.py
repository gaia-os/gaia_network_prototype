"""
Query module for the Gaia Network.

This module provides classes for representing queries between nodes in the network.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Query:
    """
    A query in the Gaia Network.
    
    A query is a request for information from one node to another.
    """
    source_node_id: str
    target_node_id: str
    query_type: str  # e.g., "schema", "posterior", "update"
    parameters: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the query to a dictionary."""
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "query_type": self.query_type,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """Create a query from a dictionary."""
        return cls(
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            query_type=data["query_type"],
            parameters=data.get("parameters", {}),
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )
    
    def serialize(self) -> str:
        """Serialize the query to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data_str: str) -> 'Query':
        """Deserialize a JSON string to a Query object."""
        data = json.loads(data_str)
        return cls.from_dict(data)


@dataclass
class QueryResponse:
    """
    A response to a query in the Gaia Network.
    """
    query_id: str
    response_type: str  # e.g., "schema", "posterior", "update", "error"
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the query response to a dictionary."""
        return {
            "query_id": self.query_id,
            "response_type": self.response_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResponse':
        """Create a query response from a dictionary."""
        return cls(
            query_id=data["query_id"],
            response_type=data["response_type"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )
    
    def serialize(self) -> str:
        """Serialize the query response to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data_str: str) -> 'QueryResponse':
        """Deserialize a JSON string to a QueryResponse object."""
        data = json.loads(data_str)
        return cls.from_dict(data)


@dataclass
class SchemaQuery(Query):
    """
    A query for a node's schema.
    """
    def __init__(self, source_node_id: str, target_node_id: str, 
                metadata: Dict[str, Any] = None):
        super().__init__(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            query_type="schema",
            parameters={},
            metadata=metadata or {}
        )


@dataclass
class PosteriorQuery(Query):
    """
    A query for a node's posterior distribution.
    """
    def __init__(self, source_node_id: str, target_node_id: str, 
                variable_name: str, covariates: Dict[str, Any] = None,
                rationale: bool = False, metadata: Dict[str, Any] = None):
        parameters = {
            "variable_name": variable_name,
            "covariates": covariates or {},
            "rationale": rationale
        }
        super().__init__(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            query_type="posterior",
            parameters=parameters,
            metadata=metadata or {}
        )


@dataclass
class UpdateQuery(Query):
    """
    A query to update a node with new observations.
    """
    def __init__(self, source_node_id: str, target_node_id: str, 
                observations: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        parameters = {"observations": observations}
        super().__init__(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            query_type="update",
            parameters=parameters,
            metadata=metadata or {}
        )
