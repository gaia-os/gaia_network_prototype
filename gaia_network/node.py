"""
Node module for the Gaia Network.

This module provides the Node class, which is the main interface for interacting with
the Gaia Network. Each node represents a probabilistic model that can be queried and updated.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .schema import Schema
from .state import State, Observation, StateCheckpoint
from .query import Query, QueryResponse, SchemaQuery, PosteriorQuery, UpdateQuery
from .distribution import Distribution, MarginalDistribution


@dataclass
class Node:
    """
    A node in the Gaia Network.
    
    A node represents a probabilistic model that can be queried and updated.
    """
    name: str
    description: str = ""
    schema: Schema = field(default_factory=Schema)
    state: State = field(default_factory=State)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Handlers for different query types
    _query_handlers: Dict[str, Callable] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Initialize the node with default query handlers."""
        self._query_handlers = {
            "schema": self._handle_schema_query,
            "posterior": self._handle_posterior_query,
            "update": self._handle_update_query
        }
    
    def process_query(self, query: Query) -> QueryResponse:
        """Process a query and return a response."""
        if query.target_node_id != self.id:
            return QueryResponse(
                query_id=query.id,
                response_type="error",
                content={"error": f"Query target node ID {query.target_node_id} does not match this node's ID {self.id}"}
            )
        
        handler = self._query_handlers.get(query.query_type)
        if handler is None:
            return QueryResponse(
                query_id=query.id,
                response_type="error",
                content={"error": f"Unsupported query type: {query.query_type}"}
            )
        
        return handler(query)
    
    def _handle_schema_query(self, query: Query) -> QueryResponse:
        """Handle a schema query."""
        return QueryResponse(
            query_id=query.id,
            response_type="schema",
            content={"schema": self.schema.to_dict()}
        )
    
    def _handle_posterior_query(self, query: Query) -> QueryResponse:
        """
        Handle a posterior query.
        
        This method should be overridden by subclasses to implement the specific
        posterior computation logic.
        """
        variable_name = query.parameters.get("variable_name")
        covariates = query.parameters.get("covariates", {})
        rationale = query.parameters.get("rationale", False)
        
        # Default implementation returns an error
        return QueryResponse(
            query_id=query.id,
            response_type="error",
            content={"error": "Posterior computation not implemented"}
        )
    
    def _handle_update_query(self, query: Query) -> QueryResponse:
        """
        Handle an update query.
        
        This method should be overridden by subclasses to implement the specific
        update logic.
        """
        observations_data = query.parameters.get("observations", [])
        observations = [Observation.from_dict(obs) for obs in observations_data]
        
        # Default implementation returns an error
        return QueryResponse(
            query_id=query.id,
            response_type="error",
            content={"error": "Update not implemented"}
        )
    
    def query_schema(self, target_node_id: str) -> QueryResponse:
        """Query another node for its schema."""
        # In a real implementation, this would send the query over the network
        # For this prototype, we'll assume direct method calls
        query = SchemaQuery(
            source_node_id=self.id,
            target_node_id=target_node_id
        )
        return self._send_query(query)
    
    def query_posterior(self, target_node_id: str, variable_name: str, 
                       covariates: Dict[str, Any] = None, 
                       rationale: bool = False) -> QueryResponse:
        """Query another node for a posterior distribution."""
        query = PosteriorQuery(
            source_node_id=self.id,
            target_node_id=target_node_id,
            variable_name=variable_name,
            covariates=covariates,
            rationale=rationale
        )
        return self._send_query(query)
    
    def send_update(self, target_node_id: str, 
                   observations: List[Dict[str, Any]]) -> QueryResponse:
        """Send an update to another node."""
        query = UpdateQuery(
            source_node_id=self.id,
            target_node_id=target_node_id,
            observations=observations
        )
        return self._send_query(query)
    
    def _send_query(self, query: Query) -> QueryResponse:
        """
        Send a query to another node.
        
        In a real implementation, this would use a network protocol.
        For this prototype, we'll use a simple registry of nodes.
        """
        from .registry import get_node
        target_node = get_node(query.target_node_id)
        if target_node is None:
            return QueryResponse(
                query_id=query.id,
                response_type="error",
                content={"error": f"Target node not found: {query.target_node_id}"}
            )
        
        return target_node.process_query(query)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "schema": self.schema.to_dict(),
            "state": self.state.to_dict(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create a node from a dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            schema=Schema.from_dict(data["schema"]),
            state=State.from_dict(data["state"]),
            id=data.get("id", str(uuid.uuid4())),
            metadata=data.get("metadata", {})
        )
    
    def serialize(self) -> str:
        """Serialize the node to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data_str: str) -> 'Node':
        """Deserialize a JSON string to a Node object."""
        data = json.loads(data_str)
        return cls.from_dict(data)
