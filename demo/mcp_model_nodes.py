"""
MCP-enabled model nodes for the Gaia Network demo.

This module implements MCP-enabled versions of the model nodes from model_nodes.py,
which extend the base model nodes with Model Context Protocol (MCP) support.
"""

from typing import Optional

from gaia_network.mcp_node import MCPNode
from demo.model_nodes import RealEstateFinanceNode, ClimateRiskNode, ActuarialDataNode


class MCPRealEstateFinanceNode(RealEstateFinanceNode, MCPNode):
    """
    MCP-enabled version of the Real Estate Finance Node.
    """
    
    def __init__(self, id: str, name: str):
        """Initialize the MCP-enabled Real Estate Finance Node."""
        # Initialize the base node first
        RealEstateFinanceNode.__init__(self)
        
        # Override the ID and name
        self.id = id
        self.name = name
        
        # Initialize the MCP functionality
        MCPNode.__init__(self, self.id, self.name, self.schema, self.state)
    
    async def update(self, observations):
        """
        Update the node with new observations.
        
        Args:
            observations: List of observations to update the node with
        """
        # Import the Observation class
        from gaia_network.state import Observation
        
        # Process each observation
        for observation in observations:
            variable = observation.get("variable")
            value = observation.get("value")
            metadata = observation.get("metadata", {})
            
            # Update the state with the observation
            if variable and value is not None:
                # Create an Observation object
                obs = Observation(
                    variable_name=variable,
                    value=value,
                    metadata=metadata
                )
                self.state.add_observation(obs)


class MCPClimateRiskNode(ClimateRiskNode, MCPNode):
    """
    MCP-enabled version of the Climate Risk Node.
    """
    
    def __init__(self, id: str, name: str):
        """Initialize the MCP-enabled Climate Risk Node."""
        # Initialize the base node first
        ClimateRiskNode.__init__(self)
        
        # Override the ID and name
        self.id = id
        self.name = name
        
        # Initialize the MCP functionality
        MCPNode.__init__(self, self.id, self.name, self.schema, self.state)
    
    async def update(self, observations):
        """
        Update the node with new observations.
        
        Args:
            observations: List of observations to update the node with
        """
        # Import the Observation class
        from gaia_network.state import Observation
        
        # Process each observation
        for observation in observations:
            variable = observation.get("variable")
            value = observation.get("value")
            metadata = observation.get("metadata", {})
            
            # Update the state with the observation
            if variable and value is not None:
                # Create an Observation object
                obs = Observation(
                    variable_name=variable,
                    value=value,
                    metadata=metadata
                )
                self.state.add_observation(obs)


class MCPActuarialDataNode(ActuarialDataNode, MCPNode):
    """
    MCP-enabled version of the Actuarial Data Node.
    """
    
    def __init__(self, id: str, name: str):
        """Initialize the MCP-enabled Actuarial Data Node."""
        # Initialize the base node first
        ActuarialDataNode.__init__(self)
        
        # Override the ID and name
        self.id = id
        self.name = name
        
        # Initialize the MCP functionality
        MCPNode.__init__(self, self.id, self.name, self.schema, self.state)
    
    async def update(self, observations):
        """
        Update the node with new observations.
        
        Args:
            observations: List of observations to update the node with
        """
        # Import the Observation class
        from gaia_network.state import Observation
        
        # Process each observation
        for observation in observations:
            variable = observation.get("variable")
            value = observation.get("value")
            metadata = observation.get("metadata", {})
            
            # Update the state with the observation
            if variable and value is not None:
                # Create an Observation object
                obs = Observation(
                    variable_name=variable,
                    value=value,
                    metadata=metadata
                )
                self.state.add_observation(obs)
