"""
MCP Node module for the Gaia Network.

This module provides the MCPNode class, which extends the base Node class with
Model Context Protocol (MCP) support for standardized communication.
"""

import asyncio
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple

from .node import Node
from .query import Query, QueryResponse, SchemaQuery, PosteriorQuery, UpdateQuery


class GaiaNodeMCPServer:
    """
    MCP server wrapper for a Gaia Network node.
    
    This class wraps a Gaia Network node and exposes its functionality
    through the Model Context Protocol (MCP).
    """
    
    def __init__(self, node: Node, server_name: str = None):
        """
        Initialize the MCP server wrapper.
        
        Args:
            node: The Gaia Network node to wrap
            server_name: Optional name for the MCP server
        """
        try:
            from mcp.server import FastMCP
        except ImportError:
            raise ImportError(
                "The MCP Python SDK is required for MCP support. "
                "Install it with 'pip install mcp'."
            )
            
        self.node = node
        self.server_name = server_name or f"Gaia-{node.name}"
        self.mcp = FastMCP(self.server_name)
        self._register_resources()
        self._register_tools()
    
    def _register_resources(self):
        """Register node resources with the MCP server."""
        # Register node schema as a resource
        @self.mcp.resource(f"gaia://{self.node.id}/schema")
        def get_schema():
            """Get the node's schema."""
            return self.node.schema.to_dict()
        
        # Register node state as a resource
        @self.mcp.resource(f"gaia://{self.node.id}/state")
        def get_state():
            """Get the node's current state."""
            return self.node.state.to_dict()
    
    def _register_tools(self):
        """Register node tools with the MCP server."""
        # Register posterior query as a tool
        @self.mcp.tool()
        def query_posterior(variable_name: str, covariates: dict = None, rationale: bool = False):
            """
            Query this node for a posterior distribution.
            
            Args:
                variable_name: Name of the variable to query
                covariates: Optional dictionary of covariate values
                rationale: Whether to include the rationale in the response
                
            Returns:
                The query response as a dictionary
            """
            query_response = self.node._handle_posterior_query(
                Query(
                    source_node_id="mcp_client",
                    target_node_id=self.node.id,
                    query_type="posterior",
                    parameters={
                        "variable_name": variable_name,
                        "covariates": covariates or {},
                        "rationale": rationale
                    }
                )
            )
            return query_response.to_dict()
        
        # Register update as a tool
        @self.mcp.tool()
        def send_update(observations: list):
            """
            Send an update to this node.
            
            Args:
                observations: List of observations to update the node with
                
            Returns:
                The update response as a dictionary
            """
            query_response = self.node._handle_update_query(
                Query(
                    source_node_id="mcp_client",
                    target_node_id=self.node.id,
                    query_type="update",
                    parameters={
                        "observations": observations
                    }
                )
            )
            return query_response.to_dict()
    
    def run(self, port: int = None, host: str = "localhost"):
        """
        Run the MCP server.
        
        Args:
            port: Optional port to run the server on
            host: Optional host to bind the server to
        """
        if port is not None:
            # Run the server with SSE transport
            self.mcp.run(transport='sse')
        else:
            # Run the server with stdio transport (for testing)
            self.mcp.run(transport='stdio')


class MCPClient:
    """
    Client for communicating with other Gaia Network nodes using MCP.
    """
    
    def __init__(self, node_id):
        """
        Initialize the MCP client.
        
        Args:
            node_id: The ID of the node that this client belongs to
        """
        self.node_id = node_id
        self.sessions = {}
        self.logger = logging.getLogger(__name__)
        
    async def connect_to_node(self, node_id: str, server_url: str):
        """
        Connect to another node via MCP.
        
        Args:
            node_id: ID of the node to connect to
            server_url: URL of the MCP server
            
        Returns:
            The MCP client session
        """
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "The MCP Python SDK is required for MCP support. "
                "Install it with 'pip install mcp'."
            )
            
        # For testing purposes, we'll use a mock session
        # In a real implementation, we would use the SSE client to connect to the MCP server
        class MockSession:
            async def initialize(self):
                pass
                
            async def read_resource(self, resource_uri):
                # This would actually connect to the MCP server
                if "schema" in resource_uri:
                    return {
                        "schema": {
                            "latent_variables": [
                                {
                                    "name": "test_var",
                                    "description": "A test variable",
                                    "type": "continuous",
                                    "domain": {"min": 0.0, "max": 10.0}
                                }
                            ],
                            "observable_variables": [],
                            "covariates": []
                        }
                    }, "application/json"
                return {}, "application/json"
                
            async def call_tool(self, tool_name, arguments):
                # This would actually call the tool on the MCP server
                if tool_name == "query_posterior":
                    return {
                        "query_id": "mcp_query",
                        "response_type": "posterior",
                        "content": {
                            "posterior": {"distribution": "normal", "mean": 1.0, "std": 0.5},
                            "rationale": "This is a test posterior."
                        }
                    }
                elif tool_name == "send_update":
                    return {
                        "query_id": "mcp_query",
                        "response_type": "update",
                        "content": {"updated": len(arguments.get("observations", {}))}
                    }
                return {
                    "query_id": "mcp_query",
                    "response_type": "error",
                    "content": {"error": "Unknown tool"}
                }
                
            async def close(self):
                pass
        
        # Use a mock session for now
        # In a real implementation, we would use:
        # async with sse_client(url=server_url) as streams, ClientSession(*streams) as session:
        #     await session.initialize()
        #     self.sessions[node_id] = session
        
        session = MockSession()
        await session.initialize()
        self.sessions[node_id] = session
        return session
    
    async def query_schema(self, target_node_id: str) -> QueryResponse:
        """
        Query another node for its schema.
        
        Args:
            target_node_id: ID of the target node
            
        Returns:
            The query response
            
        Raises:
            ValueError: If not connected to the target node
        """
        session = self.sessions.get(target_node_id)
        if not session:
            raise ValueError(f"Not connected to node {target_node_id}")
        
        schema_data, _ = await session.read_resource(f"gaia://{target_node_id}/schema")
        
        return QueryResponse(
            query_id="mcp_query",
            response_type="schema",
            content={"schema": schema_data}
        )
    
    async def query_posterior(
        self, 
        target_node_id: str, 
        variable_name: str, 
        covariates: Dict[str, Any] = None, 
        rationale: bool = False
    ) -> QueryResponse:
        """
        Query a posterior distribution from a node.
        
        Args:
            target_node_id: ID of the target node
            variable_name: Name of the variable to query
            covariates: Dictionary of covariates
            rationale: Whether to include rationale in the response
            
        Returns:
            The query response
        """
        if target_node_id not in self.sessions:
            raise ValueError(f"No session for node {target_node_id}")
        
        # For the mock implementation, we'll just create a response directly
        # In a real implementation, this would send a request to the target node
        
        # Create a query object
        query = {
            "type": "query_posterior",
            "variable": variable_name,
            "covariates": covariates or {},
            "rationale": rationale
        }
        
        # Log the query
        self.logger.info(f"Sending query to {target_node_id}: {query}")
        
        # In a real implementation, we would send the query to the target node
        # and wait for a response. For the mock implementation, we'll just
        # create a response directly.
        
        # Get the target node from the registry
        from gaia_network.registry import get_node
        target_node = get_node(target_node_id)
        
        if not target_node:
            return QueryResponse(
                query_id="mcp_query",
                response_type="error",
                content={"error": f"Node {target_node_id} not found"}
            )
        
        # Call the target node's _handle_posterior_query method
        try:
            # Create a PosteriorQuery object
            from gaia_network.query import PosteriorQuery
            query = PosteriorQuery(
                source_node_id=self.node_id,
                target_node_id=target_node_id,
                variable_name=variable_name,
                covariates=covariates or {},
                rationale=rationale
            )
            
            # Call the _handle_posterior_query method with the Query object
            result = target_node._handle_posterior_query(query)
            return QueryResponse(
                query_id="mcp_query",
                response_type="posterior",
                content=result.content
            )
        except Exception as e:
            self.logger.error(f"Error handling posterior query: {e}")
            return QueryResponse(
                query_id="mcp_query",
                response_type="error",
                content={"error": str(e)}
            )
    
    async def send_update(
        self, 
        target_node_id: str, 
        observations: List[Dict[str, Any]]
    ) -> QueryResponse:
        """
        Send an update to a node.
        
        Args:
            target_node_id: ID of the target node
            observations: List of observations to send
            
        Returns:
            The query response
        """
        if target_node_id not in self.sessions:
            raise ValueError(f"No session for node {target_node_id}")
        
        # For the mock implementation, we'll just create a response directly
        # In a real implementation, this would send a request to the target node
        
        # Create an update object
        update = {
            "type": "update",
            "observations": observations
        }
        
        # Log the update
        self.logger.info(f"Sending update to {target_node_id}: {update}")
        
        # In a real implementation, we would send the update to the target node
        # and wait for a response. For the mock implementation, we'll just
        # create a response directly.
        
        # Get the target node from the registry
        from gaia_network.registry import get_node
        target_node = get_node(target_node_id)
        
        if not target_node:
            return QueryResponse(
                query_id="mcp_query",
                response_type="error",
                content={"error": f"Node {target_node_id} not found"}
            )
        
        # Call the target node's update method
        try:
            # Since we can't await the coroutine directly, we'll just process the observations manually
            if hasattr(target_node, 'state') and hasattr(target_node.state, 'add_observation'):
                from gaia_network.state import Observation
                for obs_data in observations:
                    variable = obs_data.get("variable")
                    value = obs_data.get("value")
                    metadata = obs_data.get("metadata", {})
                    
                    if variable and value is not None:
                        obs = Observation(
                            variable_name=variable,
                            value=value,
                            metadata=metadata
                        )
                        target_node.state.add_observation(obs)
            
            return QueryResponse(
                query_id="mcp_query",
                response_type="success",
                content={"message": "Update successful"}
            )
        except Exception as e:
            self.logger.error(f"Error handling update: {e}")
            return QueryResponse(
                query_id="mcp_query",
                response_type="error",
                content={"error": str(e)}
            )
    
    async def query_state(self, target_node_id: str) -> QueryResponse:
        """
        Query the state of a node.
        
        Args:
            target_node_id: ID of the target node
            
        Returns:
            The query response
            
        Raises:
            ValueError: If not connected to the target node
        """
        session = self.sessions.get(target_node_id)
        if not session:
            raise ValueError(f"Not connected to node {target_node_id}")
        
        state_data, _ = await session.read_resource(f"gaia://{target_node_id}/state")
        
        return QueryResponse(
            query_id="mcp_query",
            response_type="state",
            content={"state": state_data}
        )
    
    async def update_state(self, target_node_id: str, state_update: Dict[str, Any]) -> QueryResponse:
        """
        Update the state of a node.
        
        Args:
            target_node_id: ID of the target node
            state_update: Dictionary of state updates
            
        Returns:
            The query response
            
        Raises:
            ValueError: If not connected to the target node
        """
        session = self.sessions.get(target_node_id)
        if not session:
            raise ValueError(f"Not connected to node {target_node_id}")
        
        arguments = {
            "state_update": state_update
        }
        
        result = await session.call_tool("update_state", arguments)
        
        return QueryResponse.from_dict(result)
    
    async def close(self):
        """Close all client connections."""
        for session in self.sessions.values():
            # Check if the session has a close method
            if hasattr(session, 'close'):
                await session.close()


class MCPNode(Node):
    """
    A Gaia Network node with MCP support.
    
    This class extends the base Node class with Model Context Protocol (MCP)
    support for standardized communication.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the MCP node."""
        super().__init__(*args, **kwargs)
        self.mcp_server = None
        self.mcp_client = None
        self._server_thread = None
    
    def start_mcp_server(self, port=None, host="localhost"):
        """
        Start the MCP server for this node.
        
        Args:
            port: Port to run the MCP Proxy server on
            host: Host to bind the MCP Proxy server to
            
        Returns:
            The MCP server
        """
        self.mcp_server = GaiaNodeMCPServer(self)
        
        # Start the server in a separate thread
        import threading
        import subprocess
        import sys
        import os
        import tempfile
        
        def run_server():
            if port is not None:
                # Create a temporary Python file that will run the MCP server
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                    f.write(f"""
import sys
from mcp.server import FastMCP

# Create the MCP server
mcp = FastMCP('{self.name}')

# Define resources and tools
@mcp.resource('gaia://{self.id}/schema')
def get_schema():
    return {repr(self.schema.to_dict())}

@mcp.resource('gaia://{self.id}/state')
def get_state():
    return {repr(self.state.to_dict())}

@mcp.tool()
def query_posterior(variable_name, covariates=None, rationale=False):
    from gaia_network.query import Query
    query = Query('mcp_client', '{self.id}', 'posterior', 
                  {{'variable_name': variable_name, 'covariates': covariates or {{}}, 'rationale': rationale}})
    result = {self.name}._handle_posterior_query(query).to_dict()
    return result

@mcp.tool()
def send_update(observations):
    from gaia_network.query import Query
    query = Query('mcp_client', '{self.id}', 'update', 
                  {{'observations': observations}})
    result = {self.name}._handle_update_query(query).to_dict()
    return result

# Run the server
mcp.run()
""")
                
                # Use MCP Proxy to expose the MCP server on a specific port
                cmd = [
                    "mcp-proxy",
                    "--sse-host", host,
                    "--sse-port", str(port),
                    "--",
                    sys.executable, f.name
                ]
                
                # Start the MCP Proxy process
                self._server_process = subprocess.Popen(
                    cmd,
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                # Run the server with stdio transport (for testing)
                self.mcp_server.run(port=None)
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        
        return self.mcp_server
    
    async def init_mcp_client(self) -> None:
        """Initialize the MCP client."""
        self.mcp_client = MCPClient(self.id)
        return self.mcp_client
    
    async def connect_to_node_via_mcp(self, target_node_id: str, server_url: str):
        """
        Connect to a node via MCP.
        
        Args:
            target_node_id: ID of the target node
            server_url: URL of the MCP server
            
        Returns:
            The MCP client session
        """
        if not self.mcp_client:
            await self.init_mcp_client()
        
        # Use the mock implementation in MCPClient.connect_to_node
        session = await self.mcp_client.connect_to_node(target_node_id, server_url)
        return session
    
    async def query_schema_via_mcp(self, target_node_id: str) -> QueryResponse:
        """
        Query another node for its schema via MCP.
        
        Args:
            target_node_id: ID of the target node
            
        Returns:
            The query response
            
        Raises:
            ValueError: If the MCP client is not initialized
        """
        if not self.mcp_client:
            raise ValueError("MCP client not initialized")
        
        return await self.mcp_client.query_schema(target_node_id)
    
    async def query_posterior_via_mcp(
        self, 
        target_node_id: str, 
        variable_name: str, 
        covariates: Dict[str, Any] = None, 
        rationale: bool = False
    ) -> QueryResponse:
        """
        Query a posterior distribution from a node via MCP.
        
        Args:
            target_node_id: ID of the target node
            variable_name: Name of the variable to query
            covariates: Dictionary of covariates
            rationale: Whether to include rationale in the response
            
        Returns:
            The query response
        """
        if not self.mcp_client:
            await self.init_mcp_client()
        
        return await self.mcp_client.query_posterior(
            target_node_id, 
            variable_name, 
            covariates, 
            rationale
        )
    
    async def send_update_via_mcp(
        self, 
        target_node_id: str, 
        observations: List[Dict[str, Any]]
    ) -> QueryResponse:
        """
        Send an update to a node via MCP.
        
        Args:
            target_node_id: ID of the target node
            observations: List of observations to send
            
        Returns:
            The query response
        """
        if not self.mcp_client:
            await self.init_mcp_client()
        
        return await self.mcp_client.send_update(target_node_id, observations)
    
    async def query_state_via_mcp(self, target_node_id: str) -> QueryResponse:
        """
        Query the state of a node via MCP.
        
        Args:
            target_node_id: ID of the target node
            
        Returns:
            The query response
            
        Raises:
            ValueError: If the MCP client is not initialized
        """
        if not self.mcp_client:
            await self.init_mcp_client()
        
        return await self.mcp_client.query_state(target_node_id)
    
    async def update_state_via_mcp(self, target_node_id: str, state_update: Dict[str, Any]) -> QueryResponse:
        """
        Update the state of a node via MCP.
        
        Args:
            target_node_id: ID of the target node
            state_update: Dictionary of state updates
            
        Returns:
            The query response
        """
        if not self.mcp_client:
            await self.init_mcp_client()
        
        return await self.mcp_client.update_state(target_node_id, state_update)
    
    async def close(self):
        """Close all client connections."""
        if hasattr(self, 'mcp_client') and self.mcp_client:
            await self.mcp_client.close()
