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
        
    async def connect_to_node(self, target_node_id: str, server_url: str):
        """
        Connect to another node via MCP.
        
        Args:
            target_node_id: ID of the target node
            server_url: URL of the MCP server
            
        Returns:
            The MCP client session
        """
        if target_node_id in self.sessions:
            self.logger.info(f"Already connected to node {target_node_id}")
            return self.sessions[target_node_id]
        
        self.logger.info(f"Connecting to node {target_node_id} at {server_url}")
        
        try:
            # Import the MCP client
            from mcp.client import SSEClient
            
            # Create a new SSE client session
            client = SSEClient(server_url)
            await client.connect()
            
            # Store the session
            self.sessions[target_node_id] = client
            self.logger.info(f"Connected to node {target_node_id}")
            
            return client
        except Exception as e:
            self.logger.error(f"Failed to connect to node {target_node_id}: {e}")
            raise
        
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
        
        try:
            # Use the real MCP client to read the schema resource
            schema_data, content_type = await session.read_resource(f"gaia://{target_node_id}/schema")
            
            self.logger.info(f"Received schema from {target_node_id}: {schema_data}")
            
            return QueryResponse(
                query_id="mcp_query",
                response_type="schema",
                content={"schema": schema_data}
            )
        except Exception as e:
            self.logger.error(f"Error querying schema from {target_node_id}: {e}")
            return QueryResponse(
                query_id="error",
                response_type="error",
                content={"error": str(e)}
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
        
        session = self.sessions[target_node_id]
        
        # Create a query object
        arguments = {
            "variable_name": variable_name,
            "covariates": covariates or {},
            "rationale": rationale
        }
        
        # Log the query
        self.logger.info(f"Sending posterior query to {target_node_id}: {arguments}")
        
        try:
            # Call the query_posterior tool on the MCP server
            result = await session.call_tool("query_posterior", arguments)
            
            self.logger.info(f"Received posterior response from {target_node_id}: {result}")
            
            # Convert the result to a QueryResponse
            return QueryResponse(
                query_id=result.get("query_id", "mcp_query"),
                response_type=result.get("response_type", "posterior"),
                content=result.get("content", {})
            )
        except Exception as e:
            self.logger.error(f"Error querying posterior from {target_node_id}: {e}")
            return QueryResponse(
                query_id="error",
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
        
        session = self.sessions[target_node_id]
        
        # Create an update object
        arguments = {
            "observations": observations
        }
        
        # Log the update
        self.logger.info(f"Sending update to {target_node_id}: {arguments}")
        
        try:
            # Call the send_update tool on the MCP server
            result = await session.call_tool("send_update", arguments)
            
            self.logger.info(f"Received update response from {target_node_id}: {result}")
            
            # Convert the result to a QueryResponse
            return QueryResponse(
                query_id=result.get("query_id", "mcp_query"),
                response_type=result.get("response_type", "update"),
                content=result.get("content", {})
            )
        except Exception as e:
            self.logger.error(f"Error sending update to {target_node_id}: {e}")
            return QueryResponse(
                query_id="error",
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
        
        try:
            # Use the real MCP client to read the state resource
            state_data, content_type = await session.read_resource(f"gaia://{target_node_id}/state")
            
            self.logger.info(f"Received state from {target_node_id}: {state_data}")
            
            return QueryResponse(
                query_id="mcp_query",
                response_type="state",
                content={"state": state_data}
            )
        except Exception as e:
            self.logger.error(f"Error querying state from {target_node_id}: {e}")
            return QueryResponse(
                query_id="error",
                response_type="error",
                content={"error": str(e)}
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
        
        # Create an update object
        arguments = {
            "state_update": state_update
        }
        
        # Log the update
        self.logger.info(f"Sending state update to {target_node_id}: {arguments}")
        
        try:
            # Call the update_state tool on the MCP server
            result = await session.call_tool("update_state", arguments)
            
            self.logger.info(f"Received state update response from {target_node_id}: {result}")
            
            # Convert the result to a QueryResponse
            return QueryResponse(
                query_id=result.get("query_id", "mcp_query"),
                response_type=result.get("response_type", "update"),
                content=result.get("content", {})
            )
        except Exception as e:
            self.logger.error(f"Error updating state for {target_node_id}: {e}")
            return QueryResponse(
                query_id="error",
                response_type="error",
                content={"error": str(e)}
            )
    
    async def close(self):
        """Close all client sessions."""
        for node_id, session in self.sessions.items():
            try:
                await session.close()
            except Exception as e:
                self.logger.error(f"Error closing session for {node_id}: {e}")
        self.sessions = {}


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
        import threading
        import subprocess
        import sys
        import os
        import tempfile
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting MCP server for node {self.id} on port {port}")
        
        # Initialize the MCP server (needed for the test)
        self.mcp_server = None
        
        if port is not None:
            # Create a temporary Python file that will run the MCP server
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                server_file_path = f.name
                # Write the server code to the file
                f.write(f"""
import sys
import logging
from mcp.server import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create the MCP server
logger.info("Creating MCP server for {self.name}")
mcp = FastMCP('{self.name}')

# Define resources and tools
@mcp.resource('gaia://{self.id}/schema')
def get_schema():
    logger.info("Received request for schema")
    schema_dict = {repr(self.schema.to_dict())}
    logger.info(f"Returning schema: {{schema_dict}}")
    return schema_dict

@mcp.resource('gaia://{self.id}/state')
def get_state():
    logger.info("Received request for state")
    state_dict = {repr(self.state.to_dict())}
    logger.info(f"Returning state: {{state_dict}}")
    return state_dict

@mcp.tool()
def query_posterior(variable_name, covariates=None, rationale=False):
    logger.info(f"Received posterior query for {{variable_name}} with covariates {{covariates}}")
    from gaia_network.query import Query
    query = Query('mcp_client', '{self.id}', 'posterior', 
                  {{'variable_name': variable_name, 'covariates': covariates or {{}}, 'rationale': rationale}})
    result = {self.name}._handle_posterior_query(query).to_dict()
    logger.info(f"Returning posterior query result: {{result}}")
    return result

@mcp.tool()
def send_update(observations):
    logger.info(f"Received update with observations {{observations}}")
    from gaia_network.query import Query
    query = Query('mcp_client', '{self.id}', 'update', 
                  {{'observations': observations}})
    result = {self.name}._handle_update_query(query).to_dict()
    logger.info(f"Returning update result: {{result}}")
    return result

@mcp.tool()
def update_state(state_update):
    logger.info(f"Received state update: {{state_update}}")
    # Implement state update logic here
    result = {{"query_id": "mcp_query", "response_type": "update", "content": {{"updated": True}}}}
    logger.info(f"Returning state update result: {{result}}")
    return result

# Run the server
logger.info("Starting MCP server")
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
                
                logger.info(f"Starting MCP Proxy with command: {' '.join(cmd)}")
                
                # Start the MCP Proxy process
                self._server_process = subprocess.Popen(
                    cmd,
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                
                logger.info(f"MCP server started with PID {self._server_process.pid}")
        else:
            # Run the server with stdio transport (for testing)
            logger.info("Running MCP server with stdio transport (for testing)")
            self.mcp_server.run(port=None)
        
        return self.mcp_server
    
    async def init_mcp_client(self) -> None:
        """Initialize the MCP client for this node."""
        if not hasattr(self, 'mcp_client') or not self.mcp_client:
            from gaia_network.mcp_node import MCPClient
            self.mcp_client = MCPClient(self.id)
            logging.getLogger(__name__).info(f"Initialized MCP client for node {self.id}")
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
        
        # Connect to the target node using the real MCP client
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
        observations: List[Dict[str, Any]] = None,
        observation_dict: Dict[str, Any] = None
    ) -> QueryResponse:
        """
        Send an update to a node via MCP.
        
        Args:
            target_node_id: ID of the target node
            observations: List of observations to send (format: [{"variable": "var_name", "value": value}])
            observation_dict: Dictionary of observations to send (format: {"var_name": value})
                             Will be converted to the list format
            
        Returns:
            The query response
        """
        if observation_dict is not None:
            # Convert the dictionary to a list of observations
            observations = [
                {"variable": var_name, "value": value, "metadata": {}}
                for var_name, value in observation_dict.items()
            ]
        
        if not observations:
            observations = []
            
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
        """Close all client connections and stop the server if running."""
        # Close client connections
        if hasattr(self, 'mcp_client') and self.mcp_client:
            await self.mcp_client.close()
        
        # Stop the server if running
        if hasattr(self, '_server_process') and self._server_process:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
                logging.getLogger(__name__).info(f"MCP server process terminated for node {self.id}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Error terminating MCP server process for node {self.id}: {e}")
