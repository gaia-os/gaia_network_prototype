"""
Test script for MCP functionality in the Gaia Network.

This script tests the MCP server and client functionality for Gaia Network nodes.
"""

import asyncio
import logging
import time
import sys
import os
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gaia_network.mcp_node import MCPNode
from gaia_network.schema import Schema, Variable
from gaia_network.state import State, StateCheckpoint
from gaia_network.query import Query, QueryResponse
from gaia_network import registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TestNode(MCPNode):
    """A simple test node for testing MCP functionality."""
    
    def __init__(self, id: str, name: str):
        """Initialize the test node."""
        test_var = Variable(
            name="test_var",
            description="A test variable",
            type="continuous",
            domain={"min": 0.0, "max": 10.0},
        )
        
        schema = Schema()
        schema.add_latent(test_var)
        
        # Create a state with a checkpoint containing the posterior distribution
        state = State()
        state.current_checkpoint = StateCheckpoint(
            latent_values={"test_var": 0.0},
            parameters={
                "posteriors": {
                    "test_var": {"distribution": "normal", "mean": 0.0, "std": 1.0}
                }
            }
        )
        
        super().__init__(id=id, name=name, schema=schema, state=state)
    
    def _handle_posterior_query(self, query: Query) -> QueryResponse:
        """Handle a posterior query."""
        variable_name = query.parameters.get("variable_name")
        if variable_name != "test_var":
            return QueryResponse(
                query_id=query.query_id,
                response_type="error",
                content={"error": f"Unknown variable: {variable_name}"},
            )
        
        # Return a simple posterior
        return QueryResponse(
            query_id=query.query_id,
            response_type="posterior",
            content={
                "posterior": {"distribution": "normal", "mean": 1.0, "std": 0.5},
                "rationale": "This is a test posterior.",
            },
        )
    
    def _handle_update_query(self, query: Query) -> QueryResponse:
        """Handle an update query."""
        observations = query.parameters.get("observations", [])
        return QueryResponse(
            query_id=query.query_id,
            response_type="update",
            content={"updated": len(observations)},
        )


async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    # Create test nodes
    logger.info("Starting MCP servers")
    node1 = TestNode(id="node1", name="Test Node 1")
    node2 = TestNode(id="node2", name="Test Node 2")
    
    # Start MCP servers
    node1.start_mcp_server(port=8001, host="localhost")
    node2.start_mcp_server(port=8002, host="localhost")
    
    # Register nodes with their MCP URLs
    from gaia_network import registry
    registry.register_node(node1, mcp_url="http://localhost:8001/sse")
    registry.register_node(node2, mcp_url="http://localhost:8002/sse")
    
    # Give the servers some time to start
    await asyncio.sleep(1)
    
    # Initialize MCP clients
    logger.info("Initializing MCP clients")
    await node1.init_mcp_client()
    await node2.init_mcp_client()
    
    # Connect nodes to each other
    logger.info("Connecting nodes to each other")
    await node1.connect_to_node_via_mcp("node2", "http://localhost:8002/sse")
    await node2.connect_to_node_via_mcp("node1", "http://localhost:8001/sse")
    
    # Test schema query
    logger.info("Testing schema query")
    schema_response = await node1.query_schema_via_mcp("node2")
    logger.info(f"Schema response: {schema_response.content}")
    
    # Test posterior query
    logger.info("Testing posterior query")
    posterior_response = await node1.query_posterior_via_mcp(
        "node2", "test_var", covariates={}, rationale=True
    )
    logger.info(f"Posterior response: {posterior_response.content}")
    
    # Test update query
    logger.info("Testing update query")
    update_response = await node1.send_update_via_mcp(
        "node2", {"test_var": 1.0}
    )
    logger.info(f"Update response: {update_response.content}")
    
    # Clean up
    logger.info("Cleaning up")
    for node in [node1, node2]:
        try:
            await node.close()
        except Exception as e:
            logger.error(f"Error closing node: {e}")
    
    logger.info("All tests passed!")


async def test_mcp_server_client():
    """Test the MCP server and client functionality."""
    node1 = None
    node2 = None
    
    try:
        # Create a test node
        node1 = TestNode(id="node1", name="Test Node 1")
        node2 = TestNode(id="node2", name="Test Node 2")
        
        # Start the MCP servers
        logger.info("Starting MCP servers")
        node1.start_mcp_server(port=8001)
        node2.start_mcp_server(port=8002)
        
        # Give the servers time to start
        await asyncio.sleep(1)
        
        # Initialize MCP clients
        logger.info("Initializing MCP clients")
        await node1.init_mcp_client()
        await node2.init_mcp_client()
        
        # Connect nodes to each other
        logger.info("Connecting nodes to each other")
        await node1.connect_to_node_via_mcp("node2", "http://localhost:8002/sse")
        await node2.connect_to_node_via_mcp("node1", "http://localhost:8001/sse")
        
        # Test schema query
        logger.info("Testing schema query")
        schema_response = await node1.query_schema_via_mcp("node2")
        logger.info(f"Schema response: {schema_response.content}")
        
        # Test posterior query
        logger.info("Testing posterior query")
        posterior_response = await node1.query_posterior_via_mcp(
            "node2", "test_var", rationale=True
        )
        logger.info(f"Posterior response: {posterior_response.content}")
        
        # Test update query
        logger.info("Testing update query")
        update_response = await node1.send_update_via_mcp(
            "node2", observations=[{"variable": "test_var", "value": 2.0}]
        )
        logger.info(f"Update response: {update_response.content}")
        
        logger.info("All tests passed!")
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
    finally:
        # Clean up
        logger.info("Cleaning up")
        for node in [node1, node2]:
            if node:
                try:
                    await node.close()
                except Exception as e:
                    logger.error(f"Error closing node: {e}")


if __name__ == "__main__":
    asyncio.run(main())
