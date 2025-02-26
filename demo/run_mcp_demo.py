"""
Gaia Network MCP Demo Script

This script demonstrates the Gaia Network prototype with MCP-enabled nodes.
"""

import asyncio
import logging
import sys
from typing import Dict

from gaia_network.registry import register_node, get_node, get_all_mcp_urls, clear_registry
from demo.mcp_model_nodes import (
    MCPRealEstateFinanceNode,
    MCPClimateRiskNode,
    MCPActuarialDataNode
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def setup_nodes():
    """
    Set up the MCP-enabled nodes for the demo.
    
    Returns:
        A dictionary of node IDs to nodes
    """
    # Create the nodes
    re_finance_node = MCPRealEstateFinanceNode(
        id="re_finance_node",
        name="Real Estate Finance Node",
    )
    climate_risk_node = MCPClimateRiskNode(
        id="climate_risk_node",
        name="Climate Risk Node",
    )
    actuarial_node = MCPActuarialDataNode(
        id="actuarial_node",
        name="Actuarial Data Node",
    )
    
    # Register the nodes with their MCP URLs
    register_node(re_finance_node, mcp_url="http://localhost:8001/sse")
    register_node(climate_risk_node, mcp_url="http://localhost:8002/sse")
    register_node(actuarial_node, mcp_url="http://localhost:8003/sse")
    
    # Initialize MCP clients for each node
    await re_finance_node.init_mcp_client()
    await climate_risk_node.init_mcp_client()
    await actuarial_node.init_mcp_client()
    
    # Connect nodes to each other using mock connections
    # This will use the mock implementation in MCPClient.connect_to_node
    for node in [re_finance_node, climate_risk_node, actuarial_node]:
        logger.info(f"Connecting node {node.id} to other nodes")
        for target_id, url in get_all_mcp_urls().items():
            if target_id != node.id:
                logger.info(f"  Connecting to {target_id} at {url}")
                await node.connect_to_node_via_mcp(target_id, url)
                
    # Log the sessions
    for node in [re_finance_node, climate_risk_node, actuarial_node]:
        logger.info(f"Node {node.id} sessions: {list(node.mcp_client.sessions.keys())}")
    
    return {
        "re_finance_node": re_finance_node,
        "climate_risk_node": climate_risk_node,
        "actuarial_node": actuarial_node,
    }


async def run_demo(nodes):
    """
    Run the MCP-enabled demo.
    
    Args:
        nodes: A dictionary of node IDs to nodes
    """
    try:
        logger.info("Running MCP-enabled demo")
        
        # Get the nodes
        re_finance_node = nodes["re_finance_node"]
        climate_risk_node = nodes["climate_risk_node"]
        actuarial_node = nodes["actuarial_node"]
        
        # Step 1: Query the climate risk node for its schema
        logger.info("Step 1: Querying climate risk node for its schema")
        schema_response = await re_finance_node.query_schema_via_mcp(climate_risk_node.id)
        logger.info(f"Schema response: {schema_response}")
        
        # Step 2: Query the actuarial node for its schema
        logger.info("Step 2: Querying actuarial node for its schema")
        schema_response = await re_finance_node.query_schema_via_mcp(actuarial_node.id)
        logger.info(f"Schema response: {schema_response}")
        
        # Step 3: Query the climate risk node for flood risk posterior
        logger.info("Step 3: Querying climate risk node for flood risk posterior")
        flood_risk_response = await re_finance_node.query_posterior_via_mcp(
            climate_risk_node.id,
            "flood_risk",
            covariates={"location": "Miami", "ipcc_scenario": "SSP2-4.5"},
            rationale=True
        )
        logger.info(f"Flood risk response: {flood_risk_response}")
        
        # Step 4: Query the actuarial node for insurance premium posterior
        logger.info("Step 4: Querying actuarial node for insurance premium posterior")
        premium_response = await re_finance_node.query_posterior_via_mcp(
            actuarial_node.id,
            "insurance_premium",
            covariates={
                "location": "Miami",
                "property_value": 1000000,
                "flood_risk": 0.3  # Using a placeholder value
            },
            rationale=True
        )
        logger.info(f"Premium response: {premium_response}")
        
        # Step 5: Send an update to the climate risk node
        logger.info("Step 5: Sending update to climate risk node")
        update_response = await re_finance_node.send_update_via_mcp(
            climate_risk_node.id,
            observations=[
                {"variable": "flood_event", "value": 1, "metadata": {"location": "Miami", "date": "2023-01-15"}}
            ]
        )
        logger.info(f"Update response: {update_response}")
        
        # Step 6: Query the climate risk node for updated flood risk posterior
        logger.info("Step 6: Querying climate risk node for updated flood risk posterior")
        updated_flood_risk_response = await re_finance_node.query_posterior_via_mcp(
            climate_risk_node.id,
            "flood_risk",
            covariates={"location": "Miami", "ipcc_scenario": "SSP2-4.5"},
            rationale=True
        )
        logger.info(f"Updated flood risk response: {updated_flood_risk_response}")
        
        logger.info("Demo completed successfully!")
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        raise


async def main():
    """Main function to run the demo."""
    try:
        logger.info("Setting up MCP-enabled nodes")
        nodes = await setup_nodes()
        
        await run_demo(nodes)
    finally:
        # Clean up
        for node_id, node in nodes.items():
            if hasattr(node, 'mcp_client') and node.mcp_client:
                await node.close()


if __name__ == "__main__":
    asyncio.run(main())
