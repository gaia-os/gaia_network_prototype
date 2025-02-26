# Gaia Network - MCP Integration Plan

## Overview

This document outlines the plan for integrating the Gaia Network prototype with the Model Context Protocol (MCP). The integration will enable nodes in the Gaia Network to communicate with each other using the standardized MCP protocol, enhancing interoperability and providing a more robust communication framework.

## Background

### Gaia Network
The Gaia Network is a distributed network for probabilistic models, allowing nodes to share information and update their beliefs based on observations from other nodes. The current implementation includes three nodes:
- **Node A**: Models project finance for a real estate development in Miami
- **Node B**: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- **Node C**: Serves actuarial data relevant for climate risk

### Model Context Protocol (MCP)
The Model Context Protocol is a standardized protocol that allows applications to provide context for LLMs in a structured way. It separates the concerns of providing context from the actual LLM interaction. MCP follows a client-server architecture where:
- **Hosts** are LLM applications that initiate connections
- **Clients** maintain 1:1 connections with servers
- **Servers** provide context, tools, and prompts to clients

## Integration Goals

1. Enable Gaia Network nodes to communicate using MCP
2. Maintain the existing probabilistic model functionality
3. Enhance interoperability with other MCP-compatible systems
4. Provide a standardized API for node interactions
5. Support both synchronous and asynchronous communication patterns

## MCP-Enabled Demo

Before diving into the implementation details, let's sketch a first version of how the MCP-enabled Gaia Network demo would work. This will help us understand the integration in practical terms.

### Demo Script Overview

```python
#!/usr/bin/env python3
"""
Gaia Network MCP Demo Script

This script demonstrates the Gaia Network prototype with MCP integration:
- Node A: Real Estate Finance Model exposed as an MCP server
- Node B: Climate Risk Model exposed as an MCP server
- Node C: Actuarial Data Service exposed as an MCP server

The demo follows a similar flow to the original demo but uses MCP for communication.
"""

import asyncio
import json
from pprint import pprint

from gaia_network.registry import register_node, get_node, get_mcp_url, clear_registry
from gaia_network.node import MCPNode
from demo.mcp_model_nodes import create_mcp_demo_nodes


async def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


async def main():
    """Run the Gaia Network MCP demo."""
    await print_separator("Gaia Network MCP Demo")
    
    # Create and register the demo nodes with MCP servers
    print("Creating MCP-enabled demo nodes...")
    node_a, node_b, node_c = await create_mcp_demo_nodes()
    print(f"Node A (Real Estate Finance): {node_a.id} - MCP URL: {get_mcp_url(node_a.id)}")
    print(f"Node B (Climate Risk): {node_b.id} - MCP URL: {get_mcp_url(node_b.id)}")
    print(f"Node C (Actuarial Data): {node_c.id} - MCP URL: {get_mcp_url(node_c.id)}")
    
    # Initialize MCP clients for each node
    await node_a.init_mcp_client()
    await node_b.init_mcp_client()
    await node_c.init_mcp_client()
    
    # Connect nodes to each other via MCP
    await node_a.connect_to_node_via_mcp(node_b.id, get_mcp_url(node_b.id))
    await node_b.connect_to_node_via_mcp(node_c.id, get_mcp_url(node_c.id))
    
    # Step 1: Node A gets a hardcoded location and IPCC scenario
    await print_separator("Step 1: Node A gets a hardcoded location and IPCC scenario")
    location = "Miami"
    ipcc_scenario = "SSP2-4.5"
    print(f"Location: {location}")
    print(f"IPCC Scenario: {ipcc_scenario}")
    
    # Step 2: Node A queries Node B's state space schema via MCP
    await print_separator("Step 2: Node A queries Node B's state space schema via MCP")
    schema_response = await node_a.query_schema_via_mcp(node_b.id)
    print("Schema response type:", schema_response.response_type)
    
    schema = schema_response.content["schema"]
    print("\nLatent variables:")
    for var in schema["latents"]:
        print(f"  - {var['name']}: {var['description']}")
    
    print("\nObservable variables:")
    for var in schema["observables"]:
        print(f"  - {var['name']}: {var['description']}")
    
    print("\nCovariates:")
    for var in schema["covariates"]:
        print(f"  - {var['name']}: {var['description']}")
    
    # Step 3: Node A formulates a query to Node B for flood probability via MCP
    await print_separator("Step 3: Node A queries Node B for flood probability via MCP")
    print(f"Querying flood probability for {location} under {ipcc_scenario} scenario...")
    
    posterior_response = await node_a.query_posterior_via_mcp(
        target_node_id=node_b.id,
        variable_name="flood_probability",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario
        }
    )
    
    print("Posterior response type:", posterior_response.response_type)
    
    if posterior_response.response_type == "posterior":
        distribution_data = posterior_response.content["distribution"]
        print("\nFlood probability distribution:")
        print(f"  - Name: {distribution_data['name']}")
        print(f"  - Type: {distribution_data['distribution']['type']}")
        print(f"  - Parameters: {distribution_data['distribution']['parameters']}")
        print(f"  - Metadata: {distribution_data['metadata']}")
        
        # Calculate expected value for Beta distribution
        alpha = distribution_data['distribution']['parameters']['alpha']
        beta = distribution_data['distribution']['parameters']['beta']
        expected_value = alpha / (alpha + beta)
        print(f"\nExpected flood probability: {expected_value:.4f}")
    
    # Step 4: Node A uses the flood probability to calculate ROI
    await print_separator("Step 4: Node A calculates ROI based on flood probability")
    print("Calculating expected ROI for the real estate project...")
    
    # This is still a direct call since it's internal to Node A
    roi_response = node_a._handle_posterior_query(
        Query(
            source_node_id=node_a.id,
            target_node_id=node_a.id,
            query_type="posterior",
            parameters={
                "variable_name": "expected_roi",
                "covariates": {
                    "location": location,
                    "ipcc_scenario": ipcc_scenario
                }
            }
        )
    )
    
    if roi_response.response_type == "posterior":
        roi_dist_data = roi_response.content["distribution"]
        print("\nROI distribution:")
        print(f"  - Name: {roi_dist_data['name']}")
        print(f"  - Type: {roi_dist_data['distribution']['type']}")
        print(f"  - Parameters: {roi_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {roi_dist_data['metadata']}")
        
        expected_roi = roi_dist_data['distribution']['parameters']['mean']
        print(f"\nExpected ROI: {expected_roi:.2%}")
    
    # Step 5: Node C gets new actuarial data
    await print_separator("Step 5: Node C gets new actuarial data")
    print("Node C receives new actuarial data...")
    
    # Add new data to Node C
    node_c.add_new_data(location=location, value=0.4)
    print("New actuarial data added to Node C")
    
    # Step 6: Node B queries Node C for updated data via MCP
    await print_separator("Step 6: Node B queries Node C for updated data via MCP")
    print("Node B queries Node C for updated actuarial data...")
    
    actuarial_response = await node_b.query_posterior_via_mcp(
        target_node_id=node_c.id,
        variable_name="historical_flood_data",
        covariates={"location": location}
    )
    
    if actuarial_response.response_type == "posterior":
        actuarial_dist_data = actuarial_response.content["distribution"]
        print("\nActuarial data distribution:")
        print(f"  - Name: {actuarial_dist_data['name']}")
        print(f"  - Type: {actuarial_dist_data['distribution']['type']}")
        print(f"  - Parameters: {actuarial_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {actuarial_dist_data['metadata']}")
    
    # Step 7: Node B updates its state based on the new data via MCP
    await print_separator("Step 7: Node B updates its state based on the new data via MCP")
    print("Node B updates its state with the new actuarial data...")
    
    update_response = await node_b.send_update_via_mcp(
        target_node_id=node_b.id,  # Self-update
        observations=[
            {
                "variable_name": "historical_flood_data",
                "value": 0.4,
                "metadata": {"location": location}
            }
        ]
    )
    
    print("Update response type:", update_response.response_type)
    print("Update status:", update_response.content.get("status"))
    
    # Step 8: Node A repeats the query to Node B via MCP
    await print_separator("Step 8: Node A repeats the query to Node B via MCP")
    print(f"Querying updated flood probability for {location} under {ipcc_scenario} scenario...")
    
    updated_posterior_response = await node_a.query_posterior_via_mcp(
        target_node_id=node_b.id,
        variable_name="flood_probability",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario
        }
    )
    
    if updated_posterior_response.response_type == "posterior":
        updated_dist_data = updated_posterior_response.content["distribution"]
        print("\nUpdated flood probability distribution:")
        print(f"  - Name: {updated_dist_data['name']}")
        print(f"  - Type: {updated_dist_data['distribution']['type']}")
        print(f"  - Parameters: {updated_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {updated_dist_data['metadata']}")
        
        # Calculate expected value for Beta distribution
        alpha = updated_dist_data['distribution']['parameters']['alpha']
        beta = updated_dist_data['distribution']['parameters']['beta']
        updated_expected_value = alpha / (alpha + beta)
        print(f"\nUpdated expected flood probability: {updated_expected_value:.4f}")
        
        # Compare with previous value
        print(f"Change in flood probability: {(updated_expected_value - expected_value):.4f}")
    
    # Step 9: Node A includes the rationale parameter via MCP
    await print_separator("Step 9: Node A queries Node B with rationale=True via MCP")
    print(f"Querying flood probability with rationale for {location} under {ipcc_scenario} scenario...")
    
    rationale_response = await node_a.query_posterior_via_mcp(
        target_node_id=node_b.id,
        variable_name="flood_probability",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario
        },
        rationale=True
    )
    
    if rationale_response.response_type == "posterior":
        rationale_dist_data = rationale_response.content["distribution"]
        print("\nFlood probability distribution with rationale:")
        print(f"  - Name: {rationale_dist_data['name']}")
        print(f"  - Type: {rationale_dist_data['distribution']['type']}")
        print(f"  - Parameters: {rationale_dist_data['distribution']['parameters']}")
        
        print("\nRationale:")
        rationale = rationale_response.content.get("rationale", {})
        
        print("\nCalculation details:")
        calculation = rationale.get("calculation", {})
        for key, value in calculation.items():
            print(f"  - {key}: {value}")
        
        print("\nObservations that caused the update:")
        observations = rationale.get("observations", [])
        for obs in observations:
            print(f"  - {obs['variable_name']}: {obs['value']} (timestamp: {obs['timestamp']})")
    
    # Step 10: Node A recalculates ROI based on updated flood probability
    await print_separator("Step 10: Node A recalculates ROI based on updated flood probability")
    print("Recalculating expected ROI for the real estate project...")
    
    # This is still a direct call since it's internal to Node A
    updated_roi_response = node_a._handle_posterior_query(
        Query(
            source_node_id=node_a.id,
            target_node_id=node_a.id,
            query_type="posterior",
            parameters={
                "variable_name": "expected_roi",
                "covariates": {
                    "location": location,
                    "ipcc_scenario": ipcc_scenario
                }
            }
        )
    )
    
    if updated_roi_response.response_type == "posterior":
        updated_roi_dist_data = updated_roi_response.content["distribution"]
        print("\nUpdated ROI distribution:")
        print(f"  - Name: {updated_roi_dist_data['name']}")
        print(f"  - Type: {updated_roi_dist_data['distribution']['type']}")
        print(f"  - Parameters: {updated_roi_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {updated_roi_dist_data['metadata']}")
        
        updated_expected_roi = updated_roi_dist_data['distribution']['parameters']['mean']
        print(f"\nUpdated expected ROI: {updated_expected_roi:.2%}")
        print(f"Change in ROI: {(updated_expected_roi - expected_roi):.2%}")
    
    # Clean up MCP connections
    await node_a.mcp_client.close()
    await node_b.mcp_client.close()
    await node_c.mcp_client.close()
    
    await print_separator("Demo Complete")


if __name__ == "__main__":
    # Import necessary classes
    from gaia_network.query import Query
    
    # Run the async demo
    asyncio.run(main())
```

### MCP Model Nodes Implementation

To support the demo, we would need to implement the MCP-enabled model nodes:

```python
"""
MCP-enabled model nodes for the Gaia Network demo.

This module implements the three nodes with MCP support:
- Node A: Models project finance for a real estate development in Miami
- Node B: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- Node C: Serves actuarial data relevant for climate risk
"""

import asyncio
from typing import Tuple

from gaia_network.node import MCPNode
from gaia_network.registry import register_node, get_mcp_url
from demo.model_nodes import RealEstateFinanceNode, ClimateRiskNode, ActuarialDataNode


class MCPRealEstateFinanceNode(RealEstateFinanceNode, MCPNode):
    """MCP-enabled Real Estate Finance Node"""
    pass


class MCPClimateRiskNode(ClimateRiskNode, MCPNode):
    """MCP-enabled Climate Risk Node"""
    pass


class MCPActuarialDataNode(ActuarialDataNode, MCPNode):
    """MCP-enabled Actuarial Data Node"""
    pass


async def create_mcp_demo_nodes() -> Tuple[MCPNode, MCPNode, MCPNode]:
    """Create and register MCP-enabled demo nodes."""
    # Create the nodes
    node_a = MCPRealEstateFinanceNode()
    node_b = MCPClimateRiskNode()
    node_c = MCPActuarialDataNode()
    
    # Start MCP servers for each node
    # In a real implementation, these would be on different ports
    server_a = node_a.start_mcp_server(port=8001)
    server_b = node_b.start_mcp_server(port=8002)
    server_c = node_c.start_mcp_server(port=8003)
    
    # Register nodes with their MCP URLs
    register_node(node_a, mcp_url="http://localhost:8001")
    register_node(node_b, mcp_url="http://localhost:8002")
    register_node(node_c, mcp_url="http://localhost:8003")
    
    # Wait for servers to start
    await asyncio.sleep(1)
    
    return node_a, node_b, node_c
```

### Key Differences from Original Demo

1. **Asynchronous Execution**: The MCP demo uses `async/await` for all network operations.
2. **MCP Server Initialization**: Each node starts its own MCP server on a different port.
3. **MCP Client Connections**: Nodes connect to each other via MCP clients.
4. **Query Methods**: Nodes use the `*_via_mcp` methods for inter-node communication.
5. **Resource Exposure**: Node schemas and states are exposed as MCP resources.
6. **Tool Exposure**: Node query and update operations are exposed as MCP tools.

This demo shows how the Gaia Network nodes can communicate using the Model Context Protocol while maintaining the same functional flow as the original demo.

## Implementation Strategy

### 1. MCP Server Wrapper for Gaia Nodes

Create an MCP server wrapper for each Gaia Network node. This wrapper will:
- Expose the node's schema as MCP resources
- Provide query functionality as MCP tools
- Handle state updates through MCP notifications

```python
from mcp.server.fastmcp import FastMCP
from gaia_network.node import Node

class GaiaNodeMCPServer:
    def __init__(self, node: Node, server_name: str = None):
        self.node = node
        self.server_name = server_name or f"Gaia-{node.name}"
        self.mcp = FastMCP(self.server_name)
        self._register_resources()
        self._register_tools()
    
    def _register_resources(self):
        # Register node schema as a resource
        @self.mcp.resource(f"gaia://{self.node.id}/schema")
        def get_schema():
            return self.node.schema.to_dict()
        
        # Register node state as a resource
        @self.mcp.resource(f"gaia://{self.node.id}/state")
        def get_state():
            return self.node.state.to_dict()
    
    def _register_tools(self):
        # Register posterior query as a tool
        @self.mcp.tool()
        def query_posterior(variable_name: str, covariates: dict = None, rationale: bool = False):
            """Query this node for a posterior distribution"""
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
            """Send an update to this node"""
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
    
    def run(self, port=None):
        """Run the MCP server"""
        # Implementation depends on how we want to deploy the server
        pass
```

### 2. MCP Client for Inter-Node Communication

Create an MCP client that Gaia Network nodes can use to communicate with each other:

```python
from mcp.client import Client
from gaia_network.query import Query, QueryResponse

class GaiaNodeMCPClient:
    def __init__(self, source_node_id: str):
        self.source_node_id = source_node_id
        self.clients = {}  # Map of node_id to MCP client
    
    async def connect_to_node(self, node_id: str, server_url: str):
        """Connect to a Gaia node MCP server"""
        client = await Client.connect_sse(server_url)
        self.clients[node_id] = client
        return client
    
    async def query_schema(self, target_node_id: str) -> QueryResponse:
        """Query another node for its schema"""
        client = self.clients.get(target_node_id)
        if not client:
            raise ValueError(f"Not connected to node {target_node_id}")
        
        schema_data = await client.read_resource(f"gaia://{target_node_id}/schema")
        return QueryResponse(
            query_id="mcp_query",
            response_type="schema",
            content={"schema": schema_data}
        )
    
    async def query_posterior(self, target_node_id: str, variable_name: str, 
                             covariates: dict = None, rationale: bool = False) -> QueryResponse:
        """Query another node for a posterior distribution"""
        client = self.clients.get(target_node_id)
        if not client:
            raise ValueError(f"Not connected to node {target_node_id}")
        
        result = await client.execute_tool(
            "query_posterior",
            variable_name=variable_name,
            covariates=covariates or {},
            rationale=rationale
        )
        
        return QueryResponse.from_dict(result)
    
    async def send_update(self, target_node_id: str, observations: list) -> QueryResponse:
        """Send an update to another node"""
        client = self.clients.get(target_node_id)
        if not client:
            raise ValueError(f"Not connected to node {target_node_id}")
        
        result = await client.execute_tool(
            "send_update",
            observations=observations
        )
        
        return QueryResponse.from_dict(result)
    
    async def close(self):
        """Close all client connections"""
        for client in self.clients.values():
            await client.close()
```

### 3. Extend Node Class with MCP Support

Extend the existing `Node` class to support MCP-based communication:

```python
class MCPNode(Node):
    """A Gaia Network node with MCP support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp_server = None
        self.mcp_client = None
    
    def start_mcp_server(self, port=None):
        """Start the MCP server for this node"""
        self.mcp_server = GaiaNodeMCPServer(self)
        self.mcp_server.run(port=port)
        return self.mcp_server
    
    async def init_mcp_client(self):
        """Initialize the MCP client for this node"""
        self.mcp_client = GaiaNodeMCPClient(self.id)
        return self.mcp_client
    
    async def connect_to_node_via_mcp(self, target_node_id: str, server_url: str):
        """Connect to another node via MCP"""
        if not self.mcp_client:
            await self.init_mcp_client()
        
        await self.mcp_client.connect_to_node(target_node_id, server_url)
    
    async def query_schema_via_mcp(self, target_node_id: str) -> QueryResponse:
        """Query another node for its schema via MCP"""
        if not self.mcp_client:
            raise ValueError("MCP client not initialized")
        
        return await self.mcp_client.query_schema(target_node_id)
    
    async def query_posterior_via_mcp(self, target_node_id: str, variable_name: str, 
                                     covariates: dict = None, rationale: bool = False) -> QueryResponse:
        """Query another node for a posterior distribution via MCP"""
        if not self.mcp_client:
            raise ValueError("MCP client not initialized")
        
        return await self.mcp_client.query_posterior(
            target_node_id, variable_name, covariates, rationale
        )
    
    async def send_update_via_mcp(self, target_node_id: str, observations: list) -> QueryResponse:
        """Send an update to another node via MCP"""
        if not self.mcp_client:
            raise ValueError("MCP client not initialized")
        
        return await self.mcp_client.send_update(target_node_id, observations)
```

### 4. Registry Updates for MCP Discovery

Enhance the node registry to support MCP server discovery:

```python
from typing import Dict, Optional
from gaia_network.node import Node

# Global registry of nodes
_nodes: Dict[str, Node] = {}
_mcp_urls: Dict[str, str] = {}

def register_node(node: Node, mcp_url: Optional[str] = None) -> None:
    """Register a node in the global registry."""
    _nodes[node.id] = node
    if mcp_url:
        _mcp_urls[node.id] = mcp_url

def get_node(node_id: str) -> Optional[Node]:
    """Get a node from the global registry."""
    return _nodes.get(node_id)

def get_mcp_url(node_id: str) -> Optional[str]:
    """Get the MCP URL for a node."""
    return _mcp_urls.get(node_id)

def clear_registry() -> None:
    """Clear the global registry."""
    _nodes.clear()
    _mcp_urls.clear()
```

### 5. Distribution Serialization for MCP

Enhance the distribution classes to support proper serialization for MCP:

```python
class Distribution:
    # ... existing code ...
    
    def to_mcp_dict(self) -> dict:
        """Convert the distribution to a dictionary suitable for MCP."""
        return self.to_dict()
    
    @classmethod
    def from_mcp_dict(cls, data: dict) -> 'Distribution':
        """Create a distribution from an MCP dictionary."""
        return cls.from_dict(data)
```

## Implementation Phases

### Phase 1: Core MCP Integration
1. Install MCP Python SDK
2. Implement `GaiaNodeMCPServer` class
3. Implement `GaiaNodeMCPClient` class
4. Update requirements.txt with MCP dependencies

### Phase 2: Node Extensions
1. Implement `MCPNode` class extending the base `Node` class
2. Update registry with MCP discovery support
3. Enhance distribution classes for MCP serialization

### Phase 3: Demo and Testing
1. Create an MCP-enabled version of the demo
2. Test inter-node communication via MCP
3. Test with external MCP clients

### Phase 4: Documentation and Refinement
1. Update project documentation
2. Refine API based on testing feedback
3. Add more advanced MCP features (prompts, etc.)

## Dependencies

Add the following to requirements.txt:
```
mcp>=0.1.0
httpx>=0.24.0
```

## Challenges and Considerations

1. **Asynchronous Communication**: MCP client operations are asynchronous, while the current Gaia Network implementation is synchronous. We'll need to handle this mismatch.

2. **Serialization**: Ensure proper serialization/deserialization of complex objects like probability distributions.

3. **Discovery**: Implement a robust mechanism for nodes to discover each other's MCP endpoints.

4. **Authentication**: Consider adding authentication for secure node-to-node communication.

5. **Error Handling**: Implement robust error handling for network failures and protocol errors.

## Conclusion

Integrating the Gaia Network with MCP will provide a standardized communication protocol for probabilistic models, enhancing interoperability and enabling integration with other MCP-compatible systems. The implementation plan outlined above provides a roadmap for achieving this integration while maintaining the core functionality of the Gaia Network.
