#!/usr/bin/env python3
"""
Gaia Network Web Demo

This script demonstrates the Gaia Network prototype with three nodes exposed as web services:
- Node A: Models project finance for a real estate development in Miami
- Node B: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- Node C: Serves actuarial data relevant for climate risk

Each node is exposed as a separate ASGI web service using Starlette.
"""

import json
import asyncio
import uvicorn
from typing import Dict, Any, Optional
from datetime import datetime

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from gaia_network.registry import register_node, get_node, clear_registry
from gaia_network.query import Query
from demo.model_nodes import create_demo_nodes


# Create and register the demo nodes
node_a, node_b, node_c = create_demo_nodes()


# Business Logic Classes

class NodeHandler:
    """Base class for handling node operations."""
    
    def __init__(self, node):
        self.node = node
    
    def get_info(self):
        """Return information about the node."""
        return {
            "id": self.node.id,
            "name": self.node.name,
            "description": self.node.description
        }
    
    def get_schema(self):
        """Return the schema of the node."""
        return self.node.schema.to_dict()
    
    def query(self, variable_name, covariates):
        """Query the node based on variable_name and covariates."""
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name=variable_name,
            covariates=covariates
        )
        return response.to_dict()
    
    def update(self, data):
        """Update the node with new observations."""
        observations = data.get("observations", [])
        self.node.send_update(
            target_node_id=self.node.id,
            observations=observations
        )
        return {
            "response_type": "update",
            "content": {
                "status": "success"
            }
        }
    
    def add_data(self, data):
        """Add new data to the node."""
        return {
            "error": f"Adding data to node {self.node.id} is not supported"
        }


class RealEstateFinanceHandler(NodeHandler):
    """Handler for the Real Estate Finance node (Node A)."""
    
    def query(self, variable_name, covariates):
        """Query Node A based on variable_name and covariates."""
        if variable_name == "roi":
            return self._query_roi(covariates)
        return super().query(variable_name, covariates)
    
    def _query_roi(self, covariates):
        """Query Node A for expected ROI."""
        location = covariates.get("location", "Miami")
        ipcc_scenario = covariates.get("ipcc_scenario", "SSP2-4.5")
        
        # Query Node B for flood probability
        flood_response = self.node.query_posterior(
            target_node_id=node_b.id,
            variable_name="flood_probability",
            covariates={
                "location": location,
                "ipcc_scenario": ipcc_scenario
            }
        )
        
        # Extract the flood probability
        if flood_response.response_type == "posterior":
            distribution_data = flood_response.content["distribution"]
            alpha = distribution_data['distribution']['parameters']['alpha']
            beta = distribution_data['distribution']['parameters']['beta']
            flood_probability = alpha / (alpha + beta)
        else:
            raise Exception("Failed to get flood probability")
        
        # Calculate expected ROI based on flood probability
        roi_response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="expected_roi",
            covariates={
                "location": location,
                "ipcc_scenario": ipcc_scenario,
                "flood_probability": flood_probability
            }
        )
        
        return roi_response.to_dict()


class ClimateRiskHandler(NodeHandler):
    """Handler for the Climate Risk node (Node B)."""
    
    def query(self, variable_name, covariates):
        """Query Node B based on variable_name and covariates."""
        if variable_name == "flood-probability":
            return self._query_flood_probability(covariates)
        return super().query(variable_name, covariates)
    
    def _query_flood_probability(self, covariates):
        """Query Node B for flood probability."""
        location = covariates.get("location", "Miami")
        ipcc_scenario = covariates.get("ipcc_scenario", "SSP2-4.5")
        rationale = covariates.get("rationale", False)
        
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="flood_probability",
            covariates={
                "location": location,
                "ipcc_scenario": ipcc_scenario
            },
            rationale=rationale
        )
        
        return response.to_dict()
    
    def update(self, data):
        """Update Node B with new observations."""
        location = data.get("location", "Miami")
        
        # Query Node C for historical flood data
        historical_response = self.node.query_posterior(
            target_node_id=node_c.id,
            variable_name="historical_flood_data",
            covariates={"location": location}
        )
        
        if historical_response.response_type != "posterior":
            raise Exception("Failed to get historical data")
            
        # Extract the historical data value from the distribution
        historical_data = None
        if hasattr(historical_response, 'distribution') and historical_response.distribution:
            historical_data = historical_response.distribution.expected_value()
        else:
            # If we can't get the distribution, try to extract from content
            content = getattr(historical_response, 'content', {})
            if isinstance(content, dict) and 'distribution' in content:
                dist_data = content['distribution']
                if 'distribution' in dist_data and 'parameters' in dist_data['distribution']:
                    params = dist_data['distribution']['parameters']
                    if 'mean' in params:
                        historical_data = params['mean']
        
        if historical_data is None:
            raise Exception("Could not extract historical data value")
        
        # Update Node B with the historical data
        self.node.send_update(
            target_node_id=self.node.id,
            observations=[
                {
                    "variable_name": "historical_flood_data",
                    "value": historical_data,
                    "metadata": {"location": location}
                }
            ]
        )
        
        # Return a standardized response
        return {
            "response_type": "update",
            "content": {
                "status": "success"
            }
        }


class ActuarialDataHandler(NodeHandler):
    """Handler for the Actuarial Data node (Node C)."""
    
    def query(self, variable_name, covariates):
        """Query Node C based on variable_name and covariates."""
        if variable_name == "historical-data":
            return self._query_historical_data(covariates)
        return super().query(variable_name, covariates)
    
    def _query_historical_data(self, covariates):
        """Query Node C for historical flood data."""
        location = covariates.get("location", "Miami")
        
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="historical_flood_data",
            covariates={"location": location}
        )
        
        return response.to_dict()
    
    def add_data(self, data):
        """Add new data to Node C."""
        location = data.get("location", "Miami")
        value = data.get("value", 0.4)
        self.node.add_new_data(location=location, value=value)
        return {
            "response_type": "update",
            "content": {
                "status": "success"
            }
        }


# Create handler instances for each node
node_a_handler = RealEstateFinanceHandler(node_a)
node_b_handler = ClimateRiskHandler(node_b)
node_c_handler = ActuarialDataHandler(node_c)

# Map node IDs to handlers
node_handlers = {
    node_a.id: node_a_handler,
    node_b.id: node_b_handler,
    node_c.id: node_c_handler
}

# Business Logic Functions that delegate to the appropriate handler

def get_node_info(node):
    """Return information about the node."""
    handler = node_handlers.get(node.id)
    return handler.get_info()


def get_node_schema(node):
    """Return the schema of the node."""
    handler = node_handlers.get(node.id)
    return handler.get_schema()


def query_node(node, variable_name, covariates):
    """Query the node based on variable_name and covariates."""
    handler = node_handlers.get(node.id)
    return handler.query(variable_name, covariates)


def update_node(node, data):
    """Update the node with new observations."""
    handler = node_handlers.get(node.id)
    return handler.update(data)


def add_data_to_node(node, data):
    """Add new data to the node."""
    handler = node_handlers.get(node.id)
    return handler.add_data(data)


# Create route handlers for each node
def create_node_routes(node):
    """Create route handlers for a specific node."""
    
    async def handle_info(request):
        """Return information about the node."""
        try:
            return JSONResponse(get_node_info(node))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    
    async def handle_schema(request):
        """Return the schema of the node."""
        try:
            return JSONResponse(get_node_schema(node))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    
    async def handle_query(request):
        """Query the node for a specific variable."""
        try:
            variable_name = request.path_params["variable_name"]
            data = await request.json()
            return JSONResponse(query_node(node, variable_name, data))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    
    async def handle_update(request):
        """Update the node with new observations."""
        try:
            data = await request.json()
            return JSONResponse(update_node(node, data))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    
    async def handle_add_data(request):
        """Add new data to the node."""
        try:
            data = await request.json()
            return JSONResponse(add_data_to_node(node, data))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    
    return [
        Route("/info", handle_info, methods=["GET"]),
        Route("/schema", handle_schema, methods=["GET"]),
        Route("/query/{variable_name}", handle_query, methods=["POST"]),
        Route("/update", handle_update, methods=["POST"]),
        Route("/add-data", handle_add_data, methods=["POST"]),
    ]

# Create middleware for CORS
middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

# Create separate apps for each node
apps = []

for node, port in zip([node_a, node_b, node_c], [8011, 8012, 8013]):
    app = Starlette(
        debug=True,
        routes=create_node_routes(node),
        middleware=middleware
    )
    apps.append((app, port))

# Run the applications
class Server:
    def __init__(self, app, host, port):
        self.app = app
        self.host = host
        self.port = port
        self.server = None
    
    async def start(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        self.server = uvicorn.Server(config)
        print(f"Starting Gaia Network Web Services on http://{self.host}:{self.port}")
        await self.server.serve()


async def run_apps():
    await asyncio.gather(*[Server(app, "127.0.0.1", port).start() for app, port in apps])

# Run the application
if __name__ == "__main__":
    print("Starting Gaia Network Web Services")
    print(f"Node A (Real Estate Finance): http://127.0.0.1:8011")
    print(f"Node B (Climate Risk): http://127.0.0.1:8012")
    print(f"Node C (Actuarial Data): http://127.0.0.1:8013")
    asyncio.run(run_apps())
