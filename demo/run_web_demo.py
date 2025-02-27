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


# Node A routes
async def node_a_info(request):
    """Return information about Node A."""
    return JSONResponse({
        "id": node_a.id,
        "name": node_a.name,
        "description": node_a.description
    })


async def node_a_schema(request):
    """Return the schema of Node A."""
    return JSONResponse(node_a.schema.to_dict())


async def node_a_query_roi(request):
    """Query Node A for expected ROI."""
    try:
        data = await request.json()
        location = data.get("location", "Miami")
        ipcc_scenario = data.get("ipcc_scenario", "SSP2-4.5")
        
        # Query Node B for flood probability
        flood_response = node_a.query_posterior(
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
            return JSONResponse({"error": "Failed to get flood probability"}, status_code=400)
        
        # Calculate expected ROI based on flood probability
        roi_response = node_a.query_posterior(
            target_node_id=node_a.id,
            variable_name="expected_roi",
            covariates={
                "location": location,
                "ipcc_scenario": ipcc_scenario,
                "flood_probability": flood_probability
            }
        )
        
        return JSONResponse(roi_response.to_dict())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Node B routes
async def node_b_info(request):
    """Return information about Node B."""
    return JSONResponse({
        "id": node_b.id,
        "name": node_b.name,
        "description": node_b.description
    })


async def node_b_schema(request):
    """Return the schema of Node B."""
    return JSONResponse(node_b.schema.to_dict())


async def node_b_query_flood_probability(request):
    """Query Node B for flood probability."""
    try:
        data = await request.json()
        location = data.get("location", "Miami")
        ipcc_scenario = data.get("ipcc_scenario", "SSP2-4.5")
        rationale = data.get("rationale", False)
        
        response = node_b.query_posterior(
            target_node_id=node_b.id,
            variable_name="flood_probability",
            covariates={
                "location": location,
                "ipcc_scenario": ipcc_scenario
            },
            rationale=rationale
        )
        
        return JSONResponse(response.to_dict())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def node_b_update(request):
    """Update Node B with new observations."""
    try:
        data = await request.json()
        location = data.get("location", "Miami")
        
        # Query Node C for historical flood data
        historical_response = node_b.query_posterior(
            target_node_id=node_c.id,
            variable_name="historical_flood_data",
            covariates={"location": location}
        )
        
        if historical_response.response_type != "posterior":
            return JSONResponse({"error": "Failed to get historical data"}, status_code=400)
            
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
            return JSONResponse({"error": "Could not extract historical data value"}, status_code=400)
        
        # Update Node B with the historical data
        try:
            node_b.send_update(
                target_node_id=node_b.id,
                observations=[
                    {
                        "variable_name": "historical_flood_data",
                        "value": historical_data,
                        "metadata": {"location": location}
                    }
                ]
            )
            
            # Return a standardized response
            return JSONResponse({
                "response_type": "update",
                "content": {
                    "status": "success"
                }
            })
        except Exception as e:
            return JSONResponse({"error": f"Failed to update Node B: {str(e)}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Node C routes
async def node_c_info(request):
    """Return information about Node C."""
    return JSONResponse({
        "id": node_c.id,
        "name": node_c.name,
        "description": node_c.description
    })


async def node_c_schema(request):
    """Return the schema of Node C."""
    return JSONResponse(node_c.schema.to_dict())


async def node_c_query_historical_data(request):
    """Query Node C for historical flood data."""
    try:
        data = await request.json()
        location = data.get("location", "Miami")
        
        response = node_c.query_posterior(
            target_node_id=node_c.id,
            variable_name="historical_flood_data",
            covariates={"location": location}
        )
        
        return JSONResponse(response.to_dict())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def node_c_add_data(request):
    """Add new data to Node C."""
    try:
        data = await request.json()
        location = data.get("location", "Miami")
        value = data.get("value", 0.4)
        
        # Directly use the add_new_data method from the original demo
        node_c.add_new_data(location=location, value=value)
        
        # Return a standardized response format
        return JSONResponse({
            "response_type": "update",
            "content": {
                "status": "success"
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Create middleware for CORS
middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

# Create the Starlette applications with routes
app_a = Starlette(
    debug=True,
    routes=[
        Route("/info", node_a_info),
        Route("/schema", node_a_schema),
        Route("/query/roi", node_a_query_roi, methods=["POST"]),
    ],
    middleware=middleware
)

app_b = Starlette(
    debug=True,
    routes=[
        Route("/info", node_b_info),
        Route("/schema", node_b_schema),
        Route("/query/flood-probability", node_b_query_flood_probability, methods=["POST"]),
        Route("/update", node_b_update, methods=["POST"]),
    ],
    middleware=middleware
)

app_c = Starlette(
    debug=True,
    routes=[
        Route("/info", node_c_info),
        Route("/schema", node_c_schema),
        Route("/query/historical-data", node_c_query_historical_data, methods=["POST"]),
        Route("/add-data", node_c_add_data, methods=["POST"]),
    ],
    middleware=middleware
)


# Run the applications
class Server:
    def __init__(self, app, host, port, node_name):
        self.app = app
        self.host = host
        self.port = port
        self.node_name = node_name
        self.server = None
    
    async def start(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        self.server = uvicorn.Server(config)
        print(f"Starting {self.node_name} server on http://{self.host}:{self.port}")
        await self.server.serve()


async def main():
    server_a = Server(app_a, "127.0.0.1", 8011, "Node A (Real Estate Finance)")
    server_b = Server(app_b, "127.0.0.1", 8012, "Node B (Climate Risk)")
    server_c = Server(app_c, "127.0.0.1", 8013, "Node C (Actuarial Data)")
    
    print("Starting Gaia Network Web Services")
    print(f"Node A (Real Estate Finance): http://127.0.0.1:8011")
    print(f"Node B (Climate Risk): http://127.0.0.1:8012")
    print(f"Node C (Actuarial Data): http://127.0.0.1:8013")
    
    await asyncio.gather(
        server_a.start(),
        server_b.start(),
        server_c.start()
    )


if __name__ == "__main__":
    asyncio.run(main())
