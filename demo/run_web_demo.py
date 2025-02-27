#!/usr/bin/env python3
"""
Gaia Network Web Demo

This script demonstrates the Gaia Network prototype with three nodes exposed as web services:
- Node A: Models project finance for a real estate development in Miami
- Node B: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- Node C: Serves actuarial data relevant for climate risk

The nodes are exposed as ASGI web services using Starlette.
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
        
        # Create a query to get the expected ROI
        query = Query(
            source_node_id="web_client",
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
        
        # Handle the query
        response = node_a._handle_posterior_query(query)
        
        return JSONResponse(response.to_dict())
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
        
        # Create a query to get the flood probability
        query = Query(
            source_node_id="web_client",
            target_node_id=node_b.id,
            query_type="posterior",
            parameters={
                "variable_name": "flood_probability",
                "covariates": {
                    "location": location,
                    "ipcc_scenario": ipcc_scenario
                },
                "rationale": rationale
            }
        )
        
        # Handle the query
        response = node_b._handle_posterior_query(query)
        
        return JSONResponse(response.to_dict())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def node_b_update(request):
    """Update Node B with new observations."""
    try:
        data = await request.json()
        observations = data.get("observations", [])
        
        # Create a query to update the model
        query = Query(
            source_node_id="web_client",
            target_node_id=node_b.id,
            query_type="update",
            parameters={
                "observations": observations
            }
        )
        
        # Handle the query
        response = node_b._handle_update_query(query)
        
        return JSONResponse(response.to_dict())
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
        
        # Create a query to get the historical flood data
        query = Query(
            source_node_id="web_client",
            target_node_id=node_c.id,
            query_type="posterior",
            parameters={
                "variable_name": "historical_flood_data",
                "covariates": {
                    "location": location
                }
            }
        )
        
        # Handle the query
        response = node_c._handle_posterior_query(query)
        
        return JSONResponse(response.to_dict())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def node_c_add_data(request):
    """Add new data to Node C."""
    try:
        data = await request.json()
        location = data.get("location", "Miami")
        value = data.get("value", 0.0)
        
        # Add the new data
        node_c.add_new_data(location=location, value=value)
        
        return JSONResponse({"status": "success"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Create the Starlette application with routes
routes = [
    # Node A routes
    Route("/node-a/info", node_a_info),
    Route("/node-a/schema", node_a_schema),
    Route("/node-a/query/roi", node_a_query_roi, methods=["POST"]),
    
    # Node B routes
    Route("/node-b/info", node_b_info),
    Route("/node-b/schema", node_b_schema),
    Route("/node-b/query/flood-probability", node_b_query_flood_probability, methods=["POST"]),
    Route("/node-b/update", node_b_update, methods=["POST"]),
    
    # Node C routes
    Route("/node-c/info", node_c_info),
    Route("/node-c/schema", node_c_schema),
    Route("/node-c/query/historical-data", node_c_query_historical_data, methods=["POST"]),
    Route("/node-c/add-data", node_c_add_data, methods=["POST"])
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

app = Starlette(debug=True, routes=routes, middleware=middleware)


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
