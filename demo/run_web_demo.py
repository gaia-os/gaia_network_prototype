"""
Run the Gaia Network web demo.

This module starts web servers for each of the three nodes described in the demo:
- Node A: Models project finance for a real estate development in Miami
- Node B: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- Node C: Serves actuarial data relevant for climate risk

Each node is exposed as a web service using the Starlette ASGI framework.
"""

import asyncio
import uvicorn
import threading
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from demo.model_nodes import create_demo_nodes, create_demo_node_handlers

# Create the demo nodes
node_a, node_b, node_c, node_d = create_demo_nodes()

# Create the node handlers
node_handlers = create_demo_node_handlers()

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
        return JSONResponse(get_node_info(node))
    
    async def handle_schema(request):
        return JSONResponse(get_node_schema(node))
    
    async def handle_query(request):
        variable_name = request.path_params.get("variable_name")
        covariates = await request.json()
        return JSONResponse(query_node(node, variable_name, covariates))
    
    async def handle_update(request):
        data = await request.json()
        return JSONResponse(update_node(node, data))
    
    async def handle_add_data(request):
        data = await request.json()
        return JSONResponse(add_data_to_node(node, data))
    
    routes = [
        Route("/info", handle_info, methods=["GET"]),
        Route("/schema", handle_schema, methods=["GET"]),
        Route("/query/{variable_name}", handle_query, methods=["POST"]),
        Route("/update", handle_update, methods=["POST"]),
    ]
    
    # Add the add-data endpoint only for Node C
    if node.id == "actuarial_data_service":
        routes.append(Route("/add-data", handle_add_data, methods=["POST"]))
    
    return routes


# Create the web applications for each node
middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

app_a = Starlette(
    debug=True,
    routes=create_node_routes(node_a),
    middleware=middleware
)

app_b = Starlette(
    debug=True,
    routes=create_node_routes(node_b),
    middleware=middleware
)

app_c = Starlette(
    debug=True,
    routes=create_node_routes(node_c),
    middleware=middleware
)

app_d = Starlette(
    debug=True,
    routes=create_node_routes(node_d),
    middleware=middleware
)


def run_server(app, host, port):
    """Run a server for the given app."""
    uvicorn.run(app, host=host, port=port)


def main():
    """Run the web demo."""
    print("Starting Gaia Network web demo...")
    print(f"Node A (Real Estate Finance) available at http://127.0.0.1:8011")
    print(f"Node B (Climate Risk) available at http://127.0.0.1:8012")
    print(f"Node C (Actuarial Data) available at http://127.0.0.1:8013")
    
    # Start the servers in separate threads
    thread_a = threading.Thread(target=run_server, args=(app_a, "127.0.0.1", 8011))
    thread_b = threading.Thread(target=run_server, args=(app_b, "127.0.0.1", 8012))
    thread_c = threading.Thread(target=run_server, args=(app_c, "127.0.0.1", 8013))
    thread_d = threading.Thread(target=run_server, args=(app_d, "127.0.0.1", 8014))
    
    thread_a.daemon = True
    thread_b.daemon = True
    thread_c.daemon = True
    thread_d.daemon = True
    
    thread_a.start()
    thread_b.start()
    thread_c.start()
    thread_d.start()
    
    # Keep the main thread running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
