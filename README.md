# Gaia Network Prototype

This is a simple prototype implementation of the Gaia Network as described in [Gaia Network: An Illustrated Primer](https://engineeringideas.substack.com/p/gaia-network-an-illustrated-primer).

## Overview

The Gaia Network is a distributed network for probabilistic models, allowing nodes to share information and update their beliefs based on observations from other nodes. This prototype implements a simplified version of the network with three nodes:

- **Node A**: Models project finance for a real estate development in Miami
- **Node B**: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- **Node C**: Serves actuarial data relevant for climate risk

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gaia_network_prototype
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
gaia_network_prototype/
├── gaia_network/            # Core library
│   ├── __init__.py
│   ├── distribution.py      # Probability distributions
│   ├── node.py              # Node implementation
│   ├── query.py             # Query and response classes
│   ├── registry.py          # Node registry
│   ├── schema.py            # State space schema
│   └── state.py             # Node state
├── demo/                    # Demo implementation
│   ├── model_nodes.py       # Specific node implementations and factory functions
│   ├── node_handler.py      # Base class for web service handlers
│   ├── real_estate_finance_node.py # Node A implementation and handler
│   ├── climate_risk_node.py # Node B implementation and handler
│   ├── actuarial_data_node.py # Node C implementation and handler
│   ├── run_demo.py          # Demo script
│   ├── run_web_demo.py      # Web services demo
│   └── web_client_demo.py   # Web client demo
├── venv/                    # Virtual environment (not in version control)
└── requirements.txt         # Project dependencies
```

## Core Concepts

- **Node**: A probabilistic model that can be queried and updated
- **Schema**: Defines the state space of a node, including latent variables, observable variables, and covariates
- **State**: The current state of a node, including the values of latent variables and model parameters
- **Query**: A request for information from one node to another
- **Distribution**: A probability distribution representing uncertainty

## Demo Script

The demo script (`demo/run_demo.py`) demonstrates the following workflow:

1. Node A gets a hardcoded location (Miami) and IPCC scenario (SSP2-4.5)
2. Node A queries Node B's state space schema
3. Node A formulates a query to Node B to get the predictive posterior for flood probability
4. Node A uses the flood probability to calculate the expected ROI for the real estate project
5. Node C gets new actuarial data
6. Node B queries Node C for the updated data
7. Node B updates its internal state posterior based on the new data
8. Node A repeats the query to Node B and gets an updated answer
9. Node A includes the `rationale=True` parameter in the request, and Node B sends its state history and rationale
10. Node A recalculates ROI based on the updated flood probability

To run the standard demo:

```bash
python -m demo.run_demo
```

This will demonstrate the interaction between the three nodes, showing how they share information and update their beliefs based on observations.

## Web Services

The Gaia Network nodes are exposed as web services using the Starlette ASGI framework. Each node runs as a separate service on its own port:

- **Node A (Real Estate Finance)**: http://127.0.0.1:8011
- **Node B (Climate Risk)**: http://127.0.0.1:8012
- **Node C (Actuarial Data)**: http://127.0.0.1:8013

### Starting the Web Services

To start the web servers:

```bash
python -m demo.run_web_demo
```

### Web Client Demo

A web client demo script is provided to demonstrate how to interact with the web services:

```bash
python -m demo.web_client_demo
```

This script follows a similar flow to the standard demo but uses HTTP requests instead of direct method calls.

### API Endpoints

Each node provides the following RESTful endpoints:

#### Common Endpoints for All Nodes

- `GET /info`: Get information about the node
- `GET /schema`: Get the schema of the node
- `POST /update`: Update the node with new observations
- `POST /query/{variable_name}`: Query the node for a specific variable

#### Node-Specific Endpoints

- **Node A (Real Estate Finance)**:
  - `POST /query/roi`: Query for expected ROI
    ```json
    {
      "location": "Miami",
      "ipcc_scenario": "SSP2-4.5"
    }
    ```

- **Node B (Climate Risk)**:
  - `POST /query/flood-probability`: Query for flood probability
    ```json
    {
      "location": "Miami",
      "ipcc_scenario": "SSP2-4.5",
      "rationale": false
    }
    ```

- **Node C (Actuarial Data)**:
  - `POST /query/historical-data`: Query for historical flood data
    ```json
    {
      "location": "Miami"
    }
    ```
  - `POST /add-data`: Add new data
    ```json
    {
      "location": "Miami",
      "value": 0.4
    }
    ```

### Example Workflow

The web client demo demonstrates the following workflow:

1. Get information about all three nodes
2. Set the location (Miami) and IPCC scenario (SSP2-4.5)
3. Get Node B's schema
4. Query Node B for flood probability
5. Query Node A for expected ROI based on the flood probability
6. Add new actuarial data to Node C
7. Query Node C for the updated data
8. Update Node B with the new data
9. Query Node B again for the updated flood probability
10. Query Node B with rationale to understand the changes
11. Query Node A for the updated ROI

This generates the following output:

```
================================================================================
========================= Gaia Network Web Client Demo =========================
================================================================================

Getting node information...
Node A (Real Estate Finance): real_estate_finance_model
Node B (Climate Risk): climate_risk_model
Node C (Actuarial Data): actuarial_data_service

================================================================================
==================== Step 1: Set location and IPCC scenario ====================
================================================================================

Location: Miami
IPCC Scenario: SSP2-4.5

================================================================================
========================= Step 2: Get Node B's schema ==========================
================================================================================


Latent variables:
  - sea_level_rise: Projected sea level rise in meters
  - storm_intensity: Projected storm intensity

Observable variables:
  - flood_probability: Probability of flooding in the next 50 years
  - historical_flood_data: Historical flood data

Covariates:
  - location: City location
  - ipcc_scenario: IPCC climate scenario

================================================================================
================== Step 3: Query Node B for flood probability ==================
================================================================================

Querying flood probability for Miami under SSP2-4.5 scenario...
Raw flood response: {'query_id': '1b821515-bdef-4f94-a2a3-54b0f8824243', 'response_type': 'posterior', 'content': {'distribution': {'name': 'flood_probability', 'distribution': {'type': 'beta', 'parameters': {'alpha': 12.8, 'beta': 7.199999999999999}}, 'metadata': {'location': 'Miami', 'ipcc_scenario': 'SSP2-4.5'}}}, 'timestamp': '2025-02-27T19:00:12.722367', 'metadata': {}}
Response type: posterior

Flood probability distribution:
  - Name: flood_probability
  - Type: beta
  - Parameters: {'alpha': 12.8, 'beta': 7.199999999999999}
  - Metadata: {'location': 'Miami', 'ipcc_scenario': 'SSP2-4.5'}

Expected flood probability: 0.6400

================================================================================
========================= Step 4: Query Node A for ROI =========================
================================================================================

Calculating expected ROI for the real estate project...

ROI distribution:
  - Name: expected_roi
  - Type: normal
  - Parameters: {'mean': -0.17, 'std': 0.03}
  - Metadata: {'location': 'Miami', 'ipcc_scenario': 'SSP2-4.5', 'flood_probability': 0.64}

Expected ROI: -17.00%

================================================================================
======================== Step 5: Add new data to Node C ========================
================================================================================

Adding new actuarial data to Node C...
Status: success

================================================================================
==================== Step 6: Query Node C for updated data =====================
================================================================================

Querying Node C for updated actuarial data...

Actuarial data distribution:
  - Name: historical_flood_data
  - Type: normal
  - Parameters: {'mean': 0.3, 'std': 0.1}
  - Metadata: {'location': 'Miami'}

================================================================================
===================== Step 7: Update Node B with new data ======================
================================================================================

Updating Node B with new actuarial data...
Response type: update
Update status: success

================================================================================
=========== Step 8: Query Node B again for updated flood probability ===========
================================================================================

Querying updated flood probability for Miami under SSP2-4.5 scenario...

Updated flood probability distribution:
  - Name: flood_probability
  - Type: beta
  - Parameters: {'alpha': 13.200000000000001, 'beta': 6.799999999999999}
  - Metadata: {'location': 'Miami', 'ipcc_scenario': 'SSP2-4.5'}

Updated expected flood probability: 0.6600
Change in flood probability: 0.0200

================================================================================
===================== Step 9: Query Node B with rationale ======================
================================================================================

Querying flood probability with rationale for Miami under SSP2-4.5 scenario...

Flood probability distribution with rationale:
  - Name: flood_probability
  - Type: beta
  - Parameters: {'alpha': 13.200000000000001, 'beta': 6.799999999999999}

================================================================================
==================== Step 10: Query Node A for updated ROI =====================
================================================================================

Recalculating expected ROI for the real estate project...

Updated ROI distribution:
  - Name: expected_roi
  - Type: normal
  - Parameters: {'mean': -0.18000000000000002, 'std': 0.03}
  - Metadata: {'location': 'Miami', 'ipcc_scenario': 'SSP2-4.5', 'flood_probability': 0.66}

Updated expected ROI: -18.00%
Change in ROI: -1.00%

================================================================================
================================ Demo Complete =================================
================================================================================
```

## Implementation Notes

- The prototype uses a simplified model for demonstration purposes
- The nodes communicate via direct method calls in the standard demo and via HTTP/[ASGI](https://asgi.readthedocs.io/en/latest/) in the web demo
- The demo uses a simple in-memory registry for node discovery, which would be replaced with a distributed registry in a real implementation
- The web services implementation uses an object-oriented approach with a base `NodeHandler` class and specialized handler classes for each node type
- Each node handler is co-located with its node implementation for better code organization
- The `model_nodes.py` module provides factory functions to create nodes and their handlers

## Future Improvements

This is a simplified prototype. A full implementation will include:

- A distributed registry for node discovery
- Use of a higher-level protocol rather than low-level HTTP/ASGI. In particular, we see value in adopting [Model Context Protocol](https://github.com/modelcontextprotocol/) and only haven't done so because of the Python MCP SDK's current lack of support for running multiple nodes on the same host
- Authentication and authorization for node access
- More sophisticated probabilistic models and inference algorithms
- A web-based UI for interacting with the network
- Support for more complex queries and updates
- Distributed state management
- Fault tolerance and recovery mechanisms
- Strategies for reproduction and verification of state posteriors
- A more complete implementation of the Gaia Ontology
- Free energy-based credit assignment for information flow
- Free energy-based model management
- Free energy-based incentive mechanisms for sharing information
- Proper distributed networking capabilities

## Troubleshooting

If you encounter any issues:

1. Make sure you have activated the virtual environment
2. Verify that all dependencies are installed
3. Check that you're running the demo from the project root directory

## Requirements

- Python 3.7+
- NumPy
- SciPy

## Development Notes

- The project uses dataclasses, which require careful ordering of parameters (required parameters must come before parameters with default values)
- The demo uses a simple in-memory registry for node discovery, which would be replaced with a distributed registry in a real implementation
