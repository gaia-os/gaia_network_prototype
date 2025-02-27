# Gaia Network Web Demo

This directory contains a web service implementation of the Gaia Network prototype. The web services are built using Starlette, an ASGI framework for Python.

## Overview

The web demo exposes the three Gaia Network nodes as REST API endpoints:

- **Node A**: Real Estate Finance Model
- **Node B**: Climate Risk Model
- **Node C**: Actuarial Data Service

Each node provides endpoints for querying its schema, getting information about the node, and performing node-specific operations.

## Setup

1. Make sure you have installed the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the web server:
   ```bash
   python -m demo.run_web_demo
   ```

3. The server will start at `http://127.0.0.1:8000`

## Running the Web Client Demo

A web client demo script is provided to demonstrate how to interact with the web services:

```bash
python -m demo.web_client_demo
```

This script follows a similar flow to the original demo but uses HTTP requests instead of direct method calls.

## API Endpoints

### Node A (Real Estate Finance Model)

- `GET /node-a/info`: Get information about Node A
- `GET /node-a/schema`: Get the schema of Node A
- `POST /node-a/query/roi`: Query Node A for expected ROI
  ```json
  {
    "location": "Miami",
    "ipcc_scenario": "SSP2-4.5"
  }
  ```

### Node B (Climate Risk Model)

- `GET /node-b/info`: Get information about Node B
- `GET /node-b/schema`: Get the schema of Node B
- `POST /node-b/query/flood-probability`: Query Node B for flood probability
  ```json
  {
    "location": "Miami",
    "ipcc_scenario": "SSP2-4.5",
    "rationale": false
  }
  ```
- `POST /node-b/update`: Update Node B with new observations
  ```json
  {
    "observations": [
      {
        "variable_name": "historical_flood_data",
        "value": 0.4,
        "metadata": {"location": "Miami"}
      }
    ]
  }
  ```

### Node C (Actuarial Data Service)

- `GET /node-c/info`: Get information about Node C
- `GET /node-c/schema`: Get the schema of Node C
- `POST /node-c/query/historical-data`: Query Node C for historical flood data
  ```json
  {
    "location": "Miami"
  }
  ```
- `POST /node-c/add-data`: Add new data to Node C
  ```json
  {
    "location": "Miami",
    "value": 0.4
  }
  ```

## Example Workflow

1. Query Node B for flood probability in Miami under SSP2-4.5 scenario
2. Query Node A for expected ROI based on the flood probability
3. Add new actuarial data to Node C
4. Update Node B with the new data
5. Query Node B again for the updated flood probability
6. Query Node A for the updated ROI

This workflow demonstrates how the nodes can share information and update their beliefs based on new observations.

## Notes

- The web services use in-memory nodes, so the state will be reset when the server is restarted.
- CORS is enabled for all origins to facilitate development and testing.
- Error handling is implemented to provide meaningful error messages.
- The API follows RESTful principles where appropriate.

## Extending the Demo

To extend the demo, you can:

1. Add more endpoints to expose additional functionality
2. Implement authentication and authorization
3. Add persistence to store node states between server restarts
4. Create a web frontend to interact with the API
5. Add more nodes to the network
