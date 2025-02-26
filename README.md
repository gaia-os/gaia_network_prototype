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
│   ├── model_nodes.py       # Specific node implementations
│   └── run_demo.py          # Demo script
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

## Running the Demo

To run the demo:

```bash
# Make sure you're in the project root directory
cd /Users/rkauf/CascadeProjects/gaia_network_prototype

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the demo
python -m demo.run_demo
```

The demo will:
1. Create three nodes (A, B, and C) representing different models
2. Demonstrate how Node A queries Node B for climate risk data
3. Show how Node C provides updated actuarial data to Node B
4. Demonstrate how Node B updates its internal state based on new data
5. Show how Node A gets updated information from Node B, including the rationale for the changes

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

## Future Improvements

This is a simplified prototype. A full implementation would include:

1. Proper distributed networking capabilities
2. More sophisticated probabilistic models and inference algorithms
3. A distributed registry for node discovery
4. Authentication and authorization mechanisms
5. A more complete implementation of the Gaia Ontology
6. Credit assignment for information flow
7. Incentive mechanisms for sharing information
