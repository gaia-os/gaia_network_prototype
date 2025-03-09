# Gaia Network Prototype

This is a simple prototype implementation of the Gaia Network as described in [Gaia Network: An Illustrated Primer](https://engineeringideas.substack.com/p/gaia-network-an-illustrated-primer).

## Overview

The Gaia Network is a distributed network for probabilistic models, allowing nodes to share information and update their beliefs based on observations from other nodes. This prototype implements a simplified version of the network with four nodes:

- **Node A**: Models project finance for a real estate development in Miami
- **Node B**: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- **Node C**: Serves actuarial data relevant for climate risk
- **Node D**: Models a resilience bond that rewards project developers for achieving resilience outcomes

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
│   ├── resilience_bond_node.py # Node D implementation and handler
│   ├── run_demo.py          # Demo script
│   ├── run_web_demo.py      # Web services demo
│   └── web_client_demo.py   # Web client demo
├── venv/                    # Virtual environment (not in version control)
└── requirements.txt         # Project dependencies
```

## Core Concepts

### Technical Concepts

- **Node**: A probabilistic model that can be queried and updated
- **Schema**: Defines the state space of a node, including latent variables, observable variables, and covariates
- **Node Schema**: Structure of each node's state space:
  - Node A: ROI variables, adaptation choices, bond effects
  - Node B: Climate scenarios, flood probabilities
  - Node C: Historical flood and resilience data
  - Node D: Bond prices, payoff functions, resilience thresholds
- **State**: The current state of a node, including the values of latent variables and model parameters
- **State Updates**: How nodes share and update information:
  - Node A: Updates ROI based on flood risk and bond effects
  - Node B: Updates flood probabilities based on actuarial data
  - Node C: Updates historical data and notifies dependent nodes
  - Node D: Updates bond terms based on market and resilience data
- **Query**: A request for information from one node to another
- **Distribution**: A probability distribution representing uncertainty
- **MarginalDistribution**: A probability distribution for a single variable:
  - Used for bond payoffs and prices
  - Includes metadata about calculation inputs
  - Can be combined with outcome probabilities

### Domain Concepts

- **ROI (Return on Investment)**: The financial return relative to the investment cost:
  - Base ROI: Expected return without considering climate risks
  - Climate-Adjusted ROI: Base ROI - (Flood Probability × Flood Impact)
  - Bond-Adjusted ROI: Climate-Adjusted ROI - Bond Price + Expected Bond Payoff
  - Final ROI: Actual return after project completion and bond settlement
- **Flood Impact**: The expected financial loss if a flood occurs, expressed as a percentage of the investment:
  - Higher for BAU projects due to lack of protection
  - Lower for Adaptation projects due to resilience measures
  - Key factor in determining the value of adaptation
- **Flood Probability**: The likelihood of a flood occurring during the investment period:
  - Higher for locations with greater climate risk
  - Reduced by successful adaptation measures
  - Updated based on new actuarial data
- **IPCC Scenario**: A standardized scenario from the Intergovernmental Panel on Climate Change describing possible climate futures
- **Actuarial Data**: Historical data about flood events and project resilience:
  - Used by Node B to update flood probability estimates
  - Used by Node D to calibrate bond prices and payoffs
  - Helps validate the effectiveness of adaptation measures
- **Resilience Bond**: A financial instrument that rewards project developers for achieving resilience outcomes
- **Resilience Outcomes**: Quantitative measures of a project's resilience to climate risks:
  - Low (0.3): Basic adaptation measures implemented
  - Medium (0.5): Significant adaptation measures implemented
  - High (0.7): Comprehensive adaptation measures implemented
- **Resilience Probabilities**: The likelihood of achieving each resilience outcome:
  - Based on historical data from similar projects
  - Influenced by location and adaptation strategy
  - Used to calculate expected bond payoff
- **Probability Assignment**: How Node A determines outcome probabilities:
  - Initial Distribution: Equal weights to all outcomes
  - Location Effect: Higher probabilities for good outcomes in low-risk areas
  - Strategy Effect: Adaptation strategy shifts distribution toward better outcomes
  - Historical Effect: Past performance in similar projects updates probabilities
- **Bond Parameters**: The key variables that define the resilience bond:
  - Base Price: Fixed upfront cost to purchase the bond
  - Payoff Multipliers: How much payoff increases with better outcomes
  - Resilience Thresholds: The outcome levels (0.3, 0.5, 0.7) and their meanings
  - Default Parameters: Calibrated to make Adaptation ROI > BAU ROI when resilience is achieved
- **Bond Payoff**: The amount paid to the project developer based on achieved resilience outcomes:
  - Higher payoff for better resilience outcomes
  - Payoff distributions modeled by Node D
  - Expected payoff = Weighted average of payoffs by outcome probabilities
- **Payoff Distribution**: A dictionary mapping resilience outcomes to payoff distributions:
  - Keys are resilience outcomes (0.3, 0.5, 0.7)
  - Values are MarginalDistributions for payoffs
  - Used to calculate expected returns
- **Adaptation Strategy**: The approach to climate risk:
  - BAU (Business-as-Usual): Standard development without specific climate adaptation
  - Adaptation: Development incorporating climate resilience measures
- **Adaptation Economics**: The financial rationale for climate adaptation:
  - BAU has lower upfront costs but higher flood risk
  - Adaptation has higher upfront costs but lower flood risk
  - Resilience bond helps bridge the gap by providing additional returns for adaptation
  - When bond payoff > bond price, Adaptation ROI can exceed BAU ROI
- **Project Development**: The timeline of a real estate project with resilience bond:
  - Initial Phase: Purchase bond and implement adaptation measures
  - Development Phase: Construction and resilience measure implementation
  - Assessment Phase: Evaluate achieved resilience outcomes
  - Final Phase: Receive bond payoff based on assessment
- **Resilience Feedback Loop**: The self-reinforcing cycle created by the bond:
  - Better adaptation measures → Higher resilience outcomes
  - Higher resilience outcomes → Larger bond payoff
  - Larger bond payoff → Higher ROI
  - Higher ROI → More incentive for adaptation

### Node Interactions and Data Flow

- **Node Interactions**: How the nodes work together to calculate ROI:
  - Node A (Finance) queries Node B (Climate) for flood probability
  - Node B queries Node C (Actuarial) for historical data
  - Node A queries Node D (Bond) for price and payoff distributions
  - Node A combines all inputs to calculate final ROI  
- **Data Flow**: How information moves through the network:
  - Actuarial Data → Climate Risk → Flood Probability
  - Location + IPCC Scenario → Risk Profile → Bond Terms
  - Adaptation Strategy + Resilience → Bond Payoff → Final ROI
  - Historical Data → Outcome Probabilities → Expected Returns

## ROI Calculation with Resilience Bond

The demo compares two investment strategies:

1. **Business-as-Usual (BAU)**: Standard real estate development without climate adaptation measures
   - ROI = Base ROI - (Flood Probability × Flood Impact)

2. **Adaptation with Bond**: Development with climate adaptation measures and resilience bond
   - Initial Cost: Purchase bond at Bond Price
   - Potential Benefit: Bond Payoff based on achieved resilience
   - ROI = Base ROI - (Flood Probability × Flood Impact) - Bond Price + Bond Payoff

The resilience bond creates a financial incentive for climate adaptation by providing additional returns when the project achieves higher resilience outcomes. Node A calculates the expected bond payoff by:
1. Getting possible resilience outcomes and their probabilities
2. Querying Node D for payoff distributions for each outcome
3. Computing the weighted average based on outcome probabilities

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
11. Compare ROI for Business-as-Usual (BAU) vs Adaptation strategies
12. Consider resilience bond effects for Adaptation strategy
13. Simulate time passing and project development, including bond payoff calculation

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
- **Node D (Resilience Bond)**: http://127.0.0.1:8014

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
  - `POST /query/expected_roi`: Query for expected ROI
    ```json
    {
      "location": "Miami",
      "ipcc_scenario": "SSP2-4.5",
      "adaptation_strategy": "BAU",
      "include_bond": "no"
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

- **Node D (Resilience Bond)**:
  - `POST /query/bond_price`: Query for bond price
    ```json
    {}
    ```
  - `POST /query/bond_payoff`: Query for bond payoff
    ```json
    {
      "location": "Miami",
      "resilience_outcomes": [0.3, 0.5, 0.7]
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
11. Query Node A for BAU and Adaptation ROI
12. Query Node A for Adaptation ROI with bond effects
13. Simulate project development and calculate final ROI with actual bond payoff

This generates the following output:

```
================================================================================
========================= Gaia Network Web Client Demo =========================
================================================================================

Getting node information...
Node A (Real Estate Finance): real_estate_finance_model
Node B (Climate Risk): climate_risk_model
Node C (Actuarial Data): actuarial_data_service
Node D (Resilience Bond): resilience_bond_issuer

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
Raw flood response: {'query_id': '7c89dc98-4a1a-4aab-8e89-5a5dbce594c2', 'response_type': 'error', 'content': {'error': 'Unsupported variable: flood-probability'}, 'timestamp': '2025-03-09T17:31:48.584371', 'metadata': {}}
Response type: error

================================================================================
========================= Step 4: Query Node A for ROI =========================
================================================================================

Calculating expected ROI for the real estate project...

ROI distribution:
  - Name: expected_roi
  - Type: normal
  - Parameters: {'mean': 0.0032000000000000084, 'std': 0.03}
  - Metadata: {'location': 'Miami', 'ipcc_scenario': 'SSP2-4.5', 'adaptation_strategy': 'BAU', 'include_bond': 'no', 'flood_probability': 0.64}

Expected ROI: 0.32%

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

================================================================================
===================== Step 9: Query Node B with rationale ======================
================================================================================

Querying flood probability with rationale for Miami under SSP2-4.5 scenario...

================================================================================
==================== Step 10: Query Node A for updated ROI =====================
================================================================================

Recalculating expected ROI for the real estate project...

Updated ROI distribution:
  - Name: expected_roi
  - Type: normal
  - Parameters: {'mean': 0.0007999999999999952, 'std': 0.03}
  - Metadata: {'location': 'Miami', 'ipcc_scenario': 'SSP2-4.5', 'adaptation_strategy': 'BAU', 'include_bond': 'no', 'flood_probability': 0.66}

Updated expected ROI: 0.08%
Change in ROI: -0.24%

================================================================================
========== Step 11: Compare ROI for BAU vs Adaptation (without bond) ===========
================================================================================


Calculating ROI for Business-as-Usual (BAU) strategy...
BAU Expected ROI: 0.08%

Calculating ROI for Climate Adaptation strategy...
Adaptation Expected ROI: -4.90%

Initial ROI difference (Adaptation - BAU): -4.98%

================================================================================
====== Step 12: Consider resilience bond effects for Adaptation strategy =======
================================================================================


Calculating ROI for Adaptation strategy with resilience bond...

Adaptation + Bond Expected ROI: 0.20%
Bond price: 3.00%
Expected bond payoff: 8.10%

Resilient outcomes and probabilities:
  - Outcome 0.60: 10% probability
  - Outcome 0.70: 60% probability
  - Outcome 0.90: 30% probability

ROI improvement from bond: 5.10%
Final ROI difference (Adaptation + Bond - BAU): 0.12%

================================================================================
============ Step 13: Simulate time passing and project development ============
================================================================================


Time passes, the project is developed...

Node C receives new actuarial data...
New actuarial data added to Node C

Node D queries Node C for actuarial data to determine actual resilience outcome...
Actual resilience outcome achieved: 0.70

Calculating final project ROI with actual bond payoff...
Final project ROI: 1.10%
Improvement over initial BAU ROI: 1.02%

================================================================================
================================ Demo Complete =================================
================================================================================
```

## Implementation Notes

- The prototype uses simple hypothetical models for demonstration purposes, rather than actual probabilistic models and inference. We will implement those in an upcoming commit.
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
