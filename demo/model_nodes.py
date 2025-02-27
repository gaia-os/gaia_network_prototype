"""
Model nodes for the Gaia Network demo.

This module imports the three nodes described in the demo:
- Node A: Models project finance for a real estate development in Miami
- Node B: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- Node C: Serves actuarial data relevant for climate risk
"""

from gaia_network.registry import register_node

# Import the node classes from their respective files
from demo.real_estate_finance_node import RealEstateFinanceNode
from demo.climate_risk_node import ClimateRiskNode
from demo.actuarial_data_node import ActuarialDataNode


def create_demo_nodes():
    """Create and register the demo nodes."""
    node_a = RealEstateFinanceNode()
    node_b = ClimateRiskNode()
    node_c = ActuarialDataNode()
    
    register_node(node_a)
    register_node(node_b)
    register_node(node_c)
    
    return node_a, node_b, node_c
