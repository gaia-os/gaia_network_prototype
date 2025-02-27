"""
Model nodes for the Gaia Network demo.

This module imports the three nodes described in the demo:
- Node A: Models project finance for a real estate development in Miami
- Node B: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- Node C: Serves actuarial data relevant for climate risk
"""

from gaia_network.registry import register_node, get_node

# Import the node classes from their respective files
from demo.real_estate_finance_node import RealEstateFinanceNode, RealEstateFinanceHandler
from demo.climate_risk_node import ClimateRiskNode, ClimateRiskHandler
from demo.actuarial_data_node import ActuarialDataNode, ActuarialDataHandler

# Global variables to store node instances
_node_a = None
_node_b = None
_node_c = None


def create_demo_nodes():
    """Create and register the demo nodes."""
    global _node_a, _node_b, _node_c
    
    _node_a = RealEstateFinanceNode()
    _node_b = ClimateRiskNode()
    _node_c = ActuarialDataNode()
    
    register_node(_node_a)
    register_node(_node_b)
    register_node(_node_c)
    
    return _node_a, _node_b, _node_c


def get_node_by_id(node_id):
    """Get a node by its ID."""
    return get_node(node_id)


def create_demo_node_handlers():
    """Create handlers for the demo nodes."""
    # Ensure nodes are created
    if _node_a is None or _node_b is None or _node_c is None:
        create_demo_nodes()
    
    # Create handler instances for each node
    node_a_handler = RealEstateFinanceHandler(_node_a)
    node_b_handler = ClimateRiskHandler(_node_b)
    node_c_handler = ActuarialDataHandler(_node_c)
    
    # Map node IDs to handlers
    node_handlers = {
        _node_a.id: node_a_handler,
        _node_b.id: node_b_handler,
        _node_c.id: node_c_handler
    }
    
    return node_handlers
