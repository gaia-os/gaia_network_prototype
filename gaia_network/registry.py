"""
Registry module for the Gaia Network.

This module provides a simple registry for nodes in the network.
In a real implementation, this would be replaced with a distributed registry.
"""

from typing import Dict, Optional
from .node import Node

# A simple in-memory registry of nodes
_nodes: Dict[str, Node] = {}

def register_node(node: Node) -> None:
    """Register a node in the registry."""
    _nodes[node.id] = node

def get_node(node_id: str) -> Optional[Node]:
    """Get a node from the registry by ID."""
    return _nodes.get(node_id)

def get_all_nodes() -> Dict[str, Node]:
    """Get all nodes in the registry."""
    return _nodes.copy()

def clear_registry() -> None:
    """Clear the registry."""
    _nodes.clear()
