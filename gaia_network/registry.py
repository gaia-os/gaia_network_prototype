"""
Registry module for the Gaia Network.

This module provides a simple registry for nodes in the network.
In a real implementation, this would be replaced with a distributed registry.
"""

from typing import Dict, Optional
from .node import Node

# A simple in-memory registry of nodes
_nodes: Dict[str, Node] = {}

# A mapping of node IDs to MCP URLs
_mcp_urls: Dict[str, str] = {}

def register_node(node: Node, mcp_url: Optional[str] = None) -> None:
    """
    Register a node in the registry.
    
    Args:
        node: The node to register
        mcp_url: Optional MCP URL for the node
    """
    _nodes[node.id] = node
    if mcp_url:
        _mcp_urls[node.id] = mcp_url

def get_node(node_id: str) -> Optional[Node]:
    """
    Get a node from the registry by ID.
    
    Args:
        node_id: The ID of the node to get
        
    Returns:
        The node, or None if not found
    """
    return _nodes.get(node_id)

def get_mcp_url(node_id: str) -> Optional[str]:
    """
    Get the MCP URL for a node.
    
    Args:
        node_id: The ID of the node to get the MCP URL for
        
    Returns:
        The MCP URL, or None if not found
    """
    return _mcp_urls.get(node_id)

def get_all_nodes() -> Dict[str, Node]:
    """
    Get all nodes in the registry.
    
    Returns:
        A copy of the node registry
    """
    return _nodes.copy()

def get_all_mcp_urls() -> Dict[str, str]:
    """
    Get all MCP URLs in the registry.
    
    Returns:
        A copy of the MCP URL registry
    """
    return _mcp_urls.copy()

def clear_registry() -> None:
    """Clear the registry."""
    _nodes.clear()
    _mcp_urls.clear()
