"""
Node Handler for the Gaia Network web services.

This module defines the base NodeHandler class that provides common functionality
for handling web requests to Gaia Network nodes.
"""

class NodeHandler:
    """Base class for handling node operations."""
    
    def __init__(self, node):
        self.node = node
    
    def get_info(self):
        """Return information about the node."""
        return {
            "id": self.node.id,
            "name": self.node.name,
            "description": self.node.description
        }
    
    def get_schema(self):
        """Return the schema of the node."""
        return self.node.schema.to_dict()
    
    def query(self, variable_name, covariates):
        """Query the node based on variable_name and covariates."""
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name=variable_name,
            covariates=covariates
        )
        return response.to_dict()
    
    def update(self, data):
        """Update the node with new observations."""
        observations = data.get("observations", [])
        self.node.send_update(
            target_node_id=self.node.id,
            observations=observations
        )
        return {
            "response_type": "update",
            "content": {
                "status": "success"
            }
        }
    
    def add_data(self, data):
        """Add new data to the node."""
        return {
            "error": f"Adding data to node {self.node.id} is not supported"
        }
