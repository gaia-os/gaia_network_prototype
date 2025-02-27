"""
Actuarial Data Node for the Gaia Network demo.

This module implements Node C: Serves actuarial data relevant for climate risk.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from gaia_network.node import Node
from gaia_network.schema import Schema, Variable
from gaia_network.state import State, Observation, StateCheckpoint
from gaia_network.query import Query, QueryResponse
from gaia_network.distribution import Distribution, NormalDistribution, BetaDistribution, MarginalDistribution
from gaia_network.registry import register_node
from demo.node_handler import NodeHandler


class ActuarialDataNode(Node):
    """
    Node C: Serves actuarial data relevant for climate risk.
    """
    def __init__(self):
        schema = Schema()
        
        # Define observable variables
        schema.add_observable(Variable(
            name="historical_flood_data",
            description="Historical flood data",
            type="continuous"
        ))
        
        schema.add_observable(Variable(
            name="insurance_claims",
            description="Insurance claims data",
            type="continuous"
        ))
        
        # Define covariates
        schema.add_covariate(Variable(
            name="location",
            description="City location",
            type="categorical"
        ))
        
        schema.add_covariate(Variable(
            name="year",
            description="Year of data",
            type="continuous"
        ))
        
        super().__init__(
            name="Actuarial Data Service",
            description="Serves actuarial data relevant for climate risk",
            schema=schema,
            id="actuarial_data_service"
        )
    
    def _handle_posterior_query(self, query: Query) -> QueryResponse:
        """Handle a posterior query for the actuarial data service."""
        variable_name = query.parameters.get("variable_name")
        covariates = query.parameters.get("covariates", {})
        
        if variable_name == "historical_flood_data":
            location = covariates.get("location", "Miami")
            
            # In a real implementation, this would query a database of historical data
            # For this demo, we'll just return a simple distribution
            
            # Different mean and std for different locations
            if location == "Miami":
                mean, std = 0.3, 0.1
            elif location == "New Orleans":
                mean, std = 0.4, 0.12
            elif location == "New York":
                mean, std = 0.2, 0.08
            else:
                mean, std = 0.25, 0.1
            
            distribution = NormalDistribution(mean=mean, std=std)
            
            return QueryResponse(
                query_id=query.id,
                response_type="posterior",
                content={
                    "distribution": MarginalDistribution(
                        name="historical_flood_data",
                        distribution=distribution,
                        metadata={"location": location}
                    ).to_dict()
                }
            )
        
        return QueryResponse(
            query_id=query.id,
            response_type="error",
            content={"error": f"Unsupported variable: {variable_name}"}
        )
    
    def add_new_data(self, location: str, value: float) -> None:
        """
        Add new actuarial data.
        
        This method simulates the arrival of new data that can be used to update
        the climate risk model.
        """
        observation = Observation(
            variable_name="historical_flood_data",
            value=value,
            metadata={"location": location}
        )
        
        self.state.add_observation(observation)


class ActuarialDataHandler(NodeHandler):
    """Handler for the Actuarial Data node (Node C)."""
    
    def query(self, variable_name, covariates):
        """Query Node C based on variable_name and covariates."""
        if variable_name == "historical-data":
            return self._query_historical_data(covariates)
        return super().query(variable_name, covariates)
    
    def _query_historical_data(self, covariates):
        """Query Node C for historical flood data."""
        location = covariates.get("location", "Miami")
        
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="historical_flood_data",
            covariates={"location": location}
        )
        
        return response.to_dict()
    
    def add_data(self, data):
        """Add new data to Node C."""
        location = data.get("location", "Miami")
        value = data.get("value", 0.4)
        self.node.add_new_data(location=location, value=value)
        return {
            "response_type": "update",
            "content": {
                "status": "success"
            }
        }
