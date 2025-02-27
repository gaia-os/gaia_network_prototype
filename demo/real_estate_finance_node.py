"""
Real Estate Finance Node for the Gaia Network demo.

This module implements Node A: Models project finance for a real estate development in Miami.
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


class RealEstateFinanceNode(Node):
    """
    Node A: Models project finance for a real estate development in Miami.
    """
    def __init__(self):
        schema = Schema()
        
        # Define latent variables
        schema.add_latent(Variable(
            name="expected_roi",
            description="Expected return on investment for the project",
            type="continuous",
            domain={"min": -1.0, "max": 1.0}
        ))
        
        schema.add_latent(Variable(
            name="risk_adjusted_roi",
            description="Risk-adjusted return on investment",
            type="continuous",
            domain={"min": -1.0, "max": 1.0}
        ))
        
        # Define observable variables
        schema.add_observable(Variable(
            name="construction_cost",
            description="Total construction cost in USD",
            type="continuous",
            domain={"min": 0.0}
        ))
        
        schema.add_observable(Variable(
            name="expected_revenue",
            description="Expected revenue in USD",
            type="continuous",
            domain={"min": 0.0}
        ))
        
        schema.add_observable(Variable(
            name="flood_probability",
            description="Probability of flooding during the project lifetime",
            type="continuous",
            domain={"min": 0.0, "max": 1.0}
        ))
        
        # Define covariates
        schema.add_covariate(Variable(
            name="location",
            description="Geographic location of the project",
            type="categorical"
        ))
        
        schema.add_covariate(Variable(
            name="ipcc_scenario",
            description="IPCC climate scenario",
            type="categorical",
            domain={"categories": ["SSP1-1.9", "SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]}
        ))
        
        super().__init__(
            name="Real Estate Finance Model",
            description="Models project finance for a real estate development",
            schema=schema,
            id="real_estate_finance_model"
        )
    
    def _handle_posterior_query(self, query: Query) -> QueryResponse:
        """Handle a posterior query for the real estate finance model."""
        variable_name = query.parameters.get("variable_name")
        covariates = query.parameters.get("covariates", {})
        
        if variable_name == "expected_roi":
            # Calculate expected ROI based on flood probability
            # In a real implementation, this would use the actual model
            
            # First, we need to get the flood probability from Node B
            location = covariates.get("location", "Miami")
            ipcc_scenario = covariates.get("ipcc_scenario", "SSP2-4.5")
            
            # Query Node B for flood probability
            flood_response = self.query_posterior(
                target_node_id="climate_risk_model",
                variable_name="flood_probability",
                covariates={
                    "location": location,
                    "ipcc_scenario": ipcc_scenario
                }
            )
            
            if flood_response.response_type == "error":
                return QueryResponse(
                    query_id=query.id,
                    response_type="error",
                    content={"error": f"Failed to get flood probability: {flood_response.content.get('error')}"}
                )
            
            # Extract the flood probability distribution
            flood_dist_data = flood_response.content.get("distribution")
            if not flood_dist_data or not isinstance(flood_dist_data, dict):
                return QueryResponse(
                    query_id=query.id,
                    response_type="error",
                    content={"error": "No distribution data found in flood probability response"}
                )
            
            flood_marginal = MarginalDistribution.from_dict(flood_dist_data)
            
            # Calculate ROI based on flood probability
            # This is a simplified model for the demo
            flood_prob = flood_marginal.distribution.parameters["alpha"] / (
                flood_marginal.distribution.parameters["alpha"] + 
                flood_marginal.distribution.parameters["beta"]
            )
            
            # Higher flood probability means lower ROI
            base_roi = 0.15  # 15% base ROI
            roi_reduction = flood_prob * 0.5  # Up to 50% reduction based on flood risk
            expected_roi = base_roi - roi_reduction
            
            # Add some uncertainty
            roi_std = 0.03
            
            distribution = NormalDistribution(mean=expected_roi, std=roi_std)
            
            return QueryResponse(
                query_id=query.id,
                response_type="posterior",
                content={
                    "distribution": MarginalDistribution(
                        name="expected_roi",
                        distribution=distribution,
                        metadata={
                            "location": location,
                            "ipcc_scenario": ipcc_scenario,
                            "flood_probability": flood_prob
                        }
                    ).to_dict()
                }
            )
        
        return QueryResponse(
            query_id=query.id,
            response_type="error",
            content={"error": f"Unsupported variable: {variable_name}"}
        )


class RealEstateFinanceHandler(NodeHandler):
    """Handler for the Real Estate Finance node (Node A)."""
    
    def query(self, variable_name, covariates):
        """Query Node A based on variable_name and covariates."""
        if variable_name == "roi":
            return self._query_roi(covariates)
        return super().query(variable_name, covariates)
    
    def _query_roi(self, covariates):
        """Query Node A for expected ROI."""
        location = covariates.get("location", "Miami")
        ipcc_scenario = covariates.get("ipcc_scenario", "SSP2-4.5")
        
        # Import here to avoid circular imports
        from demo.model_nodes import get_node_by_id
        
        # Get Node B
        node_b = get_node_by_id("climate_risk_model")
        
        # Query Node B for flood probability
        flood_response = self.node.query_posterior(
            target_node_id=node_b.id,
            variable_name="flood_probability",
            covariates={
                "location": location,
                "ipcc_scenario": ipcc_scenario
            }
        )
        
        # Extract the flood probability
        if flood_response.response_type == "posterior":
            distribution_data = flood_response.content["distribution"]
            alpha = distribution_data['distribution']['parameters']['alpha']
            beta = distribution_data['distribution']['parameters']['beta']
            flood_probability = alpha / (alpha + beta)
        else:
            raise Exception("Failed to get flood probability")
        
        # Calculate expected ROI based on flood probability
        roi_response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="expected_roi",
            covariates={
                "location": location,
                "ipcc_scenario": ipcc_scenario,
                "flood_probability": flood_probability
            }
        )
        
        return roi_response.to_dict()
