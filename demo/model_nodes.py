"""
Model nodes for the Gaia Network demo.

This module implements the three nodes described in the demo:
- Node A: Models project finance for a real estate development in Miami
- Node B: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- Node C: Serves actuarial data relevant for climate risk
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
            schema=schema
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


class ClimateRiskNode(Node):
    """
    Node B: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios.
    """
    def __init__(self):
        schema = Schema()
        
        # Define latent variables
        schema.add_latent(Variable(
            name="sea_level_rise",
            description="Projected sea level rise in meters",
            type="continuous",
            domain={"min": 0.0}
        ))
        
        schema.add_latent(Variable(
            name="storm_intensity",
            description="Projected storm intensity",
            type="continuous",
            domain={"min": 0.0}
        ))
        
        # Define observable variables
        schema.add_observable(Variable(
            name="flood_probability",
            description="Probability of flooding in the next 50 years",
            type="continuous",
            domain={"min": 0.0, "max": 1.0}
        ))
        
        schema.add_observable(Variable(
            name="historical_flood_data",
            description="Historical flood data",
            type="continuous"
        ))
        
        # Define covariates
        schema.add_covariate(Variable(
            name="location",
            description="City location",
            type="categorical"
        ))
        
        schema.add_covariate(Variable(
            name="ipcc_scenario",
            description="IPCC climate scenario",
            type="categorical",
            domain={"categories": ["SSP1-1.9", "SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]}
        ))
        
        # Initialize state with some default values
        state = State()
        state.create_checkpoint(
            latent_values={
                "sea_level_rise": {
                    "SSP1-1.9": 0.3,
                    "SSP1-2.6": 0.4,
                    "SSP2-4.5": 0.6,
                    "SSP3-7.0": 0.8,
                    "SSP5-8.5": 1.1
                },
                "storm_intensity": {
                    "SSP1-1.9": 1.1,
                    "SSP1-2.6": 1.2,
                    "SSP2-4.5": 1.3,
                    "SSP3-7.0": 1.5,
                    "SSP5-8.5": 1.8
                }
            },
            parameters={
                "flood_model": {
                    "base_probability": {
                        "Miami": 0.2,
                        "New York": 0.15,
                        "New Orleans": 0.25,
                        "Houston": 0.18
                    },
                    "sea_level_coefficient": 0.3,
                    "storm_coefficient": 0.2
                }
            }
        )
        
        super().__init__(
            name="Climate Risk Model",
            description="Models climate risk data per-city in the next 50 years conditional on IPCC scenarios",
            schema=schema,
            state=state,
            id="climate_risk_model"
        )
    
    def _handle_posterior_query(self, query: Query) -> QueryResponse:
        """Handle a posterior query for the climate risk model."""
        variable_name = query.parameters.get("variable_name")
        covariates = query.parameters.get("covariates", {})
        rationale = query.parameters.get("rationale", False)
        
        if variable_name == "flood_probability":
            location = covariates.get("location", "Miami")
            ipcc_scenario = covariates.get("ipcc_scenario", "SSP2-4.5")
            
            # Get the current state
            checkpoint = self.state.current_checkpoint
            
            # Extract model parameters
            flood_model = checkpoint.parameters.get("flood_model", {})
            base_probability = flood_model.get("base_probability", {}).get(location, 0.1)
            sea_level_coefficient = flood_model.get("sea_level_coefficient", 0.3)
            storm_coefficient = flood_model.get("storm_coefficient", 0.2)
            
            # Extract latent values
            sea_level_rise = checkpoint.latent_values.get("sea_level_rise", {}).get(ipcc_scenario, 0.5)
            storm_intensity = checkpoint.latent_values.get("storm_intensity", {}).get(ipcc_scenario, 1.2)
            
            # Calculate flood probability
            # This is a simplified model for the demo
            flood_prob = base_probability + (sea_level_rise * sea_level_coefficient) + (storm_intensity * storm_coefficient)
            flood_prob = min(max(flood_prob, 0.0), 1.0)  # Clamp to [0, 1]
            
            # Create a Beta distribution for the flood probability
            # Higher alpha/beta means more certainty
            certainty = 20.0
            alpha = flood_prob * certainty
            beta = (1.0 - flood_prob) * certainty
            
            distribution = BetaDistribution(alpha=alpha, beta=beta)
            
            response_content = {
                "distribution": MarginalDistribution(
                    name="flood_probability",
                    distribution=distribution,
                    metadata={
                        "location": location,
                        "ipcc_scenario": ipcc_scenario
                    }
                ).to_dict()
            }
            
            if rationale:
                # Include rationale in the response
                response_content["rationale"] = {
                    "previous_state": self.state.checkpoint_history[-1].to_dict() if self.state.checkpoint_history else None,
                    "current_state": checkpoint.to_dict(),
                    "observations": [obs.to_dict() for obs in self.state.observations],
                    "calculation": {
                        "base_probability": base_probability,
                        "sea_level_rise": sea_level_rise,
                        "sea_level_coefficient": sea_level_coefficient,
                        "storm_intensity": storm_intensity,
                        "storm_coefficient": storm_coefficient,
                        "flood_probability": flood_prob
                    }
                }
            
            return QueryResponse(
                query_id=query.id,
                response_type="posterior",
                content=response_content
            )
        
        return QueryResponse(
            query_id=query.id,
            response_type="error",
            content={"error": f"Unsupported variable: {variable_name}"}
        )
    
    def _handle_update_query(self, query: Query) -> QueryResponse:
        """Handle an update query for the climate risk model."""
        observations_data = query.parameters.get("observations", [])
        observations = [Observation.from_dict(obs) for obs in observations_data]
        
        # Process the observations and update the model
        # In a real implementation, this would use a proper Bayesian update
        
        # For this demo, we'll just update the model parameters directly
        checkpoint = self.state.current_checkpoint
        
        # Create a new checkpoint with updated parameters
        new_parameters = checkpoint.parameters.copy()
        new_latent_values = checkpoint.latent_values.copy()
        
        for obs in observations:
            if obs.variable_name == "historical_flood_data":
                # Update the base probability based on new actuarial data
                flood_model = new_parameters.get("flood_model", {}).copy()
                base_probabilities = flood_model.get("base_probability", {}).copy()
                
                # Increase base probabilities by 10% to simulate the effect of new data
                for location in base_probabilities:
                    base_probabilities[location] = min(base_probabilities[location] * 1.1, 1.0)
                
                flood_model["base_probability"] = base_probabilities
                new_parameters["flood_model"] = flood_model
        
        # Store the observations
        for obs in observations:
            self.state.add_observation(obs)
        
        # Create a new checkpoint
        self.state.create_checkpoint(
            latent_values=new_latent_values,
            parameters=new_parameters,
            metadata={"update_source": query.source_node_id}
        )
        
        return QueryResponse(
            query_id=query.id,
            response_type="update",
            content={"status": "success"}
        )


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


# Create and register the nodes
def create_demo_nodes():
    """Create and register the demo nodes."""
    node_a = RealEstateFinanceNode()
    node_b = ClimateRiskNode()
    node_c = ActuarialDataNode()
    
    register_node(node_a)
    register_node(node_b)
    register_node(node_c)
    
    return node_a, node_b, node_c
