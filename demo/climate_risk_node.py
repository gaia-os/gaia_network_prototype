"""
Climate Risk Node for the Gaia Network demo.

This module implements Node B: Models climate risk data per-city in the next 50 years 
conditional on IPCC scenarios.
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
            # Get the covariates
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
