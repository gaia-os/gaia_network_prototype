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
        
        schema.add_latent(Variable(
            name="bond_adjusted_roi",
            description="ROI adjusted for resilience bond payoff",
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
        
        schema.add_observable(Variable(
            name="resilience_outcome",
            description="Resilience outcome achieved by the project",
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
        
        schema.add_covariate(Variable(
            name="adaptation_strategy",
            description="Project adaptation strategy (BAU or Adaptation)",
            type="categorical",
            domain={"categories": ["BAU", "Adaptation"]}
        ))
        
        schema.add_covariate(Variable(
            name="include_bond",
            description="Whether to include resilience bond effects",
            type="categorical",
            domain={"categories": ["yes", "no"]}
        ))
        
        # Create initial state with default values
        state = State()
        state.create_checkpoint(
            parameters={
                "roi_model": {
                    "base_roi": {
                        "BAU": 0.08,       # 8% base ROI for BAU (standard real estate return)
                        "Adaptation": 0.05  # 5% base ROI for Adaptation (higher upfront costs)
                    },
                    "flood_impact": {
                        "BAU": 0.12,       # 12% ROI reduction per flood probability for BAU
                        "Adaptation": 0.15  # 15% ROI reduction per flood probability for Adaptation (more to lose)
                    },
                    "resilience_outcomes": {
                        "BAU": [0.2, 0.3, 0.4],      # Worst/median/best resilience for BAU
                        "Adaptation": [0.6, 0.7, 0.9]  # Worst/median/best resilience for Adaptation
                    },
                    "outcome_probabilities": {
                        "BAU": [0.4, 0.5, 0.1],      # Probabilities for worst/median/best (BAU)
                        "Adaptation": [0.1, 0.6, 0.3]  # Probabilities for worst/median/best (Adaptation)
                    }
                }
            }
        )
        
        super().__init__(
            name="Real Estate Finance Model",
            description="Models project finance for a real estate development",
            schema=schema,
            state=state,
            id="real_estate_finance_model"
        )
    
    def _handle_posterior_query(self, query: Query) -> QueryResponse:
        """Handle a posterior query for the real estate finance model."""
        variable_name = query.parameters.get("variable_name")
        covariates = query.parameters.get("covariates", {})
        
        if variable_name == "expected_roi":
            # Get the covariates
            location = covariates.get("location", "Miami")
            ipcc_scenario = covariates.get("ipcc_scenario", "SSP2-4.5")
            adaptation_strategy = covariates.get("adaptation_strategy", "BAU")
            include_bond = covariates.get("include_bond", "no")
            
            # Get model parameters
            roi_model = self.state.current_checkpoint.parameters["roi_model"]
            base_roi = roi_model["base_roi"][adaptation_strategy]
            flood_impact = roi_model["flood_impact"][adaptation_strategy]
            resilience_outcomes = roi_model["resilience_outcomes"][adaptation_strategy]
            outcome_probabilities = roi_model["outcome_probabilities"][adaptation_strategy]
            
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
            flood_prob = flood_marginal.distribution.parameters["alpha"] / (
                flood_marginal.distribution.parameters["alpha"] + 
                flood_marginal.distribution.parameters["beta"]
            )
            

            
            # Calculate base ROI with flood impact
            roi_reduction = flood_prob * flood_impact
            expected_roi = base_roi - roi_reduction
            
            # If adaptation strategy with bond, calculate bond effects
            if adaptation_strategy == "Adaptation" and include_bond == "yes":
                # Get bond price
                price_response = self.query_posterior(
                    target_node_id="resilience_bond_issuer",
                    variable_name="bond_price",
                    covariates={}
                )
                
                if price_response.response_type == "error":
                    return QueryResponse(
                        query_id=query.id,
                        response_type="error",
                        content={"error": f"Failed to get bond price: {price_response.content.get('error')}"}
                    )
                
                price_dist = MarginalDistribution.from_dict(price_response.content["distribution"])
                bond_price = price_dist.distribution.parameters["mean"]
                
                # Check if we have actual resilience outcome
                actual_resilience = covariates.get("actual_resilience")
                
                if actual_resilience is not None:
                    # Use actual resilience outcome
                    bond_response = self.query_posterior(
                        target_node_id="resilience_bond_issuer",
                        variable_name="bond_payoff",
                        covariates={
                            "location": location,
                            "actual_resilience": actual_resilience
                        }
                    )
                    
                    if bond_response.response_type == "error":
                        return QueryResponse(
                            query_id=query.id,
                            response_type="error",
                            content={"error": f"Failed to get actual bond payoff: {bond_response.content.get('error')}"}
                        )
                    
                    payoff_dist = MarginalDistribution.from_dict(bond_response.content["distribution"])
                    actual_payoff = payoff_dist.distribution.parameters["mean"]
                    expected_roi = expected_roi - bond_price + actual_payoff
                else:
                    # Calculate expected payoff from possible outcomes
                    bond_response = self.query_posterior(
                        target_node_id="resilience_bond_issuer",
                        variable_name="bond_payoff",
                        covariates={
                            "location": location,
                            "resilience_outcomes": resilience_outcomes
                        }
                    )
                    
                    if bond_response.response_type == "error":
                        return QueryResponse(
                            query_id=query.id,
                            response_type="error",
                            content={"error": f"Failed to get bond payoffs: {bond_response.content.get('error')}"}
                        )
                    
                    payoff_distributions = bond_response.content.get("distributions", {})
                    expected_payoff = 0.0
                    for outcome, prob in zip(resilience_outcomes, outcome_probabilities):
                        payoff_dist = MarginalDistribution.from_dict(payoff_distributions[str(outcome)])
                        payoff = payoff_dist.distribution.parameters["mean"]
                        expected_payoff += prob * payoff
                    
                    expected_roi = expected_roi - bond_price + expected_payoff
            
            # Add some uncertainty
            roi_std = 0.03
            distribution = NormalDistribution(mean=expected_roi, std=roi_std)
            
            # Initialize metadata
            metadata = {
                "location": location,
                "ipcc_scenario": ipcc_scenario,
                "adaptation_strategy": adaptation_strategy,
                "include_bond": include_bond,
                "flood_probability": flood_prob
            }
            
            # Add bond-related metadata if applicable
            if adaptation_strategy == "Adaptation" and include_bond == "yes":
                metadata["bond_price"] = bond_price
                if actual_resilience is not None:
                    metadata["actual_resilience"] = actual_resilience
                    metadata["actual_bond_payoff"] = actual_payoff
                else:
                    metadata["expected_bond_payoff"] = expected_payoff
                    metadata["resilience_outcomes"] = resilience_outcomes
                    metadata["outcome_probabilities"] = outcome_probabilities
            
            return QueryResponse(
                query_id=query.id,
                response_type="posterior",
                content={
                    "distribution": MarginalDistribution(
                        name="expected_roi",
                        distribution=distribution,
                        metadata=metadata
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
        # Set default values if not provided
        if "adaptation_strategy" not in covariates:
            covariates["adaptation_strategy"] = "BAU"
        if "include_bond" not in covariates:
            covariates["include_bond"] = "no"
        
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="expected_roi",
            covariates=covariates
        )
        
        return response.to_dict()
