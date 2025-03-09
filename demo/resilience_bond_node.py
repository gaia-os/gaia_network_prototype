"""
Resilience Bond Node for the Gaia Network demo.

This module implements Node D: Models the behavior of a resilience bond which rewards
project developers for achieving resilience outcomes.
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


class ResilienceBondNode(Node):
    """
    Node D: Models the behavior of a resilience bond which rewards project developers
    for achieving resilience outcomes.
    """
    def __init__(self):
        schema = Schema()
        
        # Define latent variables
        schema.add_latent(Variable(
            name="bond_base_price",
            description="Base price of the resilience bond",
            type="continuous",
            domain={"min": 0.0}
        ))
        
        schema.add_latent(Variable(
            name="bond_risk_premium",
            description="Risk premium for the resilience bond",
            type="continuous",
            domain={"min": 0.0}
        ))
        
        # Define observable variables
        schema.add_observable(Variable(
            name="bond_payoff",
            description="Payoff of the resilience bond based on resilience outcomes",
            type="continuous",
            domain={"min": 0.0}
        ))
        
        schema.add_observable(Variable(
            name="bond_price",
            description="Purchase price of the resilience bond",
            type="continuous",
            domain={"min": 0.0}
        ))
        
        # Define covariates
        schema.add_covariate(Variable(
            name="resilience_outcomes",
            description="List of possible resilience outcomes (worst, median, best)",
            type="continuous"
        ))
        
        schema.add_covariate(Variable(
            name="location",
            description="Geographic location of the project",
            type="categorical"
        ))
        
        schema.add_covariate(Variable(
            name="actual_resilience",
            description="Actual resilience outcome achieved",
            type="continuous",
            domain={"min": 0.0, "max": 1.0}
        ))
        
        # Initialize state with some default values
        state = State()
        state.create_checkpoint(
            latent_values={
                "bond_base_price": 0.02,  # 2% of project value (very low entry cost)
                "bond_risk_premium": 0.01  # 1% risk premium (attractive)
            },
            parameters={
                "payoff_model": {
                    "worst_case_multiplier": 1.5,    # 150% of base price (guaranteed upside)
                    "median_case_multiplier": 3.5,   # 350% of base price (strong upside)
                    "best_case_multiplier": 6.0      # 600% of base price (exceptional upside)
                }
            }
        )
        
        super().__init__(
            name="Resilience Bond Issuer",
            description="Models the behavior of a resilience bond which rewards project developers for achieving resilience outcomes",
            schema=schema,
            state=state,
            id="resilience_bond_issuer"
        )
    
    def _handle_posterior_query(self, query: Query) -> QueryResponse:
        """Handle a posterior query for the resilience bond model."""
        variable_name = query.parameters.get("variable_name")
        covariates = query.parameters.get("covariates", {})
        
        if variable_name == "bond_payoff":
            # Check if we're calculating forecast or actual payoff
            if "resilience_outcomes" in covariates:
                return self._calculate_forecast_payoffs(query)
            elif "actual_resilience" in covariates:
                return self._calculate_actual_payoff(query)
            else:
                return QueryResponse(
                    query_id=query.id,
                    response_type="error",
                    content={"error": "Missing required covariates for bond_payoff query"}
                )
        
        elif variable_name == "bond_price":
            # Calculate the bond price
            checkpoint = self.state.current_checkpoint
            base_price = checkpoint.latent_values.get("bond_base_price", 0.05)
            risk_premium = checkpoint.latent_values.get("bond_risk_premium", 0.02)
            bond_price = base_price + risk_premium
            
            # Add some uncertainty
            price_std = 0.01
            distribution = NormalDistribution(mean=bond_price, std=price_std)
            
            return QueryResponse(
                query_id=query.id,
                response_type="posterior",
                content={
                    "distribution": MarginalDistribution(
                        name="bond_price",
                        distribution=distribution,
                        metadata={}
                    ).to_dict()
                }
            )
        
        return QueryResponse(
            query_id=query.id,
            response_type="error",
            content={"error": f"Unsupported variable: {variable_name}"}
        )
    
    def _calculate_forecast_payoffs(self, query: Query) -> QueryResponse:
        """Calculate forecast bond payoffs for different resilience outcomes."""
        covariates = query.parameters.get("covariates", {})
        resilience_outcomes = covariates.get("resilience_outcomes", [0.0, 0.5, 1.0])
        location = covariates.get("location", "Miami")
        
        # Get the current state
        checkpoint = self.state.current_checkpoint
        
        # Extract model parameters
        payoff_model = checkpoint.parameters.get("payoff_model", {})
        worst_case_multiplier = payoff_model.get("worst_case_multiplier", 0.5)
        median_case_multiplier = payoff_model.get("median_case_multiplier", 1.2)
        best_case_multiplier = payoff_model.get("best_case_multiplier", 2.0)
        
        # Extract base price
        base_price = checkpoint.latent_values.get("bond_base_price", 0.05)
        
        # Create payoff distributions for each resilience outcome
        payoff_distributions = {}
        
        # Worst case
        worst_payoff = base_price * worst_case_multiplier
        worst_std = worst_payoff * 0.1  # 10% standard deviation
        worst_dist = NormalDistribution(mean=worst_payoff, std=worst_std)
        
        # Median case
        median_payoff = base_price * median_case_multiplier
        median_std = median_payoff * 0.1
        median_dist = NormalDistribution(mean=median_payoff, std=median_std)
        
        # Best case
        best_payoff = base_price * best_case_multiplier
        best_std = best_payoff * 0.1
        best_dist = NormalDistribution(mean=best_payoff, std=best_std)
        
        # Map the distributions to the resilience outcomes
        payoff_distributions[str(resilience_outcomes[0])] = MarginalDistribution(
            name="bond_payoff",
            distribution=worst_dist,
            metadata={"resilience_outcome": "worst", "value": resilience_outcomes[0]}
        ).to_dict()
        
        payoff_distributions[str(resilience_outcomes[1])] = MarginalDistribution(
            name="bond_payoff",
            distribution=median_dist,
            metadata={"resilience_outcome": "median", "value": resilience_outcomes[1]}
        ).to_dict()
        
        payoff_distributions[str(resilience_outcomes[2])] = MarginalDistribution(
            name="bond_payoff",
            distribution=best_dist,
            metadata={"resilience_outcome": "best", "value": resilience_outcomes[2]}
        ).to_dict()
        
        return QueryResponse(
            query_id=query.id,
            response_type="posterior",
            content={
                "distributions": payoff_distributions
            }
        )
    
    def _calculate_actual_payoff(self, query: Query) -> QueryResponse:
        """Calculate actual bond payoff based on the achieved resilience outcome."""
        covariates = query.parameters.get("covariates", {})
        actual_resilience = covariates.get("actual_resilience", 0.5)
        location = covariates.get("location", "Miami")
        
        # Get the current state
        checkpoint = self.state.current_checkpoint
        
        # Extract model parameters
        payoff_model = checkpoint.parameters.get("payoff_model", {})
        worst_case_multiplier = payoff_model.get("worst_case_multiplier", 0.5)
        median_case_multiplier = payoff_model.get("median_case_multiplier", 1.2)
        best_case_multiplier = payoff_model.get("best_case_multiplier", 2.0)
        
        # Extract base price
        base_price = checkpoint.latent_values.get("bond_base_price", 0.05)
        
        # Calculate payoff based on actual resilience (linear interpolation)
        # 0.0 = worst case, 0.5 = median case, 1.0 = best case
        if actual_resilience <= 0.0:
            multiplier = worst_case_multiplier
        elif actual_resilience >= 1.0:
            multiplier = best_case_multiplier
        elif actual_resilience <= 0.5:
            # Interpolate between worst and median
            t = actual_resilience / 0.5
            multiplier = worst_case_multiplier + t * (median_case_multiplier - worst_case_multiplier)
        else:
            # Interpolate between median and best
            t = (actual_resilience - 0.5) / 0.5
            multiplier = median_case_multiplier + t * (best_case_multiplier - median_case_multiplier)
        
        payoff = base_price * multiplier
        
        # Add some uncertainty
        payoff_std = payoff * 0.05  # 5% standard deviation
        distribution = NormalDistribution(mean=payoff, std=payoff_std)
        
        return QueryResponse(
            query_id=query.id,
            response_type="posterior",
            content={
                "distribution": MarginalDistribution(
                    name="bond_payoff",
                    distribution=distribution,
                    metadata={
                        "location": location,
                        "actual_resilience": actual_resilience
                    }
                ).to_dict()
            }
        )
    
    def update_with_actuarial_data(self, location: str) -> float:
        """
        Update the bond model with actuarial data from Node C and return the actual resilience outcome.
        
        This method queries Node C for actuarial data and uses it to update
        the bond payoff model. It returns the calculated actual resilience value
        between 0.0 (worst) and 1.0 (best).
        
        Args:
            location: Geographic location of the project
        
        Returns:
            float: The actual resilience outcome value between 0.0 and 1.0
        """
        # Import here to avoid circular imports
        from demo.model_nodes import get_node_by_id
        
        # Get Node C
        node_c = get_node_by_id("actuarial_data_service")
        
        # Query Node C for actuarial data
        actuarial_response = self.query_posterior(
            target_node_id=node_c.id,
            variable_name="historical_flood_data",
            covariates={"location": location}
        )
        
        if actuarial_response.response_type == "posterior":
            # Extract the actuarial data
            actuarial_dist_data = actuarial_response.content.get("distribution")
            if actuarial_dist_data:
                actuarial_marginal = MarginalDistribution.from_dict(actuarial_dist_data)
                actuarial_value = actuarial_marginal.distribution.parameters.get("mean", 0.3)
                
                # Update the bond model based on actuarial data
                # For this demo, we'll just use the actuarial value to determine
                # the actual resilience outcome
                
                # Map actuarial value to resilience outcome
                # Lower actuarial value (less flooding) means better resilience
                actual_resilience = 1.0 - actuarial_value
                
                # Clamp to [0, 1]
                actual_resilience = min(max(actual_resilience, 0.0), 1.0)
                
                # Store the actual resilience as an observation
                observation = Observation(
                    variable_name="actual_resilience",
                    value=actual_resilience,
                    metadata={"location": location}
                )
                
                self.state.add_observation(observation)
                
                return actual_resilience
        
        # If query fails, store default median resilience as observation
        default_resilience = 0.5  # Default to median resilience if query fails
        observation = Observation(
            variable_name="actual_resilience",
            value=default_resilience,
            metadata={"location": location, "default": True}
        )
        self.state.add_observation(observation)
        return default_resilience


class ResilienceBondHandler(NodeHandler):
    """Handler for the Resilience Bond node (Node D)."""
    
    def query(self, variable_name, covariates):
        """Query Node D based on variable_name and covariates."""
        if variable_name == "bond_payoff":
            return self._query_bond_payoff(covariates)
        elif variable_name == "bond_price":
            return self._query_bond_price(covariates)
        return super().query(variable_name, covariates)
    
    def _query_bond_payoff(self, covariates):
        """Query Node D for bond payoff."""
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="bond_payoff",
            covariates=covariates
        )
        
        return response.to_dict()
    
    def _query_bond_price(self, covariates):
        """Query Node D for bond price."""
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="bond_price",
            covariates=covariates
        )
        
        return response.to_dict()
    
    def update_with_actuarial_data(self, data):
        """Update Node D with actuarial data from Node C."""
        location = data.get("location", "Miami")
        actual_resilience = self.node.update_with_actuarial_data(location)
        
        return {
            "response_type": "update",
            "content": {
                "status": "success",
                "actual_resilience": actual_resilience
            }
        }
