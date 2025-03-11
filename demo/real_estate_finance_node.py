"""
Real Estate Finance Node for the Gaia Network demo.

This module implements Node A: Models project finance for a real estate development in Miami.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import uuid

from gaia_network.node import Node
from gaia_network.schema import Schema, Variable
from gaia_network.state import State, Observation, StateCheckpoint
from gaia_network.query import Query, QueryResponse
from gaia_network.distribution import (
    Distribution, NormalDistribution, BetaDistribution, 
    MarginalDistribution, JointDistribution
)
from gaia_network.registry import register_node

from demo.node_handler import NodeHandler
from demo.sfe_calculator import calculate_sfe, calculate_alignment_score_dict, TARGET_RESILIENCE_DISTRIBUTION, modified_sigmoid


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
        
        schema.add_observable(Variable(
            name="alignment_score",
            description="Alignment score between profit goals and climate resilience goals",
            type="continuous",
            domain={"min": 0.0, "max": 1.0}
        ))
        
        schema.add_observable(Variable(
            name="system_free_energy",
            description="System Free Energy (SFE) measuring divergence from target resilience distribution",
            type="continuous",
            domain={"min": 0.0}
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
                        "Adaptation": 0.02  # 2% base ROI for Adaptation (higher upfront costs)
                    },
                    "flood_impact": {
                        "BAU": 0.12,       # 12% ROI reduction per flood probability for BAU
                        "Adaptation": 0.05  # 5% ROI reduction per flood probability for Adaptation (more resilient)
                    },
                    # Unified resilience distribution as dictionaries mapping outcomes to probabilities
                    "resilience_distribution": {
                        "BAU": {
                            0.2: 0.4,  # 40% probability of 0.2 resilience outcome
                            0.3: 0.5,  # 50% probability of 0.3 resilience outcome
                            0.4: 0.1   # 10% probability of 0.4 resilience outcome
                        },
                        "Adaptation": {
                            0.6: 0.1,  # 10% probability of 0.6 resilience outcome
                            0.7: 0.6,  # 60% probability of 0.7 resilience outcome
                            0.9: 0.3   # 30% probability of 0.9 resilience outcome
                        }
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
    
    def _calculate_sfe_and_alignment(self, resilience_distribution: Dict[float, float],
                                 profit_values: Dict[float, float],
                                 target_distribution: Dict[float, float]) -> Tuple[float, float]:
        """Calculate System Free Energy and alignment score for the current scenario.
        
        Args:
            resilience_distribution: Dictionary mapping resilience outcomes to their probabilities
            profit_values: Dictionary mapping resilience outcomes to expected profit/ROI
            target_distribution: Dictionary mapping resilience outcomes to target probabilities
            
        Returns:
            Tuple of (sfe, alignment_score)
        """
        # Calculate SFE as KL divergence between current and target distributions
        sfe = calculate_sfe(resilience_distribution, target_distribution)
        
        # Calculate alignment score between profit values and resilience distribution
        # Pass the full resilience distribution dictionary for proper correlation calculation
        alignment = calculate_alignment_score_dict(profit_values, resilience_distribution)
        
        return sfe, alignment

    def _handle_posterior_query(self, query: Query) -> QueryResponse:
        """Handle a posterior query for the real estate finance model."""
        variable_name = query.parameters.get("variable_name")
        covariates = query.parameters.get("covariates", {})
        
        if variable_name == "alignment_score":
            # Calculate alignment score and return response
            return self._handle_alignment_score_query(query.id, covariates)
        
        elif variable_name == "system_free_energy":
            # Calculate system free energy and return response
            return self._handle_sfe_query(query.id, covariates)
        
        elif variable_name == "expected_roi":
            # Calculate expected ROI and return response
            roi_dist = self._calculate_expected_roi(covariates)
            
            if isinstance(roi_dist, QueryResponse):
                # If an error occurred, return the error response
                return roi_dist
                
            # Return just the marginal distribution
            return QueryResponse(
                query_id=query.id,
                response_type="posterior",
                content={
                    "distribution": roi_dist.marginal_distribution.to_dict()
                }
            )
        
        elif variable_name == "conditional_roi":
            # Calculate conditional ROI distributions and return response
            roi_dist = self._calculate_expected_roi(covariates)
            
            if isinstance(roi_dist, QueryResponse):
                # If an error occurred, return the error response
                return roi_dist
                
            # Return the full joint distribution
            return QueryResponse(
                query_id=query.id,
                response_type="posterior",
                content={
                    "joint_distribution": roi_dist.to_dict()
                }
            )
        
        return super().query(variable_name, covariates)
    
    def _handle_alignment_score_query(self, query_id, covariates):
        """Handle a query for alignment score."""
        # Get the covariates
        location = covariates.get("location", "Miami")
        ipcc_scenario = covariates.get("ipcc_scenario", "SSP2-4.5")
        adaptation_strategy = covariates.get("adaptation_strategy", "BAU")
        include_bond = covariates.get("include_bond", "no")
        
        # Calculate the ROI distribution for the specified strategy
        roi_dist = self._calculate_expected_roi(covariates)
        
        if isinstance(roi_dist, QueryResponse):
            # If an error occurred, return the error response
            return roi_dist
        
        # Get the resilience distribution for the specified strategy
        resilience_dist = self._get_resilience_distribution(adaptation_strategy)
        
        # Get the target resilience distribution
        target_dist = TARGET_RESILIENCE_DISTRIBUTION
        
        # Calculate BAU ROI for comparison
        bau_roi_dist = self._calculate_expected_roi({
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "BAU",
            "include_bond": "no"
        })
        
        # Calculate Adaptation ROI for comparison
        adaptation_roi_dist = self._calculate_expected_roi({
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": include_bond
        })
        
        # Extract the expected ROI values
        bau_roi = bau_roi_dist.marginal_distribution.distribution.parameters["mean"]
        adaptation_roi = adaptation_roi_dist.marginal_distribution.distribution.parameters["mean"]
        
        # Extract profit values from the ROI distribution
        profit_values = {outcome: dist.parameters["mean"] for outcome, dist in roi_dist.conditional_distributions.items()}
        
        # Calculate SFE
        sfe = calculate_sfe(resilience_dist, target_dist)
        
        # Add ROI comparison metadata to resilience_dist for alignment calculation
        resilience_dist_with_metadata = resilience_dist.copy()
        resilience_dist_with_metadata["metadata"] = {
            "bau_roi": bau_roi,
            "adaptation_roi": adaptation_roi,
            "adaptation_strategy": adaptation_strategy
        }
        
        # Calculate alignment score using the function from sfe_calculator
        alignment = calculate_alignment_score_dict(profit_values, resilience_dist_with_metadata)
        
        # Create the response
        return QueryResponse(
            query_id=query_id,
            response_type="posterior",
            content={
                "alignment_score": alignment,
                "sfe": sfe,
                "metadata": {
                    "location": location,
                    "ipcc_scenario": ipcc_scenario,
                    "adaptation_strategy": adaptation_strategy,
                    "include_bond": include_bond,
                    "bau_roi": bau_roi,
                    "adaptation_roi": adaptation_roi
                }
            }
        )
    
    def _handle_sfe_query(self, query_id, covariates):
        """Handle system free energy calculation and response formatting.
        
        Args:
            query_id: The ID of the original query
            covariates: Dictionary of covariates for the query
            
        Returns:
            QueryResponse with SFE distribution
        """
        # Get the covariates
        location = covariates.get("location", "Miami")
        ipcc_scenario = covariates.get("ipcc_scenario", "SSP2-4.5")
        adaptation_strategy = covariates.get("adaptation_strategy", "BAU")
        include_bond = covariates.get("include_bond", "no")
        actual_resilience = covariates.get("actual_resilience")
        
        # Get resilience distribution
        resilience_distribution = self._get_resilience_distribution(adaptation_strategy)
        
        # Calculate expected ROI and get profit values
        roi_dist = self._calculate_expected_roi({
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": adaptation_strategy,
            "include_bond": include_bond,
            "actual_resilience": actual_resilience
        })
        
        if isinstance(roi_dist, QueryResponse):
            # If an error occurred, return the error response
            return roi_dist
        
        # Extract profit values from the conditional distributions
        profit_values = {}
        for outcome, dist in roi_dist.conditional_distributions.items():
            profit_values[outcome] = dist.parameters["mean"]
        
        # Get the global target distribution for climate resilience
        target_distribution = TARGET_RESILIENCE_DISTRIBUTION
        
        # Calculate SFE using the unmodified resilience distribution
        sfe, _ = self._calculate_sfe_and_alignment(
            resilience_distribution, profit_values, target_distribution
        )
        
        # Create a distribution for the SFE
        sfe_dist = NormalDistribution(mean=sfe, std=0.01)
        
        return QueryResponse(
            query_id=query_id,
            response_type="posterior",
            content={
                "distribution": MarginalDistribution(
                    name="system_free_energy",
                    distribution=sfe_dist,
                    metadata={
                        "adaptation_strategy": adaptation_strategy,
                        "include_bond": include_bond,
                        "current_distribution": resilience_distribution,
                        "target_distribution": target_distribution
                    }
                ).to_dict()
            }
        )
    
    def _get_resilience_distribution(self, adaptation_strategy: str) -> Dict[float, float]:
        """Get the resilience distribution for a given adaptation strategy.
        
        Args:
            adaptation_strategy: The adaptation strategy (BAU or Adaptation)
            
        Returns:
            Dictionary mapping resilience outcomes to probabilities
        """
        # Get model parameters
        roi_model = self.state.current_checkpoint.parameters["roi_model"]
        
        # Check if we have a resilience distribution for this strategy
        if adaptation_strategy not in roi_model["resilience_distribution"]:
            # Default to BAU if not found
            return roi_model["resilience_distribution"]["BAU"].copy()
        else:
            return roi_model["resilience_distribution"][adaptation_strategy].copy()
    
    def _calculate_expected_roi(self, covariates):
        # Get the covariates
        location = covariates.get("location", "Miami")
        ipcc_scenario = covariates.get("ipcc_scenario", "SSP2-4.5")
        adaptation_strategy = covariates.get("adaptation_strategy", "BAU")
        include_bond = covariates.get("include_bond", "no")
        
        # Get model parameters
        roi_model = self.state.current_checkpoint.parameters["roi_model"]
        base_roi = roi_model["base_roi"][adaptation_strategy]
        flood_impact = roi_model["flood_impact"][adaptation_strategy]
        
        # Define standard resilience outcomes for bond calculations
        all_resilience_outcomes = list(roi_model["resilience_distribution"][adaptation_strategy].keys())
        
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
                query_id="error",  # This will be replaced by the caller
                response_type="error",
                content={"error": f"Failed to get flood probability: {flood_response.content.get('error')}"}
            )
        
        # Extract the flood probability distribution
        flood_dist_data = flood_response.content.get("distribution")
        if not flood_dist_data or not isinstance(flood_dist_data, dict):
            return QueryResponse(
                query_id="error",  # This will be replaced by the caller
                response_type="error",
                content={"error": "No distribution data found in flood probability response"}
            )
        
        # Get the flood probability
        flood_dist = MarginalDistribution.from_dict(flood_dist_data)
        flood_probability = flood_dist.distribution.parameters["alpha"] / (flood_dist.distribution.parameters["alpha"] + flood_dist.distribution.parameters["beta"])
        
        # Calculate the base ROI reduction due to flood probability
        roi_reduction = flood_probability * flood_impact
        
        # Calculate the expected ROI
        expected_roi = base_roi - roi_reduction
        
        # Initialize metadata
        metadata = {
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": adaptation_strategy,
            "include_bond": include_bond,
            "flood_probability": flood_probability
        }
        
        # Create conditional distributions for each resilience outcome
        conditional_distributions = {}

        # Calculate ROI for each resilience outcome
        for outcome in all_resilience_outcomes:
            # Convert outcome to float for calculations
            outcome_float = float(outcome)
            
            outcome_roi = base_roi * outcome_float
            
            # Add some uncertainty to the conditional distribution
            roi_std = 0.01  # Less uncertainty in the conditional distributions
            conditional_distributions[outcome] = NormalDistribution(mean=outcome_roi, std=roi_std)
        
        # Check if we need to include the resilience bond
        if include_bond == "yes":
            # Query Node D for bond price
            price_response = self.query_posterior(
                target_node_id="resilience_bond_issuer",
                variable_name="bond_price",
                covariates={"location": location}
            )
            
            if price_response.response_type == "error":
                return QueryResponse(
                    query_id="error",  # This will be replaced by the caller
                    response_type="error",
                    content={"error": f"Failed to get bond price: {price_response.content.get('error')}"}
                )
            
            # Extract the bond price
            price_dist = MarginalDistribution.from_dict(price_response.content["distribution"])
            bond_price = price_dist.distribution.parameters["mean"]
            
            # Add bond price to metadata
            metadata["bond_price"] = bond_price
            
            # Check if we have an actual resilience outcome
            actual_resilience = covariates.get("actual_resilience")
            if actual_resilience:
                # Query Node D for actual bond payoff
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
                        query_id="error",  # This will be replaced by the caller
                        response_type="error",
                        content={"error": f"Failed to get actual bond payoff: {bond_response.content.get('error')}"}
                    )
                
                # Extract the actual bond payoff
                payoff_dist = MarginalDistribution.from_dict(bond_response.content["distribution"])
                actual_payoff = payoff_dist.distribution.parameters["mean"]
                expected_roi = expected_roi - bond_price + actual_payoff
                
                # Update the conditional distribution for the actual resilience outcome
                outcome_roi = conditional_distributions[actual_resilience].parameters["mean"] - bond_price + actual_payoff
                conditional_distributions[actual_resilience] = NormalDistribution(mean=outcome_roi, std=roi_std)
            else:
                # Calculate expected payoff from possible outcomes
                bond_response = self.query_posterior(
                    target_node_id="resilience_bond_issuer",
                    variable_name="bond_payoff",
                    covariates={
                        "location": location,
                        "resilience_outcomes": all_resilience_outcomes
                    }
                )
                
                if bond_response.response_type == "error":
                    return QueryResponse(
                        query_id="error",  # This will be replaced by the caller
                        response_type="error",
                        content={"error": f"Failed to get bond payoffs: {bond_response.content.get('error')}"}
                    )
                
                # Extract the payoff distributions
                payoff_distributions = bond_response.content["distributions"]
                
                # Calculate expected payoff
                expected_payoff = 0.0
                
                # Calculate expected payoff using the resilience distribution
                for outcome, prob in roi_model["resilience_distribution"][adaptation_strategy].items():
                    payoff_dist = MarginalDistribution.from_dict(payoff_distributions[str(outcome)])
                    payoff = payoff_dist.distribution.parameters["mean"]
                    expected_payoff += prob * payoff
                    
                    # Update conditional distributions with bond effects
                    outcome_roi = conditional_distributions[outcome].parameters["mean"] - bond_price + payoff
                    conditional_distributions[outcome] = NormalDistribution(mean=outcome_roi, std=roi_std)
                
                # Update expected ROI with bond effects
                expected_roi = expected_roi - bond_price + expected_payoff
                
                # Add expected payoff to metadata
                metadata["expected_bond_payoff"] = expected_payoff
                
                # Add resilience distribution to metadata
                metadata["resilience_distribution"] = roi_model["resilience_distribution"][adaptation_strategy]
        
        # Add some uncertainty to the marginal distribution
        roi_std = 0.03  # More uncertainty in the marginal distribution
        marginal_distribution = MarginalDistribution(
            name="expected_roi",
            distribution=NormalDistribution(mean=expected_roi, std=roi_std),
            metadata=metadata
        )
        
        # Create and return a joint distribution
        return JointDistribution(
            name="roi_distribution",
            conditional_distributions=conditional_distributions,
            marginal_distribution=marginal_distribution,
            metadata=metadata
        )


class RealEstateFinanceHandler(NodeHandler):
    """Handler for the Real Estate Finance node (Node A)."""
    
    def query(self, variable_name, covariates):
        """Query Node A based on variable_name and covariates."""
        if variable_name == "expected_roi":
            return self._query_roi(covariates)
        elif variable_name == "alignment_score":
            return self._query_alignment(covariates)
        elif variable_name == "system_free_energy":
            return self._query_sfe(covariates)
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
    
    def _query_alignment(self, covariates):
        """Query Node A for alignment score."""
        # Set default values if not provided
        if "adaptation_strategy" not in covariates:
            covariates["adaptation_strategy"] = "BAU"
        if "include_bond" not in covariates:
            covariates["include_bond"] = "no"
            
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="alignment_score",
            covariates=covariates
        )
        
        return response.to_dict()
    
    def _query_sfe(self, covariates):
        """Query Node A for system free energy."""
        # Set default values if not provided
        if "adaptation_strategy" not in covariates:
            covariates["adaptation_strategy"] = "BAU"
        if "include_bond" not in covariates:
            covariates["include_bond"] = "no"
            
        response = self.node.query_posterior(
            target_node_id=self.node.id,
            variable_name="system_free_energy",
            covariates=covariates
        )
        
        return response.to_dict()
