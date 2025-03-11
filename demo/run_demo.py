#!/usr/bin/env python3
"""
Gaia Network Demo Script

This script demonstrates the Gaia Network prototype with three nodes:
- Node A: Models project finance for a real estate development in Miami
- Node B: Models climate risk data per-city in the next 50 years conditional on IPCC scenarios
- Node C: Serves actuarial data relevant for climate risk

The demo follows the script described in the requirements.
"""

import json
import time
from pprint import pprint

from gaia_network.state import Observation
from demo.model_nodes import create_demo_nodes
from demo.sfe_calculator import pretty_print_distribution


def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def main():
    """Run the Gaia Network demo."""
    print_separator("Gaia Network Demo")
    
    # Create and register the demo nodes
    print("Creating demo nodes...")
    node_a, node_b, node_c, node_d = create_demo_nodes()
    print(f"Node A (Real Estate Finance): {node_a.id}")
    print(f"Node B (Climate Risk): {node_b.id}")
    print(f"Node C (Actuarial Data): {node_c.id}")
    print(f"Node D (Resilience Bond): {node_d.id}")
    
    # Step 1: Node A gets a hardcoded location and IPCC scenario
    print_separator("Step 1: Node A gets a hardcoded location and IPCC scenario")
    location = "Miami"
    ipcc_scenario = "SSP2-4.5"
    print(f"Location: {location}")
    print(f"IPCC Scenario: {ipcc_scenario}")
    
    # Step 2: Node A queries Node B's state space schema
    print_separator("Step 2: Node A queries Node B's state space schema")
    schema_response = node_a.query_schema(node_b.id)
    print("Schema response type:", schema_response.response_type)
    
    schema = schema_response.content["schema"]
    print("\nLatent variables:")
    for var in schema["latents"]:
        print(f"  - {var['name']}: {var['description']}")
    
    print("\nObservable variables:")
    for var in schema["observables"]:
        print(f"  - {var['name']}: {var['description']}")
    
    print("\nCovariates:")
    for var in schema["covariates"]:
        print(f"  - {var['name']}: {var['description']}")
    
    # Step 3: Node A formulates a query to Node B for flood probability
    print_separator("Step 3: Node A queries Node B for flood probability")
    print(f"Querying flood probability for {location} under {ipcc_scenario} scenario...")
    
    posterior_response = node_a.query_posterior(
        target_node_id=node_b.id,
        variable_name="flood_probability",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario
        }
    )
    
    print("Posterior response type:", posterior_response.response_type)
    
    if posterior_response.response_type == "posterior":
        distribution_data = posterior_response.content["distribution"]
        print("\nFlood probability distribution:")
        print(f"  - Name: {distribution_data['name']}")
        print(f"  - Type: {distribution_data['distribution']['type']}")
        print(f"  - Parameters: {distribution_data['distribution']['parameters']}")
        print(f"  - Metadata: {distribution_data['metadata']}")
        
        # Calculate expected value for Beta distribution
        alpha = distribution_data['distribution']['parameters']['alpha']
        beta = distribution_data['distribution']['parameters']['beta']
        expected_value = alpha / (alpha + beta)
        print(f"\nExpected flood probability: {expected_value:.4f}")
    
    # Step 4: Node A uses the flood probability to calculate ROI
    print_separator("Step 4: Node A calculates ROI based on flood probability")
    print("Calculating expected ROI for the real estate project...")
    
    roi_response = node_a._handle_posterior_query(
        Query(
            source_node_id=node_a.id,
            target_node_id=node_a.id,
            query_type="posterior",
            parameters={
                "variable_name": "expected_roi",
                "covariates": {
                    "location": location,
                    "ipcc_scenario": ipcc_scenario
                }
            }
        )
    )
    
    if roi_response.response_type == "posterior":
        roi_dist_data = roi_response.content["distribution"]
        print("\nROI distribution:")
        print(f"  - Name: {roi_dist_data['name']}")
        print(f"  - Type: {roi_dist_data['distribution']['type']}")
        print(f"  - Parameters: {roi_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {roi_dist_data['metadata']}")
        
        expected_roi = roi_dist_data['distribution']['parameters']['mean']
        print(f"\nExpected ROI: {expected_roi:.2%}")
    
    # Step 5: Node C gets new actuarial data
    print_separator("Step 5: Node C gets new actuarial data")
    print("Node C receives new actuarial data...")
    
    # Add new data to Node C
    node_c.add_new_data(location=location, value=0.4)
    print("New actuarial data added to Node C")
    
    # Step 6: Node B queries Node C for updated data
    print_separator("Step 6: Node B queries Node C for updated data")
    print("Node B queries Node C for updated actuarial data...")
    
    actuarial_response = node_b.query_posterior(
        target_node_id=node_c.id,
        variable_name="historical_flood_data",
        covariates={"location": location}
    )
    
    if actuarial_response.response_type == "posterior":
        actuarial_dist_data = actuarial_response.content["distribution"]
        print("\nActuarial data distribution:")
        print(f"  - Name: {actuarial_dist_data['name']}")
        print(f"  - Type: {actuarial_dist_data['distribution']['type']}")
        print(f"  - Parameters: {actuarial_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {actuarial_dist_data['metadata']}")
    
    # Step 7: Node B updates its state based on the new data
    print_separator("Step 7: Node B updates its state based on the new data")
    print("Node B updates its state with the new actuarial data...")
    
    update_response = node_b.send_update(
        target_node_id=node_b.id,
        observations=[
            {
                "variable_name": "historical_flood_data",
                "value": 0.4,
                "metadata": {"location": location}
            }
        ]
    )
    
    print("Update response type:", update_response.response_type)
    print("Update status:", update_response.content.get("status"))
    
    # Step 8: Node A repeats the query to Node B
    print_separator("Step 8: Node A repeats the query to Node B")
    print(f"Querying updated flood probability for {location} under {ipcc_scenario} scenario...")
    
    updated_posterior_response = node_a.query_posterior(
        target_node_id=node_b.id,
        variable_name="flood_probability",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario
        }
    )
    
    if updated_posterior_response.response_type == "posterior":
        updated_dist_data = updated_posterior_response.content["distribution"]
        print("\nUpdated flood probability distribution:")
        print(f"  - Name: {updated_dist_data['name']}")
        print(f"  - Type: {updated_dist_data['distribution']['type']}")
        print(f"  - Parameters: {updated_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {updated_dist_data['metadata']}")
        
        # Calculate expected value for Beta distribution
        alpha = updated_dist_data['distribution']['parameters']['alpha']
        beta = updated_dist_data['distribution']['parameters']['beta']
        updated_expected_value = alpha / (alpha + beta)
        print(f"\nUpdated expected flood probability: {updated_expected_value:.4f}")
        
        # Compare with previous value
        print(f"Change in flood probability: {(updated_expected_value - expected_value):.4f}")
    
    # Step 9: Node A includes the rationale parameter
    print_separator("Step 9: Node A queries Node B with rationale=True")
    print(f"Querying flood probability with rationale for {location} under {ipcc_scenario} scenario...")
    
    rationale_response = node_a.query_posterior(
        target_node_id=node_b.id,
        variable_name="flood_probability",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario
        },
        rationale=True
    )
    
    if rationale_response.response_type == "posterior":
        rationale_dist_data = rationale_response.content["distribution"]
        print("\nFlood probability distribution with rationale:")
        print(f"  - Name: {rationale_dist_data['name']}")
        print(f"  - Type: {rationale_dist_data['distribution']['type']}")
        print(f"  - Parameters: {rationale_dist_data['distribution']['parameters']}")
        
        print("\nRationale:")
        rationale = rationale_response.content.get("rationale", {})
        
        print("\nCalculation details:")
        calculation = rationale.get("calculation", {})
        for key, value in calculation.items():
            print(f"  - {key}: {value}")
        
        print("\nObservations that caused the update:")
        observations = rationale.get("observations", [])
        for obs in observations:
            print(f"  - {obs['variable_name']}: {obs['value']} (timestamp: {obs['timestamp']})")
    
    # Step 10: Node A recalculates ROI based on updated flood probability
    print_separator("Step 10: Node A recalculates ROI based on updated flood probability")
    print("Recalculating expected ROI for the real estate project...")
    
    updated_roi_response = node_a._handle_posterior_query(
        Query(
            source_node_id=node_a.id,
            target_node_id=node_a.id,
            query_type="posterior",
            parameters={
                "variable_name": "expected_roi",
                "covariates": {
                    "location": location,
                    "ipcc_scenario": ipcc_scenario
                }
            }
        )
    )
    
    if updated_roi_response.response_type == "posterior":
        updated_roi_dist_data = updated_roi_response.content["distribution"]
        print("\nUpdated ROI distribution:")
        print(f"  - Name: {updated_roi_dist_data['name']}")
        print(f"  - Type: {updated_roi_dist_data['distribution']['type']}")
        print(f"  - Parameters: {updated_roi_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {updated_roi_dist_data['metadata']}")
        
        updated_expected_roi = updated_roi_dist_data['distribution']['parameters']['mean']
        print(f"\nUpdated expected ROI: {updated_expected_roi:.2%}")
        print(f"Change in ROI: {(updated_expected_roi - expected_roi):.2%}")
    
    # Step 11: Compare ROI for BAU vs Adaptation (without bond)
    print_separator("Step 11: Compare ROI for BAU vs Adaptation (without bond)")
    
    # Query ROI for BAU
    print("\nCalculating ROI for Business-as-Usual (BAU) strategy...")
    bau_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="expected_roi",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "BAU",
            "include_bond": "no"
        }
    )
    
    if bau_response.response_type == "posterior":
        bau_dist = bau_response.content["distribution"]
        bau_roi = bau_dist["distribution"]["parameters"]["mean"]
        print(f"BAU Expected ROI: {bau_roi:.2%}")
    
    # Query ROI for Adaptation
    print("\nCalculating ROI for Climate Adaptation strategy...")
    adapt_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="expected_roi",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "no"
        }
    )
    
    if adapt_response.response_type == "posterior":
        adapt_dist = adapt_response.content["distribution"]
        adapt_roi = adapt_dist["distribution"]["parameters"]["mean"]
        print(f"Adaptation Expected ROI: {adapt_roi:.2%}")
    
    print(f"\nInitial ROI difference (Adaptation - BAU): {(adapt_roi - bau_roi):.2%}")
    
    # Step 12: Consider resilience bond effects for Adaptation strategy
    print_separator("Step 12: Consider resilience bond effects for Adaptation strategy")
    print("Calculating ROI for Adaptation strategy with resilience bond...")
    
    # Query ROI for Adaptation with bond
    bond_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="expected_roi",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes"
        }
    )
    
    if bond_response.response_type == "posterior":
        bond_dist = bond_response.content["distribution"]
        bond_roi = bond_dist["distribution"]["parameters"]["mean"]
        metadata = bond_dist["metadata"]
        
        print(f"\nAdaptation + Bond Expected ROI: {bond_roi:.2%}")
        print(f"Bond price: {metadata['bond_price']:.2%}")
        print(f"Expected bond payoff: {metadata['expected_bond_payoff']:.2%}")
        print("\nResilient outcomes and probabilities:")
        for outcome, prob in metadata['resilience_distribution'].items():
            print(f"  - Outcome {outcome:.2f}: {prob:.0%} probability")
        
        print(f"\nROI improvement from bond: {(bond_roi - adapt_roi):.2%}")
        print(f"Final ROI difference (Adaptation + Bond - BAU): {(bond_roi - bau_roi):.2%}")
    
    # Step 13: Simulate time passing and project development
    print_separator("Step 13: Simulate time passing and project development")
    print("Time passes, the project is developed...")
    
    # Add new actuarial data to Node C
    print("\nNode C receives new actuarial data...")
    node_c.add_new_data(location=location, value=0.35)  # Better than average flood data
    print("New actuarial data added to Node C")
    
    # Node D queries Node C and updates its state
    print("\nNode D queries Node C for actuarial data to determine actual resilience outcome...")
    actual_resilience = node_d.update_with_actuarial_data(location)
    print(f"Actual resilience outcome achieved: {actual_resilience:.2f}")
    
    # Calculate actual bond payoff
    print("\nCalculating actual bond payoff based on achieved resilience...")
    actual_payoff_response = node_d.query_posterior(
        target_node_id=node_d.id,
        variable_name="bond_payoff",
        covariates={
            "location": location,
            "actual_resilience": actual_resilience
        }
    )
    
    if actual_payoff_response.response_type == "posterior":
        payoff_dist = actual_payoff_response.content["distribution"]
        actual_payoff = payoff_dist["distribution"]["parameters"]["mean"]
        print(f"Actual bond payoff: {actual_payoff:.2%}")
    
    # Calculate final project ROI
    print("\nCalculating final project ROI with actual bond payoff...")
    final_roi_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="expected_roi",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes",
            "actual_resilience": actual_resilience
        }
    )
    
    if final_roi_response.response_type == "posterior":
        final_dist = final_roi_response.content["distribution"]
        final_roi = final_dist["distribution"]["parameters"]["mean"]
        print(f"Final project ROI: {final_roi:.2%}")
        print(f"Improvement over initial BAU ROI: {(final_roi - bau_roi):.2%}")
    
    # Step 14: Calculate System Free Energy (SFE) for BAU and Adaptation strategies
    print_separator("Step 14: Calculate System Free Energy (SFE)")
    print("Calculating System Free Energy (SFE) to measure divergence from target resilience distribution...")
    
    # Query SFE for BAU
    print("\nCalculating SFE for Business-as-Usual (BAU) strategy...")
    bau_sfe_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="system_free_energy",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "BAU",
            "include_bond": "no"
        }
    )
    
    if bau_sfe_response.response_type == "posterior":
        bau_sfe_dist = bau_sfe_response.content["distribution"]
        bau_sfe = bau_sfe_dist["distribution"]["parameters"]["mean"]
        bau_current_dist = bau_sfe_dist["metadata"]["current_distribution"]
        bau_target_dist = bau_sfe_dist["metadata"]["target_distribution"]
        print(f"BAU System Free Energy: {bau_sfe:.4f}")
        print("\nBAU Resilience Distribution:")
        print(f"  - Current: {pretty_print_distribution(bau_current_dist)}")
        print(f"  - Target:  {pretty_print_distribution(bau_target_dist)}")
    
    # Query SFE for Adaptation
    print("\nCalculating SFE for Climate Adaptation strategy...")
    adapt_sfe_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="system_free_energy",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "no"
        }
    )
    
    if adapt_sfe_response.response_type == "posterior":
        adapt_sfe_dist = adapt_sfe_response.content["distribution"]
        adapt_sfe = adapt_sfe_dist["distribution"]["parameters"]["mean"]
        adapt_current_dist = adapt_sfe_dist["metadata"]["current_distribution"]
        adapt_target_dist = adapt_sfe_dist["metadata"]["target_distribution"]
        print(f"Adaptation System Free Energy: {adapt_sfe:.4f}")
        print("\nAdaptation Resilience Distribution:")
        print(f"  - Current: {pretty_print_distribution(adapt_current_dist)}")
        print(f"  - Target:  {pretty_print_distribution(adapt_target_dist)}")
    
    # Query SFE for Adaptation with bond
    print("\nCalculating SFE for Adaptation strategy with resilience bond...")
    bond_sfe_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="system_free_energy",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes"
        }
    )
    
    if bond_sfe_response.response_type == "posterior":
        bond_sfe_dist = bond_sfe_response.content["distribution"]
        bond_sfe = bond_sfe_dist["distribution"]["parameters"]["mean"]
        bond_current_dist = bond_sfe_dist["metadata"]["current_distribution"]
        bond_target_dist = bond_sfe_dist["metadata"]["target_distribution"]
        print(f"Adaptation + Bond System Free Energy: {bond_sfe:.4f}")
        print(f"SFE improvement from bond: {(adapt_sfe - bond_sfe):.4f}")
        print("\nAdaptation + Bond Resilience Distribution:")
        print(f"  - Current: {pretty_print_distribution(bond_current_dist)}")
        print(f"  - Target:  {pretty_print_distribution(bond_target_dist)}")
    
    # Step 15: Calculate alignment scores for BAU and Adaptation strategies
    print_separator("Step 15: Calculate Alignment Scores")
    print("Calculating alignment scores between profit goals and climate resilience goals...")
    
    # Query alignment for BAU
    print("\nCalculating alignment score for Business-as-Usual (BAU) strategy...")
    bau_align_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="alignment_score",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "BAU",
            "include_bond": "no"
        }
    )
    
    if bau_align_response.response_type == "posterior":
        bau_align_dist = bau_align_response.content["distribution"]
        bau_align = bau_align_dist["metadata"]["alignment_value"]
        print(f"BAU Alignment Score: {bau_align:.2%}")
    
    # Query alignment for Adaptation
    print("\nCalculating alignment score for Climate Adaptation strategy...")
    adapt_align_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="alignment_score",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "no"
        }
    )
    
    if adapt_align_response.response_type == "posterior":
        adapt_align_dist = adapt_align_response.content["distribution"]
        adapt_align = adapt_align_dist["metadata"]["alignment_value"]
        print(f"Adaptation Alignment Score: {adapt_align:.2%}")
    
    # Query alignment for Adaptation with bond
    print("\nCalculating alignment score for Adaptation strategy with resilience bond...")
    bond_align_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="alignment_score",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes"
        }
    )
    
    if bond_align_response.response_type == "posterior":
        bond_align_dist = bond_align_response.content["distribution"]
        bond_align = bond_align_dist["metadata"]["alignment_value"]
        print(f"Adaptation + Bond Alignment Score: {bond_align:.2%}")
        print(f"Alignment improvement from bond: {(bond_align - adapt_align):.2%}")
    
    # Step 16: Calculate final SFE and alignment after project development
    print_separator("Step 16: Calculate Final SFE and Alignment After Project Development")
    print("Calculating final SFE and alignment scores based on actual resilience outcome...")
    
    # Create updated outcome probabilities based on actual resilience
    # In a real system, this would come from actual measurements
    # Here we simulate by creating a distribution more concentrated around the actual outcome
    print(f"\nActual resilience outcome: {actual_resilience:.2f}")
    
    # Add an observation to the real estate finance node about the actual resilience
    observation = Observation(
        variable_name="actual_resilience",
        value=actual_resilience,
        metadata={"location": location, "ipcc_scenario": ipcc_scenario}
    )
    node_a.state.add_observation(observation)
    
    # Create a delta distribution for the actual resilience outcome
    # A delta distribution puts 100% probability on the actual outcome
    # First, determine which resilience outcome bin the actual resilience falls into
    roi_model = node_a.state.current_checkpoint.parameters["roi_model"]
    resilience_distribution = roi_model["resilience_distribution"]["Adaptation"]
    
    # Find the closest resilience outcome to the actual resilience
    closest_outcome = min(resilience_distribution.keys(), 
                          key=lambda outcome: abs(float(outcome) - actual_resilience))
    
    # Create a delta distribution (all probability mass on the actual outcome)
    delta_distribution = {outcome: 0.0 for outcome in resilience_distribution.keys()}
    delta_distribution[closest_outcome] = 1.0
    
    # Add an observation about the updated resilience distribution
    observation = Observation(
        variable_name="resilience_distribution",
        value={"Adaptation": delta_distribution},  # Delta distribution on the actual outcome
        metadata={"adaptation_strategy": "Adaptation"}
    )
    node_a.state.add_observation(observation)
    
    # We'll use the global TARGET_RESILIENCE_DISTRIBUTION as our target
    # This represents the ideal distribution of resilience outcomes
    # No need to override it with a delta distribution
    
    # Query final SFE
    print("\nCalculating final System Free Energy...")
    final_sfe_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="system_free_energy",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes"
        }
    )
    
    if final_sfe_response.response_type == "posterior":
        final_sfe_dist = final_sfe_response.content["distribution"]
        final_sfe = final_sfe_dist["distribution"]["parameters"]["mean"]
        final_current_dist = final_sfe_dist["metadata"]["current_distribution"]
        print(f"Final System Free Energy: {final_sfe:.4f}")
        if isinstance(final_current_dist, dict):
            print(f"Realized resilience distribution (delta): {pretty_print_distribution(final_current_dist)}")
        else:
            # Handle the case where it might still be a list in some contexts
            print(f"Realized resilience distribution (delta): {[f'{p:.2f}' for p in final_current_dist]}")
        print(f"Total SFE improvement from initial BAU: {(bau_sfe - final_sfe):.4f}")
    
    # Query final alignment
    print("\nCalculating final alignment score...")
    final_align_response = node_a.query_posterior(
        target_node_id=node_a.id,
        variable_name="alignment_score",
        covariates={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes"
        }
    )
    
    if final_align_response.response_type == "posterior":
        final_align_dist = final_align_response.content["distribution"]
        final_align = final_align_dist["metadata"]["alignment_value"]
        print(f"Final Alignment Score: {final_align:.2%}")
        print(f"Total alignment improvement from initial BAU: {(final_align - bau_align):.2%}")
    
    # Step 17: Summary of SFE and alignment improvements
    print_separator("Step 17: Summary of SFE and Alignment Improvements")
    print("Summary of System Free Energy (SFE) and alignment improvements:")
    print("\nSystem Free Energy (lower is better):")
    print(f"  - BAU:                   {bau_sfe:.4f}")
    print(f"  - Adaptation:            {adapt_sfe:.4f}  ({(bau_sfe - adapt_sfe):.4f} improvement)")
    print(f"  - Adaptation + Bond:     {bond_sfe:.4f}  ({(adapt_sfe - bond_sfe):.4f} improvement)")
    print(f"  - Final (Actual Result): {final_sfe:.4f}  ({(bond_sfe - final_sfe):.4f} improvement)")
    print(f"  - Total Improvement:     {(bau_sfe - final_sfe):.4f}")
    
    print("\nAlignment Score (higher is better):")
    print(f"  - BAU:                   {bau_align:.2%}")
    print(f"  - Adaptation:            {adapt_align:.2%}  ({(adapt_align - bau_align):.2%} improvement)")
    print(f"  - Adaptation + Bond:     {bond_align:.2%}  ({(bond_align - adapt_align):.2%} improvement)")
    print(f"  - Final (Actual Result): {final_align:.2%}  ({(final_align - bond_align):.2%} improvement)")
    print(f"  - Total Improvement:     {(final_align - bau_align):.2%}")
    
    print("\nConclusion: The resilience bond successfully improves alignment between")
    print("private profit goals and global climate resilience goals, as measured by")
    print("reduced System Free Energy and increased alignment score.")
    
    print_separator("Demo Complete")


if __name__ == "__main__":
    # Import Query class for the demo
    from gaia_network.query import Query
    
    main()
