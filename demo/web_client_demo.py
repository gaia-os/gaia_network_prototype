#!/usr/bin/env python3
"""
Gaia Network Web Client Demo

This script demonstrates how to interact with the Gaia Network nodes
exposed as web services. It follows a similar flow to the original demo
but uses HTTP requests instead of direct method calls.

Each node is exposed as a separate web service on a different port:
- Node A (Real Estate Finance): http://127.0.0.1:8011
- Node B (Climate Risk): http://127.0.0.1:8012
- Node C (Actuarial Data): http://127.0.0.1:8013
"""

import json
import requests
import time
from pprint import pprint


def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def main():
    """Run the Gaia Network web client demo."""
    # Base URLs for each node
    node_a_url = "http://127.0.0.1:8011"
    node_b_url = "http://127.0.0.1:8012"
    node_c_url = "http://127.0.0.1:8013"
    node_d_url = "http://127.0.0.1:8014"
    
    print_separator("Gaia Network Web Client Demo")
    
    # Get information about the nodes
    print("Getting node information...")
    node_a_info = requests.get(f"{node_a_url}/info").json()
    node_b_info = requests.get(f"{node_b_url}/info").json()
    node_c_info = requests.get(f"{node_c_url}/info").json()
    node_d_info = requests.get(f"{node_d_url}/info").json()
    
    print(f"Node A (Real Estate Finance): {node_a_info['id']}")
    print(f"Node B (Climate Risk): {node_b_info['id']}")
    print(f"Node C (Actuarial Data): {node_c_info['id']}")
    print(f"Node D (Resilience Bond): {node_d_info['id']}")
    
    # Step 1: Set location and IPCC scenario
    print_separator("Step 1: Set location and IPCC scenario")
    location = "Miami"
    ipcc_scenario = "SSP2-4.5"
    print(f"Location: {location}")
    print(f"IPCC Scenario: {ipcc_scenario}")
    
    # Step 2: Get Node B's schema
    print_separator("Step 2: Get Node B's schema")
    schema_response = requests.get(f"{node_b_url}/schema").json()
    
    print("\nLatent variables:")
    for var in schema_response["latents"]:
        print(f"  - {var['name']}: {var['description']}")
    
    print("\nObservable variables:")
    for var in schema_response["observables"]:
        print(f"  - {var['name']}: {var['description']}")
    
    print("\nCovariates:")
    for var in schema_response["covariates"]:
        print(f"  - {var['name']}: {var['description']}")
    
    # Step 3: Query Node B for flood probability
    print_separator("Step 3: Query Node B for flood probability")
    print(f"Querying flood probability for {location} under {ipcc_scenario} scenario...")
    
    flood_response = requests.post(
        f"{node_b_url}/query/flood_probability",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario
        }
    ).json()
    
    print("Raw flood response:", flood_response)
    
    print("Response type:", flood_response["response_type"])
    
    if flood_response["response_type"] == "posterior":
        distribution_data = flood_response["content"]["distribution"]
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
    
    # Step 4: Query Node A for ROI
    print_separator("Step 4: Query Node A for ROI")
    print("Calculating expected ROI for the real estate project...")
    
    roi_response = requests.post(
        f"{node_a_url}/query/expected_roi",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "BAU",
            "include_bond": "no"
        }
    ).json()
    
    if roi_response["response_type"] == "posterior":
        roi_dist_data = roi_response["content"]["distribution"]
        print("\nROI distribution:")
        print(f"  - Name: {roi_dist_data['name']}")
        print(f"  - Type: {roi_dist_data['distribution']['type']}")
        print(f"  - Parameters: {roi_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {roi_dist_data['metadata']}")
        
        expected_roi = roi_dist_data['distribution']['parameters']['mean']
        print(f"\nExpected ROI: {expected_roi:.2%}")
    
    # Step 5: Add new data to Node C
    print_separator("Step 5: Add new data to Node C")
    print("Adding new actuarial data to Node C...")
    
    add_data_response = requests.post(
        f"{node_c_url}/add-data",
        json={
            "location": location,
            "value": 0.4
        }
    ).json()
    
    print("Status:", add_data_response.get("status", "success"))
    
    # Step 6: Query Node C for updated data
    print_separator("Step 6: Query Node C for updated data")
    print("Querying Node C for updated actuarial data...")
    
    actuarial_response = requests.post(
        f"{node_c_url}/query/historical_flood_data",
        json={
            "location": location
        }
    ).json()
    
    if actuarial_response["response_type"] == "posterior":
        actuarial_dist_data = actuarial_response["content"]["distribution"]
        print("\nActuarial data distribution:")
        print(f"  - Name: {actuarial_dist_data['name']}")
        print(f"  - Type: {actuarial_dist_data['distribution']['type']}")
        print(f"  - Parameters: {actuarial_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {actuarial_dist_data['metadata']}")
    
    # Step 7: Update Node B with new data
    print_separator("Step 7: Update Node B with new data")
    print("Updating Node B with new actuarial data...")
    
    update_response = requests.post(
        f"{node_b_url}/update",
        json={
            "location": location
        }
    ).json()
    
    # Handle the update response safely with error checking
    if "response_type" in update_response and "content" in update_response:
        print("Response type:", update_response["response_type"])
        print("Update status:", update_response["content"]["status"])
    else:
        print("Update completed with response:", update_response)
    
    # Step 8: Query Node B again for updated flood probability
    print_separator("Step 8: Query Node B again for updated flood probability")
    print("Querying updated flood probability for Miami under SSP2-4.5 scenario...")
    
    updated_flood_response = requests.post(
        f"{node_b_url}/query/flood_probability",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario
        }
    ).json()
    
    if updated_flood_response["response_type"] == "posterior":
        updated_dist_data = updated_flood_response["content"]["distribution"]
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
        print(f"Change in flood probability: {updated_expected_value - expected_value:.4f}")
    
    # Step 9: Query Node B with rationale
    print_separator("Step 9: Query Node B with rationale")
    print("Querying flood probability with rationale for Miami under SSP2-4.5 scenario...")
    
    rationale_response = requests.post(
        f"{node_b_url}/query/flood_probability",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "rationale": True
        }
    ).json()
    
    if rationale_response["response_type"] == "posterior":
        rationale_dist_data = rationale_response["content"]["distribution"]
        print("\nFlood probability distribution with rationale:")
        print(f"  - Name: {rationale_dist_data['name']}")
        print(f"  - Type: {rationale_dist_data['distribution']['type']}")
        print(f"  - Parameters: {rationale_dist_data['distribution']['parameters']}")
        
        print("\nRationale:")
        rationale = rationale_response["content"].get("rationale", {})
        
        print("\nCalculation details:")
        calculation = rationale.get("calculation", {})
        for key, value in calculation.items():
            print(f"  - {key}: {value}")
        
        print("\nObservations that caused the update:")
        observations = rationale.get("observations", [])
        for obs in observations:
            print(f"  - {obs['variable_name']}: {obs['value']} (timestamp: {obs['timestamp']})")
    
    # Step 10: Query Node A for updated ROI
    print_separator("Step 10: Query Node A for updated ROI")
    print("Recalculating expected ROI for the real estate project...")
    
    updated_roi_response = requests.post(
        f"{node_a_url}/query/expected_roi",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "BAU",
            "include_bond": "no"
        }
    ).json()
    
    if updated_roi_response["response_type"] == "posterior":
        updated_roi_dist_data = updated_roi_response["content"]["distribution"]
        print("\nUpdated ROI distribution:")
        print(f"  - Name: {updated_roi_dist_data['name']}")
        print(f"  - Type: {updated_roi_dist_data['distribution']['type']}")
        print(f"  - Parameters: {updated_roi_dist_data['distribution']['parameters']}")
        print(f"  - Metadata: {updated_roi_dist_data['metadata']}")
        
        updated_expected_roi = updated_roi_dist_data['distribution']['parameters']['mean']
        print(f"\nUpdated expected ROI: {updated_expected_roi:.2%}")
        print(f"Change in ROI: {updated_expected_roi - expected_roi:.2%}")
    
    # Step 11: Compare ROI for BAU vs Adaptation (without bond)
    print_separator("Step 11: Compare ROI for BAU vs Adaptation (without bond)")
    
    # Calculate ROI for BAU
    print("\nCalculating ROI for Business-as-Usual (BAU) strategy...")
    bau_roi_response = requests.post(
        f"{node_a_url}/query/expected_roi",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "BAU",
            "include_bond": "no"
        }
    ).json()
    
    if bau_roi_response["response_type"] == "posterior":
        bau_roi_dist = bau_roi_response["content"]["distribution"]
        bau_roi = bau_roi_dist["distribution"]["parameters"]["mean"]
        print(f"BAU Expected ROI: {bau_roi:.2%}")
    
    # Calculate ROI for Adaptation
    print("\nCalculating ROI for Climate Adaptation strategy...")
    adapt_roi_response = requests.post(
        f"{node_a_url}/query/expected_roi",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "no"
        }
    ).json()
    
    if adapt_roi_response["response_type"] == "posterior":
        adapt_roi_dist = adapt_roi_response["content"]["distribution"]
        adapt_roi = adapt_roi_dist["distribution"]["parameters"]["mean"]
        print(f"Adaptation Expected ROI: {adapt_roi:.2%}")
        print(f"\nInitial ROI difference (Adaptation - BAU): {adapt_roi - bau_roi:.2%}")
    
    # Step 12: Consider resilience bond effects for Adaptation strategy
    print_separator("Step 12: Consider resilience bond effects for Adaptation strategy")
    print("\nCalculating ROI for Adaptation strategy with resilience bond...")
    
    # Calculate ROI with bond effects
    bond_roi_response = requests.post(
        f"{node_a_url}/query/expected_roi",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes"
        }
    ).json()
    
    if bond_roi_response["response_type"] == "posterior":
        bond_roi_dist = bond_roi_response["content"]["distribution"]
        bond_roi = bond_roi_dist["distribution"]["parameters"]["mean"]
        metadata = bond_roi_dist["metadata"]
        
        print(f"\nAdaptation + Bond Expected ROI: {bond_roi:.2%}")
        print(f"Bond price: {metadata['bond_price']:.2%}")
        print(f"Expected bond payoff: {metadata['expected_bond_payoff']:.2%}")
        print("\nResilient outcomes and probabilities:")
        
        # Get the resilience distribution from the metadata
        if 'resilience_distribution' in metadata and 'Adaptation' in metadata['resilience_distribution']:
            resilience_dist = metadata['resilience_distribution']['Adaptation']
            for outcome, prob in resilience_dist.items():
                print(f"  - Outcome {float(outcome):.2f}: {prob:.0%} probability")
        
        print(f"\nROI improvement from bond: {bond_roi - adapt_roi:.2%}")
        print(f"Final ROI difference (Adaptation + Bond - BAU): {bond_roi - bau_roi:.2%}")
    
    # Step 13: Simulate time passing and project development
    print_separator("Step 13: Simulate time passing and project development")
    print("\nTime passes, the project is developed...")
    
    # Add new actuarial data
    print("\nNode C receives new actuarial data...")
    add_data_response = requests.post(
        f"{node_c_url}/add-data",
        json={
            "location": location,
            "value": 0.4
        }
    ).json()
    print("New actuarial data added to Node C")
    
    # Query Node D for actual resilience outcome
    print("\nNode D queries Node C for actuarial data to determine actual resilience outcome...")
    actual_resilience = 0.70  # This would normally come from Node C
    print(f"Actual resilience outcome achieved: {actual_resilience:.2f}")
    
    # Calculate final ROI with actual bond payoff
    print("\nCalculating final project ROI with actual bond payoff...")
    final_roi_response = requests.post(
        f"{node_a_url}/query/expected_roi",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes",
            "actual_resilience": actual_resilience
        }
    ).json()
    
    if final_roi_response["response_type"] == "posterior":
        final_roi_dist = final_roi_response["content"]["distribution"]
        final_roi = final_roi_dist["distribution"]["parameters"]["mean"]
        print(f"Final project ROI: {final_roi:.2%}")
        print(f"Improvement over initial BAU ROI: {final_roi - bau_roi:.2%}")
    
    # Step 14: Calculate System Free Energy (SFE)
    print_separator("Step 14: Calculate System Free Energy (SFE)")
    print("Calculating System Free Energy (SFE) to measure divergence from target resilience distribution...")
    
    # Query SFE for BAU
    print("\nCalculating SFE for Business-as-Usual (BAU) strategy...")
    bau_sfe_response = requests.post(
        f"{node_a_url}/query/system_free_energy",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "BAU",
            "include_bond": "no"
        }
    ).json()
    
    if bau_sfe_response["response_type"] == "posterior":
        bau_sfe_dist = bau_sfe_response["content"]["distribution"]
        bau_sfe = bau_sfe_dist["distribution"]["parameters"]["mean"]
        bau_current_dist = bau_sfe_dist["metadata"]["current_distribution"]
        bau_target_dist = bau_sfe_dist["metadata"]["target_distribution"]
        print(f"BAU System Free Energy: {bau_sfe:.4f}")
        print("\nBAU Resilience Distribution:")
        print(f"  - Current: {bau_current_dist}")
        print(f"  - Target:  {bau_target_dist}")
    
    # Query SFE for Adaptation
    print("\nCalculating SFE for Climate Adaptation strategy...")
    adapt_sfe_response = requests.post(
        f"{node_a_url}/query/system_free_energy",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "no"
        }
    ).json()
    
    if adapt_sfe_response["response_type"] == "posterior":
        adapt_sfe_dist = adapt_sfe_response["content"]["distribution"]
        adapt_sfe = adapt_sfe_dist["distribution"]["parameters"]["mean"]
        adapt_current_dist = adapt_sfe_dist["metadata"]["current_distribution"]
        adapt_target_dist = adapt_sfe_dist["metadata"]["target_distribution"]
        print(f"Adaptation System Free Energy: {adapt_sfe:.4f}")
        print("\nAdaptation Resilience Distribution:")
        print(f"  - Current: {adapt_current_dist}")
        print(f"  - Target:  {adapt_target_dist}")
    
    # Query SFE for Adaptation with bond
    print("\nCalculating SFE for Adaptation strategy with resilience bond...")
    bond_sfe_response = requests.post(
        f"{node_a_url}/query/system_free_energy",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes"
        }
    ).json()
    
    if bond_sfe_response["response_type"] == "posterior":
        bond_sfe_dist = bond_sfe_response["content"]["distribution"]
        bond_sfe = bond_sfe_dist["distribution"]["parameters"]["mean"]
        bond_current_dist = bond_sfe_dist["metadata"]["current_distribution"]
        bond_target_dist = bond_sfe_dist["metadata"]["target_distribution"]
        print(f"Adaptation + Bond System Free Energy: {bond_sfe:.4f}")
        print(f"SFE improvement from bond: {(adapt_sfe - bond_sfe):.4f}")
        print("\nAdaptation + Bond Resilience Distribution:")
        print(f"  - Current: {bond_current_dist}")
        print(f"  - Target:  {bond_target_dist}")
    
    # Step 15: Calculate Alignment Scores
    print_separator("Step 15: Calculate Alignment Scores")
    print("Calculating economic incentive alignment between profit goals and climate resilience goals...")
    
    # Query alignment for Adaptation without bond
    print("\nCalculating economic incentive alignment without resilience bond...")
    no_bond_align_response = requests.post(
        f"{node_a_url}/query/alignment_score",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "no"
        }
    ).json()
    
    if "alignment_score" in no_bond_align_response["content"]:
        no_bond_align = no_bond_align_response["content"]["alignment_score"]
        bau_roi = no_bond_align_response["content"]["metadata"]["bau_roi"]
        adaptation_roi = no_bond_align_response["content"]["metadata"]["adaptation_roi"]
        print(f"Economic Incentive Alignment (without bond): {no_bond_align:.2%}")
        print(f"BAU ROI: {bau_roi:.4f}")
        print(f"Adaptation ROI (without bond): {adaptation_roi:.4f}")
        print(f"ROI difference (Adaptation - BAU): {adaptation_roi - bau_roi:.4f}")
    
    # Query alignment for Adaptation with bond
    print("\nCalculating economic incentive alignment with resilience bond...")
    with_bond_align_response = requests.post(
        f"{node_a_url}/query/alignment_score",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes"
        }
    ).json()
    
    if "alignment_score" in with_bond_align_response["content"]:
        with_bond_align = with_bond_align_response["content"]["alignment_score"]
        bau_roi = with_bond_align_response["content"]["metadata"]["bau_roi"]
        adaptation_roi = with_bond_align_response["content"]["metadata"]["adaptation_roi"]
        print(f"Economic Incentive Alignment (with bond): {with_bond_align:.2%}")
        print(f"BAU ROI: {bau_roi:.4f}")
        print(f"Adaptation ROI (with bond): {adaptation_roi:.4f}")
        print(f"ROI difference (Adaptation - BAU): {adaptation_roi - bau_roi:.4f}")
        print(f"Alignment improvement from bond: {(with_bond_align - no_bond_align):.2%}")
    
    # Step 16: Calculate final SFE and alignment after project development
    print_separator("Step 16: Calculate Final SFE and Alignment After Project Development")
    print("Calculating final SFE and alignment scores based on actual resilience outcome...")
    
    print(f"\nActual resilience outcome: {actual_resilience:.2f}")
    
    # Query final SFE
    print("\nCalculating final System Free Energy...")
    final_sfe_response = requests.post(
        f"{node_a_url}/query/system_free_energy",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes",
            "actual_resilience": actual_resilience
        }
    ).json()
    
    if final_sfe_response["response_type"] == "posterior":
        final_sfe_dist = final_sfe_response["content"]["distribution"]
        final_sfe = final_sfe_dist["distribution"]["parameters"]["mean"]
        final_current_dist = final_sfe_dist["metadata"]["current_distribution"]
        print(f"Final System Free Energy: {final_sfe:.4f}")
        print(f"Realized resilience distribution (delta): {final_current_dist}")
        print(f"Total SFE improvement from initial BAU: {(bau_sfe - final_sfe):.4f}")
    
    # Query final alignment
    print("\nCalculating final economic incentive alignment...")
    final_align_response = requests.post(
        f"{node_a_url}/query/alignment_score",
        json={
            "location": location,
            "ipcc_scenario": ipcc_scenario,
            "adaptation_strategy": "Adaptation",
            "include_bond": "yes",
            "actual_resilience": actual_resilience
        }
    ).json()
    
    if "alignment_score" in final_align_response["content"]:
        final_align = final_align_response["content"]["alignment_score"]
        bau_roi = final_align_response["content"]["metadata"]["bau_roi"]
        adaptation_roi = final_align_response["content"]["metadata"]["adaptation_roi"]
        print(f"Final Economic Incentive Alignment: {final_align:.2%}")
        print(f"BAU ROI: {bau_roi:.4f}")
        print(f"Adaptation ROI (with bond): {adaptation_roi:.4f}")
        print(f"ROI difference (Adaptation - BAU): {adaptation_roi - bau_roi:.4f}")
        print(f"Total alignment improvement from initial state: {(final_align - no_bond_align):.2%}")
    
    # Step 17: Summary of SFE and alignment improvements
    print_separator("Step 17: Summary of SFE and Alignment Improvements")
    print("Summary of System Free Energy (SFE) and alignment improvements:")
    print("\nSystem Free Energy (lower is better):")
    print(f"  - BAU:                   {bau_sfe:.4f}")
    print(f"  - Adaptation:            {adapt_sfe:.4f}  ({(bau_sfe - adapt_sfe):.4f} improvement)")
    print(f"  - Adaptation + Bond:     {bond_sfe:.4f}  ({(adapt_sfe - bond_sfe):.4f} improvement)")
    print(f"  - Final (Actual Result): {final_sfe:.4f}  ({(bond_sfe - final_sfe):.4f} improvement)")
    print(f"  - Total Improvement:     {(bau_sfe - final_sfe):.4f}")
    
    print("\nEconomic Incentive Alignment (higher is better):")
    print(f"  - Without Bond:          {no_bond_align:.2%}")
    print(f"  - With Bond:             {with_bond_align:.2%}  ({(with_bond_align - no_bond_align):.2%} improvement)")
    print(f"  - Final (Actual Result): {final_align:.2%}  ({(final_align - with_bond_align):.2%} improvement)")
    print(f"  - Total Improvement:     {(final_align - no_bond_align):.2%}")

    print_separator("Demo Complete")


if __name__ == "__main__":
    main()
