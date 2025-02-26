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

from gaia_network.registry import register_node, get_node, clear_registry
from demo.model_nodes import create_demo_nodes


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
    node_a, node_b, node_c = create_demo_nodes()
    print(f"Node A (Real Estate Finance): {node_a.id}")
    print(f"Node B (Climate Risk): {node_b.id}")
    print(f"Node C (Actuarial Data): {node_c.id}")
    
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
    
    print_separator("Demo Complete")


if __name__ == "__main__":
    # Import Query class for the demo
    from gaia_network.query import Query
    
    main()
