"""
System Free Energy (SFE) Calculator for the Gaia Network demo.

This module provides functions to calculate System Free Energy (SFE) as the KL divergence
between outcome distributions and target distributions, as well as alignment scores between
private profit goals and global climate resilience goals.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy
from scipy.special import kl_div
from scipy.stats import pearsonr
from typing import Dict, List, Any, Tuple, Optional, Union

from gaia_network.distribution import Distribution, NormalDistribution, BetaDistribution, MarginalDistribution


def pretty_print_distribution(distribution: Dict[float, float]) -> str:
    """
    Format a distribution dictionary as a pretty string with values ordered by outcome.
    
    Args:
        distribution: Dictionary mapping outcomes to probabilities
        
    Returns:
        A formatted string representation of the distribution
    """
    # Sort by outcome value
    sorted_items = sorted(distribution.items(), key=lambda x: float(x[0]))
    
    # Format each item as "outcome: probability"
    formatted_items = [f'{float(v):.2f}: {p:.2f}' for v, p in sorted_items]
    
    return '[' + ', '.join(formatted_items) + ']'


def calculate_kl_divergence_discrete(p_dist: List[float], q_dist: List[float]) -> float:
    """
    Calculate KL divergence between two discrete probability distributions.
    
    Args:
        p_dist: The current probability distribution
        q_dist: The target probability distribution
        
    Returns:
        The KL divergence (System Free Energy)
    """
    # Ensure distributions sum to 1
    p_dist = np.array(p_dist) / np.sum(p_dist)
    q_dist = np.array(q_dist) / np.sum(q_dist)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    p_dist = np.clip(p_dist, epsilon, 1.0)
    q_dist = np.clip(q_dist, epsilon, 1.0)
    
    # Calculate KL divergence
    return np.sum(p_dist * np.log(p_dist / q_dist))


def calculate_kl_divergence_normal(p_dist: NormalDistribution, q_dist: NormalDistribution) -> float:
    """
    Calculate KL divergence between two normal distributions.
    
    Args:
        p_dist: The current normal distribution
        q_dist: The target normal distribution
        
    Returns:
        The KL divergence (System Free Energy)
    """
    # KL divergence for normal distributions has an analytical form
    p_mean, p_var = p_dist.mean, p_dist.std**2
    q_mean, q_var = q_dist.mean, q_dist.std**2
    
    return 0.5 * (
        np.log(q_var / p_var) + 
        p_var / q_var + 
        ((p_mean - q_mean)**2) / q_var - 
        1
    )


def calculate_kl_divergence_beta(p_dist: BetaDistribution, q_dist: BetaDistribution, num_samples: int = 1000) -> float:
    """
    Calculate KL divergence between two beta distributions using numerical integration.
    
    Args:
        p_dist: The current beta distribution
        q_dist: The target beta distribution
        num_samples: Number of samples for numerical integration
        
    Returns:
        The KL divergence (System Free Energy)
    """
    # Generate samples from the support of the beta distribution
    x = np.linspace(0.001, 0.999, num_samples)
    
    # Calculate PDFs
    p_pdf = p_dist.pdf(x)
    q_pdf = q_dist.pdf(x)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    p_pdf = np.clip(p_pdf, epsilon, None)
    q_pdf = np.clip(q_pdf, epsilon, None)
    
    # Calculate KL divergence using numerical integration
    kl = np.sum(p_pdf * np.log(p_pdf / q_pdf)) * (x[1] - x[0])
    
    return kl


def calculate_kl_divergence_dict(p_dist: Dict[float, float], q_dist: Dict[float, float]) -> float:
    """
    Calculate KL divergence between two discrete probability distributions represented as dictionaries.
    
    Args:
        p_dist: The current probability distribution as a dictionary mapping outcomes to probabilities
        q_dist: The target probability distribution as a dictionary mapping outcomes to probabilities
        
    Returns:
        The KL divergence (System Free Energy)
    """
    import math
    
    # Ensure both distributions have the same keys
    all_outcomes = set(p_dist.keys()) | set(q_dist.keys())
    
    # Initialize distributions with zero probabilities for missing outcomes
    p = {outcome: p_dist.get(outcome, 0.0) for outcome in all_outcomes}
    q = {outcome: q_dist.get(outcome, 0.0) for outcome in all_outcomes}
    
    # Ensure distributions sum to 1
    p_sum = sum(p.values())
    q_sum = sum(q.values())
    
    if p_sum == 0 or q_sum == 0:
        return float('inf')  # Return infinity if either distribution sums to zero
    
    # Normalize the distributions
    p = {k: v/p_sum for k, v in p.items()}
    q = {k: v/q_sum for k, v in q.items()}
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Calculate KL divergence
    kl = 0.0
    for outcome in all_outcomes:
        p_val = max(p[outcome], epsilon)
        q_val = max(q[outcome], epsilon)
        kl += p_val * math.log(p_val / q_val)
    
    return kl

def calculate_sfe(current_dist: Any, target_dist: Any) -> float:
    """
    Calculate System Free Energy (SFE) between current and target distributions.
    
    Args:
        current_dist: The current distribution (could be discrete list, dictionary, or Distribution object)
        target_dist: The target distribution (could be discrete list, dictionary, or Distribution object)
        
    Returns:
        The System Free Energy (KL divergence)
    """
    # Handle dictionary-based probability distributions
    if isinstance(current_dist, dict) and isinstance(target_dist, dict):
        return calculate_kl_divergence_dict(current_dist, target_dist)
    
    # Handle discrete probability distributions as lists
    elif isinstance(current_dist, list) and isinstance(target_dist, list):
        return calculate_kl_divergence_discrete(current_dist, target_dist)
    
    # Handle normal distributions
    elif isinstance(current_dist, NormalDistribution) and isinstance(target_dist, NormalDistribution):
        return calculate_kl_divergence_normal(current_dist, target_dist)
    
    # Handle beta distributions
    elif isinstance(current_dist, BetaDistribution) and isinstance(target_dist, BetaDistribution):
        return calculate_kl_divergence_beta(current_dist, target_dist)
    
    # Handle marginal distributions
    elif isinstance(current_dist, MarginalDistribution) and isinstance(target_dist, MarginalDistribution):
        return calculate_sfe(current_dist.distribution, target_dist.distribution)
    
    else:
        raise ValueError(f"Unsupported distribution types: {type(current_dist)} and {type(target_dist)}")


def calculate_alignment_score(profit_dist: Dict[float, float], resilience_dist: Dict[float, float]) -> float:
    """
    Calculate alignment score between profit distribution and resilience distribution using dictionary format.
    
    The alignment score measures how well the profit incentives align with better resilience outcomes.
    A score of 1.0 means perfect alignment (profit-maximizing behavior leads to best resilience outcomes).
    A score of 0.0 means no alignment (profit-maximizing behavior leads to worst resilience outcomes).
    
    Args:
        profit_dist: Dictionary mapping resilience outcomes to profit values
        resilience_dist: Dictionary mapping resilience outcomes to probabilities
        
    Returns:
        Alignment score between 0 and 1, where 1 is perfect alignment
    """
    
    # Check if we have additional metadata for economic incentive calculation
    if isinstance(resilience_dist, dict) and "metadata" in resilience_dist:
        metadata = resilience_dist["metadata"]
        if "adaptation_strategy" in metadata and "bau_roi" in metadata and "adaptation_roi" in metadata:
            # Use economic incentive alignment based on marginal ROI comparison
            bau_roi = metadata["bau_roi"]
            adaptation_roi = metadata["adaptation_roi"]
            
            # Calculate the alignment score based on the ROI difference
            # Using sigmoid function for a smoother transition between no alignment and perfect alignment
            roi_diff = adaptation_roi - bau_roi
            
            # Sigmoid function: 1 / (1 + e^(-k*x))
            # k controls the steepness of the transition
            # For our specific ROI differences, we need a much smaller k value
            # to get intermediate values between 0 and 1
            def sigmoid(x, k=50):
                import math
                return 1.0 / (1.0 + math.exp(-k * x))
            
            # Apply sigmoid to the ROI difference
            alignment = sigmoid(roi_diff)
            
            return alignment
    
    return 0.5

def format_sfe_results(sfe: float, alignment: float) -> str:
    """
    Format SFE and alignment results for display.
    
    Args:
        sfe: System Free Energy value
        alignment: Alignment score
        
    Returns:
        Formatted string for display
    """
    return (
        f"System Free Energy (SFE): {sfe:.4f}\n"
        f"Alignment Score: {alignment:.2%}\n"
    )


def visualize_distributions(current_dist: List[float], target_dist: List[float], 
                           labels: List[str], title: str = "Distribution Comparison") -> None:
    """
    Create a simple text-based visualization of distributions.
    
    Args:
        current_dist: Current probability distribution
        target_dist: Target probability distribution
        labels: Labels for the distribution values
        title: Title for the visualization
    """
    result = f"\n{title}\n" + "=" * len(title) + "\n\n"
    result += f"{'Outcome':<15} {'Current':<10} {'Target':<10} {'Gap':<10}\n"
    result += "-" * 45 + "\n"
    
    for i, label in enumerate(labels):
        current = current_dist[i]
        target = target_dist[i]
        gap = target - current
        result += f"{label:<15} {current:.2%}      {target:.2%}      {gap:+.2%}\n"
    
    print(result)


# Define a modified sigmoid function for alignment score calculation
def modified_sigmoid(x, k=10):
    """
    A modified sigmoid function for calculating alignment scores based on ROI differences.
    
    Args:
        x: The input value (typically the ROI difference between Adaptation and BAU)
        k: The steepness parameter (smaller values give a smoother transition)
        
    Returns:
        A value between 0 and 1 representing the alignment score
    """
    import math
    return 1.0 / (1.0 + math.exp(-k * x))


# Define global target distribution for climate resilience
# This represents the ideal distribution of resilience outcomes
# Higher values for better resilience outcomes - prioritizing high resilience
TARGET_RESILIENCE_DISTRIBUTION = {
    # Global target distribution that prioritizes high resilience outcomes
    # The same target is used for all adaptation strategies as it represents
    # the ideal state we want to achieve regardless of the strategy
    0.2: 0.05,  # 5% probability for 0.2 resilience outcome (low resilience)
    0.3: 0.05,  # 5% probability for 0.3 resilience outcome (low resilience)
    0.4: 0.10,  # 10% probability for 0.4 resilience outcome (medium-low resilience)
    0.6: 0.15,  # 15% probability for 0.6 resilience outcome (medium resilience)
    0.7: 0.25,  # 25% probability for 0.7 resilience outcome (medium-high resilience)
    0.9: 0.40   # 40% probability for 0.9 resilience outcome (high resilience)
}
