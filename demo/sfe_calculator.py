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


def calculate_alignment_score_dict(profit_dict: Dict[float, float], resilience_dist: Dict[float, float]) -> float:
    """
    Calculate alignment score between profit distribution and resilience distribution using dictionary format.
    
    Args:
        profit_dict: Dictionary mapping resilience outcomes to profit values
        resilience_dist: Dictionary mapping resilience outcomes to probabilities
        
    Returns:
        Alignment score between 0 and 1, where 1 is perfect alignment
    """
    # Ensure both dictionaries have the same keys
    common_outcomes = sorted(set(profit_dict.keys()) & set(resilience_dist.keys()))
    
    if len(common_outcomes) < 2:
        return 0.5  # Not enough data points for correlation
    
    # Extract outcomes and profit values
    outcomes = np.array([float(outcome) for outcome in common_outcomes])
    profits = np.array([profit_dict[outcome] for outcome in common_outcomes])
    
    # Calculate Pearson correlation
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(outcomes, profits)
    
    # Map correlation from [-1, 1] to [0, 1] range
    # This is a simple linear mapping
    alignment_score = (correlation + 1) / 2
    
    # No printing here - let the calling code handle the printing if needed
    
    # Ensure the final score is in [0, 1] range
    return max(0.0, min(1.0, alignment_score))

def calculate_alignment_score(profit_dist: Any, resilience_dist: Any) -> float:
    """
    Calculate alignment score between profit distribution and resilience distribution.
    
    Args:
        profit_dist: Distribution representing profit outcomes (list, dict, or Distribution object)
        resilience_dist: Distribution representing resilience outcomes (list, dict, or Distribution object)
        
    Returns:
        Alignment score between 0 and 1, where 1 is perfect alignment
    """
    # For dictionary-based profit values and dictionary-based resilience distribution
    if isinstance(profit_dist, dict) and isinstance(resilience_dist, dict):
        return calculate_alignment_score_dict(profit_dist, resilience_dist)
    
    # For discrete distributions as lists, use weighted correlation
    elif isinstance(profit_dist, list) and isinstance(resilience_dist, list):
        # Ensure equal length
        min_len = min(len(profit_dist), len(resilience_dist))
        profit_values = profit_dist[:min_len]
        resilience_values = resilience_dist[:min_len]
        
        # Normalize profit values to [0,1] range for better comparison
        # Use min-max scaling if values are all positive or all negative
        if all(p >= 0 for p in profit_values) or all(p <= 0 for p in profit_values):
            min_profit = min(profit_values)
            max_profit = max(profit_values)
            # Avoid division by zero
            if max_profit == min_profit:
                norm_profit_values = [0.5 for _ in profit_values]
            else:
                norm_profit_values = [(p - min_profit) / (max_profit - min_profit) for p in profit_values]
        else:
            # Use sigmoid for mixed positive/negative values
            def sigmoid(x):
                return 1 / (1 + np.exp(-x * 5))
            norm_profit_values = [sigmoid(p) for p in profit_values]
        
        # Calculate weighted similarity (higher weight for better resilience outcomes)
        similarity_sum = 0
        weight_sum = 0
        
        for i in range(min_len):
            # Weight by resilience value (higher resilience = higher weight)
            weight = resilience_values[i] + 0.1  # Add small constant to avoid zero weights
            # Calculate similarity as 1 - absolute difference between normalized values
            similarity = 1 - abs(norm_profit_values[i] - resilience_values[i])
            similarity_sum += similarity * weight
            weight_sum += weight
        
        # Calculate weighted average similarity
        if weight_sum > 0:
            alignment = similarity_sum / weight_sum
        else:
            alignment = 0.5  # Default if all weights are zero
            
        return alignment
    
    # For normal distributions, use a function of the means
    elif isinstance(profit_dist, NormalDistribution) and isinstance(resilience_dist, NormalDistribution):
        # Use sigmoid function for normalization of profit mean
        # Sigmoid maps any real number to (0, 1) range in a smooth way
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # Scale the profit mean before applying sigmoid (steepness factor)
        # For ROI values, a factor of 5 gives good separation in the typical range
        norm_profit = sigmoid(profit_dist.mean * 5)
        
        # Resilience is already in [0, 1] range, but ensure it stays there
        norm_resilience = min(max(resilience_dist.mean, 0), 1)
        
        # Calculate alignment as a function of the normalized means
        alignment = 1 - abs(norm_profit - norm_resilience)
        return alignment
    
    # For beta distributions, use a function of the means
    elif isinstance(profit_dist, BetaDistribution) and isinstance(resilience_dist, BetaDistribution):
        # Calculate means
        profit_mean = profit_dist.alpha / (profit_dist.alpha + profit_dist.beta)
        resilience_mean = resilience_dist.alpha / (resilience_dist.alpha + resilience_dist.beta)
        
        # Use sigmoid function for normalization of profit mean
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # Scale the profit mean before applying sigmoid
        norm_profit = sigmoid(profit_mean * 5)
        
        # Calculate alignment as a function of the normalized means
        alignment = 1 - abs(norm_profit - resilience_mean)
        return alignment
    
    # Handle marginal distributions
    elif isinstance(profit_dist, MarginalDistribution) and isinstance(resilience_dist, MarginalDistribution):
        return calculate_alignment_score(profit_dist.distribution, resilience_dist.distribution)
    
    else:
        raise ValueError(f"Unsupported distribution types: {type(profit_dist)} and {type(resilience_dist)}")


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
