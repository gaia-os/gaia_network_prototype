"""
Plots results from Experiment 1 (Trust Modeling) saved in a pickle file.

Usage:
  python -m experiments.plot_results <path_to_results.pkl>
"""

import argparse
import pickle as pkl
import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from experiments.sim_params import SimParams

# Set plot style
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib') # Ignore tight_layout warning


def load_results(filepath: str) -> tuple[SimParams, list[dict]]:
    """Loads parameters and results list from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            data = pkl.load(f)
        if not isinstance(data, dict) or 'params' not in data or 'results' not in data:
            raise ValueError("Pickle file does not contain expected 'params' and 'results' keys.")
        params = data['params']
        results_list = data['results']
        if not isinstance(results_list, list):
             raise ValueError("'results' key does not contain a list.")
        print(f"Successfully loaded results from: {filepath}")
        print(f"Parameters used: Pooling={params.pool_trust}, Rounds={params.num_rounds}, Budget={params.customer_query_budget}")
        print(f"Results list contains {len(results_list)} records.")
        return params, results_list
    except Exception as e:
        print(f"Error loading pickle file {filepath}: {e}")
        raise


def plot_trust_evolution(results_df: pd.DataFrame, params: SimParams, output_dir: str = "."):
    """Plots the average trust mixture weight (w) over rounds for providers."""
    # Use 'trust_w_before' from the per-query records
    if 'trust_w_before' not in results_df.columns or 'query_in_round' not in results_df.columns:
        print("Required columns ('trust_w_before', 'query_in_round') not found. Skipping trust plot.")
        return

    # Filter out end-of-round markers and entries where trust wasn't applicable (e.g., random selection before first query)
    plot_data = results_df[(results_df['query_in_round'] != -1) & results_df['trust_w_before'].notna()].copy()

    if plot_data.empty:
        print("No valid per-query trust data found. Skipping trust evolution plot.")
        return

    # Calculate mean 'w' per round, provider, modality
    trust_evo = plot_data.groupby(['round', 'provider_id', 'modality_id'])['trust_w_before'].mean().reset_index()
    num_providers = len(trust_evo['provider_id'].unique())

    if num_providers == 0:
        print("No provider data found for trust plot.")
        return

    # Determine grid size (aim for roughly square, max columns ~4)
    max_cols = 4
    num_cols = min(num_providers, max_cols)
    num_rows = (num_providers + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False, sharey=True)
    axes_flat = axes.flatten()

    provider_ids = sorted(trust_evo['provider_id'].unique())

    for i, provider_id in enumerate(provider_ids):
        ax = axes_flat[i]
        provider_data = trust_evo[trust_evo['provider_id'] == provider_id]

        # Get modality for this provider (assuming one per provider)
        modality = params.provider_assignments.get(provider_id) # Check SimParams for correct structure if needed
        if not modality and 'modality_id' in provider_data.columns:
             modality = provider_data['modality_id'].iloc[0]

        sns.lineplot(data=provider_data, x='round', y='trust_w_before', ax=ax, marker='.', errorbar=None) # errorbar=None if mean is already calculated
        ax.set_title(f"{provider_id} ({modality})", fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Trust Weight (w)")
        ax.set_ylim(0, 1.05) # Set y-axis limits

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle("Evolution of Mean Trust Mixture Weight (w)", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout

    # Save plot
    pool_suffix = "pooled" if params.pool_trust else "unpooled"
    filename = os.path.join(output_dir, f"trust_evolution_{pool_suffix}.png")
    plt.savefig(filename, dpi=300)
    print(f"Saved trust evolution plot to: {filename}")
    # plt.show() # Optionally show plot


def plot_belief_accuracy(results_df: pd.DataFrame, params: SimParams, output_dir: str = "."):
    """Plots the average belief probability assigned to the true latent state."""
    # Use end-of-round records for final belief accuracy
    required_cols = ['round', 'customer_id', 'true_latent_state', 'belief_after', 'query_in_round']
    if not all(col in results_df.columns for col in required_cols):
        print(f"Missing one or more required columns for belief accuracy plot: {required_cols}. Skipping.")
        return

    # Filter for end-of-round records
    plot_data = results_df[results_df['query_in_round'] == -1].copy()

    if plot_data.empty:
        print("No end-of-round belief data found. Skipping belief accuracy plot.")
        return

    if 'latent_states' not in params.__dict__:
        print("'latent_states' not found in SimParams. Cannot calculate belief accuracy. Skipping.")
        return

    # Determine the belief column corresponding to the true state for each row
    def get_true_belief(row):
        try:
            true_state_idx = params.latent_states.index(row['true_latent_state'])
            # belief_after is the full numpy array
            belief_array = row['belief_after']
            if isinstance(belief_array, np.ndarray) and belief_array.ndim == 1 and true_state_idx < len(belief_array):
                 return belief_array[true_state_idx]
            else:
                return np.nan # Indicate error or missing data
        except (ValueError, IndexError, TypeError):
            return np.nan # Error if state not found or belief format is wrong

    plot_data['belief_in_true_state'] = plot_data.apply(get_true_belief, axis=1)

    # Calculate mean accuracy per round
    belief_acc = plot_data.groupby('round')['belief_in_true_state'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=belief_acc, x='round', y='belief_in_true_state', marker='.')
    plt.title(f"Mean Belief Probability Assigned to True State (Pooling: {params.pool_trust})")
    plt.xlabel("Round")
    plt.ylabel("Mean P(True State)")
    plt.ylim(0, 1.05)

    # Save plot
    pool_suffix = "pooled" if params.pool_trust else "unpooled"
    filename = os.path.join(output_dir, f"belief_accuracy_{pool_suffix}.png")
    plt.savefig(filename, dpi=300)
    print(f"Saved belief accuracy plot to: {filename}")
    # plt.show() # Optionally show plot


def main():
    parser = argparse.ArgumentParser(description="Plot results from Experiment 1 (Trust Modeling).")
    parser.add_argument("results_file", help="Path to the .pkl file containing simulation results.")
    parser.add_argument("-o", "--output_dir", default="plots", help="Directory to save the plots.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        params, results_list = load_results(args.results_file)

        # Convert list to DataFrame for plotting
        results_df = pd.DataFrame(results_list)
        print(f"Converted results list to DataFrame with shape: {results_df.shape}")

        # Generate plots
        plot_trust_evolution(results_df, params, args.output_dir)
        plot_belief_accuracy(results_df, params, args.output_dir)

        print(f"\nPlots saved in directory: {args.output_dir}")

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"\nError during plotting: {e}")
        # Optional: exit with error code
        # sys.exit(1)

if __name__ == "__main__":
    main()
