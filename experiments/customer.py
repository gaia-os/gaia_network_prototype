import numpy as np
from scipy.special import betaln, xlogy # xlogy for entropy calculation 0*log(0)=0
import logging
import copy # For deep copying beliefs during EIG calculation

from .sim_params import SimParams

# Helper function for Shannon entropy
def entropy(prob_vector: np.ndarray) -> float:
    """Calculates Shannon entropy H(P) = -sum(p_i * log(p_i))."""
    # Use np.where to handle p=0 cases gracefully (0 * log(0) = 0)
    return -np.sum(xlogy(prob_vector, prob_vector))
    # Alternative using masked array, potentially slower:
    # log_p = np.log(np.ma.masked_equal(prob_vector, 0))
    # return -np.sum(prob_vector * log_p)


class Customer:
    """
    Represents a customer/decision-maker in the simulation.
    Customers query providers for observations, update beliefs about a latent state,
    and update trust in providers based on observed accuracy (evaluated against their own model).
    """
    def __init__(self, customer_id: str, params: SimParams):
        self.id = customer_id
        self.params = params
        # Customer's own likelihood models: {modality: np.array P_i(X|Z)}
        self.likelihood_models = params.customer_likelihood_models[customer_id]

        # Trust parameters (Beta mixture): {(provider_id, modality_id): {'alpha': float, 'beta': float, 'w': float}}
        self.trust_params = {}
        for modality, provider_ids in params.provider_assignments.items():
            for provider_id in provider_ids:
                key = (provider_id, modality)
                self.trust_params[key] = {
                    'alpha': params.trust_prior_alpha,
                    'beta': params.trust_prior_beta,
                    'w': params.trust_prior_weight
                }

        # Beliefs about latent state P(Z) - initialized uniformly or from params
        num_latent_states = len(params.latent_states)
        # Ensure it's float dtype for calculations
        self.latent_belief = np.ones(num_latent_states, dtype=float) / num_latent_states

        # Logging setup
        self.log = logging.getLogger(f"Customer_{self.id}")
        self.log.setLevel(params.log_level.upper())
        # Basic handler if none exists, configure globally if needed
        if not self.log.handlers:
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             self.log.addHandler(handler)
             self.log.propagate = False # Avoid duplicate logs if root logger is configured


    def reset_belief(self):
        """Resets the latent belief to a uniform prior for a new round."""
        num_latent_states = len(self.params.latent_states)
        # Ensure it's float dtype for calculations
        self.latent_belief = np.ones(num_latent_states, dtype=float) / num_latent_states
        self.log.debug("Latent belief reset to uniform.")


    def get_expected_reliability(self, provider_id: str, modality_id: str) -> float:
        """Calculates the expected reliability based on the Beta mixture model."""
        trust_key = (provider_id, modality_id)
        if trust_key not in self.trust_params:
            self.log.warning(f"No trust params found for {trust_key}. Returning default 0.5")
            return 0.5

        trust = self.trust_params[trust_key]
        alpha_norm, beta_norm, w_norm = trust['alpha'], trust['beta'], trust['w']
        alpha_dec = self.params.trust_deceptive_alpha
        beta_dec = self.params.trust_deceptive_beta

        mean_norm = alpha_norm / (alpha_norm + beta_norm) if (alpha_norm + beta_norm) > 1e-9 else 0.5
        mean_dec = alpha_dec / (alpha_dec + beta_dec) if (alpha_dec + beta_dec) > 1e-9 else 0.5

        expected_reliability = w_norm * mean_norm + (1 - w_norm) * mean_dec
        self.log.debug(f"Expected reliability for {trust_key}: {expected_reliability:.3f} (w={w_norm:.3f}, N={alpha_norm:.1f}/{beta_norm:.1f}, D={alpha_dec:.1f}/{beta_dec:.1f})")
        return expected_reliability

    def update_trust(self, provider_id: str, modality_id: str, observation: int, true_latent_state_idx: int, prior_params: dict | None = None) -> dict | None:
        """
        Updates trust parameters for a provider based on observation correctness,
        evaluated using the customer's own likelihood model.

        Args:
            provider_id: The ID of the provider.
            modality_id: The ID of the modality observed.
            observation: The observed value index.
            true_latent_state_idx: The index of the true latent state Z.
            prior_params: Dictionary containing pooled {'alpha': ..., 'beta': ..., 'w': ...}
                          from the start of the round if pooling is enabled. If None,
                          pooling is disabled, and customer's own prior is used.

        Returns:
             A dictionary containing the updated {'alpha': ..., 'beta': ..., 'w': ...}
             for this customer, provider, and modality, or None if update fails.
        """
        trust_key = (provider_id, modality_id)
        if trust_key not in self.trust_params:
            self.log.warning(f"Trust parameters for {trust_key} not initialized. Skipping update.")
            return None

        # 1. Determine Correctness based on Customer's Model
        customer_likelihood_matrix = self.likelihood_models.get(modality_id)
        if customer_likelihood_matrix is None:
            self.log.error(f"Missing likelihood model for modality {modality_id}. Cannot update trust.")
            return self.trust_params[trust_key] # Return current params

        # Validate indices
        num_states, num_obs = customer_likelihood_matrix.shape
        if not (0 <= observation < num_obs):
             self.log.error(f"Invalid observation index {observation} for modality {modality_id} with {num_obs} possible values. Skipping update.")
             return self.trust_params[trust_key]
        if not (0 <= true_latent_state_idx < num_states):
             self.log.error(f"Invalid true_latent_state_idx {true_latent_state_idx} for modality {modality_id} with {num_states} states. Skipping update.")
             return self.trust_params[trust_key]

        # L_i_true = P_i(X_m=observation | Z=true_latent_state)
        likelihood_of_observation_given_true_state = customer_likelihood_matrix[true_latent_state_idx, observation]
        delta_alpha = likelihood_of_observation_given_true_state
        delta_beta = 1.0 - likelihood_of_observation_given_true_state

        # 2. Determine Priors based on Pooling Mode
        if prior_params: # Pooling is ON
             prior_alpha_norm = prior_params['alpha']
             prior_beta_norm = prior_params['beta']
             prior_w_norm = prior_params['w']
             mode = "POOLED"
        else: # Pooling is OFF
             current_trust = self.trust_params[trust_key]
             prior_alpha_norm = current_trust['alpha']
             prior_beta_norm = current_trust['beta']
             prior_w_norm = current_trust['w']
             mode = "OWN"
        self.log.debug(f"Updating {trust_key} using {mode} priors: a={prior_alpha_norm:.2f}, b={prior_beta_norm:.2f}, w={prior_w_norm:.3f}")

        # 3. Update Normal Component Parameters
        new_alpha_norm = prior_alpha_norm + delta_alpha
        new_beta_norm = prior_beta_norm + delta_beta

        # 4. Update Mixture Weight
        alpha_dec = self.params.trust_deceptive_alpha
        beta_dec = self.params.trust_deceptive_beta

        # Add epsilon to prevent log(0) or division by zero in betaln for alpha/beta=0 cases
        eps = 1e-9

        # Log Evidence calculations using log beta function betaln = log(B(x,y))
        log_ev_norm = betaln(new_alpha_norm + eps, new_beta_norm + eps) - betaln(prior_alpha_norm + eps, prior_beta_norm + eps)
        log_ev_dec = betaln(alpha_dec + delta_alpha + eps, beta_dec + delta_beta + eps) - betaln(alpha_dec + eps, beta_dec + eps)

        # Log Prior Odds: log(w / (1-w))
        if prior_w_norm >= 1.0 - eps: log_prior_odds = np.inf
        elif prior_w_norm <= eps: log_prior_odds = -np.inf
        else: log_prior_odds = np.log(prior_w_norm) - np.log1p(-prior_w_norm) # np.log1p(x) = log(1+x)

        # Log Posterior Odds = Log Likelihood Ratio + Log Prior Odds
        log_posterior_odds = (log_ev_norm - log_ev_dec) + log_prior_odds

        # Convert log odds back to probability (mixture weight w)
        if np.isinf(log_posterior_odds):
            new_w_norm = 1.0 if log_posterior_odds > 0 else 0.0
        else:
            # w = odds / (1 + odds) = 1 / (1 + exp(-log_odds)) = sigmoid(log_odds)
            new_w_norm = 1.0 / (1.0 + np.exp(-log_posterior_odds))

        # Clip to ensure validity
        new_w_norm = max(0.0, min(1.0, new_w_norm))

        # 5. Store Updated Parameters in Customer State
        # Ensure parameters are standard Python floats for consistency
        final_alpha = float(new_alpha_norm)
        final_beta = float(new_beta_norm)
        final_w = float(new_w_norm)

        updated_params = {
            'alpha': final_alpha,
            'beta': final_beta,
            'w': final_w
        }
        self.trust_params[trust_key] = updated_params

        self.log.debug(f"Updated {trust_key}: Obs={observation}, TrueZ={true_latent_state_idx}, L_i={likelihood_of_observation_given_true_state:.3f} -> "
                       f"a={final_alpha:.2f}, b={final_beta:.2f}, w={final_w:.3f}")

        # 6. Return the updated parameters for potential pooling calculation in Simulation
        return updated_params


    def update_belief(self, provider_id: str, modality_id: str, observation: int):
        """
        Update the belief P(Z) using the observation and the customer's likelihood model,
        adjusted by the provider's expected reliability.

        P(Z | X_new) proportional to P(X_new | Z) * P(Z)
        where P(X_new | Z) is the reliability-adjusted likelihood.
        """
        reliability = self.get_expected_reliability(provider_id, modality_id)
        customer_likelihood_matrix = self.likelihood_models.get(modality_id) # L_i(X|Z)

        if customer_likelihood_matrix is None:
            self.log.error(f"Cannot update belief, missing likelihood for {modality_id}")
            return

        num_states, num_obs = customer_likelihood_matrix.shape
        if not (0 <= observation < num_obs):
            self.log.error(f"Invalid observation {observation} for belief update.")
            return

        # Get the likelihood vector for the specific observation P_i(X=obs | Z)
        likelihood_vec = customer_likelihood_matrix[:, observation] # Shape: (num_states,)

        # Create the uniform distribution vector
        uniform_vec = np.ones(num_states) / num_states

        # Create reliability-adjusted likelihood vector for the observation
        # P_adj(X=obs | Z) = rho * P_i(X=obs | Z) + (1-rho) * Uniform(Z)
        # Note: The formula in the doc seemed slightly off. Adjusting P(X|Z) makes more sense
        # than adjusting P(Z|X). Let's use:
        # Likelihood of *this* obs = rho * L_i(obs|Z) + (1-rho) * (1/N_x) -- affects P(X)? No.
        # Let's use the soft observation approach: The *effective* likelihood update term
        # for P(Z | obs) is rho * L_i(obs|Z) + (1-rho)*Uniform(Z) ?? No, that doesn't make sense.

        # Revisit the EIG doc's math:
        # Predictive P(X=x | Data) = sum_z P(X=x | Z) P(Z | Data) -- using adjusted likelihood
        # Adjusted Likelihood L_tilde(x|z) = rho * L_i(x|z) + (1-rho) * (1/N_x)
        # Posterior P(Z=z | Data, x) propto L_tilde(x|z) * P(Z=z | Data)

        # Adjusted likelihood for THIS observation x: L_tilde[:, x]
        uniform_likelihood = 1.0 / num_obs # P(X=x|Z) under uniform model
        adjusted_likelihood_vec = reliability * likelihood_vec + (1 - reliability) * uniform_likelihood

        # Bayesian update: P(Z|obs) propto P_adj(obs|Z) * P(Z|prev)
        unnormalized_posterior = adjusted_likelihood_vec * self.latent_belief
        norm_constant = unnormalized_posterior.sum()

        if norm_constant > 1e-9:
            self.latent_belief = unnormalized_posterior / norm_constant
            self.log.debug(f"Belief updated using {provider_id}/{modality_id} obs={observation} (rho={reliability:.3f}). New belief: {self.latent_belief}")
        else:
            # This can happen if prior belief is zero for states where likelihood is non-zero
            self.log.warning(f"Belief update resulted in zero probability mass for {provider_id}/{modality_id} obs={observation}. Belief not updated.")


    def _calculate_eig(self, provider_id: str, modality_id: str) -> float:
        """
        Calculates the Expected Information Gain (EIG) for querying a specific provider.
        EIG = H(P(Z|current_data)) - E_{X~P(X|current_data)}[H(P(Z|current_data, X))]
        Based on formulas in docs/experiments.md Section 1.2.2.
        """
        trust_key = (provider_id, modality_id)
        customer_likelihood_matrix = self.likelihood_models.get(modality_id) # L_i(X|Z)

        if customer_likelihood_matrix is None:
            self.log.warning(f"EIG calc failed: Missing likelihood for {modality_id}")
            return -np.inf # Cannot calculate EIG

        # Check if trust params exist (should always, but good practice)
        if trust_key not in self.trust_params:
             self.log.warning(f"EIG calc failed: Missing trust params for {trust_key}")
             return -np.inf

        num_states, num_obs = customer_likelihood_matrix.shape
        reliability = self.get_expected_reliability(provider_id, modality_id)

        # Calculate reliability-adjusted likelihood matrix L_tilde(X|Z)
        # L_tilde(x|z) = rho * L_i(x|z) + (1-rho) * (1/N_x)
        uniform_likelihood_val = 1.0 / num_obs
        # L_tilde = reliability * customer_likelihood_matrix + (1 - reliability) * np.full_like(customer_likelihood_matrix, uniform_likelihood_val)
        # Broadcasting version:
        L_tilde = reliability * customer_likelihood_matrix + (1 - reliability) * uniform_likelihood_val

        # Current belief P(Z | current_data)
        current_belief = self.latent_belief # Shape (num_states,)
        current_entropy = entropy(current_belief)

        # Predictive distribution P(X | current_data)
        # p^x = L_tilde^T P(Z|current_data)
        predictive_dist_x = L_tilde.T @ current_belief # Shape (num_obs,)
        # Normalize predictive distribution (should sum to 1, but enforce for safety)
        predictive_dist_x /= predictive_dist_x.sum()

        # Expected Posterior Entropy E_{X}[H(P(Z|Data, X))]
        expected_posterior_entropy = 0.0
        for obs_idx in range(num_obs):
            prob_x = predictive_dist_x[obs_idx]
            if prob_x < 1e-9: # Skip if probability of this observation is near zero
                continue

            # Calculate posterior P(Z | Data, X=obs_idx)
            # P(Z | Data, x) propto L_tilde(x|Z) * P(Z | Data)
            likelihood_vec_for_obs = L_tilde[:, obs_idx] # Shape (num_states,)
            unnormalized_posterior = likelihood_vec_for_obs * current_belief
            norm_constant = unnormalized_posterior.sum()

            if norm_constant < 1e-9:
                # If this specific outcome leads to zero belief, its entropy is 0
                # but this case might indicate issues if prob_x was non-negligible.
                posterior_entropy = 0.0
                self.log.warning(f"EIG calc for {trust_key}, obs={obs_idx}: Posterior has zero mass.")
            else:
                posterior = unnormalized_posterior / norm_constant
                posterior_entropy = entropy(posterior)

            expected_posterior_entropy += prob_x * posterior_entropy

        # Calculate EIG
        eig = current_entropy - expected_posterior_entropy
        self.log.debug(f"EIG for {trust_key}: H(prior)={current_entropy:.4f} - E[H(post)]={expected_posterior_entropy:.4f} = {eig:.4f}")

        # EIG should generally be non-negative; small negatives possible due to float errors
        return max(0.0, eig)


    def select_next_provider_via_eig(self, available_providers: dict[str, list[str]], queried_this_round: set[tuple[str, str]]) -> tuple[str, str] | None:
        """
        Selects the single best provider/modality to query next based on EIG,
        excluding those already queried this round.

        Args:
            available_providers: Dict {modality: [provider_ids]} potentially usable.
            queried_this_round: Set of (provider_id, modality_id) already queried by
                                 this customer in the current round.

        Returns:
            tuple[str, str] | None: The single best provider tuple (provider_id, modality_id)
                                     to query next, or None if no suitable provider found
                                     (e.g., none available or max EIG <= 0).
        """
        # Create a list of candidate providers not already queried this round
        candidate_providers = []
        for modality, providers in available_providers.items():
            for provider_id in providers:
                key = (provider_id, modality)
                if key not in queried_this_round:
                    candidate_providers.append(key)

        if not candidate_providers:
            # self.log.info("No available providers left to select (already queried or none provided).")
            return None # No candidates left

        # Calculate EIG for all candidates based on *current* belief
        eigs = {}
        for provider_key in candidate_providers:
            provider_id, modality_id = provider_key
            eigs[provider_key] = self._calculate_eig(provider_id, modality_id)

        # Sort candidates by EIG descending
        sorted_candidates = sorted(candidate_providers, key=lambda k: eigs[k], reverse=True)

        # Select the best provider with positive EIG
        best_provider_key: tuple[str, str] | None = None
        max_eig = -np.inf
        if sorted_candidates:
            potential_best_key = sorted_candidates[0]
            potential_max_eig = eigs[potential_best_key]
            if potential_max_eig > 1e-9: # Use a small threshold to avoid float noise
                 best_provider_key = potential_best_key
                 max_eig = potential_max_eig
                 # self.log.info(f"Selected provider {best_key[0]}/{best_key[1]} with EIG={best_eig:.4f}")
            # else:
                 # self.log.info(f"Stopping selection: Max EIG ({best_eig:.4f}) is not positive.")

        # Return the single best provider tuple, or None
        return best_provider_key
