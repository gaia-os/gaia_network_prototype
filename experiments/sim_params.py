from dataclasses import dataclass, field
import numpy as np

@dataclass
class SimParams:
    """Simulation parameters for Experiment 1: Trust Modeling."""
    num_customers: int = 5
    num_providers: int = 10
    modalities: list[str] = field(default_factory=lambda: ['modality_A', 'modality_B'])
    latent_states: list[str] = field(default_factory=lambda: ['state_0', 'state_1'])
    num_rounds: int = 50
    customer_query_budget: int = 5 # Max queries per customer per round
    # Provider assignment: {modality: [provider_ids]}
    provider_assignments: dict[str, list[str]] = field(default_factory=dict)
    # True underlying reliability: {provider_id: reliability} (0.0 to 1.0)
    provider_reliabilities: dict[str, float] = field(default_factory=dict)
    # True data generating likelihoods P(X|Z): {modality: np.array[num_states, num_observations]}
    true_likelihoods: dict[str, np.ndarray] = field(default_factory=dict)
    # Customer's potentially incorrect models L_i(X|Z): {customer_id: {modality: np.array}}
    customer_likelihood_models: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # Trust model prior parameters (Beta mixture)
    trust_prior_alpha: float = 1.0
    trust_prior_beta: float = 1.0
    trust_prior_weight: float = 0.99 # Prior weight for the 'normal' component
    trust_deceptive_alpha: float = 1.0 # Fixed alpha for the 'deceptive' component
    trust_deceptive_beta: float = 10.0 # Fixed beta for the 'deceptive' component
    # --- Experiment Settings ---
    pool_trust: bool = False # Flag to enable/disable trust pooling globally
    log_level: str = "INFO" # Logging level for simulation components

    def __post_init__(self):
        """Validate and potentially generate default parameter values."""
        rng_state = np.random.default_rng(seed=42) # For consistent state generation
        rng_likelihood = np.random.default_rng(seed=43) # For consistent likelihood generation
        rng_customer = np.random.default_rng(seed=44) # For consistent customer model generation

        if not self.provider_assignments:
            # Basic assignment: Assign providers round-robin to modalities
            num_modalities = len(self.modalities)
            self.provider_assignments = {mod: [] for mod in self.modalities}
            for i in range(self.num_providers):
                provider_id = f"P{i:02d}"
                modality = self.modalities[i % num_modalities]
                self.provider_assignments[modality].append(provider_id)

        if not self.provider_reliabilities:
             # Assign random reliabilities
             for modality, providers in self.provider_assignments.items():
                 for provider_id in providers:
                      # Assign higher reliability more often using rng_state
                      self.provider_reliabilities[provider_id] = rng_state.beta(a=5, b=2)

        # Define dimensions
        num_latent_states = len(self.latent_states)
        num_observations = 2 # Assume binary observations {0, 1}

        if not self.true_likelihoods:
             # Generate true likelihoods P(X|Z) using rng_likelihood
             for modality in self.modalities:
                 # Simple example: State 0 prefers obs 0, State 1 prefers obs 1
                 noise = 0.1 # Chance of observing the 'wrong' thing
                 likelihood = np.full((num_latent_states, num_observations), noise / (num_observations -1 if num_observations > 1 else 1))
                 # Create identity matrix trace essentially
                 diag_val = 1.0 - noise
                 np.fill_diagonal(likelihood, diag_val)
                 # Ensure rows sum to 1 (should already be close due to fill_diagonal)
                 likelihood /= likelihood.sum(axis=1, keepdims=True)
                 self.true_likelihoods[modality] = likelihood

        if not self.customer_likelihood_models:
            # Assign models L_i(X|Z) using rng_customer
            # IMPORTANT: Keys MUST be strings '0', '1', ... to match Customer init
            for i in range(self.num_customers):
                customer_id_str = str(i) # Use string key!
                customer_model = {}
                for modality in self.modalities:
                    # Simplest case: Customers know the true likelihood
                    # Could add noise/perturbation here using rng_customer later
                    customer_model[modality] = self.true_likelihoods[modality].copy()
                self.customer_likelihood_models[customer_id_str] = customer_model

        # Add any other validation or default generation here
