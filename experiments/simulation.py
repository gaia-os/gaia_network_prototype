import numpy as np
import logging
from typing import TypeAlias
import time
from dataclasses import dataclass
import copy

from .sim_params import SimParams
from .customer import Customer
from .provider import Provider, PassiveProvider


# Define a type alias for the results structure if it becomes complex
QueryResult: TypeAlias = dict
RoundSummary: TypeAlias = dict


class Simulation:
    """Manages the simulation setup, execution, and results collection."""
    def __init__(self, params: SimParams):
        """Initializes the simulation environment."""
        self.params = params
        # Use dict for easier lookup by ID
        self.customers = {str(i): Customer(str(i), params) for i in range(params.num_customers)}
        # Use nested dictionary for results: {round: {query_num: data}}
        self.results: dict[int, dict[int, dict]] = {} # Initialized in run

        # Initialize Providers based on assignments
        self.providers: dict[tuple[str, str], Provider] = {}
        self._initialize_providers()

        # Logging setup
        self._setup_logger(params.log_level)

        num_unique_providers = len(set(pid for pid, _ in self.providers.keys()))
        self.log.info(f"Simulation initialized with {params.num_customers} customers, {num_unique_providers} unique providers ({len(self.providers)} provider-modality pairs).")
        self.log.info(f"Trust pooling: {params.pool_trust}")
        # Removed selection method log


    def _initialize_providers(self):
        for modality_id, provider_ids in self.params.provider_assignments.items():
            for provider_id in provider_ids:
                provider_key = (provider_id, modality_id)
                # Ensure provider_id is string if params use ints/other types
                str_provider_id = str(provider_id)
                if provider_key in self.providers:
                     # Initialize logger here to use it
                     if not hasattr(self, 'log'):
                         self._setup_logger(self.params.log_level)
                     self.log.warning(f"Duplicate provider assignment found for {provider_key}. Overwriting.")
                self.providers[provider_key] = PassiveProvider(str_provider_id, modality_id, self.params)


    def _setup_logger(self, log_level: str):
         """Sets up the instance logger."""
         self.log = logging.getLogger("Simulation")
         self.log.setLevel(log_level.upper())
         if not self.log.handlers:
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             self.log.addHandler(handler)
             self.log.propagate = False


    def run(self):
        """Runs the entire simulation for the specified number of rounds."""
        self.log.info(f"Starting simulation for {self.params.num_rounds} rounds.")
        # Initialize results dict with keys for each round
        self.results = {r: {} for r in range(self.params.num_rounds)}

        for r in range(self.params.num_rounds):
            round_start_time = time.time()
            self.log.info(f"--- Starting Round {r} ---")

            # 1. Reset customer beliefs for the round
            for customer in self.customers.values():
                customer.reset_belief()

            # 2. Determine True Latent State for the round
            current_true_latent_state = np.random.choice(self.params.latent_states)
            try:
                current_true_latent_state_idx = self.params.latent_states.index(current_true_latent_state)
            except ValueError:
                 self.log.error(f"Could not find true latent state '{current_true_latent_state}' in params.latent_states. Aborting round.")
                 continue # Skip to next round
            self.log.info(f"Round {r}: True Latent State = {current_true_latent_state} (Index: {current_true_latent_state_idx})")

            # 3. Initialize tracking for the round
            active_customers = list(self.customers.keys())
            queries_made_this_round = {cust_id: 0 for cust_id in self.customers}
            queried_providers_this_round = {cust_id: set() for cust_id in self.customers}
            all_observations_this_round = [] # Initialize list for trust update phase
            total_queries_in_round = 0 # Counter for final log message

            # 4. Pre-calculate pooled trust parameters if pooling is enabled
            initial_pooled_trust: dict[tuple[str, str], TrustParamsDict] | None = None
            if self.params.pool_trust:
                 # Calculate average initial trust params across all customers for each provider/modality
                 # This assumes a simple averaging; more complex pooling might be needed
                 pooled_params_sum = {}
                 pooled_params_count = {}
                 for cust in self.customers.values():
                     for key, params_dict in cust.trust_params.items():
                         if key not in pooled_params_sum:
                             pooled_params_sum[key] = {'alpha': 0.0, 'beta': 0.0, 'w': 0.0}
                             pooled_params_count[key] = 0
                         pooled_params_sum[key]['alpha'] += params_dict['alpha']
                         pooled_params_sum[key]['beta'] += params_dict['beta']
                         pooled_params_sum[key]['w'] += params_dict['w']
                         pooled_params_count[key] += 1

                 initial_pooled_trust = {}
                 num_customers = len(self.customers)
                 if num_customers > 0:
                     for key, sums in pooled_params_sum.items():
                          count = pooled_params_count[key]
                          if count > 0:
                            initial_pooled_trust[key] = {
                                'alpha': sums['alpha'] / count,
                                'beta': sums['beta'] / count,
                                'w': sums['w'] / count
                            }
                          else: # Should not happen if trust_params are initialized
                             initial_pooled_trust[key] = {'alpha': 1.0, 'beta': 1.0, 'w': 0.5} # Default fallback
                 self.log.debug(f"Round {r}: Initial pooled trust calculated.")


            # --- Query Loop (Revised Structure) ---
            # Outer loop controls max queries per customer
            for query_attempt in range(self.params.customer_query_budget):
                if not active_customers: # Stop if no customers are active
                    break

                # Keep track of customers who become inactive *during this attempt*
                newly_inactive = []

                # Inner loop gives each active customer a chance to query once per attempt
                # Use a copy of active_customers list to allow safe removal during iteration
                for customer_id in list(active_customers):

                    # Skip if customer already reached budget (shouldn't happen with this structure but safe)
                    if queries_made_this_round[customer_id] >= self.params.customer_query_budget:
                        if customer_id in active_customers:
                           active_customers.remove(customer_id)
                        continue

                    customer = self.customers[customer_id]
                    self.log.debug(f"Round {r}, Attempt {query_attempt}: Considering Customer {customer_id} (Queries made: {queries_made_this_round[customer_id]})")

                    # --- Provider Selection (using new method) ---
                    # available_providers = self.providers.keys() # Incorrect, need dict format
                    # TODO: Need to reconstruct available_providers based on self.providers
                    # For now, assume all providers are available for selection check
                    # We need a mapping {modality: [providers]} from self.providers
                    available_providers_map = {mod: [] for mod in self.params.modalities}
                    for prov_id, mod_id in self.providers.keys():
                        if mod_id in available_providers_map:
                            available_providers_map[mod_id].append(prov_id)
                        else:
                            self.log.warning(f"Provider {prov_id} has unknown modality {mod_id}. Skipping.")

                    best_provider_key = customer.select_next_provider_via_eig(
                        available_providers=available_providers_map, # Pass the map
                        queried_this_round=queried_providers_this_round[customer_id]
                    )

                    if best_provider_key:
                        best_provider_id, best_modality_id = best_provider_key
                        provider = self.providers.get(best_provider_key)

                        if not provider:
                             self.log.error(f"Selected provider key {best_provider_key} not found in self.providers despite selection. Skipping query.")
                             newly_inactive.append(customer_id)
                             continue

                        self.log.info(f"Round {r}, Query {total_queries_in_round + 1}: Customer {customer_id} selects Provider {best_provider_id} (Modality: {best_modality_id}) via EIG")

                        # --- Perform Query ---
                        observation_idx = provider.get_observation(current_true_latent_state_idx)

                        # --- Update Customer Belief First ---
                        belief_before_update = copy.deepcopy(customer.latent_belief)
                        customer.update_belief(best_provider_id, best_modality_id, observation_idx)

                        # --- Build Query Data Dictionary (Bottom-Up) ---
                        query_num_this_customer = queries_made_this_round[customer_id]
                        query_data = {
                            'round': r,
                            'customer_id': customer_id,
                            'query_in_round': query_num_this_customer, # Query index for THIS customer
                            'provider_id': best_provider_id,
                            'modality_id': best_modality_id,
                            'selection_method': 'EIG',
                            'observation': observation_idx,
                            'true_latent_state': current_true_latent_state,
                            'true_latent_state_idx': current_true_latent_state_idx,
                            'belief_before': copy.deepcopy(belief_before_update),
                            'belief_after': copy.deepcopy(customer.latent_belief),
                            'timestamp': time.time()
                            # EIG value could be added if returned by select_next_provider_via_eig
                        }
                        # Assign the fully constructed dict using a unique key (round, customer, query_num)
                        query_result_key = (r, customer_id, query_num_this_customer)
                        self.results[r][query_result_key] = query_data

                        # --- Collect data for Trust Update Phase ---
                        prior_trust_for_update = None
                        if self.params.pool_trust and initial_pooled_trust:
                            prior_trust_for_update = initial_pooled_trust.get(best_provider_key)

                        observation_details = {
                            'customer_id': customer_id,
                            'provider_id': best_provider_id,
                            'modality_id': best_modality_id,
                            'observation': observation_idx,
                            'true_latent_state_idx': current_true_latent_state_idx,
                            'prior_params_for_update': prior_trust_for_update
                        }
                        all_observations_this_round.append(observation_details)

                        # --- Update Tracking ---
                        queries_made_this_round[customer_id] += 1
                        queried_providers_this_round[customer_id].add(best_provider_key)
                        total_queries_in_round += 1

                    else:
                        # No suitable provider found (EIG <= 0 or none left)
                        self.log.debug(f"Customer {customer_id} found no provider with positive EIG. Marking inactive for round.")
                        newly_inactive.append(customer_id)

                # Remove customers who became inactive during this attempt
                for cust_id in newly_inactive:
                     if cust_id in active_customers:
                          active_customers.remove(cust_id)


            # --- End of Query Loop for Round ---
            self.log.info(f"Round {r}: Querying phase complete. Total queries made: {total_queries_in_round}")

            # --- Log End-of-Round State (Bottom-Up) --- #
            eor_customer_summaries = {} # Temporary dict to build summaries
            for cust_id, customer in self.customers.items():
                 # Nested final trust parameters
                 final_trust = {}
                 for key, params_dict in customer.trust_params.items():
                     final_trust[key] = {
                         'w': params_dict['w'],
                         'alpha': params_dict['alpha'],
                         'beta': params_dict['beta']
                     }

                 # Build complete end-of-round data for this customer
                 customer_eor_data = {
                     'customer_id': cust_id,
                     'provider_id': None,
                     'modality_id': None,
                     'selection_method': 'EIG',
                     'eig_value': None,
                     'observation': None,
                     'true_latent_state': current_true_latent_state,
                     'true_latent_state_idx': current_true_latent_state_idx,
                     'belief_before': None, # Not applicable for EOR
                     'belief_after': copy.deepcopy(customer.latent_belief),
                     'trust_w_before': None, # Not applicable for EOR
                     'trust_alpha_before': None, # Not applicable for EOR
                     'trust_beta_before': None, # Not applicable for EOR
                     'timestamp': time.time(),
                     'final_trust': final_trust # Add the fully built nested dict
                 }
                 # Add this customer's complete summary to the temporary dict
                 eor_customer_summaries[cust_id] = customer_eor_data

            # Assign the complete dict of customer summaries to the round's -1 key
            self.results[r][-1] = eor_customer_summaries

            # --- Trust Update Phase --- #
            self.log.debug(f"--- Round {r}: Trust Update Phase ---")
            # Decide whether to pool trust based on parameters
            if self.params.pool_trust:
                # Pool trust requires all observations from the round
                # Updated trust parameters are calculated based on initial pooled priors
                # and then THESE updated parameters become the new single pooled set for the NEXT round's START.
                # This implementation updates the customer's individual trust params based on the POOLED prior
                # and assumes the 'pooled' prior is static for the round (calculated at start).
                # A different interpretation might re-calculate a single pooled set *after* all observations.

                # Update each customer's trust using the observations they made,
                # but starting from the common prior calculated at the beginning of the round.
                for obs_info in all_observations_this_round:
                    cust = self.customers[obs_info['customer_id']]
                    cust.update_trust(
                        provider_id=obs_info['provider_id'],
                        modality_id=obs_info['modality_id'],
                        observation=obs_info['observation'],
                        true_latent_state_idx=obs_info['true_latent_state_idx'],
                        prior_params=obs_info['prior_params_for_update'] # Use the initial pooled prior
                    )
                self.log.info(f"Round {r}: Trust updated for all customers using initial pooled priors.")

            else: # Update trust individually
                # Update each customer's trust based on their *own* prior from the start of the update step
                # (which is effectively the state after the previous update or initialization)
                for obs_info in all_observations_this_round:
                     cust = self.customers[obs_info['customer_id']]
                     cust.update_trust(
                         provider_id=obs_info['provider_id'],
                         modality_id=obs_info['modality_id'],
                         observation=obs_info['observation'],
                         true_latent_state_idx=obs_info['true_latent_state_idx'],
                         prior_params=None # Use customer's own current trust as prior
                     )
                self.log.info(f"Round {r}: Trust updated individually for customers based on their observations.")


            # 4. Log round completion time
            round_end_time = time.time()
            self.log.info(f"--- Finished Round {r} (Duration: {round_end_time - round_start_time:.2f}s) ---")

        self.log.info("Simulation finished.")
        return self.results
