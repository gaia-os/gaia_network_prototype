import numpy as np
import logging
from abc import ABC, abstractmethod
import typing # For assert TypeGuard

from .sim_params import SimParams

class Provider(ABC):
    """Abstract base class for data providers."""
    def __init__(self, provider_id: str, modality_id: str, params: SimParams):
        self.id = provider_id
        self.modality_id = modality_id
        self.params = params
        self.log = logging.getLogger(f"Provider_{self.id}_{self.modality_id}")
        self.log.setLevel(params.log_level.upper())
        # Basic handler if none exists, configure globally if needed
        if not self.log.handlers:
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             self.log.addHandler(handler)
             self.log.propagate = False

    @abstractmethod
    def get_observation(self, true_latent_state_idx: int) -> int:
        """Generate an observation based on the true latent state."""
        pass

    def receive_reward(self, reward: float):
        """Process any reward received (optional for passive providers)."""
        self.log.debug(f"Received reward: {reward}")
        # Passive providers might not do anything with rewards
        pass


class PassiveProvider(Provider):
    """
    A provider with a fixed true reliability.
    Generates observations based on P_true(X|Z) and its reliability.
    """

    true_reliability: float

    def __init__(self, provider_id: str, modality_id: str, params: SimParams):
        super().__init__(provider_id, modality_id, params)
        self.true_reliability = params.provider_reliabilities.get(provider_id, 0.5)

        self.true_likelihood_matrix = params.true_likelihoods[modality_id]
        if self.true_likelihood_matrix is None:
            self.log.error(f"True likelihood matrix not found for modality {modality_id}. Cannot generate observations.")
            # Raise an error or handle appropriately
            raise ValueError(f"Missing true likelihood for modality {modality_id}")
        self.num_observations = self.true_likelihood_matrix.shape[1]

    def get_observation(self, true_latent_state_idx: int) -> int:
        """
        Generates an observation.
        With probability `true_reliability`, samples from P_true(X | Z=true_state).
        With probability `1 - true_reliability`, samples uniformly from possible observations.
        """
        rng = np.random.default_rng()

        if rng.random() < self.true_reliability:
            # Sample from the true conditional likelihood P(X | Z = true_state)
            probabilities = self.true_likelihood_matrix[true_latent_state_idx, :]
            observation = rng.choice(self.num_observations, p=probabilities)
            self.log.debug(f"Generating HONEST observation based on Z={true_latent_state_idx}. Result: {observation}")
        else:
            # Sample uniformly (uninformative/random)
            observation = rng.choice(self.num_observations)
            self.log.debug(f"Generating RANDOM observation (reliability miss). Result: {observation}")

        return observation

# TODO: Implement other provider types (Active, Strategic) from experiments.md
# class ActiveProvider(Provider): ...
# class StrategicProvider(Provider): ...
