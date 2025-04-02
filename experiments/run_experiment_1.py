import logging
import time
import os
import pickle as pkl

from experiments.sim_params import SimParams
from experiments.simulation import Simulation

def setup_logging(level="INFO"):
    """Configures basic logging for the experiment run."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Configure root logger
    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # Output to console
    logging.getLogger("Simulation").info("Logging configured.")

def run_experiment(params: SimParams, results_dir="results"):
    """
    Runs a single simulation experiment with the given parameters
    and saves the results.
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Generate a filename based on parameters (optional, could use timestamp)
    filename = f"exp1_results_{int(time.time())}.pkl"
    results_path = os.path.join(results_dir, filename)

    logging.info(f"Starting experiment run. Pooling: {params.pool_trust}")
    logging.info(f"Results will be saved to: {results_path}")

    # Initialize and run simulation
    simulation = Simulation(params)
    start_time = time.time()
    results = simulation.run() # Returns a list of dicts
    end_time = time.time()

    logging.info(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    # Save results
    try:
        saveDict = {"params": params, "results": results}
        pkl.dump(saveDict, open(results_path, "wb"))
        logging.info(f"Results successfully saved to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save results to {results_path}: {e}")

    return saveDict

if __name__ == "__main__":
    # --- Configuration --- # 
    LOG_LEVEL = "INFO" # Or DEBUG, WARNING, ERROR
    RESULTS_DIRECTORY = "results/experiment_1"

    setup_logging(LOG_LEVEL)

    # --- Run Experiment without Pooling --- #
    params_unpooled = SimParams(
        num_customers=5,
        num_providers=10,
        num_rounds=1,
        pool_trust=False, # Explicitly disable pooling
        log_level=LOG_LEVEL
        # Other parameters will use defaults from SimParams __post_init__
    )
    logging.info("\n=== Running Experiment: NO Trust Pooling ===")
    run_experiment(params_unpooled, results_dir=RESULTS_DIRECTORY)


    # --- Run Experiment WITH Pooling --- #
    params_pooled = SimParams(
        num_customers=5,
        num_providers=10,
        num_rounds=1,
        pool_trust=True, # Explicitly enable pooling
        log_level=LOG_LEVEL
        # Other parameters will use defaults from SimParams __post_init__
    )
    logging.info("\n=== Running Experiment: WITH Trust Pooling ===")
    run_experiment(params_pooled, results_dir=RESULTS_DIRECTORY)

    logging.info("\nExperiment script finished.")
