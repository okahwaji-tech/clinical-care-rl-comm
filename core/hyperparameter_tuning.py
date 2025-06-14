"""Hyperparameter tuning framework using Ray Tune for Temporal Factored Rainbow DQN.

This module implements comprehensive hyperparameter optimization using Ray Tune
with ASHA scheduler for efficient parallel trials on healthcare RL systems.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.temporal_rainbow_dqn import TemporalFactoredDQN
from core.temporal_training_data import generate_stratified_temporal_training_data
from core.temporal_training_loop import TemporalFactoredTrainer
from core.replay_buffer import PrioritizedReplayBuffer
from core.temporal_actions import EnhancedPatientState
from core.logging_system import get_logger

logger = get_logger(__name__)

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray Tune not available. Install with: pip install ray[tune] hyperopt")


def train_tfrdqn(config: Dict[str, Any]) -> Dict[str, float]:
    """Train Temporal Factored Rainbow DQN with given hyperparameter configuration.
    
    This function is designed to be called by Ray Tune for hyperparameter optimization.
    
    Args:
        config (Dict[str, Any]): Hyperparameter configuration from Ray Tune
        
    Returns:
        Dict[str, float]: Training metrics for optimization
    """
    try:
        # Extract hyperparameters
        lr = config["learning_rate"]
        per_alpha = config["per_alpha"]
        per_beta = config["per_beta"]
        cql_alpha = config["cql_alpha"]
        
        # Reward weight coefficients
        reward_weights = {
            "base_weight": config["base_weight"],
            "medication_penalty_weight": config["medication_penalty_weight"],
            "high_risk_bonus_weight": config["high_risk_bonus_weight"],
            "action_bonus_weight": config["action_bonus_weight"],
            "time_bonus_weight": config["time_bonus_weight"]
        }
        
        # Generate training data
        training_data = generate_stratified_temporal_training_data(
            num_samples=config.get("num_samples", 50000)
        )
        
        # Create networks
        policy_network = TemporalFactoredDQN(
            input_dim=27,
            use_dueling=True,
            use_noisy=True,
            use_distributional=True
        )
        
        target_network = TemporalFactoredDQN(
            input_dim=27,
            use_dueling=True,
            use_noisy=True,
            use_distributional=True
        )
        
        # Initialize target network
        target_network.load_state_dict(policy_network.state_dict())
        
        # Create replay buffer with tuned parameters
        replay_buffer = PrioritizedReplayBuffer(
            capacity=config.get("buffer_capacity", 100000),
            alpha=per_alpha,
            beta_start=per_beta,
            beta_frames=config.get("beta_frames", 100000)
        )
        
        # Create trainer
        trainer = TemporalFactoredTrainer(
            policy_network=policy_network,
            target_network=target_network,
            replay_buffer=replay_buffer,
            learning_rate=lr,
            gamma=config.get("gamma", 0.99),
            target_update_freq=config.get("target_update_freq", 1000)
        )
        
        # Fill replay buffer
        for _, row in training_data.iterrows():
            state_features = row[EnhancedPatientState.get_feature_names()].values
            
            # Extract action components
            healthcare_action = int(row["healthcare_action"])
            timing_action = int(row["time_horizon"])
            schedule_action = int(row["time_of_day"])
            communication_action = int(row["communication_channel"])
            
            # Encode as single integer
            action = (
                healthcare_action * 8 * 4 * 4
                + timing_action * 4 * 4
                + schedule_action * 4
                + communication_action
            )
            
            reward = row["reward"]
            done = row["done"]
            next_state = state_features  # Simplified
            
            replay_buffer.add(state_features, action, reward, next_state, done)
        
        # Training loop
        num_episodes = config.get("num_episodes", 1000)
        losses = []
        
        for episode in range(num_episodes):
            # Reset noise
            policy_network.reset_noise()
            target_network.reset_noise()
            
            # Training step
            loss_info = trainer.train_step(batch_size=config.get("batch_size", 64))
            losses.append(loss_info["total_loss"])
            
            # Report intermediate results to Ray Tune
            if episode % 100 == 0 and episode > 0:
                avg_loss = np.mean(losses[-100:])
                tune.report(
                    loss=avg_loss,
                    episode=episode,
                    avg_loss_100=avg_loss
                )
        
        # Final evaluation metrics
        final_avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
        loss_improvement = losses[0] - losses[-1] if len(losses) > 1 else 0.0
        convergence_rate = loss_improvement / len(losses) if len(losses) > 0 else 0.0
        
        # Return metrics for optimization (Ray Tune minimizes the first metric)
        return {
            "loss": final_avg_loss,
            "loss_improvement": loss_improvement,
            "convergence_rate": convergence_rate,
            "final_loss": losses[-1] if losses else float('inf')
        }
        
    except Exception as e:
        logger.error(f"Training failed in Ray Tune trial: {e}")
        return {
            "loss": float('inf'),
            "loss_improvement": 0.0,
            "convergence_rate": 0.0,
            "final_loss": float('inf')
        }


def run_hyperparameter_optimization(
    num_samples: int = 16,
    max_num_epochs: int = 20,
    grace_period: int = 5,
    reduction_factor: int = 2,
    local_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run hyperparameter optimization using Ray Tune with ASHA scheduler.
    
    Args:
        num_samples (int): Number of parallel trials to run. Defaults to 16.
        max_num_epochs (int): Maximum number of epochs per trial. Defaults to 20.
        grace_period (int): Minimum number of epochs before early stopping. Defaults to 5.
        reduction_factor (int): Factor by which to reduce trials. Defaults to 2.
        local_dir (Optional[str]): Directory to save results. Defaults to temp directory.
        
    Returns:
        Dict[str, Any]: Best hyperparameter configuration and results
        
    Raises:
        ImportError: If Ray Tune is not available
        RuntimeError: If optimization fails
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune is required for hyperparameter optimization. Install with: pip install ray[tune] hyperopt")
    
    logger.info("Starting hyperparameter optimization with Ray Tune")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Define search space
    search_space = {
        # Core learning parameters
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "per_alpha": tune.uniform(0.4, 0.8),
        "per_beta": tune.uniform(0.3, 0.6),
        "cql_alpha": tune.uniform(0.3, 1.5),
        
        # Reward weight coefficients (must sum to 1.0)
        "base_weight": tune.uniform(0.2, 0.4),
        "medication_penalty_weight": tune.uniform(0.15, 0.25),
        "high_risk_bonus_weight": tune.uniform(0.1, 0.2),
        "action_bonus_weight": tune.uniform(0.1, 0.2),
        "time_bonus_weight": tune.uniform(0.05, 0.15),
        
        # Training parameters
        "batch_size": tune.choice([32, 64, 128]),
        "gamma": tune.uniform(0.95, 0.99),
        "target_update_freq": tune.choice([500, 1000, 2000]),
        
        # Fixed parameters for trials
        "num_samples": 50000,  # Smaller for faster trials
        "num_episodes": 1000,
        "buffer_capacity": 100000,
        "beta_frames": 100000
    }
    
    # ASHA scheduler for efficient early stopping
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor
    )
    
    # HyperOpt search algorithm
    search_alg = HyperOptSearch(metric="loss", mode="min")
    
    # Create temporary directory if not provided
    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix="ray_tune_tfrdqn_")
        cleanup_dir = True
    else:
        cleanup_dir = False
    
    try:
        # Run optimization
        analysis = tune.run(
            train_tfrdqn,
            config=search_space,
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            local_dir=local_dir,
            name="tfrdqn_hyperopt",
            stop={"training_iteration": max_num_epochs},
            resources_per_trial={"cpu": 1, "gpu": 0.1 if torch.cuda.is_available() else 0},
            verbose=1
        )
        
        # Get best configuration
        best_config = analysis.best_config
        best_result = analysis.best_result
        
        logger.info(
            "Hyperparameter optimization completed",
            best_loss=best_result["loss"],
            best_config=best_config
        )
        
        # Compile results
        results = {
            "best_config": best_config,
            "best_result": best_result,
            "all_trials": analysis.results_df.to_dict("records") if hasattr(analysis, "results_df") else [],
            "analysis": analysis
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        raise RuntimeError(f"Optimization failed: {e}") from e
        
    finally:
        # Cleanup temporary directory
        if cleanup_dir and os.path.exists(local_dir):
            shutil.rmtree(local_dir, ignore_errors=True)
        
        # Shutdown Ray
        ray.shutdown()


def demonstrate_hyperparameter_tuning() -> None:
    """Demonstrate hyperparameter tuning functionality."""
    logger.info("Hyperparameter Tuning Demonstration")
    logger.info("=" * 60)
    
    if not RAY_AVAILABLE:
        logger.warning("Ray Tune not available - skipping demonstration")
        return
    
    try:
        # Run small-scale optimization
        results = run_hyperparameter_optimization(
            num_samples=4,  # Small number for demo
            max_num_epochs=5,
            grace_period=2
        )
        
        logger.info("Best hyperparameters found:")
        for key, value in results["best_config"].items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"Best loss: {results['best_result']['loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")


if __name__ == "__main__":
    demonstrate_hyperparameter_tuning()
