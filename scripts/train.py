#!/usr/bin/env python3
"""Enhanced Training Script for Temporal Factored Rainbow DQN Healthcare System.

This script provides comprehensive training with all 7 enhanced features:
1. Reward Re-shaping with clinical penalties/bonuses
2. Stratified Synthetic Data Generator with risk-tier partitioning  
3. Risk-Aware Prioritized Replay with 1.5x multiplier
4. Adaptive CQL Strength (α=0.5 for high-risk, α=1.0 otherwise)
5. Monotonicity Regularizer (β=0.1)
6. Feature Scaling Audit (risk_score rescaled to [-1,1])
7. Hyperparameter Tuning Framework (optional)

Optimized for Apple Silicon M3 Ultra with parallel processing.

Usage:
    python scripts/train.py                    # Default training (10K samples)
    python scripts/train.py --samples 1000000  # 1M samples
    python scripts/train.py --quick            # Quick test run (1K samples)
    python scripts/train.py --tune             # With hyperparameter tuning
"""

import argparse
import sys
import os
import time
from typing import Dict, Any
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.temporal_rainbow_dqn import TemporalFactoredDQN
from core.temporal_training_data import generate_stratified_temporal_training_data
from core.temporal_training_loop import TemporalFactoredTrainer
from core.replay_buffer import PrioritizedReplayBuffer
from core.temporal_actions import EnhancedPatientState
from core.logging_system import get_logger

logger = get_logger(__name__)

try:
    from core.hyperparameter_tuning import run_hyperparameter_optimization
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    logger.warning("Hyperparameter optimization not available (Ray Tune not installed)")


def train_enhanced_model(
    num_samples: int = 10000,
    batch_size: int = 128,
    num_epochs: int = 10,
    learning_rate: float = 0.0003,
    use_hyperopt: bool = False
) -> Dict[str, Any]:
    """Train enhanced model with all improvements.
    
    Args:
        num_samples: Number of training samples to generate
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate (if not using hyperopt)
        use_hyperopt: Whether to use hyperparameter optimization
        
    Returns:
        Dict containing training results and metrics
    """
    logger.info("STARTING ENHANCED TEMPORAL FACTORED RAINBOW DQN TRAINING")
    logger.info("=" * 70)

    start_time = time.time()
    results = {}

    try:
        # Step 1: Hyperparameter Optimization (optional)
        if use_hyperopt and HYPEROPT_AVAILABLE:
            logger.info("STEP 1: Hyperparameter Optimization with Ray Tune")
            hyperopt_results = run_hyperparameter_optimization(
                num_samples=16, max_num_epochs=10, grace_period=3
            )
            results["hyperopt"] = hyperopt_results

            # Use best hyperparameters
            best_config = hyperopt_results["best_config"]
            learning_rate = best_config["learning_rate"]
            batch_size = best_config.get("batch_size", batch_size)
        else:
            logger.info("STEP 1: Using default hyperparameters")
            results["hyperopt"] = {"skipped": True}
        
        # Step 2: Enhanced Data Generation
        logger.info("STEP 2: Stratified Synthetic Data Generation")
        training_data = generate_stratified_temporal_training_data(num_samples)
        results["data_generation"] = {
            "samples": len(training_data),
            "risk_tier_distribution": training_data["risk_tier"].value_counts().to_dict()
        }
        
        # Step 3: Model Creation
        logger.info("STEP 3: Enhanced Model Creation")
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
        target_network.load_state_dict(policy_network.state_dict())
        
        # Step 4: Enhanced Replay Buffer
        logger.info("STEP 4: Risk-Aware Prioritized Replay Buffer")
        replay_buffer = PrioritizedReplayBuffer(
            capacity=min(100000, num_samples * 2),
            alpha=0.6,
            beta_start=0.4,
            beta_frames=num_samples // 10
        )
        
        # Step 5: Enhanced Trainer
        logger.info("STEP 5: Enhanced Trainer with All Components")
        trainer = TemporalFactoredTrainer(
            policy_network=policy_network,
            target_network=target_network,
            replay_buffer=replay_buffer,
            learning_rate=learning_rate,
            gamma=0.99,
            target_update_freq=1000
        )
        
        # Step 6: Fill Replay Buffer
        logger.info("STEP 6: Filling Enhanced Replay Buffer")
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
            next_state = state_features
            
            replay_buffer.add(state_features, action, reward, next_state, done)
        
        # Step 7: Enhanced Training Loop
        logger.info(f"STEP 7: Enhanced Training for {num_epochs} epochs")
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Multiple training steps per epoch
            steps_per_epoch = min(100, len(replay_buffer) // batch_size)
            
            for step in range(steps_per_epoch):
                # Reset noise
                policy_network.reset_noise()
                target_network.reset_noise()
                
                # Enhanced training step with all improvements
                loss_info = trainer.train_step(batch_size=batch_size)
                epoch_losses.append(loss_info["total_loss"])
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            losses.append(avg_epoch_loss)
            
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} completed",
                avg_loss=avg_epoch_loss,
                steps=len(epoch_losses)
            )
        
        # Step 8: Save Model
        logger.info("STEP 8: Saving Enhanced Model")

        # Ensure models directory exists
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        model_filename = f"enhanced_tfrdqn_{num_samples}_samples_{int(time.time())}.pt"
        model_path = os.path.join(models_dir, model_filename)
        
        checkpoint = {
            "model_state_dict": policy_network.state_dict(),
            "training_results": results,
            "enhancements": [
                "reward_reshaping",
                "stratified_data_generation", 
                "risk_aware_prioritized_replay",
                "adaptive_cql_strength",
                "monotonicity_regularizer",
                "feature_scaling_audit"
            ],
            "timestamp": time.time(),
            "samples": num_samples
        }
        
        torch.save(checkpoint, model_path)
        
        # Compile final results
        total_time = time.time() - start_time
        training_stats = {
            "epochs": num_epochs,
            "final_loss": losses[-1] if losses else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
            "total_steps": num_epochs * min(100, len(replay_buffer) // batch_size)
        }
        
        results.update({
            "training": training_stats,
            "model_path": model_path,
            "total_time": total_time,
            "success": True
        })
        
        # Final Summary
        logger.info("=" * 70)
        logger.info("ENHANCED TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Total Time: {total_time/60:.1f} minutes")
        logger.info(f"Training Samples: {num_samples:,}")
        logger.info(f"Final Loss: {training_stats['final_loss']:.4f}")
        logger.info(f"Model Saved: {model_path}")
        logger.info("Enhancements Applied:")
        logger.info("  - Reward Re-shaping (-0.3 for low-risk medication, +0.3 for high-risk interventions)")
        logger.info("  - Stratified Synthetic Data Generator (risk-tier partitioning)")
        logger.info("  - Risk-Aware Prioritized Replay (1.5x multiplier for high-risk actions)")
        logger.info("  - Adaptive CQL Strength (α=0.5 for high-risk, α=1.0 otherwise)")
        logger.info("  - Monotonicity Regularizer (β=0.1)")
        logger.info("  - Feature Scaling Audit (risk_score rescaled to [-1,1])")

        # Step 9: Sample Prediction Demo
        logger.info("STEP 9: Running Sample Prediction Demo")
        try:
            from scripts.predict import run_prediction_demo
            run_prediction_demo(model_path=model_path, use_untrained=False)
        except Exception as e:
            logger.warning(f"Prediction demo failed: {e}")
            logger.info("You can run predictions manually with: python scripts/predict.py")

        return results
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {e}")
        results.update({
            "success": False,
            "error": str(e),
            "total_time": time.time() - start_time
        })
        return results


def main():
    """Main entry point for enhanced training."""
    parser = argparse.ArgumentParser(description="Enhanced Temporal Factored Rainbow DQN Training")
    parser.add_argument("--samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--quick", action="store_true", help="Quick test run (1K samples, 3 epochs)")
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.samples = 1000
        args.epochs = 3
        args.batch_size = 32
        print("Quick mode: 1K samples, 3 epochs")

    # Run training
    results = train_enhanced_model(
        num_samples=args.samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_hyperopt=args.tune
    )

    if results["success"]:
        print(f"\nTRAINING COMPLETED SUCCESSFULLY!")
        print(f"Model saved: {results['model_path']}")
        print(f"Total time: {results['total_time']/60:.1f} minutes")
        sys.exit(0)
    else:
        print(f"\nTRAINING FAILED: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
