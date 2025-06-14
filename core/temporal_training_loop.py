"""
Temporal Training Loop for Factored Rainbow DQN Healthcare Communication Optimization.

This module implements the training infrastructure for the temporal factored DQN,
including specialized loss functions and training procedures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.temporal_rainbow_dqn import TemporalFactoredDQN
from core.temporal_training_data import generate_temporal_training_data
from core.temporal_actions import EnhancedPatientState
from core.replay_buffer import PrioritizedReplayBuffer
from core.cql_components import compute_adaptive_cql_loss, compute_monotonicity_regularizer
from core.logging_system import get_logger

logger = get_logger(__name__)


class TemporalFactoredTrainer:
    """Trainer for temporal factored Rainbow DQN."""

    def __init__(
        self,
        policy_network: TemporalFactoredDQN,
        target_network: TemporalFactoredDQN,
        replay_buffer: PrioritizedReplayBuffer,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        combination_method: str = "additive",
    ):
        self.policy_network = policy_network
        self.target_network = target_network
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.combination_method = combination_method

        # Optimizer
        self.optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

        # Training statistics
        self.training_step = 0
        self.losses = []

        logger.info(
            "TemporalFactoredTrainer initialized",
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_freq=target_update_freq,
            combination_method=combination_method,
        )

    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """Perform one training step."""

        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0}

        # Sample batch from replay buffer
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            _,  # n_step_returns (unused)
            _,  # n_step_states (unused)
            weights,
            indices,
        ) = self.replay_buffer.sample(batch_size)

        # OPTIMIZED: Convert to numpy arrays first, then to tensors
        states_np = np.array(states) if not isinstance(states, np.ndarray) else states
        next_states_np = (
            np.array(next_states)
            if not isinstance(next_states, np.ndarray)
            else next_states
        )
        rewards_np = (
            np.array(rewards) if not isinstance(rewards, np.ndarray) else rewards
        )
        dones_np = np.array(dones) if not isinstance(dones, np.ndarray) else dones
        weights_np = (
            np.array(weights) if not isinstance(weights, np.ndarray) else weights
        )

        # Get device from policy network
        device = next(self.policy_network.parameters()).device

        # Convert numpy arrays to tensors and move to device
        states = torch.FloatTensor(states_np).to(device)
        next_states = torch.FloatTensor(next_states_np).to(device)
        rewards = torch.FloatTensor(rewards_np).to(device)
        dones = torch.BoolTensor(dones_np).to(device)
        weights = torch.FloatTensor(weights_np).to(device)

        # Decode action components from encoded integers
        healthcare_actions = []
        timing_actions = []
        schedule_actions = []
        communication_actions = []

        for action in actions:
            # Decode the action components
            communication_action = action % 4
            schedule_action = (action // 4) % 4
            timing_action = (action // 16) % 8
            healthcare_action = action // 128

            healthcare_actions.append(healthcare_action)
            timing_actions.append(timing_action)
            schedule_actions.append(schedule_action)
            communication_actions.append(communication_action)

        healthcare_actions = torch.LongTensor(healthcare_actions).to(device)
        timing_actions = torch.LongTensor(timing_actions).to(device)
        schedule_actions = torch.LongTensor(schedule_actions).to(device)
        communication_actions = torch.LongTensor(communication_actions).to(device)

        # Forward pass through policy network
        policy_outputs = self.policy_network(states)

        # Forward pass through target network
        with torch.no_grad():
            target_outputs = self.target_network(next_states)

        # Compute factored losses
        losses = {}
        total_loss = 0.0

        # Healthcare action loss
        if self.policy_network.use_distributional:
            healthcare_loss = self._compute_distributional_loss(
                policy_outputs["healthcare"],
                target_outputs["healthcare"],
                healthcare_actions,
                rewards,
                dones,
            )
        else:
            healthcare_loss = self._compute_standard_loss(
                policy_outputs["healthcare"],
                target_outputs["healthcare"],
                healthcare_actions,
                rewards,
                dones,
            )
        losses["healthcare"] = healthcare_loss
        total_loss += healthcare_loss

        # Timing action loss
        if self.policy_network.use_distributional:
            timing_loss = self._compute_distributional_loss(
                policy_outputs["timing"],
                target_outputs["timing"],
                timing_actions,
                rewards,
                dones,
            )
        else:
            timing_loss = self._compute_standard_loss(
                policy_outputs["timing"],
                target_outputs["timing"],
                timing_actions,
                rewards,
                dones,
            )
        losses["timing"] = timing_loss
        total_loss += timing_loss

        # Schedule action loss
        if self.policy_network.use_distributional:
            schedule_loss = self._compute_distributional_loss(
                policy_outputs["schedule"],
                target_outputs["schedule"],
                schedule_actions,
                rewards,
                dones,
            )
        else:
            schedule_loss = self._compute_standard_loss(
                policy_outputs["schedule"],
                target_outputs["schedule"],
                schedule_actions,
                rewards,
                dones,
            )
        losses["schedule"] = schedule_loss
        total_loss += schedule_loss

        # Communication action loss
        if self.policy_network.use_distributional:
            communication_loss = self._compute_distributional_loss(
                policy_outputs["communication"],
                target_outputs["communication"],
                communication_actions,
                rewards,
                dones,
            )
        else:
            communication_loss = self._compute_standard_loss(
                policy_outputs["communication"],
                target_outputs["communication"],
                communication_actions,
                rewards,
                dones,
            )
        losses["communication"] = communication_loss
        total_loss += communication_loss

        # Add adaptive CQL loss and monotonicity regularizer for healthcare actions only
        try:
            adaptive_cql_loss, _, _ = compute_adaptive_cql_loss(
                policy_outputs["healthcare"],
                healthcare_actions,
                target_outputs["healthcare"],
                rewards,
                dones,
                states,
                gamma=self.gamma
            )

            # Add monotonicity regularizer
            monotonicity_loss = compute_monotonicity_regularizer(
                policy_outputs["healthcare"],
                states,
                refer_action_idx=1,  # REFER action index
                beta=0.1
            )

            # Combine all losses
            enhanced_total_loss = total_loss + adaptive_cql_loss + monotonicity_loss

        except Exception as e:
            # Fallback to standard loss if enhanced components fail
            logger.warning(f"Enhanced loss components failed, using standard loss: {e}")
            enhanced_total_loss = total_loss
            adaptive_cql_loss = torch.tensor(0.0, device=total_loss.device)
            monotonicity_loss = torch.tensor(0.0, device=total_loss.device)

        # Apply importance sampling weights
        weighted_loss = (enhanced_total_loss * weights).mean()

        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()

        # Update priorities in replay buffer with risk-aware adjustment
        # Compute per-sample TD errors for priority updates
        with torch.no_grad():
            # Use the healthcare loss as a proxy for TD error (per sample)
            # In a full implementation, this would be the actual TD error per sample
            per_sample_errors = torch.abs(healthcare_loss.detach()).expand(len(indices))
            if per_sample_errors.dim() == 0:
                per_sample_errors = per_sample_errors.unsqueeze(0).expand(len(indices))

        # Pass states and actions for risk-aware priority adjustment
        self.replay_buffer.update_priorities(
            indices,
            per_sample_errors,
            states=states,
            actions=healthcare_actions
        )

        # Record loss
        loss_value = weighted_loss.item()
        self.losses.append(loss_value)

        # Return enhanced loss components
        result = {
            "total_loss": loss_value,
            "healthcare_loss": healthcare_loss.item(),
            "timing_loss": timing_loss.item(),
            "schedule_loss": schedule_loss.item(),
            "communication_loss": communication_loss.item(),
            "adaptive_cql_loss": adaptive_cql_loss.item(),
            "monotonicity_loss": monotonicity_loss.item(),
        }

        return result

    def _compute_standard_loss(
        self,
        policy_q_values: torch.Tensor,
        target_q_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard DQN loss for one dimension."""

        # Get Q-values for selected actions
        current_q_values = policy_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        next_q_values = target_q_values.max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        return loss

    def _compute_distributional_loss(
        self,
        policy_dist: torch.Tensor,
        target_dist: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distributional loss for one dimension."""

        batch_size = policy_dist.size(0)
        num_atoms = policy_dist.size(-1)

        # Get distribution for selected actions
        policy_dist_selected = policy_dist[range(batch_size), actions]

        # For simplicity, use cross-entropy loss with target distribution
        # In a full implementation, this would be the Wasserstein loss
        with torch.no_grad():
            # Simple target: use max action from target network
            target_actions = target_dist.sum(dim=-1).argmax(dim=1)
            target_dist_selected = target_dist[range(batch_size), target_actions]

        # Cross-entropy loss
        loss = -torch.sum(
            target_dist_selected * torch.log(policy_dist_selected + 1e-8), dim=1
        ).mean()

        return loss

    def _update_target_network(self):
        """Update target network with policy network weights."""
        self.target_network.load_state_dict(self.policy_network.state_dict())
        logger.info("🎯 Target network updated", step=self.training_step)

    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        if not self.losses:
            return {"avg_loss": 0.0, "recent_loss": 0.0}

        return {
            "avg_loss": np.mean(self.losses),
            "recent_loss": np.mean(self.losses[-100:])
            if len(self.losses) >= 100
            else np.mean(self.losses),
            "total_steps": self.training_step,
        }


def train_temporal_factored_dqn(
    training_data: pd.DataFrame,
    num_episodes: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    replay_buffer_capacity: int = 10000,
) -> Tuple[TemporalFactoredDQN, TemporalFactoredTrainer]:
    """
    Train a temporal factored DQN on healthcare communication data.

    Args:
        training_data: DataFrame with temporal training data
        num_episodes: Number of training episodes
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        replay_buffer_capacity: Capacity of replay buffer

    Returns:
        Tuple of (trained_network, trainer)
    """
    logger.info(
        "Training Temporal Factored DQN",
        training_samples=len(training_data),
        episodes=num_episodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Create networks
    policy_network = TemporalFactoredDQN(
        input_dim=27, use_dueling=True, use_noisy=True, use_distributional=True
    )

    target_network = TemporalFactoredDQN(
        input_dim=27, use_dueling=True, use_noisy=True, use_distributional=True
    )

    # Initialize target network with policy network weights
    target_network.load_state_dict(policy_network.state_dict())

    # Create replay buffer
    replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_capacity)

    # Create trainer
    trainer = TemporalFactoredTrainer(
        policy_network=policy_network,
        target_network=target_network,
        replay_buffer=replay_buffer,
        learning_rate=learning_rate,
    )

    # Fill replay buffer with training data
    logger.info("Filling replay buffer...")
    for _, row in training_data.iterrows():
        # Extract state features (first 27 columns)
        state_features = row[EnhancedPatientState.get_feature_names()].values

        # Extract action components - convert to single integer for replay buffer compatibility
        healthcare_action = int(row["healthcare_action"])
        timing_action = int(row["time_horizon"])
        schedule_action = int(row["time_of_day"])
        communication_action = int(row["communication_channel"])

        # Encode as single integer (we'll decode in training)
        # This is a temporary solution - ideally we'd modify the replay buffer
        action = (
            healthcare_action * 8 * 4 * 4
            + timing_action * 4 * 4
            + schedule_action * 4
            + communication_action
        )

        reward = row["reward"]
        done = row["done"]

        # For simplicity, use same state as next state (could be improved)
        next_state = state_features

        replay_buffer.add(state_features, action, reward, next_state, done)

    logger.info("Replay buffer filled", experiences=len(replay_buffer))

    # Training loop
    logger.info("Starting training...")

    for episode in range(num_episodes):
        # Reset noise in noisy layers
        policy_network.reset_noise()
        target_network.reset_noise()

        # Perform training step
        loss_info = trainer.train_step(batch_size)

        # Log progress
        if episode % 100 == 0:
            stats = trainer.get_training_stats()
            logger.log_training_progress(
                episode=episode,
                total_episodes=num_episodes,
                metrics={
                    "current_loss": loss_info["total_loss"],
                    "avg_loss": stats["avg_loss"],
                },
            )

    logger.info("Training completed!")

    # Final statistics
    final_stats = trainer.get_training_stats()
    logger.info(
        "Final Training Statistics",
        total_steps=final_stats["total_steps"],
        avg_loss=final_stats["avg_loss"],
        recent_loss=final_stats["recent_loss"],
    )

    return policy_network, trainer


def demonstrate_temporal_training():
    """Demonstrate temporal factored DQN training."""
    logger.info("Temporal Factored DQN Training Demo")
    logger.info("=" * 60)

    # Generate training data
    training_data = generate_temporal_training_data(2000)

    # Train model
    trained_network, trainer = train_temporal_factored_dqn(
        training_data=training_data,
        num_episodes=200,
        batch_size=32,
        learning_rate=0.001,
    )

    # Test prediction
    logger.info("Testing trained model...")

    # Create sample patient state
    sample_features = np.random.randn(27)
    predicted_action = trained_network.predict_temporal_action(sample_features)

    logger.info("Sample prediction", prediction=predicted_action.to_string())

    logger.info("Temporal training demonstration completed!")


if __name__ == "__main__":
    demonstrate_temporal_training()
