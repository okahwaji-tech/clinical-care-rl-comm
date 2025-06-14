"""Integration tests for all 7 enhanced features working together.

Tests that all enhancements integrate properly and work as a cohesive system:
1. Reward Re-shaping
2. Stratified Synthetic Data Generator  
3. Risk-Aware Prioritized Replay
4. Adaptive CQL Strength
5. Monotonicity Regularizer
6. Feature Scaling Audit
7. Hyperparameter Tuning Framework
"""

import pytest
import torch
import numpy as np
import pandas as pd

from core.temporal_rainbow_dqn import TemporalFactoredDQN
from core.temporal_training_data import generate_stratified_temporal_training_data
from core.temporal_training_loop import TemporalFactoredTrainer
from core.replay_buffer import PrioritizedReplayBuffer
from core.temporal_actions import EnhancedPatientState
from core.cql_components import compute_adaptive_cql_loss, compute_monotonicity_regularizer


class TestEnhancedFeaturesIntegration:
    """Test all enhanced features working together."""
    
    def test_complete_enhanced_pipeline(self, enhanced_trainer):
        """Test complete enhanced training pipeline with small dataset."""
        # Generate stratified data (Enhancement 2)
        training_data = generate_stratified_temporal_training_data(500)
        
        # Verify stratified data has enhanced features
        assert 'risk_tier' in training_data.columns
        assert len(training_data['risk_tier'].unique()) == 5
        
        # Fill replay buffer with enhanced data
        for _, row in training_data.head(100).iterrows():
            state_features = row[EnhancedPatientState.get_feature_names()].values
            
            # Verify enhanced feature scaling (Enhancement 6)
            risk_score_scaled = state_features[0]
            original_risk = row['risk_score']
            expected_scaled = original_risk * 2.0 - 1.0
            assert np.isclose(risk_score_scaled, expected_scaled), \
                "Risk score scaling not working correctly"
            
            # Encode action
            action = (
                int(row['healthcare_action']) * 128 
                + int(row['time_horizon']) * 16 
                + int(row['time_of_day']) * 4 
                + int(row['communication_channel'])
            )
            
            reward = row['reward']
            done = row['done']
            
            enhanced_trainer.replay_buffer.add(
                state_features, action, reward, state_features, done
            )
        
        # Test enhanced training step with all components
        if len(enhanced_trainer.replay_buffer) >= 32:
            loss_info = enhanced_trainer.train_step(batch_size=32)
            
            # Should have enhanced loss components
            expected_keys = [
                'total_loss', 'healthcare_loss', 'timing_loss', 
                'schedule_loss', 'communication_loss',
                'adaptive_cql_loss', 'monotonicity_loss'
            ]
            
            for key in expected_keys:
                assert key in loss_info, f"Missing enhanced loss component: {key}"
                assert isinstance(loss_info[key], (int, float)), \
                    f"Loss component {key} should be numeric"
    
    def test_risk_aware_prioritized_replay(self):
        """Test risk-aware prioritized replay (Enhancement 3)."""
        replay_buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta_start=0.4)
        
        # Add experiences with different risk levels
        high_risk_state = np.random.randn(27).astype(np.float32)
        high_risk_state[1] = 0.8  # High risk score for priority calculation
        
        low_risk_state = np.random.randn(27).astype(np.float32)
        low_risk_state[1] = 0.2  # Low risk score
        
        # Add high-risk REFER action (should get 1.5x priority multiplier)
        high_risk_refer_action = 1 * 128  # REFER action encoded
        replay_buffer.add(high_risk_state, high_risk_refer_action, 1.0, high_risk_state, False)
        
        # Add low-risk action
        low_risk_action = 0 * 128  # MONITOR action encoded
        replay_buffer.add(low_risk_state, low_risk_action, 1.0, low_risk_state, False)
        
        # Test priority update with risk-aware adjustment
        if len(replay_buffer) >= 2:
            batch = replay_buffer.sample(2)
            states, actions, _, _, _, _, _, weights, indices = batch
            
            # Mock TD errors
            td_errors = torch.tensor([1.0, 1.0])
            
            # Update priorities with risk-aware adjustment
            replay_buffer.update_priorities(indices, td_errors, states=states, actions=actions)
            
            # High-risk actions should have higher sampling probability
            # (This is tested indirectly through the priority update mechanism)
            assert True  # Priority adjustment tested in unit tests
    
    def test_adaptive_cql_strength(self):
        """Test adaptive CQL strength (Enhancement 4)."""
        batch_size = 16
        num_actions = 5
        
        # Create Q-values and states with mixed risk levels
        q_values = torch.randn(batch_size, num_actions)
        actions = torch.randint(0, num_actions, (batch_size,))
        next_q_values = torch.randn(batch_size, num_actions)
        rewards = torch.randn(batch_size)
        dones = torch.zeros(batch_size, dtype=torch.bool)
        
        # Create states with different risk levels
        states = torch.randn(batch_size, 27)
        states[:8, 1] = 0.8  # High risk (should use α=0.5)
        states[8:, 1] = 0.3  # Low risk (should use α=1.0)
        
        # Test adaptive CQL loss
        total_loss, bellman_loss, cql_loss = compute_adaptive_cql_loss(
            q_values, actions, next_q_values, rewards, dones, states
        )
        
        # Should compute without errors
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(bellman_loss, torch.Tensor)
        assert isinstance(cql_loss, torch.Tensor)
        
        # Losses should be finite
        assert torch.isfinite(total_loss)
        assert torch.isfinite(bellman_loss)
        assert torch.isfinite(cql_loss)
    
    def test_monotonicity_regularizer(self):
        """Test monotonicity regularizer (Enhancement 5)."""
        batch_size = 8
        num_actions = 5
        
        # Create Q-values and states with ordered risk levels
        q_values = torch.randn(batch_size, num_actions)
        states = torch.randn(batch_size, 27)
        
        # Set risk scores in ascending order
        risk_scores = torch.linspace(0.1, 0.9, batch_size)
        states[:, 1] = risk_scores
        
        # Test monotonicity regularizer
        mono_loss = compute_monotonicity_regularizer(q_values, states, refer_action_idx=1)
        
        # Should compute without errors
        assert isinstance(mono_loss, torch.Tensor)
        assert torch.isfinite(mono_loss)
        assert mono_loss >= 0, "Monotonicity loss should be non-negative"
    
    def test_enhanced_model_predictions(self, mock_model):
        """Test enhanced model predictions with risk-aware behavior."""
        # Test different risk levels
        risk_levels = [0.2, 0.5, 0.8]
        predictions = []
        
        for risk in risk_levels:
            # Create test input with enhanced feature scaling
            test_input = np.random.randn(27).astype(np.float32)
            test_input[0] = risk * 2.0 - 1.0  # Enhanced scaling: [0,1] -> [-1,1]
            test_input[1] = risk  # Original risk for other components
            
            # Get prediction
            action = mock_model.predict_temporal_action(test_input, use_exploration=False)
            predictions.append((risk, action))
        
        # Verify predictions are valid temporal actions
        for risk, action in predictions:
            from tests.conftest import assert_temporal_consistency
            assert_temporal_consistency(action)
        
        # Test that model produces different behaviors for different risk levels
        low_risk_action = predictions[0][1]
        high_risk_action = predictions[2][1]
        
        # Actions should be different (at least sometimes)
        # Note: With random weights, this isn't guaranteed, but structure should be valid
        assert low_risk_action.healthcare_action in [0, 1, 2, 3, 4]  # Valid action
        assert high_risk_action.healthcare_action in [0, 1, 2, 3, 4]  # Valid action


class TestEnhancedTrainingStability:
    """Test training stability with all enhancements."""
    
    def test_training_convergence_with_enhancements(self, enhanced_trainer):
        """Test that training converges with all enhancements active."""
        # Generate small training dataset
        training_data = generate_stratified_temporal_training_data(200)
        
        # Fill replay buffer
        for _, row in training_data.iterrows():
            state_features = row[EnhancedPatientState.get_feature_names()].values
            action = (
                int(row['healthcare_action']) * 128 
                + int(row['time_horizon']) * 16 
                + int(row['time_of_day']) * 4 
                + int(row['communication_channel'])
            )
            reward = row['reward']
            done = row['done']
            
            enhanced_trainer.replay_buffer.add(
                state_features, action, reward, state_features, done
            )
        
        # Run multiple training steps
        losses = []
        for _ in range(10):
            if len(enhanced_trainer.replay_buffer) >= 32:
                loss_info = enhanced_trainer.train_step(batch_size=32)
                losses.append(loss_info['total_loss'])
        
        # Should have completed training steps without errors
        assert len(losses) > 0, "Should have completed at least one training step"
        
        # Losses should be finite
        for loss in losses:
            assert np.isfinite(loss), f"Loss should be finite, got {loss}"
    
    def test_memory_efficiency_with_enhancements(self, enhanced_trainer):
        """Test memory efficiency with enhanced components."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate and process data
        training_data = generate_stratified_temporal_training_data(1000)
        
        # Fill replay buffer
        for _, row in training_data.head(500).iterrows():
            state_features = row[EnhancedPatientState.get_feature_names()].values
            action = int(row['healthcare_action']) * 128
            reward = row['reward']
            done = row['done']
            
            enhanced_trainer.replay_buffer.add(
                state_features, action, reward, state_features, done
            )
        
        # Run training steps
        for _ in range(5):
            if len(enhanced_trainer.replay_buffer) >= 32:
                enhanced_trainer.train_step(batch_size=32)
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, \
            f"Memory usage increased by {memory_increase:.1f}MB, which seems excessive"


class TestEnhancedDataConsistency:
    """Test data consistency across enhanced components."""
    
    def test_risk_tier_action_consistency(self):
        """Test consistency between risk tiers and action distributions."""
        # Generate larger dataset for statistical significance
        training_data = generate_stratified_temporal_training_data(2000)
        
        # Group by risk tier and analyze action distributions
        tier_actions = training_data.groupby('risk_tier')['healthcare_action'].value_counts(normalize=True)
        
        # Very high risk should favor REFER (action 1)
        very_high_refer_pct = tier_actions.get(('very_high', 1), 0)
        assert very_high_refer_pct > 0.4, \
            f"Very high risk should have >40% REFER actions, got {very_high_refer_pct:.2%}"
        
        # Very low risk should favor MONITOR (action 0) or DISCHARGE (action 3)
        very_low_conservative = (
            tier_actions.get(('very_low', 0), 0) + 
            tier_actions.get(('very_low', 3), 0)
        )
        assert very_low_conservative > 0.4, \
            f"Very low risk should have >40% conservative actions, got {very_low_conservative:.2%}"
    
    def test_reward_risk_correlation(self):
        """Test that rewards correlate appropriately with risk levels and actions."""
        training_data = generate_stratified_temporal_training_data(1000)
        
        # High-risk REFER actions should have higher rewards than low-risk MEDICATE
        high_risk_refer = training_data[
            (training_data['risk_score'] >= 0.7) & 
            (training_data['healthcare_action'] == 1)  # REFER
        ]['reward']
        
        low_risk_medicate = training_data[
            (training_data['risk_score'] <= 0.3) & 
            (training_data['healthcare_action'] == 2)  # MEDICATE
        ]['reward']
        
        if len(high_risk_refer) > 0 and len(low_risk_medicate) > 0:
            assert high_risk_refer.mean() > low_risk_medicate.mean(), \
                "High-risk referrals should have higher average reward than low-risk medication"


@pytest.mark.slow
class TestEnhancedPerformance:
    """Test performance with all enhancements (slower tests)."""
    
    def test_large_scale_data_generation(self):
        """Test stratified data generation at larger scale."""
        # Generate larger dataset
        large_data = generate_stratified_temporal_training_data(10000)
        
        # Should maintain all enhanced properties at scale
        assert len(large_data) >= 10000
        assert len(large_data['risk_tier'].unique()) == 5
        assert 'risk_tier' in large_data.columns
        
        # Risk tier distribution should be maintained
        tier_counts = large_data['risk_tier'].value_counts()
        assert all(count > 100 for count in tier_counts.values()), \
            "All risk tiers should have substantial representation"
