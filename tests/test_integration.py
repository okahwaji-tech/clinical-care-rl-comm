"""
Integration tests for the healthcare DQN system.

Tests the interaction between different components and
end-to-end functionality of the system.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from typing import Dict, Any

from core.temporal_actions import create_temporal_action_space, EnhancedPatientState
from core.temporal_rainbow_dqn import create_temporal_factored_dqn
from core.factory_patterns import (
    create_standard_healthcare_dqn, 
    create_refactored_healthcare_dqn,
    create_lightweight_healthcare_dqn
)
from core.dependency_injection import service_scope
from tests import (
    TestBase, PerformanceTestMixin, TestConfig,
    assert_valid_temporal_action, assert_valid_q_values, assert_valid_network_output
)


@pytest.mark.integration
class TestSystemIntegration(TestBase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_prediction(self, sample_patient_state):
        """Test end-to-end prediction pipeline."""
        # Create action space
        action_space = create_temporal_action_space()
        
        # Create network
        network = create_temporal_factored_dqn()
        
        # Create patient state
        patient = EnhancedPatientState(
            risk_score=0.7, age=65, comorbidities=2, bmi=28.5,
            systolic_bp=140, diastolic_bp=90, heart_rate=85,
            current_hour=14, day_of_week=2, urgency_level=0.6,
            last_contact_hours_ago=48, next_appointment_days=7,
            preferred_channel=action_space.communication_channels[2],  # PHONE
            preferred_time=action_space.times_of_day[0],  # MORNING_9AM
            sms_success_rate=0.6, email_success_rate=0.8,
            phone_success_rate=0.95, mail_success_rate=0.4,
            medication_adherence=0.75, appointment_compliance=0.85,
            response_time_preference=0.8
        )
        
        # Get features
        features = patient.to_feature_vector()
        
        # Predict action
        predicted_action = network.predict_temporal_action(features)
        
        # Validate prediction
        assert_valid_temporal_action(predicted_action)
        
        # Test that prediction is deterministic (without exploration)
        predicted_action2 = network.predict_temporal_action(features, use_exploration=False)
        assert predicted_action.to_string() == predicted_action2.to_string()
    
    def test_batch_processing(self, sample_patient_state):
        """Test batch processing of multiple patients."""
        network = create_temporal_factored_dqn()
        
        # Create batch of patient states
        batch_size = TestConfig.BATCH_SIZE
        batch_states = np.tile(sample_patient_state, (batch_size, 1))
        
        # Add some noise to make them different
        batch_states += np.random.normal(0, 0.01, batch_states.shape)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(batch_states)
        
        # Forward pass
        outputs = network(state_tensor)
        
        # Validate outputs
        assert_valid_network_output(outputs, batch_size)
        
        # Check each dimension has correct shape
        expected_shapes = {
            'healthcare': (batch_size, 5),
            'timing': (batch_size, 8),
            'schedule': (batch_size, 4),
            'communication': (batch_size, 4)
        }
        
        for dimension, expected_shape in expected_shapes.items():
            assert dimension in outputs
            self.assert_tensor_shape(outputs[dimension], expected_shape)
    
    def test_factory_pattern_integration(self):
        """Test integration with factory patterns."""
        # Test standard system
        standard_system = create_standard_healthcare_dqn()
        self._validate_system_components(standard_system)
        
        # Test refactored system
        refactored_system = create_refactored_healthcare_dqn()
        self._validate_system_components(refactored_system)
        
        # Test lightweight system
        lightweight_system = create_lightweight_healthcare_dqn()
        self._validate_system_components(lightweight_system)
    
    def _validate_system_components(self, system: Dict[str, Any]):
        """Validate that system has all required components."""
        required_components = [
            "policy_network", "target_network", "replay_buffer", 
            "trainer", "device", "network_config", "training_config"
        ]
        
        for component in required_components:
            assert component in system, f"Missing component: {component}"
        
        # Test that networks can process input
        sample_input = torch.randn(1, TestConfig.INPUT_DIM)
        policy_output = system["policy_network"](sample_input)
        target_output = system["target_network"](sample_input)
        
        assert_valid_network_output(policy_output, 1)
        assert_valid_network_output(target_output, 1)
    
    def test_dependency_injection_integration(self):
        """Test integration with dependency injection."""
        with service_scope() as container:
            # Test that we can get services
            from core.base_interfaces import NetworkConfig, TrainingConfig
            
            network_config = container.get(NetworkConfig, "default_network_config")
            training_config = container.get(TrainingConfig, "default_training_config")
            
            assert isinstance(network_config, NetworkConfig)
            assert isinstance(training_config, TrainingConfig)
            
            # Test that configurations are valid
            assert network_config.input_dim > 0
            assert network_config.hidden_dim > 0
            assert training_config.learning_rate > 0
            assert training_config.batch_size > 0


@pytest.mark.integration
class TestTrainingIntegration(TestBase):
    """Integration tests for training pipeline."""
    
    def test_training_step_integration(self, sample_batch):
        """Test a complete training step."""
        # Create lightweight system for faster testing
        system = create_lightweight_healthcare_dqn()
        
        policy_network = system["policy_network"]
        target_network = system["target_network"]
        trainer = system["trainer"]
        
        # Perform training step
        metrics = trainer.train_step(sample_batch)
        
        # Validate metrics
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert np.isfinite(metrics["loss"])
        
        # Check that networks are still functional
        sample_input = torch.randn(1, TestConfig.INPUT_DIM)
        policy_output = policy_network(sample_input)
        target_output = target_network(sample_input)
        
        assert_valid_network_output(policy_output, 1)
        assert_valid_network_output(target_output, 1)
    
    def test_replay_buffer_integration(self):
        """Test replay buffer integration."""
        system = create_lightweight_healthcare_dqn()
        replay_buffer = system["replay_buffer"]
        
        # Add experiences
        for _ in range(100):
            experience = self.create_sample_experience()
            replay_buffer.add(
                state=experience["state"],
                action=experience["action"],
                reward=experience["reward"],
                next_state=experience["next_state"],
                done=experience["done"]
            )
        
        # Sample batch
        if len(replay_buffer) >= TestConfig.BATCH_SIZE:
            batch = replay_buffer.sample(TestConfig.BATCH_SIZE)
            
            # Validate batch
            assert len(batch) >= 5  # states, actions, rewards, next_states, dones
            
            states, actions, rewards, next_states, dones = batch[:5]
            
            self.assert_tensor_shape(states, (TestConfig.BATCH_SIZE, TestConfig.INPUT_DIM))
            self.assert_tensor_shape(actions, (TestConfig.BATCH_SIZE,))
            self.assert_tensor_shape(rewards, (TestConfig.BATCH_SIZE,))
            self.assert_tensor_shape(next_states, (TestConfig.BATCH_SIZE, TestConfig.INPUT_DIM))
            self.assert_tensor_shape(dones, (TestConfig.BATCH_SIZE,))


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration(TestBase, PerformanceTestMixin):
    """Performance integration tests."""
    
    def test_prediction_throughput(self):
        """Test prediction throughput."""
        network = create_temporal_factored_dqn()
        sample_state = np.random.randn(TestConfig.INPUT_DIM)
        
        def predict():
            return network.predict_temporal_action(sample_state, use_exploration=False)
        
        # Measure throughput
        throughput = self.measure_throughput(predict, iterations=1000, warmup=100)
        
        # Should achieve at least 2000 predictions per second
        min_throughput = 2000
        assert throughput >= min_throughput, f"Prediction throughput too low: {throughput:.1f} < {min_throughput}"
    
    def test_batch_processing_throughput(self):
        """Test batch processing throughput."""
        network = create_temporal_factored_dqn()
        batch_size = 32
        sample_batch = torch.randn(batch_size, TestConfig.INPUT_DIM)
        
        def process_batch():
            with torch.no_grad():
                return network(sample_batch)
        
        # Measure throughput
        throughput = self.measure_throughput(process_batch, iterations=100, warmup=10)
        
        # Calculate samples per second
        samples_per_second = throughput * batch_size
        
        # Should process at least 10,000 samples per second
        min_samples_per_second = 10000
        assert samples_per_second >= min_samples_per_second, \
            f"Batch processing too slow: {samples_per_second:.1f} < {min_samples_per_second}"
    
    def test_memory_usage(self):
        """Test memory usage during operation."""
        def create_and_use_network():
            network = create_temporal_factored_dqn()
            sample_input = torch.randn(TestConfig.BATCH_SIZE, TestConfig.INPUT_DIM)
            
            # Forward pass
            outputs = network(sample_input)
            
            # Backward pass simulation
            if isinstance(outputs, dict):
                loss = sum(output.sum() for output in outputs.values())
            else:
                loss = outputs.sum()
            
            loss.backward()
            
            return network
        
        # Measure memory usage
        peak_memory_mb = self.measure_memory_usage(create_and_use_network)
        
        # Should use less than 500MB
        max_memory_mb = 500
        assert peak_memory_mb <= max_memory_mb, \
            f"Memory usage too high: {peak_memory_mb:.1f}MB > {max_memory_mb}MB"


@pytest.mark.integration
class TestRobustnessIntegration(TestBase):
    """Robustness integration tests."""
    
    def test_edge_case_inputs(self):
        """Test system with edge case inputs."""
        network = create_temporal_factored_dqn()
        
        # Test with zeros
        zero_input = np.zeros(TestConfig.INPUT_DIM)
        action = network.predict_temporal_action(zero_input)
        assert_valid_temporal_action(action)
        
        # Test with large values
        large_input = np.full(TestConfig.INPUT_DIM, 1000.0)
        action = network.predict_temporal_action(large_input)
        assert_valid_temporal_action(action)
        
        # Test with negative values
        negative_input = np.full(TestConfig.INPUT_DIM, -100.0)
        action = network.predict_temporal_action(negative_input)
        assert_valid_temporal_action(action)
    
    def test_network_state_consistency(self):
        """Test that network state remains consistent."""
        network = create_temporal_factored_dqn()
        sample_input = np.random.randn(TestConfig.INPUT_DIM)
        
        # Get initial prediction
        network.eval()
        initial_action = network.predict_temporal_action(sample_input, use_exploration=False)
        
        # Perform some operations
        batch_input = torch.randn(TestConfig.BATCH_SIZE, TestConfig.INPUT_DIM)
        _ = network(batch_input)
        
        # Check prediction is still the same
        network.eval()
        final_action = network.predict_temporal_action(sample_input, use_exploration=False)
        
        assert initial_action.to_string() == final_action.to_string()
    
    def test_concurrent_access(self):
        """Test concurrent access to system components."""
        import threading
        import time
        
        network = create_temporal_factored_dqn()
        results = []
        errors = []
        
        def worker():
            try:
                sample_input = np.random.randn(TestConfig.INPUT_DIM)
                action = network.predict_temporal_action(sample_input)
                results.append(action)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        
        # Validate all results
        for action in results:
            assert_valid_temporal_action(action)
