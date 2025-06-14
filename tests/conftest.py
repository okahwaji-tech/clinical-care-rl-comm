"""Pytest configuration and shared fixtures for Enhanced Healthcare DQN tests.

This module provides common fixtures and configuration for testing the enhanced
Temporal Factored Rainbow DQN healthcare system with all 7 enhancements.
"""

import pytest
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any
import tempfile
import os

from core.temporal_rainbow_dqn import TemporalFactoredDQN
from core.temporal_actions import (
    EnhancedPatientState, TemporalAction, TemporalActionSpace,
    HealthcareAction, TimeHorizon, TimeOfDay, CommunicationChannel
)
from core.temporal_training_data import (
    generate_stratified_temporal_training_data, TemporalRewardFunction
)
from core.replay_buffer import PrioritizedReplayBuffer
from core.temporal_training_loop import TemporalFactoredTrainer


@pytest.fixture
def sample_patient_states() -> List[EnhancedPatientState]:
    """Generate sample patient states across different risk levels."""
    states = []
    
    # Low risk patient
    states.append(EnhancedPatientState(
        risk_score=0.2, age=35, comorbidities=1, bmi=24, systolic_bp=120, diastolic_bp=80,
        heart_rate=70, current_hour=10, day_of_week=2, urgency_level=0.3,
        last_contact_hours_ago=48, next_appointment_days=14,
        preferred_channel=CommunicationChannel.SMS, preferred_time=TimeOfDay.MORNING_9AM,
        sms_success_rate=0.9, email_success_rate=0.8, phone_success_rate=0.7, mail_success_rate=0.6,
        medication_adherence=0.9, appointment_compliance=0.9, response_time_preference=0.6
    ))
    
    # Medium risk patient
    states.append(EnhancedPatientState(
        risk_score=0.5, age=55, comorbidities=3, bmi=28, systolic_bp=140, diastolic_bp=90,
        heart_rate=80, current_hour=14, day_of_week=3, urgency_level=0.5,
        last_contact_hours_ago=12, next_appointment_days=7,
        preferred_channel=CommunicationChannel.EMAIL, preferred_time=TimeOfDay.AFTERNOON_2PM,
        sms_success_rate=0.7, email_success_rate=0.9, phone_success_rate=0.8, mail_success_rate=0.5,
        medication_adherence=0.7, appointment_compliance=0.8, response_time_preference=0.4
    ))
    
    # High risk patient
    states.append(EnhancedPatientState(
        risk_score=0.8, age=75, comorbidities=6, bmi=35, systolic_bp=170, diastolic_bp=105,
        heart_rate=95, current_hour=16, day_of_week=1, urgency_level=0.8,
        last_contact_hours_ago=2, next_appointment_days=1,
        preferred_channel=CommunicationChannel.PHONE, preferred_time=TimeOfDay.EVENING_6PM,
        sms_success_rate=0.5, email_success_rate=0.6, phone_success_rate=0.9, mail_success_rate=0.4,
        medication_adherence=0.6, appointment_compliance=0.7, response_time_preference=0.2
    ))
    
    return states


@pytest.fixture
def mock_model() -> TemporalFactoredDQN:
    """Create a lightweight model for testing."""
    return TemporalFactoredDQN(
        input_dim=27,
        use_dueling=True,
        use_noisy=True,
        use_distributional=True,
        hidden_dim=64  # Smaller for faster tests
    )


@pytest.fixture
def small_training_data() -> pd.DataFrame:
    """Generate small training dataset for fast tests."""
    return generate_stratified_temporal_training_data(1000)


@pytest.fixture
def action_space() -> TemporalActionSpace:
    """Create temporal action space for testing."""
    return TemporalActionSpace()


@pytest.fixture
def reward_function() -> TemporalRewardFunction:
    """Create reward function for testing."""
    return TemporalRewardFunction()


@pytest.fixture
def replay_buffer() -> PrioritizedReplayBuffer:
    """Create replay buffer for testing."""
    return PrioritizedReplayBuffer(
        capacity=1000,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=1000
    )


@pytest.fixture
def enhanced_trainer(mock_model, replay_buffer) -> TemporalFactoredTrainer:
    """Create enhanced trainer with all components."""
    target_model = TemporalFactoredDQN(
        input_dim=27,
        use_dueling=True,
        use_noisy=True,
        use_distributional=True,
        hidden_dim=64
    )
    target_model.load_state_dict(mock_model.state_dict())
    
    return TemporalFactoredTrainer(
        policy_network=mock_model,
        target_network=target_model,
        replay_buffer=replay_buffer,
        learning_rate=0.001
    )


@pytest.fixture
def temp_model_path():
    """Create temporary file path for model saving/loading tests."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# Test utilities
def assert_clinical_appropriateness(action: TemporalAction, patient_state: EnhancedPatientState) -> None:
    """Assert that an action is clinically appropriate for the patient state."""
    # High-risk patients should get more intensive actions
    if patient_state.risk_score >= 0.7:
        assert action.healthcare_action in [HealthcareAction.REFER, HealthcareAction.MEDICATE], \
            f"High-risk patient should get REFER or MEDICATE, got {action.healthcare_action.name}"
    
    # Low-risk patients should get conservative actions
    elif patient_state.risk_score <= 0.3:
        assert action.healthcare_action in [HealthcareAction.MONITOR, HealthcareAction.DISCHARGE], \
            f"Low-risk patient should get MONITOR or DISCHARGE, got {action.healthcare_action.name}"


def assert_risk_stratified_behavior(
    low_risk_action: TemporalAction, 
    high_risk_action: TemporalAction
) -> None:
    """Assert that actions differ appropriately between risk levels."""
    # High-risk should have more urgent time horizons
    high_risk_urgency = high_risk_action.time_horizon.value
    low_risk_urgency = low_risk_action.time_horizon.value
    
    # Lower values = more urgent (IMMEDIATE=0, ONE_HOUR=1, etc.)
    assert high_risk_urgency <= low_risk_urgency, \
        "High-risk patients should have more urgent time horizons"


def assert_temporal_consistency(action: TemporalAction) -> None:
    """Assert that temporal action components are consistent."""
    # All components should be valid enum values
    assert isinstance(action.healthcare_action, HealthcareAction)
    assert isinstance(action.time_horizon, TimeHorizon)
    assert isinstance(action.time_of_day, TimeOfDay)
    assert isinstance(action.communication_channel, CommunicationChannel)
    
    # Action string should be properly formatted
    action_string = action.to_string()
    assert len(action_string.split('_')) == 4, "Action string should have 4 components"


def assert_enhanced_features_active(trainer: TemporalFactoredTrainer) -> None:
    """Assert that all enhanced features are properly configured."""
    # Check that trainer has enhanced components
    assert hasattr(trainer, 'policy_network')
    assert hasattr(trainer, 'target_network')
    assert hasattr(trainer, 'replay_buffer')
    
    # Check that replay buffer has risk-aware prioritization
    assert hasattr(trainer.replay_buffer, '_apply_risk_aware_adjustment')
    
    # Check that networks have enhanced features
    assert trainer.policy_network.use_dueling
    assert trainer.policy_network.use_noisy
    assert trainer.policy_network.use_distributional


@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        'small_dataset': 1000,
        'medium_dataset': 10000,
        'large_dataset': 100000,
        'training_steps': 100,
        'batch_size': 32
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "clinical: Clinical validation tests")
    config.addinivalue_line("markers", "slow: Slow tests that take more than 30 seconds")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "clinical" in str(item.fspath):
            item.add_marker(pytest.mark.clinical)
        
        # Add slow marker for tests that might take long
        if any(keyword in item.name.lower() for keyword in ['large', 'benchmark', 'training']):
            item.add_marker(pytest.mark.slow)
