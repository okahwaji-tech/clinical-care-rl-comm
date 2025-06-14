"""
Unit tests for temporal actions module.

Tests the temporal action space, enhanced patient state,
and action sampling functionality.
"""

from __future__ import annotations

import pytest
import numpy as np
from typing import Set

from core.temporal_actions import (
    TemporalActionSpace, TemporalAction, EnhancedPatientState,
    HealthcareAction, TimeHorizon, TimeOfDay, CommunicationChannel,
    create_temporal_action_space
)
from tests import (
    TestBase, assert_valid_temporal_action, TestConfig
)


class TestTemporalActionSpace(TestBase):
    """Test cases for TemporalActionSpace."""
    
    def test_initialization(self):
        """Test action space initialization."""
        action_space = TemporalActionSpace()
        
        # Check dimensions
        assert action_space.n_healthcare == 5
        assert action_space.n_time_horizons == 8
        assert action_space.n_times_of_day == 4
        assert action_space.n_communication == 4
        
        # Check total combinations
        expected_total = 5 * 8 * 4 * 4
        assert action_space.total_combinations == expected_total
    
    def test_sample_random_action(self):
        """Test random action sampling."""
        action_space = TemporalActionSpace()
        
        # Sample multiple actions
        actions = [action_space.sample_random_action() for _ in range(100)]
        
        # Check all actions are valid
        for action in actions:
            assert_valid_temporal_action(action)
        
        # Check diversity (should have different actions)
        action_strings = [action.to_string() for action in actions]
        unique_actions = set(action_strings)
        assert len(unique_actions) > 10, "Not enough diversity in random actions"
    
    def test_indices_to_action(self):
        """Test conversion from indices to action."""
        action_space = TemporalActionSpace()
        
        # Test specific indices
        action = action_space.indices_to_action(0, 0, 0, 0)
        assert action.healthcare_action == HealthcareAction.MONITOR
        assert action.time_horizon == TimeHorizon.IMMEDIATE
        assert action.time_of_day == TimeOfDay.MORNING_9AM
        assert action.communication_channel == CommunicationChannel.SMS
        
        # Test maximum indices
        action = action_space.indices_to_action(4, 7, 3, 3)
        assert action.healthcare_action == HealthcareAction.FOLLOWUP
        assert action.time_horizon == TimeHorizon.ONE_MONTH
        assert action.time_of_day == TimeOfDay.NIGHT_11PM
        assert action.communication_channel == CommunicationChannel.MAIL
    
    def test_action_to_indices(self):
        """Test conversion from action to indices."""
        action_space = TemporalActionSpace()
        
        # Create specific action
        action = TemporalAction(
            healthcare_action=HealthcareAction.MEDICATE,
            time_horizon=TimeHorizon.ONE_WEEK,
            time_of_day=TimeOfDay.EVENING_6PM,
            communication_channel=CommunicationChannel.PHONE
        )
        
        indices = action_space.action_to_indices(action)
        assert indices == (2, 5, 2, 2)  # Expected indices for this action
        
        # Test round-trip conversion
        reconstructed = action_space.indices_to_action(*indices)
        assert reconstructed.healthcare_action == action.healthcare_action
        assert reconstructed.time_horizon == action.time_horizon
        assert reconstructed.time_of_day == action.time_of_day
        assert reconstructed.communication_channel == action.communication_channel
    
    def test_all_possible_actions(self):
        """Test that all possible action combinations are valid."""
        action_space = TemporalActionSpace()
        
        # Generate all possible actions
        all_actions = []
        for h in range(action_space.n_healthcare):
            for t in range(action_space.n_time_horizons):
                for s in range(action_space.n_times_of_day):
                    for c in range(action_space.n_communication):
                        action = action_space.indices_to_action(h, t, s, c)
                        all_actions.append(action)
        
        # Check we have the right number
        assert len(all_actions) == action_space.total_combinations
        
        # Check all are valid
        for action in all_actions:
            assert_valid_temporal_action(action)
        
        # Check all are unique
        action_strings = [action.to_string() for action in all_actions]
        assert len(set(action_strings)) == len(all_actions)


class TestTemporalAction(TestBase):
    """Test cases for TemporalAction."""
    
    def test_creation(self):
        """Test temporal action creation."""
        action = TemporalAction(
            healthcare_action=HealthcareAction.REFER,
            time_horizon=TimeHorizon.THREE_DAYS,
            time_of_day=TimeOfDay.AFTERNOON_2PM,
            communication_channel=CommunicationChannel.EMAIL
        )
        
        assert action.healthcare_action == HealthcareAction.REFER
        assert action.time_horizon == TimeHorizon.THREE_DAYS
        assert action.time_of_day == TimeOfDay.AFTERNOON_2PM
        assert action.communication_channel == CommunicationChannel.EMAIL
    
    def test_to_string(self):
        """Test string representation."""
        action = TemporalAction(
            healthcare_action=HealthcareAction.MONITOR,
            time_horizon=TimeHorizon.ONE_DAY,
            time_of_day=TimeOfDay.MORNING_9AM,
            communication_channel=CommunicationChannel.SMS
        )
        
        string_repr = action.to_string()
        assert "MONITOR" in string_repr
        assert "ONE_DAY" in string_repr
        assert "MORNING_9AM" in string_repr
        assert "SMS" in string_repr
    
    def test_to_indices(self):
        """Test conversion to indices."""
        action = TemporalAction(
            healthcare_action=HealthcareAction.DISCHARGE,
            time_horizon=TimeHorizon.TWO_WEEKS,
            time_of_day=TimeOfDay.NIGHT_11PM,
            communication_channel=CommunicationChannel.MAIL
        )
        
        indices = action.to_indices()
        assert indices == (3, 6, 3, 3)  # Expected indices


class TestEnhancedPatientState(TestBase):
    """Test cases for EnhancedPatientState."""
    
    def test_creation(self):
        """Test patient state creation."""
        patient = EnhancedPatientState(
            risk_score=0.7,
            age=65,
            comorbidities=2,
            bmi=28.5,
            systolic_bp=140,
            diastolic_bp=90,
            heart_rate=85,
            current_hour=14,
            day_of_week=2,
            urgency_level=0.6,
            last_contact_hours_ago=48,
            next_appointment_days=7,
            preferred_channel=CommunicationChannel.PHONE,
            preferred_time=TimeOfDay.MORNING_9AM,
            sms_success_rate=0.6,
            email_success_rate=0.8,
            phone_success_rate=0.95,
            mail_success_rate=0.4,
            medication_adherence=0.75,
            appointment_compliance=0.85,
            response_time_preference=0.8
        )
        
        assert patient.risk_score == 0.7
        assert patient.age == 65
        assert patient.preferred_channel == CommunicationChannel.PHONE
    
    def test_to_feature_vector(self):
        """Test conversion to feature vector."""
        patient = EnhancedPatientState(
            risk_score=0.7,
            age=65,
            comorbidities=2,
            bmi=28.5,
            systolic_bp=140,
            diastolic_bp=90,
            heart_rate=85,
            current_hour=14,
            day_of_week=2,
            urgency_level=0.6,
            last_contact_hours_ago=48,
            next_appointment_days=7,
            preferred_channel=CommunicationChannel.PHONE,
            preferred_time=TimeOfDay.MORNING_9AM,
            sms_success_rate=0.6,
            email_success_rate=0.8,
            phone_success_rate=0.95,
            mail_success_rate=0.4,
            medication_adherence=0.75,
            appointment_compliance=0.85,
            response_time_preference=0.8
        )
        
        features = patient.to_feature_vector()
        
        # Check shape
        assert features.shape == (TestConfig.INPUT_DIM,)
        
        # Check values are finite
        assert np.isfinite(features).all()
        
        # Check some specific values
        assert features[0] == 0.7  # risk_score
        assert features[1] == 65.0  # age
        assert features[2] == 2.0   # comorbidities
    
    def test_feature_vector_consistency(self):
        """Test that feature vector is consistent across calls."""
        patient = EnhancedPatientState(
            risk_score=0.5,
            age=45,
            comorbidities=1,
            bmi=25.0,
            systolic_bp=120,
            diastolic_bp=80,
            heart_rate=70,
            current_hour=10,
            day_of_week=1,
            urgency_level=0.3,
            last_contact_hours_ago=24,
            next_appointment_days=14,
            preferred_channel=CommunicationChannel.EMAIL,
            preferred_time=TimeOfDay.AFTERNOON_2PM,
            sms_success_rate=0.7,
            email_success_rate=0.9,
            phone_success_rate=0.8,
            mail_success_rate=0.5,
            medication_adherence=0.9,
            appointment_compliance=0.95,
            response_time_preference=0.6
        )
        
        features1 = patient.to_feature_vector()
        features2 = patient.to_feature_vector()
        
        np.testing.assert_array_equal(features1, features2)


class TestFactoryFunction(TestBase):
    """Test cases for factory functions."""
    
    def test_create_temporal_action_space(self):
        """Test action space factory function."""
        action_space = create_temporal_action_space()
        
        assert isinstance(action_space, TemporalActionSpace)
        assert action_space.total_combinations == 640  # 5*8*4*4
    
    def test_action_space_consistency(self):
        """Test that multiple action spaces are consistent."""
        space1 = create_temporal_action_space()
        space2 = create_temporal_action_space()
        
        assert space1.total_combinations == space2.total_combinations
        assert space1.n_healthcare == space2.n_healthcare
        assert space1.n_time_horizons == space2.n_time_horizons


@pytest.mark.performance
class TestTemporalActionsPerformance(TestBase):
    """Performance tests for temporal actions."""
    
    def test_action_sampling_performance(self):
        """Test action sampling performance."""
        action_space = TemporalActionSpace()
        
        def sample_action():
            return action_space.sample_random_action()
        
        # Measure throughput
        from tests import PerformanceTestMixin
        mixin = PerformanceTestMixin()
        throughput = mixin.measure_throughput(sample_action, iterations=1000)
        
        # Should be able to sample at least 10,000 actions per second
        assert throughput > 10000, f"Action sampling too slow: {throughput:.1f} ops/sec"
    
    def test_feature_vector_performance(self):
        """Test feature vector conversion performance."""
        patient = EnhancedPatientState(
            risk_score=0.7, age=65, comorbidities=2, bmi=28.5,
            systolic_bp=140, diastolic_bp=90, heart_rate=85,
            current_hour=14, day_of_week=2, urgency_level=0.6,
            last_contact_hours_ago=48, next_appointment_days=7,
            preferred_channel=CommunicationChannel.PHONE,
            preferred_time=TimeOfDay.MORNING_9AM,
            sms_success_rate=0.6, email_success_rate=0.8,
            phone_success_rate=0.95, mail_success_rate=0.4,
            medication_adherence=0.75, appointment_compliance=0.85,
            response_time_preference=0.8
        )
        
        def convert_to_features():
            return patient.to_feature_vector()
        
        # Measure throughput
        from tests import PerformanceTestMixin
        mixin = PerformanceTestMixin()
        throughput = mixin.measure_throughput(convert_to_features, iterations=1000)
        
        # Should be able to convert at least 50,000 times per second
        assert throughput > 50000, f"Feature conversion too slow: {throughput:.1f} ops/sec"
