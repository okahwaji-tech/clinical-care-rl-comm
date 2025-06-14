"""Unit tests for temporal actions and patient state components.

Tests the enhanced temporal action space, patient state representation,
and feature scaling improvements including the risk score rescaling.
"""

import pytest
import numpy as np
from typing import List

from core.temporal_actions import (
    TemporalAction, TemporalActionSpace, EnhancedPatientState,
    HealthcareAction, TimeHorizon, TimeOfDay, CommunicationChannel
)


class TestTemporalAction:
    """Test TemporalAction dataclass and methods."""
    
    def test_temporal_action_creation(self):
        """Test creating a temporal action with all components."""
        action = TemporalAction(
            healthcare_action=HealthcareAction.REFER,
            time_horizon=TimeHorizon.IMMEDIATE,
            time_of_day=TimeOfDay.MORNING_9AM,
            communication_channel=CommunicationChannel.PHONE
        )
        
        assert action.healthcare_action == HealthcareAction.REFER
        assert action.time_horizon == TimeHorizon.IMMEDIATE
        assert action.time_of_day == TimeOfDay.MORNING_9AM
        assert action.communication_channel == CommunicationChannel.PHONE
    
    def test_action_to_string(self):
        """Test temporal action string representation."""
        action = TemporalAction(
            healthcare_action=HealthcareAction.MEDICATE,
            time_horizon=TimeHorizon.ONE_DAY,
            time_of_day=TimeOfDay.EVENING_6PM,
            communication_channel=CommunicationChannel.SMS
        )
        
        action_string = action.to_string()
        expected = "medicate_one_day_evening_6pm_sms"
        assert action_string == expected
        
        # Should have 4 components separated by underscores
        components = action_string.split('_')
        assert len(components) == 4
    
    def test_action_to_indices(self):
        """Test converting action to indices."""
        action = TemporalAction(
            healthcare_action=HealthcareAction.MONITOR,  # value = 0
            time_horizon=TimeHorizon.ONE_HOUR,           # value = 1
            time_of_day=TimeOfDay.AFTERNOON_2PM,         # value = 1
            communication_channel=CommunicationChannel.EMAIL  # value = 1
        )
        
        indices = action.to_indices()
        assert indices == (0, 1, 1, 1)
        assert len(indices) == 4


class TestTemporalActionSpace:
    """Test TemporalActionSpace functionality."""
    
    def test_action_space_initialization(self, action_space):
        """Test action space is properly initialized."""
        assert action_space.n_healthcare == 5  # MONITOR, REFER, MEDICATE, DISCHARGE, FOLLOWUP
        assert action_space.n_time_horizons == 8  # IMMEDIATE to ONE_MONTH
        assert action_space.n_times_of_day == 4  # MORNING, AFTERNOON, EVENING, NIGHT
        assert action_space.n_communication == 4  # SMS, EMAIL, PHONE, MAIL
        
        # Total combinations should be product of all dimensions
        expected_total = 5 * 8 * 4 * 4
        assert action_space.total_combinations == expected_total
    
    def test_sample_random_action(self, action_space):
        """Test random action sampling."""
        action = action_space.sample_random_action()
        
        assert isinstance(action, TemporalAction)
        assert isinstance(action.healthcare_action, HealthcareAction)
        assert isinstance(action.time_horizon, TimeHorizon)
        assert isinstance(action.time_of_day, TimeOfDay)
        assert isinstance(action.communication_channel, CommunicationChannel)
    
    def test_indices_to_action(self, action_space):
        """Test converting indices back to action."""
        # Test specific indices
        action = action_space.indices_to_action(1, 2, 3, 0)  # REFER, FOUR_HOURS, NIGHT_9PM, SMS
        
        assert action.healthcare_action == HealthcareAction.REFER
        assert action.time_horizon == TimeHorizon.FOUR_HOURS
        assert action.time_of_day == TimeOfDay.NIGHT_9PM
        assert action.communication_channel == CommunicationChannel.SMS
    
    def test_get_action_names(self, action_space):
        """Test getting human-readable action names."""
        names = action_space.get_action_names()
        
        assert 'healthcare' in names
        assert 'time_horizons' in names
        assert 'times_of_day' in names
        assert 'communication' in names
        
        # Check some expected values
        assert 'monitor' in names['healthcare']
        assert 'refer' in names['healthcare']
        assert 'immediate' in names['time_horizons']
        assert 'morning_9am' in names['times_of_day']


class TestEnhancedPatientState:
    """Test EnhancedPatientState and feature scaling."""
    
    def test_patient_state_creation(self, sample_patient_states):
        """Test creating enhanced patient states."""
        state = sample_patient_states[0]  # Low risk patient
        
        assert state.risk_score == 0.2
        assert state.age == 35
        assert state.comorbidities == 1
        assert isinstance(state.preferred_channel, CommunicationChannel)
        assert isinstance(state.preferred_time, TimeOfDay)
    
    def test_enhanced_feature_scaling(self, sample_patient_states):
        """Test the enhanced risk score scaling from [0,1] to [-1,1]."""
        for state in sample_patient_states:
            features = state.to_feature_vector()
            
            # First feature should be scaled risk score
            scaled_risk = features[0]
            expected_scaled = state.risk_score * 2.0 - 1.0
            
            assert np.isclose(scaled_risk, expected_scaled), \
                f"Risk score {state.risk_score} should scale to {expected_scaled}, got {scaled_risk}"
            
            # Scaled risk should be in [-1, 1] range
            assert -1.0 <= scaled_risk <= 1.0, \
                f"Scaled risk score {scaled_risk} should be in [-1, 1] range"
    
    def test_feature_vector_properties(self, sample_patient_states):
        """Test feature vector properties and normalization."""
        for state in sample_patient_states:
            features = state.to_feature_vector()
            
            # Should have correct number of features
            assert len(features) == 27
            assert features.dtype == np.float32
            
            # Most features should be normalized to reasonable ranges
            # (excluding one-hot encoded features which are 0 or 1)
            continuous_features = features[:12]  # First 12 are continuous
            assert np.all(continuous_features >= -1.0), "Continuous features should be >= -1"
            assert np.all(continuous_features <= 2.0), "Continuous features should be <= 2"  # Some can be > 1
    
    def test_feature_names_consistency(self):
        """Test that feature names match feature vector length."""
        feature_names = EnhancedPatientState.get_feature_names()
        feature_count = EnhancedPatientState.get_feature_count()
        
        assert len(feature_names) == feature_count == 27
        
        # Check that risk score is properly named as scaled
        assert feature_names[0] == "risk_score_scaled"
    
    def test_risk_stratification(self, sample_patient_states):
        """Test that different risk levels produce different feature patterns."""
        low_risk = sample_patient_states[0]   # risk_score = 0.2
        high_risk = sample_patient_states[2]  # risk_score = 0.8
        
        low_features = low_risk.to_feature_vector()
        high_features = high_risk.to_feature_vector()
        
        # Risk scores should be different when scaled
        assert low_features[0] < high_features[0], \
            "High-risk patient should have higher scaled risk score"
        
        # Should have different clinical features
        assert not np.array_equal(low_features, high_features), \
            "Different risk patients should have different feature vectors"


class TestClinicalAppropriatenessValidation:
    """Test clinical appropriateness validation utilities."""
    
    def test_risk_based_action_validation(self, sample_patient_states):
        """Test that risk-based action validation works correctly."""
        from tests.conftest import assert_clinical_appropriateness
        
        low_risk_state = sample_patient_states[0]
        high_risk_state = sample_patient_states[2]
        
        # Test appropriate actions
        appropriate_low_risk = TemporalAction(
            healthcare_action=HealthcareAction.MONITOR,
            time_horizon=TimeHorizon.ONE_DAY,
            time_of_day=TimeOfDay.MORNING_9AM,
            communication_channel=CommunicationChannel.SMS
        )
        
        appropriate_high_risk = TemporalAction(
            healthcare_action=HealthcareAction.REFER,
            time_horizon=TimeHorizon.IMMEDIATE,
            time_of_day=TimeOfDay.MORNING_9AM,
            communication_channel=CommunicationChannel.PHONE
        )
        
        # These should not raise assertions
        assert_clinical_appropriateness(appropriate_low_risk, low_risk_state)
        assert_clinical_appropriateness(appropriate_high_risk, high_risk_state)
        
        # Test inappropriate actions should raise assertions
        inappropriate_high_risk = TemporalAction(
            healthcare_action=HealthcareAction.DISCHARGE,  # Inappropriate for high-risk
            time_horizon=TimeHorizon.ONE_WEEK,
            time_of_day=TimeOfDay.NIGHT_9PM,
            communication_channel=CommunicationChannel.MAIL_LETTER
        )
        
        with pytest.raises(AssertionError):
            assert_clinical_appropriateness(inappropriate_high_risk, high_risk_state)


@pytest.mark.parametrize("risk_score,expected_range", [
    (0.0, (-1.0, -1.0)),  # Minimum risk
    (0.5, (-0.1, 0.1)),   # Medium risk (around 0)
    (1.0, (1.0, 1.0)),    # Maximum risk
])
def test_risk_score_scaling_edge_cases(risk_score, expected_range):
    """Test risk score scaling for edge cases."""
    state = EnhancedPatientState(
        risk_score=risk_score, age=50, comorbidities=2, bmi=25, systolic_bp=130, diastolic_bp=85,
        heart_rate=75, current_hour=12, day_of_week=3, urgency_level=0.5,
        last_contact_hours_ago=24, next_appointment_days=7,
        preferred_channel=CommunicationChannel.EMAIL, preferred_time=TimeOfDay.AFTERNOON_2PM,
        sms_success_rate=0.8, email_success_rate=0.9, phone_success_rate=0.7, mail_success_rate=0.5,
        medication_adherence=0.8, appointment_compliance=0.8, response_time_preference=0.5
    )
    
    features = state.to_feature_vector()
    scaled_risk = features[0]
    
    assert expected_range[0] <= scaled_risk <= expected_range[1], \
        f"Risk score {risk_score} scaled to {scaled_risk}, expected in range {expected_range}"
