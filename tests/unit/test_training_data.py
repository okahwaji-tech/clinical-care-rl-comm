"""Unit tests for enhanced training data generation.

Tests the stratified data generation, enhanced reward shaping,
and risk-tier partitioning functionality.
"""

import pytest
import pandas as pd
import numpy as np

from core.temporal_training_data import (
    generate_stratified_temporal_training_data,
    TemporalRewardFunction
)
from core.temporal_actions import (
    TemporalAction, EnhancedPatientState, HealthcareAction, 
    TimeHorizon, TimeOfDay, CommunicationChannel
)


class TestStratifiedDataGeneration:
    """Test enhanced stratified data generation."""
    
    def test_stratified_data_basic_properties(self, small_training_data):
        """Test basic properties of stratified training data."""
        df = small_training_data
        
        # Should have expected columns
        expected_columns = [
            'risk_score', 'healthcare_action', 'time_horizon', 
            'time_of_day', 'communication_channel', 'reward', 
            'done', 'risk_tier'
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        # Should have data
        assert len(df) > 0
        assert len(df) >= 1000  # At least the requested samples
    
    def test_risk_tier_distribution(self, small_training_data):
        """Test that risk tiers are properly distributed."""
        df = small_training_data
        
        # Should have all 5 risk tiers
        risk_tiers = df['risk_tier'].unique()
        expected_tiers = ['very_low', 'low', 'medium', 'high', 'very_high']
        
        for tier in expected_tiers:
            assert tier in risk_tiers, f"Missing risk tier: {tier}"
        
        # Distribution should be reasonable (not all in one tier)
        tier_counts = df['risk_tier'].value_counts()
        assert len(tier_counts) == 5, "Should have exactly 5 risk tiers"
        
        # No tier should be empty
        assert all(count > 0 for count in tier_counts.values()), \
            "All risk tiers should have samples"
    
    def test_risk_score_tier_alignment(self, small_training_data):
        """Test that risk scores align with their assigned tiers."""
        df = small_training_data
        
        # Check risk score ranges for each tier
        tier_ranges = {
            'very_low': (0.0, 0.2),
            'low': (0.2, 0.4),
            'medium': (0.4, 0.6),
            'high': (0.6, 0.8),
            'very_high': (0.8, 1.0)
        }
        
        for tier, (min_risk, max_risk) in tier_ranges.items():
            tier_data = df[df['risk_tier'] == tier]
            if len(tier_data) > 0:  # Only check if tier has data
                tier_risks = tier_data['risk_score']
                assert tier_risks.min() >= min_risk, \
                    f"{tier} tier has risk scores below {min_risk}"
                assert tier_risks.max() <= max_risk, \
                    f"{tier} tier has risk scores above {max_risk}"
    
    def test_clinically_appropriate_action_distribution(self, small_training_data):
        """Test that action distributions are clinically appropriate per tier."""
        df = small_training_data
        
        # Very high risk should have high proportion of REFER actions
        very_high_risk = df[df['risk_tier'] == 'very_high']
        if len(very_high_risk) > 10:  # Only test if sufficient samples
            refer_pct = (very_high_risk['healthcare_action'] == 1).mean()  # REFER = 1
            assert refer_pct > 0.5, \
                f"Very high risk should have >50% REFER actions, got {refer_pct:.2%}"
        
        # Very low risk should have high proportion of MONITOR/DISCHARGE
        very_low_risk = df[df['risk_tier'] == 'very_low']
        if len(very_low_risk) > 10:
            monitor_discharge_pct = (
                (very_low_risk['healthcare_action'] == 0) |  # MONITOR = 0
                (very_low_risk['healthcare_action'] == 3)    # DISCHARGE = 3
            ).mean()
            assert monitor_discharge_pct > 0.5, \
                f"Very low risk should have >50% MONITOR/DISCHARGE, got {monitor_discharge_pct:.2%}"
    
    @pytest.mark.parametrize("num_samples", [100, 500, 1000])
    def test_different_sample_sizes(self, num_samples):
        """Test stratified generation with different sample sizes."""
        df = generate_stratified_temporal_training_data(num_samples)
        
        # Should generate at least the requested number of samples
        assert len(df) >= num_samples
        
        # Should have all risk tiers represented
        assert len(df['risk_tier'].unique()) == 5


class TestEnhancedRewardShaping:
    """Test enhanced reward calculation with new penalties and bonuses."""
    
    def test_low_risk_medication_penalty(self, reward_function):
        """Test -0.3 penalty for low-risk medication."""
        # Create low-risk patient (risk_score < 0.4)
        low_risk_patient = EnhancedPatientState(
            risk_score=0.3, age=30, comorbidities=1, bmi=24, systolic_bp=120, diastolic_bp=80,
            heart_rate=70, current_hour=10, day_of_week=2, urgency_level=0.2,
            last_contact_hours_ago=48, next_appointment_days=14,
            preferred_channel=CommunicationChannel.SMS, preferred_time=TimeOfDay.MORNING_9AM,
            sms_success_rate=0.9, email_success_rate=0.8, phone_success_rate=0.7, mail_success_rate=0.6,
            medication_adherence=0.9, appointment_compliance=0.9, response_time_preference=0.6
        )
        
        # Create medication action
        medicate_action = TemporalAction(
            healthcare_action=HealthcareAction.MEDICATE,
            time_horizon=TimeHorizon.ONE_DAY,
            time_of_day=TimeOfDay.MORNING_9AM,
            communication_channel=CommunicationChannel.SMS
        )
        
        reward = reward_function.calculate_temporal_reward(medicate_action, low_risk_patient)
        
        # Should be negative due to penalty
        assert reward < 0, f"Low-risk medication should have negative reward, got {reward}"
        
        # Compare with non-medication action for same patient
        monitor_action = medicate_action._replace(healthcare_action=HealthcareAction.MONITOR)
        monitor_reward = reward_function.calculate_temporal_reward(monitor_action, low_risk_patient)
        
        assert reward < monitor_reward, \
            "Medication should have lower reward than monitoring for low-risk patient"
    
    def test_high_risk_intervention_bonus(self, reward_function):
        """Test +0.3 bonus for high-risk REFER/MEDICATE."""
        # Create high-risk patient (risk_score >= 0.7)
        high_risk_patient = EnhancedPatientState(
            risk_score=0.8, age=70, comorbidities=5, bmi=35, systolic_bp=170, diastolic_bp=105,
            heart_rate=95, current_hour=16, day_of_week=1, urgency_level=0.8,
            last_contact_hours_ago=2, next_appointment_days=1,
            preferred_channel=CommunicationChannel.PHONE, preferred_time=TimeOfDay.EVENING_6PM,
            sms_success_rate=0.5, email_success_rate=0.6, phone_success_rate=0.9, mail_success_rate=0.4,
            medication_adherence=0.6, appointment_compliance=0.7, response_time_preference=0.2
        )
        
        # Test REFER action
        refer_action = TemporalAction(
            healthcare_action=HealthcareAction.REFER,
            time_horizon=TimeHorizon.IMMEDIATE,
            time_of_day=TimeOfDay.EVENING_6PM,
            communication_channel=CommunicationChannel.PHONE
        )
        
        refer_reward = reward_function.calculate_temporal_reward(refer_action, high_risk_patient)
        
        # Should be positive due to bonus
        assert refer_reward > 0, f"High-risk referral should have positive reward, got {refer_reward}"
        
        # Test MEDICATE action
        medicate_action = refer_action._replace(healthcare_action=HealthcareAction.MEDICATE)
        medicate_reward = reward_function.calculate_temporal_reward(medicate_action, high_risk_patient)
        
        assert medicate_reward > 0, f"High-risk medication should have positive reward, got {medicate_reward}"
        
        # Compare with less intensive action
        monitor_action = refer_action._replace(healthcare_action=HealthcareAction.MONITOR)
        monitor_reward = reward_function.calculate_temporal_reward(monitor_action, high_risk_patient)
        
        assert refer_reward > monitor_reward, \
            "Referral should have higher reward than monitoring for high-risk patient"
    
    def test_reward_range_validity(self, reward_function, sample_patient_states):
        """Test that rewards are in valid range after enhancements."""
        action = TemporalAction(
            healthcare_action=HealthcareAction.MONITOR,
            time_horizon=TimeHorizon.ONE_DAY,
            time_of_day=TimeOfDay.MORNING_9AM,
            communication_channel=CommunicationChannel.SMS
        )
        
        for patient_state in sample_patient_states:
            reward = reward_function.calculate_temporal_reward(action, patient_state)
            
            # Rewards should be reasonable (not extreme values)
            assert -2.0 <= reward <= 2.0, \
                f"Reward {reward} outside reasonable range for risk {patient_state.risk_score}"
    
    @pytest.mark.parametrize("risk_score,action_type,expected_sign", [
        (0.2, HealthcareAction.MEDICATE, -1),  # Low risk medication -> negative
        (0.8, HealthcareAction.REFER, 1),      # High risk referral -> positive
        (0.8, HealthcareAction.MEDICATE, 1),   # High risk medication -> positive
        (0.5, HealthcareAction.MONITOR, 1),    # Medium risk monitoring -> positive
    ])
    def test_reward_sign_patterns(self, reward_function, risk_score, action_type, expected_sign):
        """Test that reward signs match expected patterns."""
        patient_state = EnhancedPatientState(
            risk_score=risk_score, age=50, comorbidities=3, bmi=28, systolic_bp=140, diastolic_bp=90,
            heart_rate=80, current_hour=12, day_of_week=3, urgency_level=risk_score,
            last_contact_hours_ago=12, next_appointment_days=7,
            preferred_channel=CommunicationChannel.EMAIL, preferred_time=TimeOfDay.AFTERNOON_2PM,
            sms_success_rate=0.7, email_success_rate=0.9, phone_success_rate=0.8, mail_success_rate=0.5,
            medication_adherence=0.7, appointment_compliance=0.8, response_time_preference=0.4
        )
        
        action = TemporalAction(
            healthcare_action=action_type,
            time_horizon=TimeHorizon.ONE_DAY,
            time_of_day=TimeOfDay.AFTERNOON_2PM,
            communication_channel=CommunicationChannel.EMAIL
        )
        
        reward = reward_function.calculate_temporal_reward(action, patient_state)
        
        if expected_sign > 0:
            assert reward > 0, f"Expected positive reward for {action_type.name} with risk {risk_score}, got {reward}"
        else:
            assert reward < 0, f"Expected negative reward for {action_type.name} with risk {risk_score}, got {reward}"


class TestDataQuality:
    """Test data quality and consistency."""
    
    def test_no_missing_values(self, small_training_data):
        """Test that generated data has no missing values."""
        df = small_training_data
        
        # Check for NaN values
        assert not df.isnull().any().any(), "Training data should not have missing values"
        
        # Check for infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            assert np.isfinite(df[col]).all(), f"Column {col} contains infinite values"
    
    def test_data_types_consistency(self, small_training_data):
        """Test that data types are consistent."""
        df = small_training_data
        
        # Risk scores should be float in [0, 1]
        assert df['risk_score'].dtype in [np.float32, np.float64]
        assert (df['risk_score'] >= 0).all() and (df['risk_score'] <= 1).all()
        
        # Actions should be integers
        assert df['healthcare_action'].dtype in [np.int32, np.int64]
        assert df['time_horizon'].dtype in [np.int32, np.int64]
        
        # Rewards should be numeric
        assert df['reward'].dtype in [np.float32, np.float64]
        
        # Risk tier should be string
        assert df['risk_tier'].dtype == object
    
    def test_action_value_ranges(self, small_training_data):
        """Test that action values are in valid ranges."""
        df = small_training_data
        
        # Healthcare actions: 0-4 (5 actions)
        assert df['healthcare_action'].min() >= 0
        assert df['healthcare_action'].max() <= 4
        
        # Time horizons: 0-7 (8 horizons)
        assert df['time_horizon'].min() >= 0
        assert df['time_horizon'].max() <= 7
        
        # Time of day: 0-3 (4 times)
        assert df['time_of_day'].min() >= 0
        assert df['time_of_day'].max() <= 3
        
        # Communication channels: 0-3 (4 channels)
        assert df['communication_channel'].min() >= 0
        assert df['communication_channel'].max() <= 3
