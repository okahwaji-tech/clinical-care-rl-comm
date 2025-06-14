"""Healthcare temporal training data generation for DQN optimization.

This module generates realistic temporal healthcare training samples,
including patient states, composite actions, and rewards with nuanced
shaping for healthcare communication decision-making.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from core.temporal_actions import (
    TemporalActionSpace,
    EnhancedPatientState,
    TemporalAction,
    HealthcareAction,
    TimeHorizon,
    TimeOfDay,
    CommunicationChannel,
)
from core.logging_system import get_logger

logger = get_logger(__name__)


class TemporalRewardFunction:
    """Sophisticated reward function for temporal healthcare actions.

    Computes comprehensive rewards based on risk, urgency, demographics,
    time appropriateness, communication preferences, and random perturbations.
    """

    def __init__(self) -> None:
        """Initialize TemporalRewardFunction with predefined reward components."""
        self.action_space = TemporalActionSpace()

        # Base reward matrices for different risk levels (REBALANCED)
        self.risk_action_rewards = {
            # Very low risk (0.0-0.2): Strongly prefer monitoring and discharge, penalize medication
            "very_low": {
                HealthcareAction.MONITOR: 0.95,  # Increased: monitoring is best for low risk
                HealthcareAction.DISCHARGE: 0.90,  # High reward for appropriate discharge
                HealthcareAction.FOLLOWUP: 0.75,  # Good for preventive care
                HealthcareAction.MEDICATE: 0.15,  # Heavily penalized: unnecessary medication
                HealthcareAction.REFER: 0.10,  # Penalized: unnecessary escalation
            },
            # Low risk (0.2-0.4): Prefer monitoring and follow-up, moderate medication penalty
            "low": {
                HealthcareAction.MONITOR: 0.92,  # Best choice for low risk
                HealthcareAction.FOLLOWUP: 0.85,  # Good for ongoing care
                HealthcareAction.DISCHARGE: 0.70,  # Reasonable if stable
                HealthcareAction.MEDICATE: 0.35,  # Reduced penalty but still discouraged
                HealthcareAction.REFER: 0.25,  # Discouraged escalation
            },
            # Medium risk (0.4-0.6): Balanced approach, medication more appropriate
            "medium": {
                HealthcareAction.MONITOR: 0.85,  # Still good for medium risk
                HealthcareAction.MEDICATE: 0.80,  # Reduced from 0.9 to balance
                HealthcareAction.REFER: 0.75,  # Appropriate for some cases
                HealthcareAction.FOLLOWUP: 0.70,  # Good for ongoing management
                HealthcareAction.DISCHARGE: 0.25,  # Discouraged for medium risk
            },
            # High risk (0.6-0.8): Prefer referral and medication, discourage monitoring alone
            "high": {
                HealthcareAction.REFER: 0.95,  # Best for high risk
                HealthcareAction.MEDICATE: 0.85,  # Appropriate intervention
                HealthcareAction.MONITOR: 0.45,  # Reduced: monitoring alone insufficient
                HealthcareAction.FOLLOWUP: 0.40,  # Insufficient for high risk
                HealthcareAction.DISCHARGE: 0.05,  # Dangerous for high risk
            },
            # Very high risk (0.8-1.0): Urgent referral required, penalize delays
            "very_high": {
                HealthcareAction.REFER: 0.98,  # Critical for very high risk
                HealthcareAction.MEDICATE: 0.75,  # May be needed but referral better
                HealthcareAction.MONITOR: 0.20,  # Dangerous delay
                HealthcareAction.FOLLOWUP: 0.15,  # Dangerous delay
                HealthcareAction.DISCHARGE: 0.02,  # Extremely dangerous
            },
        }

        # Time appropriateness bonuses/penalties
        self.time_appropriateness = {
            # High urgency requires immediate action
            "urgent": {
                TimeHorizon.IMMEDIATE: 0.3,
                TimeHorizon.ONE_HOUR: 0.2,
                TimeHorizon.FOUR_HOURS: 0.0,
                TimeHorizon.ONE_DAY: -0.2,
                TimeHorizon.THREE_DAYS: -0.4,
                TimeHorizon.ONE_WEEK: -0.6,
                TimeHorizon.TWO_WEEKS: -0.8,
                TimeHorizon.ONE_MONTH: -1.0,
            },
            # Medium urgency allows some delay
            "medium": {
                TimeHorizon.IMMEDIATE: 0.1,
                TimeHorizon.ONE_HOUR: 0.2,
                TimeHorizon.FOUR_HOURS: 0.3,
                TimeHorizon.ONE_DAY: 0.2,
                TimeHorizon.THREE_DAYS: 0.0,
                TimeHorizon.ONE_WEEK: -0.2,
                TimeHorizon.TWO_WEEKS: -0.4,
                TimeHorizon.ONE_MONTH: -0.6,
            },
            # Low urgency prefers longer monitoring
            "low": {
                TimeHorizon.IMMEDIATE: -0.2,
                TimeHorizon.ONE_HOUR: -0.1,
                TimeHorizon.FOUR_HOURS: 0.0,
                TimeHorizon.ONE_DAY: 0.1,
                TimeHorizon.THREE_DAYS: 0.2,
                TimeHorizon.ONE_WEEK: 0.3,
                TimeHorizon.TWO_WEEKS: 0.2,
                TimeHorizon.ONE_MONTH: 0.1,
            },
        }

        # Communication channel effectiveness by demographics
        self.channel_effectiveness = {
            # Young patients (18-40)
            "young": {
                CommunicationChannel.SMS: 0.9,
                CommunicationChannel.EMAIL: 0.8,
                CommunicationChannel.PHONE: 0.6,
                CommunicationChannel.MAIL_LETTER: 0.3,
            },
            # Middle-aged patients (40-65)
            "middle": {
                CommunicationChannel.EMAIL: 0.8,
                CommunicationChannel.PHONE: 0.9,
                CommunicationChannel.SMS: 0.7,
                CommunicationChannel.MAIL_LETTER: 0.5,
            },
            # Elderly patients (65+)
            "elderly": {
                CommunicationChannel.PHONE: 0.9,
                CommunicationChannel.MAIL_LETTER: 0.8,
                CommunicationChannel.EMAIL: 0.5,
                CommunicationChannel.SMS: 0.4,
            },
        }

        # Time of day preferences
        self.time_preferences = {
            TimeOfDay.MORNING_9AM: 0.8,  # Generally good time
            TimeOfDay.AFTERNOON_2PM: 0.9,  # Best time for most
            TimeOfDay.EVENING_6PM: 0.7,  # Good for working people
            TimeOfDay.NIGHT_9PM: 0.4,  # Less preferred
        }

    def calculate_temporal_reward(
        self, action: TemporalAction, patient_state: EnhancedPatientState
    ) -> float:
        """Calculate comprehensive reward for a temporal action with enhanced shaping.

        Args:
            action (TemporalAction): Composite temporal action.
            patient_state (EnhancedPatientState): Patient state features.

        Returns:
            float: Clipped reward value between 0 and 1.
        """

        # Base healthcare action reward
        risk_category = self._get_risk_category(patient_state.risk_score)
        base_reward = self.risk_action_rewards[risk_category][action.healthcare_action]

        # ENHANCED MEDICATION OVERUSE PENALTY: Penalize unnecessary medication
        medication_penalty = 0.0
        if action.healthcare_action == HealthcareAction.MEDICATE:
            # NEW: Subtract 0.3 when risk_score < 0.4 and action == MEDICATE
            if patient_state.risk_score < 0.4:
                medication_penalty = -0.3  # Enhanced penalty for low-risk medication
            elif patient_state.risk_score < 0.5:
                medication_penalty = -0.15  # Moderate penalty
            # No penalty for high-risk patients who need medication

        # ENHANCED HIGH-RISK BONUS: Add +0.3 for appropriate high-risk actions
        high_risk_bonus = 0.0
        if patient_state.risk_score >= 0.7 and action.healthcare_action in [HealthcareAction.REFER, HealthcareAction.MEDICATE]:
            high_risk_bonus = 0.3  # NEW: Strong bonus for appropriate high-risk interventions

        # ACTION-SPECIFIC BONUSES: More nuanced reward shaping
        action_bonus = 0.0

        if action.healthcare_action == HealthcareAction.MONITOR:
            # Monitoring bonus varies by risk level
            if patient_state.risk_score < 0.3:
                action_bonus = 0.15  # Strong bonus for low-risk monitoring
            elif patient_state.risk_score < 0.6:
                action_bonus = 0.10  # Good bonus for medium-risk monitoring
            else:
                action_bonus = 0.02  # Small bonus for high-risk monitoring

        elif action.healthcare_action == HealthcareAction.MEDICATE:
            # Medication bonus for appropriate cases
            if patient_state.risk_score > 0.5:
                action_bonus = 0.12  # Good bonus for high-risk medication
            elif patient_state.risk_score > 0.3:
                action_bonus = 0.05  # Small bonus for medium-risk medication
            # No bonus for low-risk medication (handled by penalty)

        elif action.healthcare_action == HealthcareAction.REFER:
            # Referral bonus for high-risk cases
            if patient_state.risk_score > 0.7:
                action_bonus = 0.18  # Strong bonus for high-risk referral
            elif patient_state.risk_score > 0.5:
                action_bonus = 0.08  # Moderate bonus for medium-high risk

        elif action.healthcare_action == HealthcareAction.FOLLOWUP:
            # Follow-up bonus for ongoing care
            if 0.2 < patient_state.risk_score < 0.7:
                action_bonus = 0.08  # Good for medium-risk follow-up

        elif action.healthcare_action == HealthcareAction.DISCHARGE:
            # Discharge bonus for low-risk stable patients
            if patient_state.risk_score < 0.2:
                action_bonus = 0.12  # Good bonus for appropriate discharge

        # Time appropriateness bonus
        urgency_category = self._get_urgency_category(patient_state.urgency_level)
        time_bonus = self.time_appropriateness[urgency_category][action.time_horizon]

        # Communication channel effectiveness
        age_category = self._get_age_category(patient_state.age)
        channel_bonus = self.channel_effectiveness[age_category][
            action.communication_channel
        ]

        # Time of day preference
        schedule_bonus = self.time_preferences[action.time_of_day]

        # Patient preference alignment
        preference_bonus = 0.0
        if action.communication_channel == patient_state.preferred_channel:
            preference_bonus += 0.15  # Reduced from 0.2
        if action.time_of_day == patient_state.preferred_time:
            preference_bonus += 0.08  # Reduced from 0.1

        # ENHANCED REWARD COMBINATION with high-risk bonus and action-specific bonuses
        total_reward = (
            base_reward * 0.30  # 30% base healthcare appropriateness (reduced)
            + medication_penalty * 0.20  # 20% medication penalty
            + high_risk_bonus * 0.15  # 15% high-risk intervention bonus (NEW)
            + action_bonus * 0.15  # 15% action-specific bonus
            + time_bonus * 0.10  # 10% time appropriateness
            + channel_bonus * 0.06  # 6% channel effectiveness
            + schedule_bonus * 0.02  # 2% time preference
            + preference_bonus * 0.02  # 2% patient preference
        )

        # WIDEN REWARD SPREAD: Much larger noise to break determinism
        base_noise = np.random.normal(0, 0.1)  # INCREASED from 0.03 to 0.1

        # Add occasional larger perturbations (10% chance)
        if np.random.random() < 0.1:
            perturbation = np.random.normal(0, 0.2)  # Large perturbation
            total_reward += perturbation

        total_reward += base_noise
        total_reward = np.clip(total_reward, 0, 1)

        return total_reward

    def _get_risk_category(self, risk_score: float) -> str:
        """Map risk score to risk category.

        Args:
            risk_score (float): Normalized risk score (0-1).

        Returns:
            str: Risk category key.
        """
        if risk_score < 0.2:
            return "very_low"
        elif risk_score < 0.4:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "very_high"

    def _get_urgency_category(self, urgency_level: float) -> str:
        """Map urgency level to urgency category.

        Args:
            urgency_level (float): Normalized urgency (0-1).

        Returns:
            str: Urgency category key.
        """
        if urgency_level < 0.3:
            return "low"
        elif urgency_level < 0.7:
            return "medium"
        else:
            return "urgent"

    def _get_age_category(self, age: int) -> str:
        """Map age to demographic category.

        Args:
            age (int): Patient age in years.

        Returns:
            str: Age demographic key.
        """
        if age < 40:
            return "young"
        elif age < 65:
            return "middle"
        else:
            return "elderly"


def generate_stratified_temporal_training_data(num_samples: int = 10000) -> pd.DataFrame:
    """Generate stratified temporal training data partitioned by risk tiers.

    Implements risk-stratified sampling where patients are partitioned by risk tier
    and their "correct" majority actions are sampled based on clinical appropriateness.

    Args:
        num_samples (int): Number of training samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing generated stratified training samples.
    """
    logger.info(
        "Generating stratified temporal healthcare training samples",
        num_samples=num_samples
    )

    action_space = TemporalActionSpace()
    reward_function = TemporalRewardFunction()

    training_data = []

    # Risk tier distribution (approximate normal distribution centered at medium risk)
    risk_tier_distribution = {
        "very_low": 0.15,   # Risk 0.0-0.2
        "low": 0.25,        # Risk 0.2-0.4
        "medium": 0.30,     # Risk 0.4-0.6
        "high": 0.20,       # Risk 0.6-0.8
        "very_high": 0.10   # Risk 0.8-1.0
    }

    # Calculate samples per risk tier
    samples_per_tier = {
        tier: int(num_samples * proportion)
        for tier, proportion in risk_tier_distribution.items()
    }

    # Adjust for rounding errors
    total_allocated = sum(samples_per_tier.values())
    if total_allocated < num_samples:
        samples_per_tier["medium"] += (num_samples - total_allocated)

    logger.info(
        "Risk tier distribution for stratified sampling",
        samples_per_tier=samples_per_tier
    )

    # Generate samples for each risk tier
    for tier, tier_samples in samples_per_tier.items():
        for i in range(tier_samples):
            # Generate patient state with risk score in the appropriate tier
            patient_state = _generate_patient_in_risk_tier(tier)

            # Sample 5-10 actions per patient for diverse training
            num_actions = np.random.randint(5, 11)

            for _ in range(num_actions):
                # Sample action based on risk tier with appropriate distribution
                action = _sample_stratified_action(patient_state, action_space, tier)

                # Calculate reward
                reward = reward_function.calculate_temporal_reward(action, patient_state)

                # Create training sample
                sample = {
                    **_patient_state_to_dict(patient_state),
                    "healthcare_action": action.healthcare_action.value,
                    "time_horizon": action.time_horizon.value,
                    "time_of_day": action.time_of_day.value,
                    "communication_channel": action.communication_channel.value,
                    "action_string": action.to_string(),
                    "reward": reward,
                    "done": np.random.choice([True, False], p=[0.1, 0.9]),
                    "risk_tier": tier,  # Add risk tier for analysis
                }

                training_data.append(sample)

        # Log progress for each tier
        if (i + 1) % (tier_samples // 10 or 1) == 0:
            logger.debug(
                "Generated patients",
                tier=tier,
                current=i + 1,
                total=tier_samples
            )

    # Log completion of tier
    logger.info(
        "Completed risk tier generation",
        tier=tier,
        samples=tier_samples
    )

    df = pd.DataFrame(training_data)

    logger.info(
        "Generated stratified training samples",
        total_samples=len(df),
        avg_reward=df["reward"].mean(),
        reward_std=df["reward"].std(),
    )

    # Show action distribution by risk tier
    logger.info("Action Distribution by Risk Tier:")

    # Group by risk tier and healthcare action
    tier_action_counts = df.groupby(["risk_tier", "healthcare_action"]).size().unstack()

    # Calculate percentages
    tier_action_pcts = tier_action_counts.div(tier_action_counts.sum(axis=1), axis=0) * 100

    # Log summary
    for tier in tier_action_pcts.index:
        logger.info(
            f"Risk tier action distribution",
            tier=tier,
            action_percentages=tier_action_pcts.loc[tier].to_dict()
        )

    return df


def generate_temporal_training_data(num_samples: int = 10000) -> pd.DataFrame:
    """Generate temporal training data using stratified sampling.

    This is a wrapper function that maintains backward compatibility while
    using the new stratified data generation approach.

    Args:
        num_samples (int): Number of training samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing generated training samples.
    """
    return generate_stratified_temporal_training_data(num_samples)


def _generate_patient_in_risk_tier(tier: str) -> EnhancedPatientState:
    """Generate a patient state with risk score in the specified tier.

    Args:
        tier (str): Risk tier ('very_low', 'low', 'medium', 'high', 'very_high')

    Returns:
        EnhancedPatientState: Patient state with risk score in the specified tier
    """
    # Define risk score ranges for each tier
    risk_ranges = {
        "very_low": (0.0, 0.2),
        "low": (0.2, 0.4),
        "medium": (0.4, 0.6),
        "high": (0.6, 0.8),
        "very_high": (0.8, 1.0)
    }

    min_risk, max_risk = risk_ranges[tier]

    # Generate base patient state
    patient_state = _generate_realistic_patient_state()

    # Override risk score to be in the specified tier
    # Use beta distribution to create realistic distribution within tier
    alpha, beta = 2, 2  # Symmetric beta distribution
    normalized_risk = np.random.beta(alpha, beta)
    risk_score = min_risk + normalized_risk * (max_risk - min_risk)

    # Create new patient state with adjusted risk score
    return EnhancedPatientState(
        risk_score=risk_score,
        age=patient_state.age,
        comorbidities=patient_state.comorbidities,
        bmi=patient_state.bmi,
        systolic_bp=patient_state.systolic_bp,
        diastolic_bp=patient_state.diastolic_bp,
        heart_rate=patient_state.heart_rate,
        current_hour=patient_state.current_hour,
        day_of_week=patient_state.day_of_week,
        urgency_level=patient_state.urgency_level,
        last_contact_hours_ago=patient_state.last_contact_hours_ago,
        next_appointment_days=patient_state.next_appointment_days,
        preferred_channel=patient_state.preferred_channel,
        preferred_time=patient_state.preferred_time,
        sms_success_rate=patient_state.sms_success_rate,
        email_success_rate=patient_state.email_success_rate,
        phone_success_rate=patient_state.phone_success_rate,
        mail_success_rate=patient_state.mail_success_rate,
        medication_adherence=patient_state.medication_adherence,
        appointment_compliance=patient_state.appointment_compliance,
        response_time_preference=patient_state.response_time_preference,
    )


def _sample_stratified_action(
    patient_state: EnhancedPatientState,
    action_space: TemporalActionSpace,
    tier: str
) -> TemporalAction:
    """Sample action based on risk tier with appropriate clinical distribution.

    Implements stratified sampling where majority actions are sampled based on
    clinical appropriateness for each risk tier.

    Args:
        patient_state (EnhancedPatientState): Patient state for sampling
        action_space (TemporalActionSpace): Action space manager
        tier (str): Risk tier for stratified sampling

    Returns:
        TemporalAction: Sampled composite action appropriate for risk tier
    """
    # Define majority action distributions for each risk tier
    tier_action_distributions = {
        "very_low": {  # >70% MONITOR/DISCHARGE for risk<0.2
            HealthcareAction.MONITOR: 0.45,
            HealthcareAction.DISCHARGE: 0.30,
            HealthcareAction.FOLLOWUP: 0.15,
            HealthcareAction.MEDICATE: 0.05,
            HealthcareAction.REFER: 0.05
        },
        "low": {  # Favor monitoring and follow-up
            HealthcareAction.MONITOR: 0.40,
            HealthcareAction.FOLLOWUP: 0.25,
            HealthcareAction.DISCHARGE: 0.15,
            HealthcareAction.MEDICATE: 0.15,
            HealthcareAction.REFER: 0.05
        },
        "medium": {  # Balanced approach
            HealthcareAction.MONITOR: 0.30,
            HealthcareAction.MEDICATE: 0.25,
            HealthcareAction.REFER: 0.20,
            HealthcareAction.FOLLOWUP: 0.15,
            HealthcareAction.DISCHARGE: 0.10
        },
        "high": {  # Favor referral and medication
            HealthcareAction.REFER: 0.40,
            HealthcareAction.MEDICATE: 0.30,
            HealthcareAction.MONITOR: 0.15,
            HealthcareAction.FOLLOWUP: 0.10,
            HealthcareAction.DISCHARGE: 0.05
        },
        "very_high": {  # Always REFER for risk>0.9
            HealthcareAction.REFER: 0.70,
            HealthcareAction.MEDICATE: 0.20,
            HealthcareAction.MONITOR: 0.05,
            HealthcareAction.FOLLOWUP: 0.03,
            HealthcareAction.DISCHARGE: 0.02
        }
    }

    # Sample healthcare action based on tier distribution
    actions = list(tier_action_distributions[tier].keys())
    probabilities = list(tier_action_distributions[tier].values())
    healthcare_action = np.random.choice(actions, p=probabilities)

    # Sample timing and channel as before (based on urgency and preferences)
    if patient_state.urgency_level > 0.7:
        time_horizon = np.random.choice(
            [TimeHorizon.IMMEDIATE, TimeHorizon.ONE_HOUR], p=[0.8, 0.2]
        )
    elif patient_state.urgency_level > 0.4:
        time_horizon = np.random.choice(
            [TimeHorizon.ONE_HOUR, TimeHorizon.FOUR_HOURS, TimeHorizon.ONE_DAY],
            p=[0.3, 0.4, 0.3],
        )
    else:
        time_horizon = np.random.choice(
            [TimeHorizon.ONE_DAY, TimeHorizon.THREE_DAYS, TimeHorizon.ONE_WEEK],
            p=[0.3, 0.4, 0.3],
        )

    # Use patient preferences for communication
    communication_channel = patient_state.preferred_channel
    time_of_day = patient_state.preferred_time

    return TemporalAction(
        healthcare_action=healthcare_action,
        time_horizon=time_horizon,
        time_of_day=time_of_day,
        communication_channel=communication_channel,
    )


def _generate_realistic_patient_state() -> EnhancedPatientState:
    """Generate a realistic patient state with correlated clinical and temporal features.

    Returns:
        EnhancedPatientState: Simulated patient state object.
    """
    # Generate correlated clinical features
    age = np.random.normal(60, 20)
    age = np.clip(age, 18, 95)

    # Risk score correlated with age and other factors
    age_risk = (age - 18) / 77  # Normalize age contribution
    comorbidities = np.random.poisson(age_risk * 3)

    base_risk = age_risk * 0.4 + (comorbidities / 5) * 0.3
    risk_score = np.clip(base_risk + np.random.normal(0, 0.2), 0, 1)

    # Urgency correlated with risk
    urgency_level = np.clip(risk_score + np.random.normal(0, 0.15), 0, 1)

    # Generate other clinical features
    bmi = np.random.normal(25 + risk_score * 10, 5)
    systolic_bp = np.random.normal(120 + risk_score * 40, 15)
    diastolic_bp = np.random.normal(80 + risk_score * 20, 10)
    heart_rate = np.random.normal(70 + risk_score * 30, 15)

    # Temporal features
    current_hour = np.random.randint(8, 20)  # Business hours mostly
    day_of_week = np.random.randint(0, 7)
    last_contact_hours_ago = np.random.exponential(48)  # Average 2 days
    next_appointment_days = np.random.exponential(14)  # Average 2 weeks

    # Communication preferences based on age
    if age < 40:
        preferred_channel = np.random.choice(
            [CommunicationChannel.SMS, CommunicationChannel.EMAIL], p=[0.7, 0.3]
        )
        preferred_time = np.random.choice(
            [TimeOfDay.MORNING_9AM, TimeOfDay.EVENING_6PM], p=[0.3, 0.7]
        )
    elif age < 65:
        preferred_channel = np.random.choice(
            [CommunicationChannel.EMAIL, CommunicationChannel.PHONE], p=[0.6, 0.4]
        )
        preferred_time = np.random.choice(
            [TimeOfDay.MORNING_9AM, TimeOfDay.AFTERNOON_2PM], p=[0.4, 0.6]
        )
    else:
        preferred_channel = np.random.choice(
            [CommunicationChannel.PHONE, CommunicationChannel.MAIL_LETTER], p=[0.8, 0.2]
        )
        preferred_time = np.random.choice(
            [TimeOfDay.MORNING_9AM, TimeOfDay.AFTERNOON_2PM], p=[0.6, 0.4]
        )

    # Communication success rates based on preferences
    base_success = 0.7
    sms_success = base_success + (
        0.2 if preferred_channel == CommunicationChannel.SMS else 0
    )
    email_success = base_success + (
        0.2 if preferred_channel == CommunicationChannel.EMAIL else 0
    )
    phone_success = base_success + (
        0.2 if preferred_channel == CommunicationChannel.PHONE else 0
    )
    mail_success = base_success + (
        0.2 if preferred_channel == CommunicationChannel.MAIL_LETTER else 0
    )

    # Add noise to success rates
    sms_success = np.clip(sms_success + np.random.normal(0, 0.1), 0, 1)
    email_success = np.clip(email_success + np.random.normal(0, 0.1), 0, 1)
    phone_success = np.clip(phone_success + np.random.normal(0, 0.1), 0, 1)
    mail_success = np.clip(mail_success + np.random.normal(0, 0.1), 0, 1)

    # Behavioral factors
    medication_adherence = np.clip(
        0.8 - risk_score * 0.3 + np.random.normal(0, 0.1), 0, 1
    )
    appointment_compliance = np.clip(
        0.85 - risk_score * 0.2 + np.random.normal(0, 0.1), 0, 1
    )
    response_time_preference = np.random.uniform(0, 1)

    return EnhancedPatientState(
        risk_score=risk_score,
        age=int(age),
        comorbidities=comorbidities,
        bmi=bmi,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        current_hour=current_hour,
        day_of_week=day_of_week,
        urgency_level=urgency_level,
        last_contact_hours_ago=last_contact_hours_ago,
        next_appointment_days=next_appointment_days,
        preferred_channel=preferred_channel,
        preferred_time=preferred_time,
        sms_success_rate=sms_success,
        email_success_rate=email_success,
        phone_success_rate=phone_success,
        mail_success_rate=mail_success,
        medication_adherence=medication_adherence,
        appointment_compliance=appointment_compliance,
        response_time_preference=response_time_preference,
    )


def _sample_appropriate_action(
    patient_state: EnhancedPatientState, action_space: TemporalActionSpace
) -> TemporalAction:
    """Sample appropriate action.

    Args:
        patient_state (EnhancedPatientState): Patient state for sampling.
        action_space (TemporalActionSpace): Action space manager.

    Returns:
        TemporalAction: Sampled composite action.
    """

    # REBALANCED: More monitoring for low-risk, less medication
    if patient_state.risk_score < 0.2:
        healthcare_action = np.random.choice(
            [HealthcareAction.MONITOR, HealthcareAction.DISCHARGE], p=[0.7, 0.3]
        )  # Increased monitoring preference
    elif patient_state.risk_score < 0.4:
        healthcare_action = np.random.choice(
            [HealthcareAction.MONITOR, HealthcareAction.FOLLOWUP], p=[0.8, 0.2]
        )  # Strong monitoring preference
    elif patient_state.risk_score < 0.6:
        healthcare_action = np.random.choice(
            [HealthcareAction.MONITOR, HealthcareAction.MEDICATE], p=[0.6, 0.4]
        )  # Balanced but favor monitoring
    elif patient_state.risk_score < 0.8:
        healthcare_action = np.random.choice(
            [HealthcareAction.REFER, HealthcareAction.MEDICATE], p=[0.7, 0.3]
        )
    else:
        healthcare_action = HealthcareAction.REFER

    # Choose time horizon based on urgency
    if patient_state.urgency_level > 0.7:
        time_horizon = np.random.choice(
            [TimeHorizon.IMMEDIATE, TimeHorizon.ONE_HOUR], p=[0.8, 0.2]
        )
    elif patient_state.urgency_level > 0.4:
        time_horizon = np.random.choice(
            [TimeHorizon.ONE_HOUR, TimeHorizon.FOUR_HOURS, TimeHorizon.ONE_DAY],
            p=[0.3, 0.4, 0.3],
        )
    else:
        time_horizon = np.random.choice(
            [TimeHorizon.ONE_DAY, TimeHorizon.THREE_DAYS, TimeHorizon.ONE_WEEK],
            p=[0.3, 0.4, 0.3],
        )

    # Use patient preferences for communication
    communication_channel = patient_state.preferred_channel
    time_of_day = patient_state.preferred_time

    return TemporalAction(
        healthcare_action=healthcare_action,
        time_horizon=time_horizon,
        time_of_day=time_of_day,
        communication_channel=communication_channel,
    )


def _sample_monitoring_focused_action(
    patient_state: EnhancedPatientState,
) -> TemporalAction:
    """Sample monitoring-focused action.

    Args:
        patient_state (EnhancedPatientState): Patient state for sampling.

    Returns:
        TemporalAction: Sampled composite action.
    """

    # Always use monitoring action
    healthcare_action = HealthcareAction.MONITOR

    # Choose appropriate time horizons for monitoring
    if patient_state.risk_score < 0.3:
        # Low risk: longer monitoring periods
        time_horizon = np.random.choice(
            [TimeHorizon.ONE_WEEK, TimeHorizon.TWO_WEEKS, TimeHorizon.ONE_MONTH],
            p=[0.4, 0.4, 0.2],
        )
    elif patient_state.risk_score < 0.6:
        # Medium risk: moderate monitoring periods
        time_horizon = np.random.choice(
            [TimeHorizon.THREE_DAYS, TimeHorizon.ONE_WEEK], p=[0.6, 0.4]
        )
    else:
        # High risk: shorter monitoring periods
        time_horizon = np.random.choice(
            [TimeHorizon.ONE_DAY, TimeHorizon.THREE_DAYS], p=[0.7, 0.3]
        )

    # Use patient preferences
    communication_channel = patient_state.preferred_channel
    time_of_day = patient_state.preferred_time

    return TemporalAction(
        healthcare_action=healthcare_action,
        time_horizon=time_horizon,
        time_of_day=time_of_day,
        communication_channel=communication_channel,
    )


def _sample_sms_focused_action(patient_state: EnhancedPatientState) -> TemporalAction:
    """Sample SMS-focused action.

    Args:
        patient_state (EnhancedPatientState): Patient state for sampling.

    Returns:
        TemporalAction: Sampled composite action.
    """

    # Choose appropriate healthcare action
    if patient_state.risk_score < 0.4:
        healthcare_action = np.random.choice(
            [HealthcareAction.MONITOR, HealthcareAction.FOLLOWUP], p=[0.7, 0.3]
        )
    elif patient_state.risk_score < 0.7:
        healthcare_action = np.random.choice(
            [
                HealthcareAction.MONITOR,
                HealthcareAction.MEDICATE,
                HealthcareAction.FOLLOWUP,
            ],
            p=[0.5, 0.3, 0.2],
        )
    else:
        # High risk: SMS might not be appropriate, but include some examples
        healthcare_action = np.random.choice(
            [HealthcareAction.REFER, HealthcareAction.MEDICATE], p=[0.6, 0.4]
        )

    # Choose time horizon
    if patient_state.urgency_level > 0.7:
        time_horizon = np.random.choice(
            [TimeHorizon.IMMEDIATE, TimeHorizon.ONE_HOUR, TimeHorizon.FOUR_HOURS],
            p=[0.4, 0.3, 0.3],
        )
    else:
        time_horizon = np.random.choice(
            [TimeHorizon.ONE_DAY, TimeHorizon.THREE_DAYS, TimeHorizon.ONE_WEEK],
            p=[0.4, 0.4, 0.2],
        )

    # Always use SMS communication
    communication_channel = CommunicationChannel.SMS

    # Prefer convenient times for SMS
    time_of_day = np.random.choice(
        [TimeOfDay.MORNING_9AM, TimeOfDay.EVENING_6PM], p=[0.4, 0.6]
    )  # Evening preferred for SMS

    return TemporalAction(
        healthcare_action=healthcare_action,
        time_horizon=time_horizon,
        time_of_day=time_of_day,
        communication_channel=communication_channel,
    )


def _sample_underrepresented_action(
    patient_state: EnhancedPatientState,
) -> TemporalAction:
    """Sample underrepresented action.

    Args:
        patient_state (EnhancedPatientState): Patient state for sampling.

    Returns:
        TemporalAction: Sampled composite action.
    """

    # DELIBERATELY SAMPLE COUNTER-INTUITIVE BUT VALID ACTIONS
    sample_type = np.random.random()

    if sample_type < 0.3:
        # Medium-risk patients getting monitoring (often under-represented)
        healthcare_action = HealthcareAction.MONITOR
        time_horizon = np.random.choice([TimeHorizon.THREE_DAYS, TimeHorizon.ONE_WEEK])

    elif sample_type < 0.6:
        # Low-risk patients getting follow-up calls (under-represented)
        healthcare_action = HealthcareAction.FOLLOWUP
        time_horizon = np.random.choice([TimeHorizon.ONE_WEEK, TimeHorizon.TWO_WEEKS])

    else:
        # Wait actions with various time horizons (often under-represented)
        healthcare_action = np.random.choice(
            [HealthcareAction.MONITOR, HealthcareAction.FOLLOWUP]
        )
        # Longer wait times that might be under-represented
        time_horizon = np.random.choice([TimeHorizon.TWO_WEEKS, TimeHorizon.ONE_MONTH])

    # Mix of communication channels and times
    communication_channel = np.random.choice(list(CommunicationChannel))
    time_of_day = np.random.choice(list(TimeOfDay))

    return TemporalAction(
        healthcare_action=healthcare_action,
        time_horizon=time_horizon,
        time_of_day=time_of_day,
        communication_channel=communication_channel,
    )


def _sample_medication_focused_action(
    patient_state: EnhancedPatientState,
) -> TemporalAction:
    """Sample medication-focused action.

    Args:
        patient_state (EnhancedPatientState): Patient state for sampling.

    Returns:
        TemporalAction: Sampled composite action.
    """

    # Always use medication action
    healthcare_action = HealthcareAction.MEDICATE

    # Choose appropriate time horizons for medication
    if patient_state.risk_score > 0.6:
        # High risk: immediate to short-term medication
        time_horizon = np.random.choice(
            [TimeHorizon.IMMEDIATE, TimeHorizon.ONE_HOUR, TimeHorizon.FOUR_HOURS],
            p=[0.4, 0.3, 0.3],
        )
    elif patient_state.risk_score > 0.3:
        # Medium risk: short to medium-term medication
        time_horizon = np.random.choice(
            [TimeHorizon.FOUR_HOURS, TimeHorizon.ONE_DAY, TimeHorizon.THREE_DAYS],
            p=[0.3, 0.4, 0.3],
        )
    else:
        # Low risk: longer medication periods (if any)
        time_horizon = np.random.choice(
            [TimeHorizon.ONE_DAY, TimeHorizon.THREE_DAYS, TimeHorizon.ONE_WEEK],
            p=[0.4, 0.4, 0.2],
        )

    # Prefer reliable communication for medication
    if patient_state.age > 65:
        communication_channel = CommunicationChannel.PHONE  # Elderly prefer phone
    elif patient_state.age < 40:
        communication_channel = CommunicationChannel.SMS  # Young prefer SMS
    else:
        communication_channel = np.random.choice(
            [CommunicationChannel.PHONE, CommunicationChannel.EMAIL], p=[0.6, 0.4]
        )

    # Prefer morning for medication instructions
    time_of_day = np.random.choice(
        [TimeOfDay.MORNING_9AM, TimeOfDay.AFTERNOON_2PM], p=[0.7, 0.3]
    )

    return TemporalAction(
        healthcare_action=healthcare_action,
        time_horizon=time_horizon,
        time_of_day=time_of_day,
        communication_channel=communication_channel,
    )


def _patient_state_to_dict(patient_state: EnhancedPatientState) -> Dict[str, Any]:
    """Convert patient state to dictionary for DataFrame.

    Args:
        patient_state (EnhancedPatientState): Patient state to convert.

    Returns:
        Dict[str, Any]: Mapping of feature names to values.
    """
    feature_vector = patient_state.to_feature_vector()
    feature_names = EnhancedPatientState.get_feature_names()
    return dict(zip(feature_names, feature_vector))


def demonstrate_temporal_training_data() -> None:
    """Demonstrate stratified temporal training data generation and summary statistics."""
    logger.info("Stratified Temporal Training Data Generation Demo")
    logger.info("=" * 60)

    # Generate small sample using stratified generator
    df = generate_stratified_temporal_training_data(1000)

    logger.info("Sample Data Shape", shape=df.shape, columns=list(df.columns))

    # Show reward distribution by risk tier
    logger.info("Reward Analysis by Risk Tier:")

    # Risk tier is already in the dataframe
    reward_by_risk = df.groupby("risk_tier")["reward"].agg(
        ["mean", "std", "count"]
    )
    logger.info("Reward by risk tier", reward_analysis=reward_by_risk.to_dict())

    # Show healthcare action distribution by risk tier
    action_by_risk = pd.crosstab(
        df["risk_tier"],
        df["healthcare_action"],
        normalize="index"
    ) * 100

    logger.info(
        "Healthcare action distribution by risk tier (%)",
        action_distribution=action_by_risk.to_dict()
    )

    logger.info("Stratified temporal training data demonstration completed!")


if __name__ == "__main__":
    demonstrate_temporal_training_data()
