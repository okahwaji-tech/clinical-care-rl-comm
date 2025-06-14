"""Healthcare prioritized temporal action definitions for DQN systems.

This module provides definitions for temporal healthcare actions, including
enums for each decision dimension, a composite action dataclass, the action
space manager, and enriched patient state representations for network input.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class HealthcareAction(Enum):
    """Primary healthcare actions."""

    MONITOR = 0
    REFER = 1
    MEDICATE = 2
    DISCHARGE = 3
    FOLLOWUP = 4


class TimeHorizon(Enum):
    """Time horizons for healthcare actions."""

    IMMEDIATE = 0  # 0-30 minutes
    ONE_HOUR = 1  # 1 hour
    FOUR_HOURS = 2  # 4 hours
    ONE_DAY = 3  # 24 hours
    THREE_DAYS = 4  # 3 days
    ONE_WEEK = 5  # 1 week
    TWO_WEEKS = 6  # 2 weeks
    ONE_MONTH = 7  # 1 month


class TimeOfDay(Enum):
    """Specific times of day for communication."""

    MORNING_9AM = 0  # 9:00 AM
    AFTERNOON_2PM = 1  # 2:00 PM
    EVENING_6PM = 2  # 6:00 PM
    NIGHT_9PM = 3  # 9:00 PM


class CommunicationChannel(Enum):
    """Communication channels for patient contact."""

    SMS = 0
    EMAIL = 1
    PHONE = 2
    MAIL_LETTER = 3


@dataclass
class TemporalAction:
    """Complete temporal healthcare action.

    Represents a single action composed of healthcare action, time horizon,
    time of day, and communication channel.

    Attributes:
        healthcare_action (HealthcareAction): Type of healthcare action.
        time_horizon (TimeHorizon): Time horizon for action execution.
        time_of_day (TimeOfDay): Specific time of day for action.
        communication_channel (CommunicationChannel): Channel used for communication.
    """

    healthcare_action: HealthcareAction
    time_horizon: TimeHorizon
    time_of_day: TimeOfDay
    communication_channel: CommunicationChannel

    def to_string(self) -> str:
        """Convert to human-readable string.

        Returns:
            str: Human-readable representation of the temporal action.
        """
        return f"{self.healthcare_action.name.lower()}_{self.time_horizon.name.lower()}_{self.time_of_day.name.lower()}_{self.communication_channel.name.lower()}"

    def to_indices(self) -> Tuple[int, int, int, int]:
        """Convert to action indices for network.

        Returns:
            Tuple[int, int, int, int]: Indices for healthcare_action, time_horizon,
                time_of_day, and communication_channel.
        """
        return (
            self.healthcare_action.value,
            self.time_horizon.value,
            self.time_of_day.value,
            self.communication_channel.value,
        )


class TemporalActionSpace:
    """Manages the temporal action space for healthcare DQN.

    Initializes lists of possible healthcare actions, time horizons,
    times of day, and communication channels, and computes total combinations.
    """

    def __init__(self) -> None:
        """Initialize the temporal action space.

        Sets up action lists and computes the total number of action combinations.
        """
        self.healthcare_actions = list(HealthcareAction)
        self.time_horizons = list(TimeHorizon)
        self.times_of_day = list(TimeOfDay)
        self.communication_channels = list(CommunicationChannel)

        # Action space dimensions
        self.n_healthcare = len(self.healthcare_actions)
        self.n_time_horizons = len(self.time_horizons)
        self.n_times_of_day = len(self.times_of_day)
        self.n_communication = len(self.communication_channels)

        # Total possible combinations
        self.total_combinations = (
            self.n_healthcare
            * self.n_time_horizons
            * self.n_times_of_day
            * self.n_communication
        )

        # REMOVED: Print statements from __init__ (side effects)
        # Use logging if needed for debugging

    def sample_random_action(self) -> TemporalAction:
        """Sample a random temporal action.

        Returns:
            TemporalAction: Randomly selected composite action.
        """
        return TemporalAction(
            healthcare_action=np.random.choice(self.healthcare_actions),
            time_horizon=np.random.choice(self.time_horizons),
            time_of_day=np.random.choice(self.times_of_day),
            communication_channel=np.random.choice(self.communication_channels),
        )

    def indices_to_action(
        self, healthcare_idx: int, time_idx: int, schedule_idx: int, comm_idx: int
    ) -> TemporalAction:
        """Convert indices back to a TemporalAction.

        Args:
            healthcare_idx (int): Index into healthcare_actions.
            time_idx (int): Index into time_horizons.
            schedule_idx (int): Index into times_of_day.
            comm_idx (int): Index into communication_channels.

        Returns:
            TemporalAction: Reconstructed composite action.
        """
        return TemporalAction(
            healthcare_action=self.healthcare_actions[healthcare_idx],
            time_horizon=self.time_horizons[time_idx],
            time_of_day=self.times_of_day[schedule_idx],
            communication_channel=self.communication_channels[comm_idx],
        )

    def get_action_names(self) -> Dict[str, List[str]]:
        """Get human-readable action names for each dimension.

        Returns:
            Dict[str, List[str]]: Mapping of dimension names to lists of action names.
        """
        return {
            "healthcare": [action.name.lower() for action in self.healthcare_actions],
            "time_horizons": [horizon.name.lower() for horizon in self.time_horizons],
            "times_of_day": [time.name.lower() for time in self.times_of_day],
            "communication": [
                channel.name.lower() for channel in self.communication_channels
            ],
        }


@dataclass
class EnhancedPatientState:
    """Enhanced patient state including clinical, temporal, and communication features.

    Combines normalized clinical measures, temporal context, communication preferences,
    and contextual factors into a single dataclass for network input.

    Attributes:
        risk_score (float): Patient risk score.
        age (int): Patient age.
        comorbidities (int): Count of comorbid conditions.
        bmi (float): Body mass index.
        systolic_bp (float): Systolic blood pressure.
        diastolic_bp (float): Diastolic blood pressure.
        heart_rate (float): Heart rate.
        current_hour (int): Hour of day (0-23).
        day_of_week (int): Day of week (0=Monday).
        urgency_level (float): Derived urgency level (0-1).
        last_contact_hours_ago (float): Hours since last contact.
        next_appointment_days (float): Days until next appointment.
        preferred_channel (CommunicationChannel): Preferred contact channel.
        preferred_time (TimeOfDay): Preferred time of day for contact.
        sms_success_rate (float): SMS success probability.
        email_success_rate (float): Email success probability.
        phone_success_rate (float): Phone success probability.
        mail_success_rate (float): Mail success probability.
        medication_adherence (float): Adherence level (0-1).
        appointment_compliance (float): Compliance level (0-1).
        response_time_preference (float): Preference for response speed (0-1).
    """

    # Clinical features (existing)
    risk_score: float
    age: int
    comorbidities: int
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float

    # NEW: Temporal features
    current_hour: int  # 0-23
    day_of_week: int  # 0-6 (Monday=0)
    urgency_level: float  # 0-1 (derived from clinical state)
    last_contact_hours_ago: float
    next_appointment_days: float

    # NEW: Communication preferences and history
    preferred_channel: CommunicationChannel
    preferred_time: TimeOfDay
    sms_success_rate: float
    email_success_rate: float
    phone_success_rate: float
    mail_success_rate: float

    # NEW: Contextual factors
    medication_adherence: float  # 0-1
    appointment_compliance: float  # 0-1
    response_time_preference: float  # 0-1 (immediate vs delayed)

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for neural network input.

        Combines and normalizes clinical, temporal, preference, and contextual
        features into a float32 numpy array.

        Returns:
            np.ndarray: Feature vector array of shape (feature_count,).
        """
        features = [
            # Clinical features - ENHANCED FEATURE SCALING
            self.risk_score * 2.0 - 1.0,  # Rescale from [0,1] to [-1,1] for better feature parity
            self.age / 100.0,  # Normalize
            self.comorbidities / 10.0,  # Normalize
            self.bmi / 50.0,  # Normalize
            self.systolic_bp / 200.0,  # Normalize
            self.diastolic_bp / 120.0,  # Normalize
            self.heart_rate / 200.0,  # Normalize
            # Temporal features
            self.current_hour / 24.0,  # Normalize
            self.day_of_week / 7.0,  # Normalize
            self.urgency_level,
            np.log1p(self.last_contact_hours_ago) / 10.0,  # Log-normalize
            self.next_appointment_days / 30.0,  # Normalize
            # Communication preferences (one-hot encoded)
            1.0 if self.preferred_channel == CommunicationChannel.SMS else 0.0,
            1.0 if self.preferred_channel == CommunicationChannel.EMAIL else 0.0,
            1.0 if self.preferred_channel == CommunicationChannel.PHONE else 0.0,
            1.0 if self.preferred_channel == CommunicationChannel.MAIL_LETTER else 0.0,
            # Preferred time (one-hot encoded)
            1.0 if self.preferred_time == TimeOfDay.MORNING_9AM else 0.0,
            1.0 if self.preferred_time == TimeOfDay.AFTERNOON_2PM else 0.0,
            1.0 if self.preferred_time == TimeOfDay.EVENING_6PM else 0.0,
            1.0 if self.preferred_time == TimeOfDay.NIGHT_9PM else 0.0,
            # Communication success rates
            self.sms_success_rate,
            self.email_success_rate,
            self.phone_success_rate,
            self.mail_success_rate,
            # Contextual factors
            self.medication_adherence,
            self.appointment_compliance,
            self.response_time_preference,
        ]

        return np.array(features, dtype=np.float32)

    @classmethod
    def get_feature_count(cls) -> int:
        """Get the total number of features in the vector.

        Returns:
            int: Number of features produced by to_feature_vector().
        """
        return 27  # Total features in to_feature_vector()

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get names of all features for interpretability.

        Returns:
            List[str]: List of feature name strings.
        """
        return [
            # Clinical features
            "risk_score_scaled",  # Scaled from [0,1] to [-1,1]
            "age_norm",
            "comorbidities_norm",
            "bmi_norm",
            "systolic_bp_norm",
            "diastolic_bp_norm",
            "heart_rate_norm",
            # Temporal features
            "current_hour_norm",
            "day_of_week_norm",
            "urgency_level",
            "last_contact_log_norm",
            "next_appointment_norm",
            # Communication preferences
            "prefers_sms",
            "prefers_email",
            "prefers_phone",
            "prefers_mail",
            # Preferred time
            "prefers_morning",
            "prefers_afternoon",
            "prefers_evening",
            "prefers_night",
            # Communication success rates
            "sms_success_rate",
            "email_success_rate",
            "phone_success_rate",
            "mail_success_rate",
            # Contextual factors
            "medication_adherence",
            "appointment_compliance",
            "response_time_preference",
        ]


def create_temporal_action_space() -> TemporalActionSpace:
    """Get a new TemporalActionSpace instance.

    Returns:
        TemporalActionSpace: Initialized action space manager.
    """
    return TemporalActionSpace()


if __name__ == "__main__":
    # For testing imports only
    from core.logging_system import get_logger

    logger = get_logger(__name__)

    action_space = create_temporal_action_space()
    logger.info(
        "Temporal action space initialized",
        total_combinations=action_space.total_combinations,
    )
