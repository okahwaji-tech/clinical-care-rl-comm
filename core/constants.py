"""Healthcare patient care optimization constants.

This module defines comprehensive clinical feature sets and constants
for healthcare patient care optimization using deep reinforcement learning.
"""

FEATURES_SET = {
    "clinical_features": [
        # Patient Demographics & Risk Assessment
        "patient_age_years",
        "patient_risk_score_ratio",
        "comorbidity_count",
        "bmi_category",
        # Vital Signs & Clinical Measurements
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "heart_rate_bpm",
        "respiratory_rate",
        "oxygen_saturation_percent",
        "temperature_celsius",
        # Laboratory Values
        "hemoglobin_level",
        "white_blood_cell_count",
        "creatinine_level",
        "glucose_level",
        # Treatment & Care Management
        "treatment_duration_hours",
        "treatment_modality_type",
        "medication_count",
        "care_unit_id",
        "treatment_delay_days",
        # Scheduling & Temporal Features
        "appointment_day_of_week",
        "appointment_day_of_month",
        "appointment_month",
        "followup_scheduled_day_of_week",
        "followup_scheduled_day_of_month",
        "followup_scheduled_hour_of_day",
    ]
}

CLINICAL_FEATURE_COUNT = len(FEATURES_SET["clinical_features"])

# Rainbow DQN Hyperparameters
RAINBOW_CONFIG = {
    "n_step": 3,  # N-step returns
    "num_atoms": 51,  # Number of atoms for distributional RL (C51)
    "v_min": -10.0,  # Minimum value for distributional RL
    "v_max": 10.0,  # Maximum value for distributional RL
    "noisy_std": 0.5,  # Standard deviation for noisy networks
    "target_update_freq": 1000,  # Target network update frequency
}

# Prioritized Experience Replay Hyperparameters
PER_CONFIG = {
    "alpha": 0.6,  # Prioritization exponent
    "beta_start": 0.4,  # Initial importance sampling weight
    "beta_frames": 100000,  # Frames over which to anneal beta to 1.0
    "epsilon": 1e-6,  # Small constant to prevent zero priorities
    "max_priority": 1.0,  # Maximum priority value
}

# Conservative Q-Learning Hyperparameters
CQL_CONFIG = {
    "cql_alpha": 1.0,  # CQL regularization coefficient
    "cql_temperature": 1.0,  # Temperature for CQL logsumexp
    "num_random_actions": 10,  # Number of random actions for CQL penalty
}

# Network Architecture Constants
NETWORK_CONFIG = {
    "default_input_dim": 27,  # Enhanced patient state features
    "default_hidden_dim": 256,  # Hidden layer dimension
    "gradient_clip_norm": 1.0,  # Gradient clipping threshold
    "dropout_rate": 0.1,  # Dropout rate for regularization
}

# Training Constants
TRAINING_CONFIG = {
    "default_learning_rate": 0.0001,
    "default_batch_size": 64,
    "default_gamma": 0.99,  # Discount factor
    "default_epsilon": 0.1,  # Epsilon-greedy exploration
    "default_temperature": 1.2,  # Softmax temperature for exploration
    "max_gradient_buffer_size": 101,  # From JacobianStore
    "numerical_perturbation_delta": 1e-6,  # From JacobianStore
}

# Temporal Action Space Constants
ACTION_SPACE_CONFIG = {
    "n_healthcare_actions": 5,  # monitor, refer, medicate, discharge, followup
    "n_time_horizons": 8,  # immediate to 1 month
    "n_times_of_day": 4,  # morning, afternoon, evening, night
    "n_communication_channels": 4,  # SMS, email, phone, mail
}

# Data Generation Constants
DATA_GENERATION_CONFIG = {
    "random_action_probability": 0.45,  # 45% random actions for diversity
    "reward_noise_std": 0.1,  # Standard deviation for reward noise
    "default_sample_count": 3000,  # Default number of training samples
    "min_samples_per_action": 50,  # Minimum samples per action type
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "metrics_log_frequency": 100,  # Log metrics every N steps
    "checkpoint_frequency": 1000,  # Save checkpoint every N steps
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "min_prediction_speed": 2000,  # predictions per second
    "max_memory_usage_mb": 500,  # Maximum memory usage in MB
    "max_import_time_seconds": 2,  # Maximum import time
    "target_test_coverage": 0.9,  # 90% test coverage target
}
