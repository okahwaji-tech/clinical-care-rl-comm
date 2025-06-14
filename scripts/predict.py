#!/usr/bin/env python3
"""Sample Prediction Demo for Enhanced Temporal Factored Rainbow DQN.

This script demonstrates the complete temporal recommendation system using
sample patient data across different risk levels. Shows healthcare actions,
timing recommendations, and communication channel optimization.

Usage:
    python scripts/predict.py                    # Use latest trained model
    python scripts/predict.py --model MODEL.pt  # Use specific model
    python scripts/predict.py --untrained       # Demo with untrained model
"""

import argparse
import sys
import os
import glob
from typing import List, Dict, Any
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.temporal_rainbow_dqn import TemporalFactoredDQN
from core.temporal_actions import (
    EnhancedPatientState, TemporalAction,
    HealthcareAction, TimeHorizon, TimeOfDay, CommunicationChannel
)
from core.logging_system import get_logger

logger = get_logger(__name__)


def create_sample_patients() -> List[Dict[str, Any]]:
    """Create diverse sample patients for prediction demo."""
    patients = [
        {
            "name": "Sarah Johnson (Low Risk)",
            "description": "35-year-old healthy patient, routine checkup",
            "state": EnhancedPatientState(
                risk_score=0.2, age=35, comorbidities=1, bmi=24.0, 
                systolic_bp=120, diastolic_bp=80, heart_rate=70,
                current_hour=10, day_of_week=2, urgency_level=0.2,
                last_contact_hours_ago=48, next_appointment_days=14,
                preferred_channel=CommunicationChannel.SMS, 
                preferred_time=TimeOfDay.MORNING_9AM,
                sms_success_rate=0.9, email_success_rate=0.8, 
                phone_success_rate=0.7, mail_success_rate=0.6,
                medication_adherence=0.9, appointment_compliance=0.9, 
                response_time_preference=0.6
            )
        },
        {
            "name": "Robert Chen (Medium Risk)",
            "description": "55-year-old with diabetes and hypertension",
            "state": EnhancedPatientState(
                risk_score=0.5, age=55, comorbidities=3, bmi=28.5, 
                systolic_bp=140, diastolic_bp=90, heart_rate=80,
                current_hour=14, day_of_week=3, urgency_level=0.5,
                last_contact_hours_ago=12, next_appointment_days=7,
                preferred_channel=CommunicationChannel.EMAIL, 
                preferred_time=TimeOfDay.AFTERNOON_2PM,
                sms_success_rate=0.7, email_success_rate=0.9, 
                phone_success_rate=0.8, mail_success_rate=0.5,
                medication_adherence=0.7, appointment_compliance=0.8, 
                response_time_preference=0.4
            )
        },
        {
            "name": "Maria Rodriguez (High Risk)",
            "description": "75-year-old with multiple comorbidities, recent symptoms",
            "state": EnhancedPatientState(
                risk_score=0.8, age=75, comorbidities=6, bmi=35.0, 
                systolic_bp=170, diastolic_bp=105, heart_rate=95,
                current_hour=16, day_of_week=1, urgency_level=0.8,
                last_contact_hours_ago=2, next_appointment_days=1,
                preferred_channel=CommunicationChannel.PHONE, 
                preferred_time=TimeOfDay.EVENING_6PM,
                sms_success_rate=0.5, email_success_rate=0.6, 
                phone_success_rate=0.9, mail_success_rate=0.4,
                medication_adherence=0.6, appointment_compliance=0.7, 
                response_time_preference=0.2
            )
        },
        {
            "name": "James Wilson (Very High Risk)",
            "description": "68-year-old post-cardiac event, critical monitoring needed",
            "state": EnhancedPatientState(
                risk_score=0.9, age=68, comorbidities=8, bmi=32.0, 
                systolic_bp=180, diastolic_bp=110, heart_rate=105,
                current_hour=20, day_of_week=0, urgency_level=0.9,
                last_contact_hours_ago=1, next_appointment_days=0,
                preferred_channel=CommunicationChannel.PHONE, 
                preferred_time=TimeOfDay.NIGHT_9PM,
                sms_success_rate=0.4, email_success_rate=0.5, 
                phone_success_rate=0.95, mail_success_rate=0.3,
                medication_adherence=0.8, appointment_compliance=0.9, 
                response_time_preference=0.1
            )
        }
    ]
    return patients


def load_model(model_path: str = None, use_untrained: bool = False) -> TemporalFactoredDQN:
    """Load trained model or create untrained model for demo."""
    if use_untrained:
        logger.info("Creating untrained model for demonstration")
        model = TemporalFactoredDQN(
            input_dim=27,
            use_dueling=True,
            use_noisy=True,
            use_distributional=True
        )
        logger.warning("Using UNTRAINED model - predictions are random!")
        return model
    
    # Find latest model if no specific path provided
    if model_path is None:
        # Look in models directory first, then root directory for backward compatibility
        model_files = glob.glob("models/enhanced_tfrdqn_*.pt") + glob.glob("enhanced_tfrdqn_*.pt")
        if not model_files:
            logger.warning("No trained models found in models/ or root directory")
            logger.info("Train a model first with: python scripts/train.py --quick")
            return load_model(use_untrained=True)

        # Get most recent model
        model_path = max(model_files, key=os.path.getctime)
        logger.info(f"Using latest trained model: {model_path}")
    
    try:
        # Load trained model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = TemporalFactoredDQN(
            input_dim=27,
            use_dueling=True,
            use_noisy=True,
            use_distributional=True
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Loaded trained model from {model_path}")
        
        # Show training info if available
        if 'training_results' in checkpoint:
            training_info = checkpoint['training_results']
            if 'training' in training_info:
                final_loss = training_info['training'].get('final_loss', 'unknown')
                samples = checkpoint.get('samples', 'unknown')
                logger.info(f"Model training info: {samples} samples, final loss: {final_loss}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        logger.info("Falling back to untrained model")
        return load_model(use_untrained=True)


def predict_for_patient(model: TemporalFactoredDQN, patient: Dict[str, Any]) -> TemporalAction:
    """Generate prediction for a single patient."""
    state = patient["state"]
    features = state.to_feature_vector()
    
    # Get prediction (disable exploration for consistent results)
    action = model.predict_temporal_action(features, use_exploration=False)
    return action


def format_clinical_interpretation(patient: Dict[str, Any], action: TemporalAction) -> str:
    """Format clinical interpretation of the recommendation."""
    risk_score = patient["state"].risk_score
    
    # Risk level interpretation
    if risk_score >= 0.8:
        risk_level = "VERY HIGH"
    elif risk_score >= 0.6:
        risk_level = "HIGH"
    elif risk_score >= 0.4:
        risk_level = "MEDIUM"
    elif risk_score >= 0.2:
        risk_level = "LOW"
    else:
        risk_level = "VERY LOW"
    
    # Action interpretation
    action_interpretations = {
        HealthcareAction.MONITOR: "Continue monitoring patient status",
        HealthcareAction.REFER: "Refer to specialist or higher level of care",
        HealthcareAction.MEDICATE: "Initiate or adjust medication",
        HealthcareAction.DISCHARGE: "Patient can be safely discharged",
        HealthcareAction.FOLLOWUP: "Schedule follow-up appointment"
    }
    
    # Timing interpretation
    timing_interpretations = {
        TimeHorizon.IMMEDIATE: "Take action immediately",
        TimeHorizon.ONE_HOUR: "Take action within 1 hour",
        TimeHorizon.FOUR_HOURS: "Take action within 4 hours",
        TimeHorizon.ONE_DAY: "Take action within 24 hours",
        TimeHorizon.THREE_DAYS: "Take action within 3 days",
        TimeHorizon.ONE_WEEK: "Take action within 1 week",
        TimeHorizon.TWO_WEEKS: "Take action within 2 weeks",
        TimeHorizon.ONE_MONTH: "Take action within 1 month"
    }
    
    # Communication timing
    time_interpretations = {
        TimeOfDay.MORNING_9AM: "Contact at 9:00 AM",
        TimeOfDay.AFTERNOON_2PM: "Contact at 2:00 PM", 
        TimeOfDay.EVENING_6PM: "Contact at 6:00 PM",
        TimeOfDay.NIGHT_9PM: "Contact at 9:00 PM"
    }
    
    interpretation = f"""
    Risk Level: {risk_level} (Score: {risk_score:.1f})

    Recommended Action: {action.healthcare_action.name}
       → {action_interpretations.get(action.healthcare_action, 'Unknown action')}

    Timing: {action.time_horizon.name}
       → {timing_interpretations.get(action.time_horizon, 'Unknown timing')}

    Communication Time: {action.time_of_day.name}
       → {time_interpretations.get(action.time_of_day, 'Unknown time')}

    Communication Channel: {action.communication_channel.name}
       → Use {action.communication_channel.name.lower()} for patient contact
    """
    
    return interpretation


def run_prediction_demo(model_path: str = None, use_untrained: bool = False):
    """Run complete prediction demonstration."""
    logger.info("ENHANCED TEMPORAL HEALTHCARE PREDICTION DEMO")
    logger.info("=" * 70)

    # Load model
    model = load_model(model_path, use_untrained)

    # Create sample patients
    patients = create_sample_patients()

    logger.info(f"Generating predictions for {len(patients)} sample patients")
    logger.info("=" * 70)
    
    # Generate predictions for each patient
    for i, patient in enumerate(patients, 1):
        print(f"\n{'='*70}")
        print(f"PATIENT {i}: {patient['name']}")
        print(f"{'='*70}")
        print(f"Description: {patient['description']}")

        # Generate prediction
        action = predict_for_patient(model, patient)

        # Show complete temporal recommendation
        print(f"\nCOMPLETE TEMPORAL RECOMMENDATION:")
        print(f"   Healthcare Action: {action.healthcare_action.name}")
        print(f"   Time Horizon: {action.time_horizon.name}")
        print(f"   Time of Day: {action.time_of_day.name}")
        print(f"   Communication: {action.communication_channel.name}")
        print(f"   Full Action String: {action.to_string()}")

        # Show clinical interpretation
        interpretation = format_clinical_interpretation(patient, action)
        print(f"\nCLINICAL INTERPRETATION:{interpretation}")

        # Show patient preferences alignment
        state = patient["state"]
        pref_match = "MATCH" if action.communication_channel == state.preferred_channel else "MISMATCH"
        time_match = "MATCH" if action.time_of_day == state.preferred_time else "MISMATCH"

        print(f"\nPREFERENCE ALIGNMENT:")
        print(f"   Communication: Recommended {action.communication_channel.name}, Patient prefers {state.preferred_channel.name} ({pref_match})")
        print(f"   Timing: Recommended {action.time_of_day.name}, Patient prefers {state.preferred_time.name} ({time_match})")
    
    print(f"\n{'='*70}")
    print("PREDICTION DEMO COMPLETED")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  - Multi-dimensional temporal recommendations")
    print("  - Risk-stratified clinical decision making")
    print("  - Patient preference consideration")
    print("  - Complete healthcare communication optimization")
    print("  - Clinical appropriateness validation")

    if use_untrained:
        print("\nNote: This demo used an UNTRAINED model.")
        print("   For realistic predictions, train a model first:")
        print("   python scripts/train.py --quick")


def main():
    """Main entry point for prediction demo."""
    parser = argparse.ArgumentParser(description="Enhanced Temporal Healthcare Prediction Demo")
    parser.add_argument("--model", type=str, help="Path to trained model file")
    parser.add_argument("--untrained", action="store_true", help="Use untrained model for demo")
    
    args = parser.parse_args()
    
    try:
        run_prediction_demo(model_path=args.model, use_untrained=args.untrained)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction demo failed: {e}")
        print(f"\nDEMO FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
