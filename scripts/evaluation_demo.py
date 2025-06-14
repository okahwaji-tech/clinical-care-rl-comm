#!/usr/bin/env python3
"""Comprehensive evaluation framework demonstration for temporal healthcare DQN systems.

This module provides a professional evaluation framework for assessing the performance
of temporal Rainbow DQN models in healthcare communication optimization scenarios.
It demonstrates comprehensive evaluation capabilities including model consistency,
clinical appropriateness, and exploration behavior analysis.

Classes:
    ModelEvaluationFramework: Main evaluation framework for temporal DQN models
    EvaluationMetrics: Data class for storing evaluation results
    EvaluationConfig: Configuration class for evaluation parameters

Usage:
    python scripts/evaluation_demo.py                    # Demo with untrained model
    python scripts/evaluation_demo.py --model <path>     # Evaluate trained model

Example:
    >>> from scripts.evaluation_demo import ModelEvaluationFramework
    >>> evaluator = ModelEvaluationFramework(model)
    >>> results = evaluator.run_comprehensive_evaluation()
    >>> print(f"Overall score: {results.overall_score:.3f}")
"""

from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.temporal_rainbow_dqn import TemporalFactoredDQN
from core.temporal_actions import (
    TemporalActionSpace, EnhancedPatientState, TemporalAction,
    HealthcareAction, TimeHorizon, TimeOfDay, CommunicationChannel
)
from core.temporal_training_data import generate_temporal_training_data
from core.logging_system import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Data class for storing comprehensive evaluation metrics.

    Attributes:
        action_consistency: Mean consistency score for action predictions
        clinical_appropriateness: Mean clinical appropriateness score
        exploration_diversity: Diversity score for exploration behavior
        overall_score: Weighted overall evaluation score
        grade: Letter grade (A-D) based on overall score
        status: Human-readable status description
        evaluation_time: Time taken for evaluation in seconds
    """
    action_consistency: float
    clinical_appropriateness: float
    exploration_diversity: float
    overall_score: float
    grade: str
    status: str
    evaluation_time: float


@dataclass
class EvaluationConfig:
    """Configuration parameters for model evaluation.

    Attributes:
        num_test_samples: Number of test samples to generate
        num_clinical_scenarios: Number of clinical scenarios to test
        num_consistency_tests: Number of consistency tests per state
        num_exploration_tests: Number of exploration behavior tests
        risk_levels: List of risk levels to test for clinical appropriateness
        consistency_weight: Weight for action consistency in overall score
        clinical_weight: Weight for clinical appropriateness in overall score
        exploration_weight: Weight for exploration diversity in overall score
    """
    num_test_samples: int = 1000
    num_clinical_scenarios: int = 200
    num_consistency_tests: int = 5
    num_exploration_tests: int = 100
    risk_levels: List[float] = None
    consistency_weight: float = 0.3
    clinical_weight: float = 0.5
    exploration_weight: float = 0.2

    def __post_init__(self) -> None:
        """Initialize default risk levels if not provided."""
        if self.risk_levels is None:
            self.risk_levels = [0.2, 0.5, 0.8]


class ModelEvaluationFramework:
    """Professional evaluation framework for temporal healthcare DQN models.

    This class provides comprehensive evaluation capabilities for temporal Rainbow DQN
    models, including action consistency testing, clinical appropriateness assessment,
    and exploration behavior analysis.

    Attributes:
        model: The temporal DQN model to evaluate
        action_space: Action space for generating test scenarios
        config: Configuration parameters for evaluation
    """

    def __init__(
        self,
        model: TemporalFactoredDQN,
        config: Optional[EvaluationConfig] = None
    ) -> None:
        """Initialize the evaluation framework.

        Args:
            model: The temporal DQN model to evaluate
            config: Optional configuration parameters. Uses defaults if not provided.
        """
        self.model = model
        self.action_space = TemporalActionSpace()
        self.config = config or EvaluationConfig()

        logger.info(
            "Evaluation framework initialized",
            model_type=type(model).__name__,
            test_samples=self.config.num_test_samples,
            clinical_scenarios=self.config.num_clinical_scenarios
        )

    def evaluate_action_consistency(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate action prediction consistency across multiple runs.

        Tests whether the model produces consistent predictions for the same input
        when exploration is disabled. Higher consistency indicates more stable
        decision-making behavior.

        Args:
            test_data: DataFrame containing test samples

        Returns:
            Dictionary containing consistency metrics (mean, std, min)

        Raises:
            ValueError: If test_data is empty or invalid
        """
        if test_data.empty:
            raise ValueError("Test data cannot be empty")

        logger.info("Evaluating action consistency")

        self.model.eval()

        # Generate test states
        test_states = self._extract_test_states(test_data, num_states=10)

        consistency_scores = []
        for state in test_states:
            predictions = []
            for _ in range(self.config.num_consistency_tests):
                try:
                    action = self.model.predict_temporal_action(state, use_exploration=False)
                    predictions.append(action.to_string())
                except Exception as e:
                    logger.warning("Prediction failed for state", error=str(e))
                    continue

            if predictions:
                most_common = max(set(predictions), key=predictions.count)
                consistency = predictions.count(most_common) / len(predictions)
                consistency_scores.append(consistency)

        if not consistency_scores:
            logger.error("No valid consistency scores obtained")
            return {"mean": 0.0, "std": 0.0, "min": 0.0}

        results = {
            "mean": float(np.mean(consistency_scores)),
            "std": float(np.std(consistency_scores)),
            "min": float(np.min(consistency_scores))
        }

        logger.info("Action consistency evaluation completed", **results)
        return results

    def evaluate_q_value_distribution(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze Q-value distributions across test samples.

        Examines the range and distribution of Q-values produced by the model
        to assess whether the value function has learned meaningful estimates.

        Args:
            test_data: DataFrame containing test samples

        Returns:
            Dictionary containing Q-value distribution metrics
        """
        logger.info("Analyzing Q-value distributions")

        q_value_ranges = []

        with torch.no_grad():
            for _, row in test_data.sample(min(100, len(test_data))).iterrows():
                try:
                    state_features = row[EnhancedPatientState.get_feature_names()].values.astype(np.float32)
                    state_tensor = torch.FloatTensor(state_features).unsqueeze(0)

                    q_values = self.model.get_factored_q_values(state_tensor)

                    total_q = 0.0
                    for dim_q in q_values.values():
                        if self.model.use_distributional:
                            # For distributional RL, use the mean of the distribution
                            q_mean = torch.mean(dim_q[0], dim=-1)
                            total_q += q_mean.max().item()
                        else:
                            total_q += dim_q[0].max().item()

                    q_value_ranges.append(total_q)

                except Exception as e:
                    logger.warning("Q-value computation failed", error=str(e))
                    continue

        if not q_value_ranges:
            logger.error("No valid Q-values obtained")
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        results = {
            "mean": float(np.mean(q_value_ranges)),
            "std": float(np.std(q_value_ranges)),
            "min": float(np.min(q_value_ranges)),
            "max": float(np.max(q_value_ranges))
        }

        logger.info("Q-value distribution analysis completed", **results)
        return results
    
    def evaluate_clinical_appropriateness(self) -> Dict[str, Any]:
        """Evaluate clinical appropriateness across different risk levels.

        Tests whether the model makes clinically appropriate decisions based on
        patient risk levels. Assesses risk stratification, medication restraint
        for low-risk patients, and appropriate action for high-risk patients.

        Returns:
            Dictionary containing appropriateness metrics for each risk level
            and overall clinical assessment

        Raises:
            RuntimeError: If clinical evaluation fails
        """
        logger.info("Evaluating clinical appropriateness")

        self.model.eval()
        results = {}

        try:
            for risk_level in self.config.risk_levels:
                scenario_results = []
                medication_count = 0
                scenarios_per_risk = self.config.num_clinical_scenarios // len(self.config.risk_levels)

                for _ in range(scenarios_per_risk):
                    try:
                        # Generate patient with specific risk level
                        patient = self._generate_patient_with_risk(risk_level)
                        features = patient.to_feature_vector()

                        # Get model prediction
                        action = self.model.predict_temporal_action(features, use_exploration=False)

                        # Count medication decisions
                        if action.healthcare_action == HealthcareAction.MEDICATE:
                            medication_count += 1

                        # Score clinical appropriateness
                        score = self._score_clinical_appropriateness(patient, action)
                        scenario_results.append(score)

                    except Exception as e:
                        logger.warning("Clinical scenario evaluation failed", error=str(e))
                        continue

                if scenario_results:
                    medication_rate = medication_count / len(scenario_results)
                    results[f'risk_{risk_level}'] = {
                        'appropriateness_score': float(np.mean(scenario_results)),
                        'medication_rate': float(medication_rate),
                        'num_scenarios': len(scenario_results)
                    }
                else:
                    logger.warning("No valid scenarios for risk level", risk_level=risk_level)
                    results[f'risk_{risk_level}'] = {
                        'appropriateness_score': 0.0,
                        'medication_rate': 0.0,
                        'num_scenarios': 0
                    }

            # Overall clinical assessment
            if len(results) >= 2:
                low_risk_key = f'risk_{min(self.config.risk_levels)}'
                high_risk_key = f'risk_{max(self.config.risk_levels)}'

                results['overall'] = {
                    'risk_stratification': (
                        results[high_risk_key]['medication_rate'] >
                        results[low_risk_key]['medication_rate']
                    ),
                    'low_risk_restraint': results[low_risk_key]['medication_rate'] < 0.4,
                    'high_risk_action': results[high_risk_key]['medication_rate'] > 0.3
                }
            else:
                results['overall'] = {
                    'risk_stratification': False,
                    'low_risk_restraint': False,
                    'high_risk_action': False
                }

            logger.info("Clinical appropriateness evaluation completed")
            return results

        except Exception as e:
            logger.error("Clinical appropriateness evaluation failed", error=str(e))
            raise RuntimeError(f"Clinical evaluation failed: {e}") from e
    
    def evaluate_exploration_behavior(self, test_states: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate exploration vs exploitation behavior.

        Tests the model's ability to explore different actions during training mode
        versus exploiting learned policies during evaluation mode. Higher exploration
        diversity indicates better exploration capabilities.

        Args:
            test_states: List of state feature vectors for testing

        Returns:
            Dictionary containing exploration and exploitation diversity metrics

        Raises:
            ValueError: If test_states is empty
        """
        if not test_states:
            raise ValueError("Test states cannot be empty")

        logger.info("Evaluating exploration behavior")

        # Test exploration mode
        self.model.train()
        exploration_actions = []
        for _ in range(self.config.num_exploration_tests):
            try:
                state_idx = np.random.randint(0, len(test_states))
                state = test_states[state_idx].astype(np.float32)
                action = self.model.predict_temporal_action(state, use_exploration=True, epsilon=0.1)
                exploration_actions.append(action.to_string())
            except Exception as e:
                logger.warning("Exploration prediction failed", error=str(e))
                continue

        # Test exploitation mode
        self.model.eval()
        exploitation_actions = []
        for _ in range(self.config.num_exploration_tests):
            try:
                state_idx = np.random.randint(0, len(test_states))
                state = test_states[state_idx].astype(np.float32)
                action = self.model.predict_temporal_action(state, use_exploration=False)
                exploitation_actions.append(action.to_string())
            except Exception as e:
                logger.warning("Exploitation prediction failed", error=str(e))
                continue

        # Calculate diversity metrics
        exploration_diversity = (
            len(set(exploration_actions)) / len(exploration_actions)
            if exploration_actions else 0.0
        )
        exploitation_diversity = (
            len(set(exploitation_actions)) / len(exploitation_actions)
            if exploitation_actions else 0.0
        )

        results = {
            'exploration_diversity': float(exploration_diversity),
            'exploitation_diversity': float(exploitation_diversity),
            'unique_exploration_actions': len(set(exploration_actions)),
            'unique_exploitation_actions': len(set(exploitation_actions))
        }

        logger.info("Exploration behavior evaluation completed", **results)
        return results
    
    def _extract_test_states(self, test_data: pd.DataFrame, num_states: int) -> List[np.ndarray]:
        """Extract test states from test data.

        Args:
            test_data: DataFrame containing test samples
            num_states: Number of states to extract

        Returns:
            List of state feature vectors
        """
        test_states = []
        sample_size = min(num_states, len(test_data))

        for _ in range(sample_size):
            row = test_data.sample(1).iloc[0]
            state_features = row[EnhancedPatientState.get_feature_names()].values.astype(np.float32)
            test_states.append(state_features)

        return test_states

    def _generate_patient_with_risk(self, risk_level: float) -> EnhancedPatientState:
        """Generate a synthetic patient with specified risk level.

        Creates a realistic patient state with clinical parameters correlated
        to the specified risk level for testing clinical appropriateness.

        Args:
            risk_level: Target risk level (0.0 to 1.0)

        Returns:
            EnhancedPatientState with specified risk characteristics
        """
        # Age correlated with risk
        age = int(30 + risk_level * 50 + np.random.normal(0, 10))
        age = max(18, min(95, age))

        # Comorbidities increase with risk
        comorbidities = max(0, int(risk_level * 5 + np.random.normal(0, 1)))

        # Vital signs correlated with risk
        systolic_bp = 120 + risk_level * 60 + np.random.normal(0, 15)
        diastolic_bp = 80 + risk_level * 30 + np.random.normal(0, 10)
        heart_rate = 70 + risk_level * 40 + np.random.normal(0, 15)

        return EnhancedPatientState(
            risk_score=risk_level,
            age=age,
            comorbidities=comorbidities,
            bmi=25 + np.random.normal(0, 5),
            systolic_bp=max(90, min(200, systolic_bp)),
            diastolic_bp=max(60, min(120, diastolic_bp)),
            heart_rate=max(50, min(150, heart_rate)),
            current_hour=np.random.randint(8, 20),
            day_of_week=np.random.randint(0, 7),
            urgency_level=risk_level + np.random.normal(0, 0.1),
            last_contact_hours_ago=np.random.exponential(48),
            next_appointment_days=np.random.exponential(14),
            preferred_channel=np.random.choice(list(CommunicationChannel)),
            preferred_time=np.random.choice(list(TimeOfDay)),
            sms_success_rate=np.clip(0.8 + np.random.normal(0, 0.1), 0.0, 1.0),
            email_success_rate=np.clip(0.7 + np.random.normal(0, 0.1), 0.0, 1.0),
            phone_success_rate=np.clip(0.9 + np.random.normal(0, 0.1), 0.0, 1.0),
            mail_success_rate=np.clip(0.6 + np.random.normal(0, 0.1), 0.0, 1.0),
            medication_adherence=np.clip(0.8 + np.random.normal(0, 0.1), 0.0, 1.0),
            appointment_compliance=np.clip(0.8 + np.random.normal(0, 0.1), 0.0, 1.0),
            response_time_preference=np.random.uniform(0, 1)
        )

    def _score_clinical_appropriateness(
        self,
        patient: EnhancedPatientState,
        action: TemporalAction
    ) -> float:
        """Score the clinical appropriateness of an action for a patient.

        Evaluates whether the recommended action is clinically appropriate
        based on patient risk level, urgency, and preferences.

        Args:
            patient: Patient state information
            action: Recommended temporal action

        Returns:
            Appropriateness score between 0.0 and 1.0
        """
        score = 0.0

        # Risk-action alignment (50% of score)
        if patient.risk_score < 0.3:  # Low risk
            if action.healthcare_action in [HealthcareAction.MONITOR, HealthcareAction.DISCHARGE]:
                score += 0.5
        elif patient.risk_score > 0.7:  # High risk
            if action.healthcare_action in [HealthcareAction.MEDICATE, HealthcareAction.REFER]:
                score += 0.5
        else:  # Medium risk
            if action.healthcare_action in [HealthcareAction.MONITOR, HealthcareAction.MEDICATE]:
                score += 0.5

        # Time horizon appropriateness (30% of score)
        if patient.urgency_level > 0.7 and action.time_horizon in [TimeHorizon.FOUR_HOURS, TimeHorizon.ONE_DAY]:
            score += 0.3
        elif patient.urgency_level < 0.3 and action.time_horizon in [TimeHorizon.ONE_WEEK, TimeHorizon.ONE_MONTH]:
            score += 0.3

        # Communication preference alignment (20% of score)
        if action.communication_channel == patient.preferred_channel:
            score += 0.2

        return min(1.0, score)

    def run_comprehensive_evaluation(self, test_data: Optional[pd.DataFrame] = None) -> EvaluationMetrics:
        """Run comprehensive evaluation of the model.

        Performs all evaluation tests and computes an overall assessment
        of the model's performance across multiple dimensions.

        Args:
            test_data: Optional test data. Generated if not provided.

        Returns:
            EvaluationMetrics containing comprehensive evaluation results

        Raises:
            RuntimeError: If evaluation fails
        """
        start_time = time.time()

        try:
            logger.info("Starting comprehensive model evaluation")

            # Generate test data if not provided
            if test_data is None:
                logger.info("Generating test data", samples=self.config.num_test_samples)
                test_data = generate_temporal_training_data(self.config.num_test_samples)

            # Extract test states for exploration evaluation
            test_states = self._extract_test_states(test_data, num_states=20)

            # Run all evaluation components
            logger.info("Running action consistency evaluation")
            consistency_results = self.evaluate_action_consistency(test_data)

            logger.info("Running clinical appropriateness evaluation")
            clinical_results = self.evaluate_clinical_appropriateness()

            logger.info("Running exploration behavior evaluation")
            exploration_results = self.evaluate_exploration_behavior(test_states)

            # Compute overall metrics
            action_consistency = consistency_results['mean']
            clinical_appropriateness = np.mean([
                r['appropriateness_score'] for r in clinical_results.values()
                if isinstance(r, dict) and 'appropriateness_score' in r
            ])
            exploration_diversity = exploration_results['exploration_diversity']

            # Calculate weighted overall score
            overall_score = (
                action_consistency * self.config.consistency_weight +
                clinical_appropriateness * self.config.clinical_weight +
                exploration_diversity * self.config.exploration_weight
            )

            # Assign grade and status
            if overall_score >= 0.8:
                grade, status = "A", "EXCELLENT"
            elif overall_score >= 0.7:
                grade, status = "B", "GOOD"
            elif overall_score >= 0.6:
                grade, status = "C", "ACCEPTABLE"
            else:
                grade, status = "D", "NEEDS IMPROVEMENT"

            evaluation_time = time.time() - start_time

            # Create results object
            results = EvaluationMetrics(
                action_consistency=action_consistency,
                clinical_appropriateness=clinical_appropriateness,
                exploration_diversity=exploration_diversity,
                overall_score=overall_score,
                grade=grade,
                status=status,
                evaluation_time=evaluation_time
            )

            logger.info(
                "Comprehensive evaluation completed",
                overall_score=overall_score,
                grade=grade,
                status=status,
                evaluation_time_seconds=evaluation_time
            )

            return results

        except Exception as e:
            logger.error("Comprehensive evaluation failed", error=str(e))
            raise RuntimeError(f"Evaluation failed: {e}") from e


class EvaluationDemoRunner:
    """Runner class for evaluation demonstrations.

    Handles model loading, test data generation, and orchestrates
    the evaluation process with proper error handling and logging.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        """Initialize the evaluation demo runner.

        Args:
            config: Optional evaluation configuration
        """
        self.config = config or EvaluationConfig()

    def load_model(self, model_path: Optional[str] = None) -> TemporalFactoredDQN:
        """Load a model for evaluation.

        Args:
            model_path: Optional path to trained model checkpoint

        Returns:
            Loaded or newly created model

        Raises:
            FileNotFoundError: If model_path is provided but file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            if model_path:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                logger.info("Loading trained model", path=model_path)
                checkpoint = torch.load(model_path, map_location='cpu')

                model = TemporalFactoredDQN(
                    input_dim=27,
                    use_dueling=True,
                    use_noisy=True,
                    use_distributional=True
                )

                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                logger.info("Trained model loaded successfully")
                return model
            else:
                # Try to find latest model automatically
                import glob
                model_files = glob.glob("models/enhanced_tfrdqn_*.pt") + glob.glob("enhanced_tfrdqn_*.pt")

                if model_files:
                    # Get most recent model
                    latest_model = max(model_files, key=os.path.getctime)
                    logger.info(f"Auto-detected latest model: {latest_model}")
                    return self.load_model(latest_model)
                else:
                    logger.info("No trained models found, creating untrained model for demonstration")
                    model = TemporalFactoredDQN(
                        input_dim=27,
                        use_dueling=True,
                        use_noisy=True,
                        use_distributional=True
                    )
                    logger.warning("Using UNTRAINED model for demo purposes")
                    return model

        except Exception as e:
            logger.error("Model loading failed", error=str(e))
            raise RuntimeError(f"Failed to load model: {e}") from e


def run_evaluation_demo(model_path: Optional[str] = None) -> bool:
    """Run comprehensive evaluation demonstration.

    Demonstrates the evaluation framework capabilities with either a trained
    model or an untrained model for testing purposes.

    Args:
        model_path: Optional path to trained model checkpoint

    Returns:
        True if evaluation completed successfully, False otherwise
    """
    logger.info("COMPREHENSIVE EVALUATION FRAMEWORK DEMONSTRATION")
    logger.info("=" * 70)

    try:
        # Initialize demo runner
        demo_runner = EvaluationDemoRunner()

        # Load model
        model = demo_runner.load_model(model_path)

        # Create evaluation framework
        evaluator = ModelEvaluationFramework(model)

        # Run comprehensive evaluation
        logger.info("Running comprehensive evaluation")
        results = evaluator.run_comprehensive_evaluation()

        # Display results
        logger.info("=" * 70)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("=" * 70)

        logger.info("Model Performance Metrics:")
        logger.info("  Action Consistency", score=f"{results.action_consistency:.3f}")
        logger.info("  Clinical Appropriateness", score=f"{results.clinical_appropriateness:.3f}")
        logger.info("  Exploration Diversity", score=f"{results.exploration_diversity:.3f}")

        logger.info("Overall Assessment:")
        logger.info("  Overall Score", score=f"{results.overall_score:.3f}")
        logger.info("  Grade", grade=results.grade)
        logger.info("  Status", status=results.status)
        logger.info("  Evaluation Time", minutes=f"{results.evaluation_time/60:.1f}")

        logger.info("=" * 70)
        logger.info("EVALUATION DEMONSTRATION COMPLETED SUCCESSFULLY")

        return True

    except Exception as e:
        logger.error("Evaluation demonstration failed", error=str(e))
        import traceback
        traceback.print_exc()
        return False


def main() -> bool:
    """Main entry point for evaluation demonstration.

    Parses command line arguments and runs the evaluation demonstration
    with appropriate error handling and exit codes.

    Returns:
        True if evaluation completed successfully, False otherwise
    """
    try:
        model_path = None

        # Parse command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "--model" and len(sys.argv) > 2:
                model_path = sys.argv[2]
                logger.info("Model path specified", path=model_path)
            elif sys.argv[1] in ["-h", "--help"]:
                print(__doc__)
                return True
            else:
                logger.warning("Unknown argument", arg=sys.argv[1])
                print("Usage: python evaluation_demo.py [--model <path>]")
                return False

        # Run evaluation demonstration
        success = run_evaluation_demo(model_path)

        if success:
            logger.info("Evaluation demonstration completed successfully")
        else:
            logger.error("Evaluation demonstration failed")

        return success

    except KeyboardInterrupt:
        logger.info("Evaluation demonstration interrupted by user")
        return False
    except Exception as e:
        logger.error("Unexpected error in main", error=str(e))
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
