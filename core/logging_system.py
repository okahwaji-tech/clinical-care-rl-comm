"""Healthcare structured logging system for temporal difference learning.

This module provides a comprehensive, structured logging system replacing print
statements with proper formatting, log levels, and metrics tracking capabilities
for healthcare reinforcement learning applications.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from core.constants import LOGGING_CONFIG


class HealthcareLogger:
    """Structured logger for healthcare DQN system.

    Provides consistent, formatted logging across all components with support
    for log levels, output formats, and optional metrics tracking.
    """

    def __init__(
        self,
        name: str,
        log_level: str = LOGGING_CONFIG["log_level"],
        log_file: Optional[Path] = None,
        enable_metrics: bool = True,
    ) -> None:
        """Initialize the healthcare logger.

        Configures console and optional file handlers, sets log level, and
        initializes metrics tracking.

        Args:
            name (str): Logger name (usually module name).
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR).
            log_file (Optional[Path]): Optional file path to write logs.
            enable_metrics (bool): Whether to enable metrics tracking.
        """
        self.name = name
        self.enable_metrics = enable_metrics
        self.metrics_history: Dict[str, list] = {}

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(LOGGING_CONFIG["log_format"])

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an informational message.

        Args:
            message (str): Message to log.
            **kwargs (Any): Additional key-value pairs to include in the log.
        """
        extra_info = self._format_extra(**kwargs)
        self.logger.info(f"{message}{extra_info}")

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message.

        Args:
            message (str): Message to log.
            **kwargs (Any): Additional key-value pairs to include in the log.
        """
        extra_info = self._format_extra(**kwargs)
        self.logger.warning(f"{message}{extra_info}")

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message.

        Args:
            message (str): Message to log.
            **kwargs (Any): Additional key-value pairs to include in the log.
        """
        extra_info = self._format_extra(**kwargs)
        self.logger.error(f"{message}{extra_info}")

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message.

        Args:
            message (str): Message to log.
            **kwargs (Any): Additional key-value pairs to include in the log.
        """
        extra_info = self._format_extra(**kwargs)
        self.logger.debug(f"{message}{extra_info}")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log training metrics at a given step.

        Stores metrics history internally and logs formatted metrics.

        Args:
            metrics (Dict[str, float]): Mapping of metric names to values.
            step (int): Training step number.
        """
        if not self.enable_metrics:
            return

        # Store metrics history
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(
                {"step": step, "value": value, "timestamp": datetime.now().isoformat()}
            )

        # Log metrics
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.info(f"Step {step} metrics: {metrics_str}")

    def log_network_initialization(
        self, network_name: str, parameters: Dict[str, Any]
    ) -> None:
        """Log details of network initialization.

        Args:
            network_name (str): Name of the initialized network.
            parameters (Dict[str, Any]): Initialization parameters, including 'parameter_count'.
        """
        param_count = parameters.get("parameter_count", "unknown")
        config = {k: v for k, v in parameters.items() if k != "parameter_count"}

        self.info(f"{network_name} initialized", parameter_count=param_count, **config)

    def log_training_progress(
        self, episode: int, total_episodes: int, metrics: Dict[str, float]
    ) -> None:
        """Log training progress with percentage and metrics.

        Args:
            episode (int): Current episode number.
            total_episodes (int): Total number of episodes.
            metrics (Dict[str, float]): Mapping of metric names to values.
        """
        progress = (episode / total_episodes) * 100

        self.info(
            f"Training progress: {progress:.1f}% ({episode}/{total_episodes})",
            **metrics,
        )

    def log_performance_benchmark(
        self, operation: str, throughput: float, unit: str = "ops/sec"
    ) -> None:
        """Log performance benchmark results.

        Args:
            operation (str): Description of the operation benchmarked.
            throughput (float): Measured throughput.
            unit (str): Unit of throughput measurement.
        """
        self.info(f"⚡ Performance: {operation}", throughput=throughput, unit=unit)

    def log_action_distribution(
        self, action_counts: Dict[str, int], total_actions: int
    ) -> None:
        """Log distribution of actions taken.

        Args:
            action_counts (Dict[str, int]): Counts per action label.
            total_actions (int): Total number of actions.
        """
        self.info("📊 Action Distribution:")

        for action, count in action_counts.items():
            percentage = (count / total_actions) * 100
            self.info(f"   {action}: {count} ({percentage:.1f}%)")

    def log_clinical_safety_check(
        self, risk_level: float, recommended_action: str, confidence: float
    ) -> None:
        """Log results of a clinical safety validation.

        Args:
            risk_level (float): Computed risk level (0.0-1.0).
            recommended_action (str): Action recommended by safety check.
            confidence (float): Confidence score for the recommendation.
        """
        safety_level = (
            "HIGH" if risk_level > 0.7 else "MEDIUM" if risk_level > 0.3 else "LOW"
        )

        self.info(
            f"🏥 Clinical Safety Check",
            risk_level=risk_level,
            safety_level=safety_level,
            recommended_action=recommended_action,
            confidence=confidence,
        )

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for tracked metrics.

        Returns:
            Dict[str, Dict[str, float]]: Summary per metric including count, mean, min, max, and latest value.
        """
        summary = {}

        for metric_name, history in self.metrics_history.items():
            if not history:
                continue

            values = [entry["value"] for entry in history]
            summary[metric_name] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else 0.0,
            }

        return summary

    def save_metrics_to_file(self, filepath: Path) -> None:
        """Save stored metrics history to a JSON file.

        Args:
            filepath (Path): File path to write JSON metrics.
        """
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        self.info(f"Metrics saved to {filepath}")

    def _format_extra(self, **kwargs: Any) -> str:
        """Format additional keyword arguments for inclusion in log messages.

        Args:
            **kwargs (Any): Extra key-value pairs to format.

        Returns:
            str: Formatted string of key-value pairs enclosed in brackets.
        """
        if not kwargs:
            return ""

        formatted_items = []
        for key, value in kwargs.items():
            if isinstance(value, float):
                formatted_items.append(f"{key}={value:.4f}")
            else:
                formatted_items.append(f"{key}={value}")

        return f" [{', '.join(formatted_items)}]"


# Global logger factory
_loggers: Dict[str, HealthcareLogger] = {}


def get_logger(
    name: str, log_level: Optional[str] = None, log_file: Optional[Path] = None
) -> HealthcareLogger:
    """Get or create a named HealthcareLogger instance.

    Args:
        name (str): Logger identifier (usually __name__).
        log_level (Optional[str]): Optional override for log level.
        log_file (Optional[Path]): Optional file path for log output.

    Returns:
        HealthcareLogger: Configured logger instance.
    """
    if name not in _loggers:
        _loggers[name] = HealthcareLogger(
            name=name,
            log_level=log_level or LOGGING_CONFIG["log_level"],
            log_file=log_file,
        )

    return _loggers[name]


def configure_global_logging(
    log_level: str = "INFO", log_file: Optional[Path] = None
) -> None:
    """Configure global Python logging settings.

    Updates LOGGING_CONFIG and reinitializes root logger handlers.

    Args:
        log_level (str): Desired global log level.
        log_file (Optional[Path]): Optional global log file path.
    """
    # Update global config
    LOGGING_CONFIG["log_level"] = log_level

    # Clear existing loggers to force recreation with new settings
    _loggers.clear()

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=LOGGING_CONFIG["log_format"],
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if log_file is None else [logging.FileHandler(log_file)]),
        ],
    )
