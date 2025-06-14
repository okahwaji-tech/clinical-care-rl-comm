"""
Test suite for the temporal healthcare DQN system.

This package contains comprehensive tests for all components of the
healthcare communication optimization system.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from core.logging_system import get_logger, configure_global_logging

# Configure test logging
configure_global_logging(log_level="DEBUG")
logger = get_logger(__name__)


class TestConfig:
    """Configuration for test environment."""
    
    # Test data paths
    TEST_DATA_DIR = Path(__file__).parent / "test_data"
    FIXTURES_DIR = Path(__file__).parent / "fixtures"
    
    # Test parameters
    BATCH_SIZE = 4
    INPUT_DIM = 27
    HIDDEN_DIM = 64  # Smaller for faster tests
    NUM_EPISODES = 10  # Short episodes for tests
    
    # Tolerances
    FLOAT_TOLERANCE = 1e-6
    PERFORMANCE_TOLERANCE = 0.1  # 10% tolerance for performance tests
    
    # Test device
    DEVICE = torch.device("cpu")  # Use CPU for consistent test results


@pytest.fixture
def test_config() -> TestConfig:
    """Provide test configuration."""
    return TestConfig()


# Network and training configurations are now handled directly in test files


@pytest.fixture
def sample_patient_state() -> np.ndarray:
    """Provide sample patient state for testing."""
    # Create realistic patient state
    state = np.array([
        0.7,   # risk_score
        65.0,  # age
        2.0,   # comorbidities
        28.5,  # bmi
        140.0, # systolic_bp
        90.0,  # diastolic_bp
        85.0,  # heart_rate
        14.0,  # current_hour
        2.0,   # day_of_week
        0.6,   # urgency_level
        48.0,  # last_contact_hours_ago
        7.0,   # next_appointment_days
        2.0,   # preferred_channel (phone)
        1.0,   # preferred_time (morning)
        0.6,   # sms_success_rate
        0.8,   # email_success_rate
        0.95,  # phone_success_rate
        0.4,   # mail_success_rate
        0.75,  # medication_adherence
        0.85,  # appointment_compliance
        0.8,   # response_time_preference
        # Additional features to reach 27
        0.5, 0.3, 0.7, 0.2, 0.9, 0.1
    ])
    
    assert len(state) == TestConfig.INPUT_DIM, f"Expected {TestConfig.INPUT_DIM} features, got {len(state)}"
    return state


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Provide sample training batch."""
    batch_size = TestConfig.BATCH_SIZE
    input_dim = TestConfig.INPUT_DIM
    
    return {
        "states": torch.randn(batch_size, input_dim),
        "actions": torch.randint(0, 640, (batch_size,)),  # 5*8*4*4 = 640 total actions
        "rewards": torch.randn(batch_size),
        "next_states": torch.randn(batch_size, input_dim),
        "dones": torch.randint(0, 2, (batch_size,)).bool(),
        "weights": torch.ones(batch_size),  # Uniform weights for testing
        "indices": torch.arange(batch_size)
    }


# Service container functionality removed with dependency injection cleanup


class TestBase:
    """Base class for all tests with common utilities."""
    
    def assert_tensor_shape(
        self, 
        tensor: torch.Tensor, 
        expected_shape: tuple,
        message: Optional[str] = None
    ) -> None:
        """Assert tensor has expected shape."""
        actual_shape = tuple(tensor.shape)
        msg = message or f"Expected shape {expected_shape}, got {actual_shape}"
        assert actual_shape == expected_shape, msg
    
    def assert_tensor_finite(
        self, 
        tensor: torch.Tensor,
        message: Optional[str] = None
    ) -> None:
        """Assert tensor contains only finite values."""
        msg = message or "Tensor contains non-finite values"
        assert torch.isfinite(tensor).all(), msg
    
    def assert_tensor_in_range(
        self, 
        tensor: torch.Tensor,
        min_val: float,
        max_val: float,
        message: Optional[str] = None
    ) -> None:
        """Assert tensor values are in expected range."""
        msg = message or f"Tensor values not in range [{min_val}, {max_val}]"
        assert tensor.min() >= min_val and tensor.max() <= max_val, msg
    
    def assert_close(
        self, 
        actual: float, 
        expected: float,
        tolerance: float = TestConfig.FLOAT_TOLERANCE,
        message: Optional[str] = None
    ) -> None:
        """Assert two values are close within tolerance."""
        diff = abs(actual - expected)
        msg = message or f"Values not close: {actual} vs {expected} (diff: {diff})"
        assert diff <= tolerance, msg
    
    def create_mock_network(self, input_dim: int = 27, hidden_dim: int = 64) -> torch.nn.Module:
        """Create a simple mock network for testing."""
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 640)  # Total action combinations
        )
    
    def create_sample_experience(self) -> Dict[str, Any]:
        """Create sample experience for replay buffer testing."""
        return {
            "state": np.random.randn(TestConfig.INPUT_DIM),
            "action": np.random.randint(0, 640),
            "reward": np.random.randn(),
            "next_state": np.random.randn(TestConfig.INPUT_DIM),
            "done": np.random.choice([True, False])
        }


class PerformanceTestMixin:
    """Mixin for performance testing utilities."""
    
    def measure_throughput(
        self, 
        func: callable, 
        iterations: int = 1000,
        warmup: int = 100
    ) -> float:
        """
        Measure function throughput.
        
        Args:
            func: Function to measure
            iterations: Number of iterations
            warmup: Number of warmup iterations
            
        Returns:
            Throughput in operations per second
        """
        import time
        
        # Warmup
        for _ in range(warmup):
            func()
        
        # Measure
        start_time = time.time()
        for _ in range(iterations):
            func()
        end_time = time.time()
        
        duration = end_time - start_time
        return iterations / duration if duration > 0 else float('inf')
    
    def measure_memory_usage(self, func: callable) -> float:
        """
        Measure peak memory usage of function.
        
        Args:
            func: Function to measure
            
        Returns:
            Peak memory usage in MB
        """
        import tracemalloc
        
        tracemalloc.start()
        func()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return peak / 1024 / 1024  # Convert to MB


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")


def pytest_collection_modifyitems(_, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ["integration", "performance", "slow"] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Test utilities for common assertions
def assert_valid_temporal_action(action):
    """Assert that an action is a valid temporal action."""
    from core.temporal_actions import TemporalAction
    
    assert isinstance(action, TemporalAction), f"Expected TemporalAction, got {type(action)}"
    assert hasattr(action, 'healthcare_action'), "Missing healthcare_action"
    assert hasattr(action, 'time_horizon'), "Missing time_horizon"
    assert hasattr(action, 'time_of_day'), "Missing time_of_day"
    assert hasattr(action, 'communication_channel'), "Missing communication_channel"


def assert_valid_q_values(q_values: torch.Tensor, expected_actions: int):
    """Assert that Q-values are valid."""
    assert isinstance(q_values, torch.Tensor), f"Expected torch.Tensor, got {type(q_values)}"
    assert q_values.dim() == 2, f"Expected 2D tensor, got {q_values.dim()}D"
    assert q_values.size(1) == expected_actions, f"Expected {expected_actions} actions, got {q_values.size(1)}"
    assert torch.isfinite(q_values).all(), "Q-values contain non-finite values"


def assert_valid_network_output(output, batch_size: int):
    """Assert that network output is valid."""
    if isinstance(output, torch.Tensor):
        assert output.size(0) == batch_size, f"Expected batch size {batch_size}, got {output.size(0)}"
        assert torch.isfinite(output).all(), "Network output contains non-finite values"
    elif isinstance(output, dict):
        for key, tensor in output.items():
            assert isinstance(tensor, torch.Tensor), f"Output[{key}] is not a tensor"
            assert tensor.size(0) == batch_size, f"Output[{key}] has wrong batch size"
            assert torch.isfinite(tensor).all(), f"Output[{key}] contains non-finite values"
    else:
        raise AssertionError(f"Invalid network output type: {type(output)}")


# Export commonly used test utilities
__all__ = [
    "TestConfig",
    "TestBase", 
    "PerformanceTestMixin",
    "assert_valid_temporal_action",
    "assert_valid_q_values", 
    "assert_valid_network_output"
]
