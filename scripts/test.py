#!/usr/bin/env python3
"""Test runner for Enhanced Healthcare DQN system.

This script provides convenient commands for running different test suites
and generating coverage reports for the enhanced temporal factored Rainbow DQN.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def run_unit_tests():
    """Run unit tests."""
    cmd = "python -m pytest tests/unit/ -v --tb=short"
    return run_command(cmd, "Running Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = "python -m pytest tests/integration/ -v --tb=short"
    return run_command(cmd, "Running Integration Tests")


def run_all_tests():
    """Run all tests."""
    cmd = "python -m pytest tests/ -v --tb=short"
    return run_command(cmd, "Running All Tests")


def run_fast_tests():
    """Run only fast tests (exclude slow marker)."""
    cmd = "python -m pytest tests/ -v --tb=short -m 'not slow'"
    return run_command(cmd, "Running Fast Tests Only")


def run_coverage_report():
    """Run tests with coverage report."""
    cmd = "python -m pytest tests/ --cov=core --cov-report=html --cov-report=term"
    return run_command(cmd, "Running Tests with Coverage Report")


def check_test_dependencies():
    """Check if required test dependencies are installed."""
    required_packages = ['pytest', 'pytest-cov', 'psutil']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required test dependencies:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False

    print("All test dependencies are installed")
    return True


def run_prediction_test():
    """Test prediction functionality."""
    cmd = "python scripts/predict.py --untrained"
    return run_command(cmd, "Testing Prediction Demo with Untrained Model")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Enhanced Healthcare DQN Test Runner")
    parser.add_argument(
        "command",
        choices=["unit", "integration", "all", "fast", "coverage", "check", "predict"],
        help="Test command to run"
    )
    
    args = parser.parse_args()
    
    print("Enhanced Healthcare DQN Test Runner")
    print("=" * 60)

    # Check dependencies first
    if not check_test_dependencies():
        sys.exit(1)

    # Ensure we're in the right directory
    if not Path("core").exists():
        print("Error: Run this script from the project root directory")
        sys.exit(1)
    
    success = True
    
    if args.command == "unit":
        success = run_unit_tests()
    elif args.command == "integration":
        success = run_integration_tests()
    elif args.command == "all":
        success = run_all_tests()
    elif args.command == "fast":
        success = run_fast_tests()
    elif args.command == "coverage":
        success = run_coverage_report()
    elif args.command == "check":
        # Already checked dependencies above
        pass
    elif args.command == "predict":
        success = run_prediction_test()
    
    if success:
        print("\nTests completed successfully!")

        if args.command == "coverage":
            print("\nCoverage report generated:")
            print("   - HTML report: htmlcov/index.html")
            print("   - Terminal report: shown above")
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
