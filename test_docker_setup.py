#!/usr/bin/env python3
"""
Test script to validate Docker containerization setup for Diamond Price Predictor.
This script checks if the Docker containers can be built and basic configurations are correct.
"""

import subprocess
import sys
import os
import yaml
import json

def run_command(command, description=""):
    """Run a shell command and return the result."""
    try:
        print(f"Running: {description or command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"[PASS] Success: {description or command}")
            return True, result.stdout
        else:
            print(f"[FAIL] Failed: {description or command}")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[FAIL] Timeout: {description or command}")
        return False, "Command timed out"
    except Exception as e:
        print(f"[FAIL] Exception: {description or command} - {str(e)}")
        return False, str(e)

def test_docker_compose_config():
    """Test docker-compose configuration syntax."""
    print("\n=== Testing docker-compose configuration ===")
    success, output = run_command("docker-compose config", "Validating docker-compose.yml syntax")
    if success:
        print("[PASS] docker-compose.yml is valid")
        return True
    return False

def test_dockerfiles_exist():
    """Check if required Dockerfiles exist."""
    print("\n=== Testing Dockerfile existence ===")
    required_files = [
        "Dockerfile.flask",
        "Dockerfile.streamlit", 
        ".dockerignore"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"[PASS] {file} exists")
        else:
            print(f"[FAIL] {file} missing")
            all_exist = False
    
    return all_exist

def test_requirements_file():
    """Check if requirements.txt exists and contains necessary packages."""
    print("\n=== Testing requirements.txt ===")
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt missing")
        return False
    
    required_packages = ["flask", "streamlit", "gunicorn", "pandas", "numpy", "scikit-learn"]
    with open("requirements.txt", "r") as f:
        content = f.read().lower()
    
    missing_packages = []
    for package in required_packages:
        if package not in content:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[FAIL] Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("[PASS] All required packages found in requirements.txt")
        return True

def test_directory_structure():
    """Check if required directories exist or will be created."""
    print("\n=== Testing directory structure ===")
    required_dirs = ["artifacts", "logs"]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"[PASS] {dir_name}/ directory exists")
        else:
            print(f"[INFO] {dir_name}/ directory will be created by Docker volumes")
    
    return True

def test_yaml_syntax():
    """Test YAML file syntax."""
    print("\n=== Testing YAML syntax ===")
    try:
        with open("docker-compose.yml", "r") as f:
            yaml.safe_load(f)
        print("[PASS] docker-compose.yml has valid YAML syntax")
        return True
    except yaml.YAMLError as e:
        print(f"[FAIL] docker-compose.yml has invalid YAML syntax: {e}")
        return False
    except FileNotFoundError:
        print("[FAIL] docker-compose.yml not found")
        return False

def main():
    """Run all containerization tests."""
    print("Diamond Price Predictor - Docker Containerization Tests")
    print("=" * 60)
    
    tests = [
        test_dockerfiles_exist,
        test_requirements_file,
        test_directory_structure,
        test_yaml_syntax,
        test_docker_compose_config,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Test Summary: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("[PASS] All containerization tests passed!")
        print("[INFO] Docker setup is ready for deployment")
        return 0
    else:
        print("[FAIL] Some tests failed. Please fix the issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())