#!/usr/bin/env python3
"""
Simple file verification script for ModelTrainer implementation.
"""

import os
from pathlib import Path


def check_file(filepath):
    """Check if a file exists and return its status and size."""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        size_kb = size / 1024
        return True, size_kb
    else:
        return False, 0


def main():
    print("=" * 70)
    print("MODELTRAINER IMPLEMENTATION - FILE VERIFICATION")
    print("=" * 70)
    
    files_to_check = [
        ("src/components/model_trainer.py", "Main ModelTrainer class"),
        ("src/config/configuration.py", "Configuration management"),
        ("src/config/__init__.py", "Config package init"),
        ("params.yaml", "Enhanced hyperparameter configuration"),
        ("src/utils.py", "Utility functions"),
        ("src/exception.py", "Custom exception handling"),
        ("src/logger.py", "Logging configuration"),
        ("src/__init__.py", "Main package init"),
        ("src/components/__init__.py", "Components package init"),
        ("demo_model_trainer.py", "Comprehensive demo script"),
        ("simple_test.py", "Basic ModelTrainer test"),
        ("advanced_test.py", "Advanced test with optimization"),
        ("requirements.txt", "ML dependencies"),
    ]
    
    print("\nCORE IMPLEMENTATION FILES:")
    print("-" * 50)
    
    total_files = 0
    existing_files = 0
    total_size = 0
    
    for filepath, description in files_to_check:
        exists, size_kb = check_file(filepath)
        total_files += 1
        
        if exists:
            existing_files += 1
            total_size += size_kb
            status = f"YES ({size_kb:.1f} KB)"
        else:
            status = "MISSING"
            
        print(f"{filepath:<40} {status}")
        print(f"{''.ljust(40)} -> {description}")
        print()
    
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    print(f"Total Files Expected: {total_files}")
    print(f"Files Found: {existing_files}")
    print(f"Files Missing: {total_files - existing_files}")
    print(f"Total Size: {total_size:.1f} KB")
    
    if existing_files == total_files:
        print("Implementation Status: COMPLETE")
    else:
        print("Implementation Status: INCOMPLETE")
    
    # Check ModelTrainer functionality
    print(f"\nFUNCTIONALITY VERIFICATION:")
    print("-" * 50)
    
    model_trainer_path = "src/components/model_trainer.py"
    if os.path.exists(model_trainer_path):
        with open(model_trainer_path, 'r', encoding='utf-8') as f:
            content = f.read()
            checks = [
                ("class ModelTrainer", "ModelTrainer class defined"),
                ("initiate_model_training", "Main training method"),
                ("hyperparameter_tuning", "Hyperparameter optimization"),
                ("evaluate_models", "Model evaluation"),
                ("GridSearchCV", "Grid search implementation"),
                ("XGBRegressor", "XGBoost integration"),
                ("cross_val_score", "Cross-validation"),
            ]
            
            for check, desc in checks:
                if check in content:
                    print(f"  YES: {desc}")
                else:
                    print(f"  NO:  {desc}")
    else:
        print("  NO: ModelTrainer file not found")
    
    # Check params.yaml
    params_path = "params.yaml"
    if os.path.exists(params_path):
        print(f"\nCONFIGURATION VERIFICATION:")
        print("-" * 50)
        with open(params_path, 'r', encoding='utf-8') as f:
            content = f.read()
            config_checks = [
                ("model_trainer:", "ModelTrainer configuration section"),
                ("target_accuracy: 0.95", "95% accuracy target"),
                ("xgboost:", "XGBoost parameters"),
                ("cv_folds:", "Cross-validation config"),
            ]
            
            for check, desc in config_checks:
                if check in content:
                    print(f"  YES: {desc}")
                else:
                    print(f"  NO:  {desc}")
    
    # Test scripts
    print(f"\nTEST SCRIPT VERIFICATION:")
    print("-" * 50)
    
    test_scripts = ["simple_test.py", "advanced_test.py", "demo_model_trainer.py"]
    for script in test_scripts:
        if os.path.exists(script):
            print(f"  YES: {script} (Ready for execution)")
        else:
            print(f"  NO:  {script} (Missing)")
    
    print(f"\nREADY TO RUN:")
    print("-" * 50)
    print("  # Test basic functionality:")
    print("  python simple_test.py")
    print()
    print("  # Test advanced features:")
    print("  python advanced_test.py")
    
    return existing_files == total_files


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\nSUCCESS: All files verified!")
    else:
        print(f"\nWARNING: Some files are missing!")