"""
Configuration classes for Diamond Price Predictor ML components.
"""

import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for model training with hyperparameter optimization."""
    
    # Core configuration
    root_dir: str = "artifacts"
    trained_model_file_path: str = "artifacts/model.pkl"
    model_name: str = "xgboost"
    test_size: float = 0.2
    random_state: int = 42
    
    # Performance requirements
    target_accuracy: float = 0.95  # 95%+ R² score
    max_training_time_minutes: int = 10
    cv_folds: int = 5
    
    # Hyperparameter optimization
    optimization_method: str = "grid_search"
    n_jobs: int = -1
    verbose: int = 1
    
    # Model evaluation
    metrics: List[str] = None
    primary_metric: str = "r2_score"
    
    # XGBoost hyperparameters
    xgboost_params: Dict[str, Any] = None
    random_forest_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.metrics is None:
            object.__setattr__(self, 'metrics', [
                "mean_absolute_error",
                "mean_squared_error", 
                "root_mean_squared_error",
                "r2_score",
                "mean_absolute_percentage_error"
            ])


class ConfigurationManager:
    """Manages configuration loading from params.yaml file."""
    
    def __init__(self, config_filepath: str = "params.yaml"):
        """Initialize configuration manager.
        
        Args:
            config_filepath: Path to the params.yaml configuration file
        """
        self.config_filepath = Path(config_filepath)
        
        if not self.config_filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_filepath}")
            
        with open(self.config_filepath) as f:
            self.config = yaml.safe_load(f)
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Get model trainer configuration from params.yaml.
        
        Returns:
            ModelTrainerConfig: Configuration object for model training
        """
        config = self.config["model_trainer"]
        
        # Create artifacts directory
        os.makedirs("artifacts", exist_ok=True)
        
        return ModelTrainerConfig(
            root_dir="artifacts",
            trained_model_file_path="artifacts/model.pkl",
            model_name=config["model_name"],
            test_size=config["test_size"],
            random_state=config["random_state"],
            target_accuracy=config["target_accuracy"],
            max_training_time_minutes=config["max_training_time_minutes"],
            cv_folds=config["cv_folds"],
            optimization_method=config["optimization_method"],
            n_jobs=config["n_jobs"],
            verbose=config["verbose"],
            metrics=config["metrics"],
            primary_metric=config["primary_metric"],
            xgboost_params=config["xgboost"],
            random_forest_params=config["random_forest"]
        )