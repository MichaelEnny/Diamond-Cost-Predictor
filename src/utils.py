"""
Utility functions for Diamond Price Predictor System
"""

import os
import sys
import yaml
import pickle
import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np


class ConfigManager:
    """Configuration management utility"""
    
    def __init__(self, config_path: str = "params.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get_data_ingestion_config(self) -> Dict[str, Any]:
        """Get data ingestion configuration"""
        return self.config.get('data_ingestion', {})
    
    def get_model_trainer_config(self) -> Dict[str, Any]:
        """Get model trainer configuration"""
        return self.config.get('model_trainer', {})
    
    def get_data_transformation_config(self) -> Dict[str, Any]:
        """Get data transformation configuration"""
        return self.config.get('data_transformation', {})
    
    def get_model_evaluation_config(self) -> Dict[str, Any]:
        """Get model evaluation configuration"""
        return self.config.get('model_evaluation', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration"""
        return self.config.get('mlflow', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})


def setup_logging(config_manager: ConfigManager = None) -> logging.Logger:
    """Setup logging configuration"""
    if config_manager is None:
        config_manager = ConfigManager()
    
    logging_config = config_manager.get_logging_config()
    
    # Create logs directory if it doesn't exist
    log_file = logging_config.get('file', 'logs/diamond_predictor.log')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, logging_config.get('level', 'INFO')),
        format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def save_object(file_path: str, obj: Any) -> None:
    """Save object as pickle file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise Exception(f"Error saving object to {file_path}: {e}")


def load_object(file_path: str) -> Any:
    """Load object from pickle file"""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Exception(f"Error loading object from {file_path}: {e}")


def create_directories(dirs: list) -> None:
    """Create directories if they don't exist"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def get_file_size(file_path: str) -> str:
    """Get human readable file size"""
    size_bytes = os.path.getsize(file_path)
    if size_bytes == 0:
        return "0B"
    
    size_name = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_dataframe_schema(df: pd.DataFrame, required_columns: list) -> tuple[bool, list]:
        """Validate DataFrame has required columns"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        return len(missing_columns) == 0, missing_columns
    
    @staticmethod
    def validate_data_types(df: pd.DataFrame, expected_types: dict) -> tuple[bool, dict]:
        """Validate DataFrame column data types"""
        type_issues = {}
        
        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if actual_type != expected_type:
                    type_issues[column] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
        
        return len(type_issues) == 0, type_issues
    
    @staticmethod
    def validate_data_ranges(df: pd.DataFrame, validation_ranges: dict) -> tuple[bool, dict]:
        """Validate data falls within expected ranges"""
        range_issues = {}
        
        for column, (min_val, max_val) in validation_ranges.items():
            if column in df.columns and df[column].dtype in ['int64', 'float64']:
                actual_min = df[column].min()
                actual_max = df[column].max()
                
                if actual_min < min_val or actual_max > max_val:
                    range_issues[column] = {
                        'expected_range': [min_val, max_val],
                        'actual_range': [actual_min, actual_max]
                    }
        
        return len(range_issues) == 0, range_issues
    
    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> dict:
        """Generate comprehensive data quality report"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'column_info': {}
        }
        
        for column in df.columns:
            col_info = {
                'dtype': str(df[column].dtype),
                'non_null_count': df[column].count(),
                'null_percentage': (df[column].isnull().sum() / len(df)) * 100
            }
            
            if df[column].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'mean': df[column].mean(),
                    'std': df[column].std()
                })
            elif df[column].dtype == 'object':
                col_info.update({
                    'unique_values': df[column].nunique(),
                    'most_frequent': df[column].mode().iloc[0] if not df[column].mode().empty else None
                })
            
            report['column_info'][column] = col_info
        
        return report