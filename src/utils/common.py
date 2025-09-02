"""
Common utility functions for Diamond Price Predictor
"""

import os
import pickle
import logging
from typing import Any


def load_object(file_path: str) -> Any:
    """Load object from pickle file"""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Exception(f"Error loading object from {file_path}: {e}")


def save_object(file_path: str, obj: Any) -> None:
    """Save object as pickle file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise Exception(f"Error saving object to {file_path}: {e}")


def setup_logging() -> logging.Logger:
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)