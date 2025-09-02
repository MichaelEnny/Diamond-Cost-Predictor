"""
Data Ingestion Component for Diamond Price Predictor System

This module handles downloading, validating, and processing the diamond dataset
for the ML pipeline. Supports both local files and remote URL sources.
"""

import os
import sys
import logging
import requests
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from urllib.request import urlretrieve

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import ConfigManager, setup_logging, DataValidator, get_file_size, create_directories

# Try to import MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Experiment tracking will be disabled.")


@dataclass
class DataIngestionConfig:
    """Data ingestion configuration dataclass"""
    source_url: str
    local_data_file: str
    raw_data_path: str
    train_test_split_ratio: float
    random_state: int
    required_columns: list
    column_types: dict
    validation_ranges: dict


class DataIngestionException(Exception):
    """Custom exception for data ingestion errors"""
    pass


class DataIngestion:
    """
    Production-ready Data Ingestion component for Diamond Price Prediction
    
    This class handles:
    - Downloading diamond dataset from remote sources
    - Validating data integrity and schema
    - Data quality checks and reporting
    - DVC-compatible data processing
    - MLflow experiment tracking
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize DataIngestion component
        
        Args:
            config_path (str): Path to configuration YAML file
        """
        try:
            self.config_manager = ConfigManager(config_path)
            self.logger = setup_logging(self.config_manager)
            self.config = self._load_data_ingestion_config()
            self.validator = DataValidator()
            
            # Create necessary directories
            create_directories([
                self.config.raw_data_path,
                "data/processed",
                "logs",
                "artifacts"
            ])
            
            self.logger.info("DataIngestion component initialized successfully")
            
        except Exception as e:
            raise DataIngestionException(f"Failed to initialize DataIngestion: {e}")
    
    def _load_data_ingestion_config(self) -> DataIngestionConfig:
        """Load and validate data ingestion configuration"""
        try:
            config_dict = self.config_manager.get_data_ingestion_config()
            
            # Validate required configuration keys
            required_keys = ['source_url', 'local_data_file', 'raw_data_path', 
                           'required_columns', 'column_types', 'validation_ranges']
            
            missing_keys = [key for key in required_keys if key not in config_dict]
            if missing_keys:
                raise DataIngestionException(f"Missing required config keys: {missing_keys}")
            
            return DataIngestionConfig(
                source_url=config_dict['source_url'],
                local_data_file=config_dict['local_data_file'],
                raw_data_path=config_dict['raw_data_path'],
                train_test_split_ratio=config_dict.get('train_test_split_ratio', 0.2),
                random_state=config_dict.get('random_state', 42),
                required_columns=config_dict['required_columns'],
                column_types=config_dict['column_types'],
                validation_ranges=config_dict['validation_ranges']
            )
            
        except Exception as e:
            raise DataIngestionException(f"Error loading data ingestion config: {e}")
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file for integrity validation"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def download_data(self) -> str:
        """
        Download diamond dataset from configured source
        
        Returns:
            str: Path to downloaded data file
            
        Raises:
            DataIngestionException: If download fails
        """
        try:
            self.logger.info(f"Starting data download from: {self.config.source_url}")
            
            # Check if file already exists
            if os.path.exists(self.config.local_data_file):
                self.logger.info(f"Data file already exists: {self.config.local_data_file}")
                file_size = get_file_size(self.config.local_data_file)
                self.logger.info(f"Existing file size: {file_size}")
                
                # For automation, skip re-download prompt
                self.logger.info("Using existing data file")
                return self.config.local_data_file
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            
            # Download data
            if self._is_url(self.config.source_url):
                self._download_from_url(self.config.source_url, self.config.local_data_file)
            else:
                # Handle local file source
                if os.path.exists(self.config.source_url):
                    import shutil
                    shutil.copy2(self.config.source_url, self.config.local_data_file)
                    self.logger.info(f"Copied local file from {self.config.source_url}")
                else:
                    raise FileNotFoundError(f"Source file not found: {self.config.source_url}")
            
            # Validate download
            if not os.path.exists(self.config.local_data_file):
                raise DataIngestionException("Downloaded file not found")
            
            file_size = get_file_size(self.config.local_data_file)
            checksum = self._calculate_file_checksum(self.config.local_data_file)
            
            self.logger.info(f"Data download completed successfully")
            self.logger.info(f"File: {self.config.local_data_file}")
            self.logger.info(f"Size: {file_size}")
            self.logger.info(f"Checksum: {checksum}")
            
            return self.config.local_data_file
            
        except Exception as e:
            error_msg = f"Data download failed: {e}"
            self.logger.error(error_msg)
            raise DataIngestionException(error_msg)
    
    def _is_url(self, path: str) -> bool:
        """Check if path is a URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _download_from_url(self, url: str, local_path: str) -> None:
        """Download file from URL with progress tracking"""
        try:
            response = requests.head(url, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            self.logger.info(f"Starting download: {total_size} bytes")
            
            # Use urllib for actual download with progress
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 50 == 0:  # Log every 50 blocks to avoid spam
                        self.logger.debug(f"Download progress: {percent}%")
            
            urlretrieve(url, local_path, reporthook=progress_hook)
            
        except requests.exceptions.RequestException as e:
            raise DataIngestionException(f"Network error during download: {e}")
        except Exception as e:
            raise DataIngestionException(f"Download error: {e}")
    
    def extract_data(self) -> pd.DataFrame:
        """
        Extract and load diamond dataset into pandas DataFrame
        
        Returns:
            pd.DataFrame: Loaded diamond dataset
            
        Raises:
            DataIngestionException: If data extraction fails
        """
        try:
            self.logger.info("Starting data extraction")
            
            if not os.path.exists(self.config.local_data_file):
                raise FileNotFoundError(f"Data file not found: {self.config.local_data_file}")
            
            # Read CSV with error handling
            try:
                df = pd.read_csv(self.config.local_data_file)
                self.logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
                
            except pd.errors.EmptyDataError:
                raise DataIngestionException("Data file is empty")
            except pd.errors.ParserError as e:
                raise DataIngestionException(f"CSV parsing error: {e}")
            except Exception as e:
                raise DataIngestionException(f"Error reading CSV file: {e}")
            
            # Log basic data info
            self.logger.info(f"Data shape: {df.shape}")
            self.logger.info(f"Columns: {list(df.columns)}")
            self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return df
            
        except Exception as e:
            error_msg = f"Data extraction failed: {e}"
            self.logger.error(error_msg)
            raise DataIngestionException(error_msg)
    
    def validate_data_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate diamond dataset schema and data quality
        
        Args:
            df (pd.DataFrame): Diamond dataset to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            DataIngestionException: If validation fails
        """
        try:
            self.logger.info("Starting data schema validation")
            validation_passed = True
            issues = []
            
            # 1. Check required columns
            schema_valid, missing_columns = self.validator.validate_dataframe_schema(
                df, self.config.required_columns
            )
            
            if not schema_valid:
                validation_passed = False
                issues.append(f"Missing required columns: {missing_columns}")
                self.logger.error(f"Schema validation failed: missing columns {missing_columns}")
            else:
                self.logger.info("✓ All required columns present")
            
            # 2. Check data types (with some flexibility)
            types_valid, type_issues = self.validator.validate_data_types(
                df, self.config.column_types
            )
            
            if not types_valid:
                self.logger.warning(f"Data type mismatches found: {type_issues}")
                # Don't fail validation for type issues, just log warnings
                for column, issue in type_issues.items():
                    self.logger.warning(f"Column '{column}': expected {issue['expected']}, got {issue['actual']}")
            else:
                self.logger.info("✓ Data types validation passed")
            
            # 3. Check data ranges
            ranges_valid, range_issues = self.validator.validate_data_ranges(
                df, self.config.validation_ranges
            )
            
            if not ranges_valid:
                validation_passed = False
                issues.append(f"Data range validation failed: {range_issues}")
                self.logger.error(f"Range validation failed: {range_issues}")
            else:
                self.logger.info("✓ Data ranges validation passed")
            
            # 4. Check for duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                self.logger.warning(f"Found {duplicate_count} duplicate rows")
                issues.append(f"Found {duplicate_count} duplicate rows")
            else:
                self.logger.info("✓ No duplicate rows found")
            
            # 5. Check for missing values
            missing_summary = df.isnull().sum()
            if missing_summary.sum() > 0:
                self.logger.warning("Missing values found:")
                for col, count in missing_summary[missing_summary > 0].items():
                    percentage = (count / len(df)) * 100
                    self.logger.warning(f"  {col}: {count} ({percentage:.2f}%)")
                    issues.append(f"Missing values in {col}: {count} ({percentage:.2f}%)")
            else:
                self.logger.info("✓ No missing values found")
            
            # Generate comprehensive data quality report
            quality_report = self.validator.get_data_quality_report(df)
            self.logger.info(f"Data quality summary: {len(df)} rows, {len(df.columns)} columns")
            
            if validation_passed:
                self.logger.info("🎉 Data validation completed successfully!")
            else:
                error_msg = f"Data validation failed. Issues: {'; '.join(issues)}"
                self.logger.error(error_msg)
                raise DataIngestionException(error_msg)
            
            return validation_passed
            
        except Exception as e:
            if isinstance(e, DataIngestionException):
                raise
            error_msg = f"Data validation error: {e}"
            self.logger.error(error_msg)
            raise DataIngestionException(error_msg)
    
    def initiate_data_ingestion(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete data ingestion pipeline
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (dataframe, ingestion_info)
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING DIAMOND PRICE PREDICTOR DATA INGESTION")
            self.logger.info("=" * 60)
            
            # Track with MLflow if available
            if MLFLOW_AVAILABLE:
                try:
                    mlflow_config = self.config_manager.get_mlflow_config()
                    mlflow.set_experiment(mlflow_config.get('experiment_name', 'diamond_price_prediction'))
                    
                    with mlflow.start_run(run_name="data_ingestion") as run:
                        return self._run_ingestion_pipeline()
                        
                except Exception as e:
                    self.logger.warning(f"MLflow tracking failed: {e}. Continuing without tracking.")
                    return self._run_ingestion_pipeline()
            else:
                return self._run_ingestion_pipeline()
            
        except Exception as e:
            error_msg = f"Data ingestion pipeline failed: {e}"
            self.logger.error(error_msg)
            raise DataIngestionException(error_msg)
    
    def _run_ingestion_pipeline(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run the core data ingestion pipeline"""
        ingestion_info = {
            'timestamp': pd.Timestamp.now(),
            'source_url': self.config.source_url,
            'local_file': self.config.local_data_file,
            'validation_passed': False
        }
        
        try:
            # Step 1: Download data
            download_path = self.download_data()
            ingestion_info['download_path'] = download_path
            ingestion_info['file_size'] = get_file_size(download_path)
            
            # Step 2: Extract data
            df = self.extract_data()
            ingestion_info['data_shape'] = df.shape
            ingestion_info['columns'] = list(df.columns)
            
            # Step 3: Validate schema
            validation_result = self.validate_data_schema(df)
            ingestion_info['validation_passed'] = validation_result
            
            # Step 4: Generate data quality report
            quality_report = self.validator.get_data_quality_report(df)
            ingestion_info['quality_report'] = quality_report
            
            # Log MLflow metrics if available
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_param("source_url", self.config.source_url)
                mlflow.log_param("data_shape", f"{df.shape[0]}x{df.shape[1]}")
                mlflow.log_metric("total_rows", df.shape[0])
                mlflow.log_metric("total_columns", df.shape[1])
                mlflow.log_metric("missing_values", df.isnull().sum().sum())
                mlflow.log_metric("duplicate_rows", df.duplicated().sum())
                
                # Log data file as artifact
                mlflow.log_artifact(download_path, "raw_data")
            
            self.logger.info("=" * 60)
            self.logger.info("DATA INGESTION COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Dataset shape: {df.shape}")
            self.logger.info(f"File location: {download_path}")
            self.logger.info("=" * 60)
            
            return df, ingestion_info
            
        except Exception as e:
            ingestion_info['error'] = str(e)
            raise DataIngestionException(f"Ingestion pipeline failed: {e}")


# Main execution for testing
if __name__ == "__main__":
    try:
        # Initialize and run data ingestion
        data_ingestion = DataIngestion()
        df, info = data_ingestion.initiate_data_ingestion()
        
        print(f"\n🎉 Data Ingestion Successful!")
        print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Validation passed: {info['validation_passed']}")
        
        # Display first few rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Display basic statistics
        print(f"\nDataset Info:")
        print(df.info())
        
    except Exception as e:
        print(f"❌ Data Ingestion Failed: {e}")
        sys.exit(1)