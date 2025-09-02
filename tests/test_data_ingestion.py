"""
Comprehensive unit tests for DataIngestion component

Tests cover:
- Configuration loading and validation
- Data downloading from URLs and local files
- Data extraction and CSV parsing
- Schema validation and data quality checks  
- Error handling and edge cases
- MLflow integration (when available)
"""

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'components'))

from components.data_ingestion import DataIngestion, DataIngestionException, DataIngestionConfig
from utils import ConfigManager, DataValidator


class TestDataIngestion:
    """Test suite for DataIngestion component"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_diamond_data(self):
        """Create sample diamond dataset for testing"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'carat': np.random.uniform(0.2, 5.0, n_samples),
            'cut': np.random.choice(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], n_samples),
            'color': np.random.choice(['D', 'E', 'F', 'G', 'H', 'I', 'J'], n_samples),
            'clarity': np.random.choice(['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2'], n_samples),
            'depth': np.random.uniform(43.0, 79.0, n_samples),
            'table': np.random.uniform(43.0, 95.0, n_samples),
            'price': np.random.randint(326, 18823, n_samples),
            'x': np.random.uniform(0.0, 10.0, n_samples),
            'y': np.random.uniform(0.0, 10.0, n_samples),
            'z': np.random.uniform(0.0, 6.0, n_samples)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def test_config_file(self, temp_dir):
        """Create test configuration file"""
        config_path = os.path.join(temp_dir, "test_params.yaml")
        config_content = """
data_ingestion:
  source_url: "https://example.com/diamonds.csv"
  local_data_file: "data/raw/diamonds.csv"
  raw_data_path: "data/raw"
  train_test_split_ratio: 0.2
  random_state: 42
  
  required_columns:
    - "carat"
    - "cut" 
    - "color"
    - "clarity"
    - "depth"
    - "table"
    - "price"
    - "x"
    - "y"
    - "z"
  
  column_types:
    carat: "float64"
    cut: "object"
    color: "object"
    clarity: "object"
    depth: "float64"
    table: "float64"
    price: "int64"
    x: "float64"
    y: "float64"
    z: "float64"
  
  validation_ranges:
    carat: [0.2, 5.01]
    depth: [43.0, 79.0]
    table: [43.0, 95.0]
    price: [326, 18823]
    x: [0.0, 10.74]
    y: [0.0, 58.9]
    z: [0.0, 31.8]

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/test.log"
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path
    
    @pytest.fixture
    def sample_csv_file(self, temp_dir, sample_diamond_data):
        """Create sample CSV file for testing"""
        csv_path = os.path.join(temp_dir, "test_diamonds.csv")
        sample_diamond_data.to_csv(csv_path, index=False)
        return csv_path
    
    def test_data_ingestion_initialization_success(self, test_config_file):
        """Test successful DataIngestion initialization"""
        data_ingestion = DataIngestion(test_config_file)
        
        assert data_ingestion.config_manager is not None
        assert data_ingestion.logger is not None
        assert data_ingestion.config is not None
        assert data_ingestion.validator is not None
        assert isinstance(data_ingestion.config, DataIngestionConfig)
    
    def test_data_ingestion_initialization_invalid_config(self, temp_dir):
        """Test DataIngestion initialization with invalid config"""
        invalid_config = os.path.join(temp_dir, "invalid.yaml")
        with open(invalid_config, 'w') as f:
            f.write("invalid yaml content: [")
        
        with pytest.raises(DataIngestionException):
            DataIngestion(invalid_config)
    
    def test_data_ingestion_initialization_missing_config(self):
        """Test DataIngestion initialization with missing config file"""
        with pytest.raises(DataIngestionException):
            DataIngestion("nonexistent_config.yaml")
    
    def test_load_data_ingestion_config_success(self, test_config_file):
        """Test successful configuration loading"""
        data_ingestion = DataIngestion(test_config_file)
        config = data_ingestion.config
        
        assert config.source_url == "https://example.com/diamonds.csv"
        assert config.local_data_file == "data/raw/diamonds.csv"
        assert config.raw_data_path == "data/raw"
        assert config.train_test_split_ratio == 0.2
        assert config.random_state == 42
        assert len(config.required_columns) == 10
        assert "carat" in config.required_columns
        assert "price" in config.required_columns
    
    def test_load_data_ingestion_config_missing_keys(self, temp_dir):
        """Test configuration loading with missing required keys"""
        config_path = os.path.join(temp_dir, "incomplete_config.yaml")
        with open(config_path, 'w') as f:
            f.write("""
data_ingestion:
  source_url: "https://example.com/diamonds.csv"
  # Missing other required keys
logging:
  level: "INFO"
""")
        
        with pytest.raises(DataIngestionException, match="Missing required config keys"):
            DataIngestion(config_path)
    
    def test_calculate_file_checksum(self, test_config_file, sample_csv_file):
        """Test file checksum calculation"""
        data_ingestion = DataIngestion(test_config_file)
        checksum = data_ingestion._calculate_file_checksum(sample_csv_file)
        
        assert checksum is not None
        assert len(checksum) == 32  # MD5 hash length
        assert isinstance(checksum, str)
        
        # Test same file produces same checksum
        checksum2 = data_ingestion._calculate_file_checksum(sample_csv_file)
        assert checksum == checksum2
    
    def test_calculate_file_checksum_nonexistent_file(self, test_config_file):
        """Test checksum calculation for non-existent file"""
        data_ingestion = DataIngestion(test_config_file)
        checksum = data_ingestion._calculate_file_checksum("nonexistent_file.csv")
        
        assert checksum == ""  # Should return empty string on error
    
    def test_is_url_valid_urls(self, test_config_file):
        """Test URL validation for valid URLs"""
        data_ingestion = DataIngestion(test_config_file)
        
        assert data_ingestion._is_url("https://example.com/data.csv") == True
        assert data_ingestion._is_url("http://example.com/data.csv") == True
        assert data_ingestion._is_url("ftp://example.com/data.csv") == True
    
    def test_is_url_invalid_urls(self, test_config_file):
        """Test URL validation for invalid URLs"""
        data_ingestion = DataIngestion(test_config_file)
        
        assert data_ingestion._is_url("/local/path/data.csv") == False
        assert data_ingestion._is_url("data.csv") == False
        assert data_ingestion._is_url("") == False
        assert data_ingestion._is_url("invalid-url") == False
    
    def test_download_data_existing_file_no_redownload(self, test_config_file, sample_csv_file, temp_dir):
        """Test download when file exists and uses existing file"""
        # Modify config to use temp directory
        data_ingestion = DataIngestion(test_config_file)
        data_ingestion.config.local_data_file = sample_csv_file
        
        result_path = data_ingestion.download_data()
        assert result_path == sample_csv_file
        assert os.path.exists(result_path)
    
    def test_download_data_local_file_success(self, test_config_file, sample_csv_file, temp_dir):
        """Test successful download from local file"""
        target_path = os.path.join(temp_dir, "target_diamonds.csv")
        
        data_ingestion = DataIngestion(test_config_file)
        data_ingestion.config.source_url = sample_csv_file  # Use local file as source
        data_ingestion.config.local_data_file = target_path
        
        result_path = data_ingestion.download_data()
        
        assert result_path == target_path
        assert os.path.exists(target_path)
        
        # Verify content is same
        original_df = pd.read_csv(sample_csv_file)
        copied_df = pd.read_csv(target_path)
        pd.testing.assert_frame_equal(original_df, copied_df)
    
    def test_download_data_local_file_not_found(self, test_config_file, temp_dir):
        """Test download failure when local source file doesn't exist"""
        data_ingestion = DataIngestion(test_config_file)
        data_ingestion.config.source_url = "/nonexistent/file.csv"
        data_ingestion.config.local_data_file = os.path.join(temp_dir, "target.csv")
        
        with pytest.raises(DataIngestionException, match="Data download failed"):
            data_ingestion.download_data()
    
    def test_extract_data_success(self, test_config_file, sample_csv_file):
        """Test successful data extraction"""
        data_ingestion = DataIngestion(test_config_file)
        data_ingestion.config.local_data_file = sample_csv_file
        
        df = data_ingestion.extract_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100  # Sample data has 100 rows
        assert len(df.columns) == 10  # Should have 10 columns
        assert 'carat' in df.columns
        assert 'price' in df.columns
    
    def test_extract_data_file_not_found(self, test_config_file):
        """Test data extraction failure when file doesn't exist"""
        data_ingestion = DataIngestion(test_config_file)
        data_ingestion.config.local_data_file = "nonexistent_file.csv"
        
        with pytest.raises(DataIngestionException, match="Data file not found"):
            data_ingestion.extract_data()
    
    def test_extract_data_empty_file(self, test_config_file, temp_dir):
        """Test data extraction failure with empty file"""
        empty_file = os.path.join(temp_dir, "empty.csv")
        open(empty_file, 'w').close()  # Create empty file
        
        data_ingestion = DataIngestion(test_config_file)
        data_ingestion.config.local_data_file = empty_file
        
        with pytest.raises(DataIngestionException, match="Data file is empty"):
            data_ingestion.extract_data()
    
    def test_validate_data_schema_success(self, test_config_file, sample_diamond_data):
        """Test successful schema validation"""
        data_ingestion = DataIngestion(test_config_file)
        
        result = data_ingestion.validate_data_schema(sample_diamond_data)
        
        assert result == True
    
    def test_validate_data_schema_missing_columns(self, test_config_file, sample_diamond_data):
        """Test schema validation failure due to missing columns"""
        data_ingestion = DataIngestion(test_config_file)
        
        # Remove required column
        df_missing = sample_diamond_data.drop(columns=['carat'])
        
        with pytest.raises(DataIngestionException, match="Missing required columns"):
            data_ingestion.validate_data_schema(df_missing)
    
    def test_validate_data_schema_invalid_ranges(self, test_config_file, sample_diamond_data):
        """Test schema validation failure due to invalid data ranges"""
        data_ingestion = DataIngestion(test_config_file)
        
        # Create data with invalid ranges
        df_invalid = sample_diamond_data.copy()
        df_invalid.loc[0, 'price'] = -1000  # Invalid price
        
        with pytest.raises(DataIngestionException, match="Data range validation failed"):
            data_ingestion.validate_data_schema(df_invalid)
    
    def test_validate_data_schema_with_duplicates(self, test_config_file, sample_diamond_data):
        """Test schema validation with duplicate rows (should pass with warning)"""
        data_ingestion = DataIngestion(test_config_file)
        
        # Add duplicate rows
        df_with_duplicates = pd.concat([sample_diamond_data, sample_diamond_data.iloc[:5]], ignore_index=True)
        
        # Should still pass validation but log warnings
        result = data_ingestion.validate_data_schema(df_with_duplicates)
        assert result == True
    
    def test_validate_data_schema_with_missing_values(self, test_config_file, sample_diamond_data):
        """Test schema validation with missing values (should pass with warning)"""
        data_ingestion = DataIngestion(test_config_file)
        
        # Add some missing values
        df_with_na = sample_diamond_data.copy()
        df_with_na.loc[0:4, 'depth'] = np.nan
        
        # Should still pass validation but log warnings
        result = data_ingestion.validate_data_schema(df_with_na)
        assert result == True
    
    @patch('components.data_ingestion.MLFLOW_AVAILABLE', False)
    def test_initiate_data_ingestion_without_mlflow(self, test_config_file, sample_csv_file):
        """Test complete data ingestion pipeline without MLflow"""
        data_ingestion = DataIngestion(test_config_file)
        data_ingestion.config.local_data_file = sample_csv_file
        data_ingestion.config.source_url = sample_csv_file  # Use local file
        
        df, info = data_ingestion.initiate_data_ingestion()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert isinstance(info, dict)
        assert info['validation_passed'] == True
        assert 'timestamp' in info
        assert 'data_shape' in info
        assert 'quality_report' in info
    
    def test_initiate_data_ingestion_download_failure(self, test_config_file):
        """Test data ingestion pipeline failure during download"""
        data_ingestion = DataIngestion(test_config_file)
        data_ingestion.config.source_url = "/nonexistent/file.csv"
        
        with pytest.raises(DataIngestionException, match="Data ingestion pipeline failed"):
            data_ingestion.initiate_data_ingestion()
    
    def test_data_validator_validate_dataframe_schema(self):
        """Test DataValidator schema validation"""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Test success case
        valid, missing = validator.validate_dataframe_schema(df, ['col1', 'col2'])
        assert valid == True
        assert missing == []
        
        # Test failure case
        valid, missing = validator.validate_dataframe_schema(df, ['col1', 'col2', 'col3'])
        assert valid == False
        assert 'col3' in missing
    
    def test_data_validator_validate_data_types(self):
        """Test DataValidator data type validation"""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })
        
        expected_types = {
            'int_col': 'int64',
            'float_col': 'float64',
            'str_col': 'object'
        }
        
        valid, issues = validator.validate_data_types(df, expected_types)
        assert valid == True
        assert issues == {}
        
        # Test type mismatch
        wrong_types = {
            'int_col': 'float64',  # Wrong type
            'float_col': 'float64',
            'str_col': 'object'
        }
        
        valid, issues = validator.validate_data_types(df, wrong_types)
        assert valid == False
        assert 'int_col' in issues
    
    def test_data_validator_validate_data_ranges(self):
        """Test DataValidator data range validation"""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0],
            'col2': [10, 20, 30]
        })
        
        # Test valid ranges
        valid_ranges = {
            'col1': [0.5, 3.5],
            'col2': [5, 35]
        }
        
        valid, issues = validator.validate_data_ranges(df, valid_ranges)
        assert valid == True
        assert issues == {}
        
        # Test invalid ranges
        invalid_ranges = {
            'col1': [1.5, 2.5],  # Too restrictive
            'col2': [5, 35]
        }
        
        valid, issues = validator.validate_data_ranges(df, invalid_ranges)
        assert valid == False
        assert 'col1' in issues
    
    def test_data_validator_get_data_quality_report(self, sample_diamond_data):
        """Test DataValidator quality report generation"""
        validator = DataValidator()
        
        report = validator.get_data_quality_report(sample_diamond_data)
        
        assert isinstance(report, dict)
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'missing_values' in report
        assert 'duplicate_rows' in report
        assert 'memory_usage' in report
        assert 'column_info' in report
        
        assert report['total_rows'] == 100
        assert report['total_columns'] == 10
        assert isinstance(report['column_info'], dict)
        
        # Check column info details
        for col in sample_diamond_data.columns:
            assert col in report['column_info']
            col_info = report['column_info'][col]
            assert 'dtype' in col_info
            assert 'non_null_count' in col_info
            assert 'null_percentage' in col_info


class TestDataIngestionIntegration:
    """Integration tests for DataIngestion component"""
    
    def test_end_to_end_pipeline_with_real_data(self, temp_dir):
        """Test complete pipeline with realistic diamond data"""
        # Create realistic diamond dataset
        np.random.seed(42)
        n_samples = 50
        
        realistic_data = {
            'carat': np.random.gamma(2, 0.5, n_samples),  # More realistic carat distribution
            'cut': np.random.choice(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], 
                                  n_samples, p=[0.1, 0.15, 0.25, 0.3, 0.2]),
            'color': np.random.choice(['D', 'E', 'F', 'G', 'H', 'I', 'J'], 
                                    n_samples, p=[0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05]),
            'clarity': np.random.choice(['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2'], 
                                      n_samples, p=[0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.2, 0.1]),
            'depth': np.random.normal(61.5, 2.5, n_samples),
            'table': np.random.normal(57.0, 3.0, n_samples),
            'x': np.random.uniform(3.0, 9.0, n_samples),
            'y': np.random.uniform(3.0, 9.0, n_samples),
            'z': np.random.uniform(2.0, 6.0, n_samples),
        }
        
        # Create realistic price based on features
        carat_factor = realistic_data['carat'] ** 1.8
        cut_factor = {'Fair': 0.8, 'Good': 0.9, 'Very Good': 1.0, 'Premium': 1.1, 'Ideal': 1.2}
        color_factor = {'D': 1.2, 'E': 1.15, 'F': 1.1, 'G': 1.05, 'H': 1.0, 'I': 0.95, 'J': 0.9}
        
        realistic_data['price'] = (
            carat_factor * 
            [cut_factor[cut] for cut in realistic_data['cut']] *
            [color_factor[color] for color in realistic_data['color']] *
            np.random.normal(3500, 500, n_samples)
        ).astype(int)
        
        # Ensure price is within valid range
        realistic_data['price'] = np.clip(realistic_data['price'], 326, 18823)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(realistic_data)
        csv_path = os.path.join(temp_dir, "realistic_diamonds.csv")
        df.to_csv(csv_path, index=False)
        
        # Create config file
        config_path = os.path.join(temp_dir, "config.yaml")
        config_content = f"""
data_ingestion:
  source_url: "{csv_path}"
  local_data_file: "{os.path.join(temp_dir, 'loaded_diamonds.csv')}"
  raw_data_path: "{temp_dir}"
  train_test_split_ratio: 0.2
  random_state: 42
  
  required_columns: ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]
  
  column_types:
    carat: "float64"
    cut: "object"
    color: "object" 
    clarity: "object"
    depth: "float64"
    table: "float64"
    price: "int64"
    x: "float64"
    y: "float64"
    z: "float64"
  
  validation_ranges:
    carat: [0.2, 5.01]
    depth: [43.0, 79.0]
    table: [43.0, 95.0]
    price: [326, 18823]
    x: [0.0, 10.74]
    y: [0.0, 58.9]
    z: [0.0, 31.8]

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "{os.path.join(temp_dir, 'test.log')}"
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Run complete pipeline
        data_ingestion = DataIngestion(config_path)
        result_df, info = data_ingestion.initiate_data_ingestion()
        
        # Verify results
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 50
        assert len(result_df.columns) == 10
        assert info['validation_passed'] == True
        assert 'quality_report' in info
        
        # Verify data quality
        quality_report = info['quality_report']
        assert quality_report['total_rows'] == 50
        assert quality_report['total_columns'] == 10
        
        # Verify statistical properties are reasonable
        assert result_df['price'].min() >= 326
        assert result_df['price'].max() <= 18823
        assert result_df['carat'].min() > 0
        assert result_df['depth'].std() > 0  # Should have variation


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])