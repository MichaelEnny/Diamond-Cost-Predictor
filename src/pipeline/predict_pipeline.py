"""
Prediction Pipeline for Diamond Price Predictor
Handles inference and real-time predictions
"""
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from typing import Union, Dict, Any, List

# Add src to path for imports
sys.path.append(str(os.path.dirname(os.path.dirname(__file__))))
from utils.common import load_object


@dataclass
class PredictionPipelineConfig:
    """Configuration for prediction pipeline"""
    model_path: str = os.path.join('artifacts', 'model.pkl')
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class CustomData:
    """
    Data class for handling input diamond data
    """
    def __init__(
        self,
        carat: float,
        cut: str,
        color: str,
        clarity: str,
        depth: float,
        table: float,
        x: float,
        y: float,
        z: float
    ):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
    
    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Convert input data to DataFrame format with categorical encoding
        """
        try:
            # Define categorical mappings to match training data
            cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
            color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
            clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7, 'FL': 8}
            
            custom_data_input_dict = {
                "carat": [self.carat],
                "cut": [cut_mapping.get(self.cut, 2)],  # Default to 'Very Good' if unknown
                "color": [color_mapping.get(self.color, 3)],  # Default to 'G' if unknown
                "clarity": [clarity_mapping.get(self.clarity, 3)],  # Default to 'VS2' if unknown
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise Exception(f"Error creating DataFrame: {str(e)}")
    
    def validate_input(self) -> Dict[str, Any]:
        """
        Validate input data against business rules
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Validate numerical ranges
            if not (0.1 <= self.carat <= 10.0):
                validation_result['warnings'].append(f"Carat {self.carat} outside typical range (0.1-10.0)")
            
            if not (40.0 <= self.depth <= 80.0):
                validation_result['warnings'].append(f"Depth {self.depth} outside typical range (40.0-80.0)")
            
            if not (40.0 <= self.table <= 80.0):
                validation_result['warnings'].append(f"Table {self.table} outside typical range (40.0-80.0)")
            
            # Validate dimensions
            if self.x <= 0 or self.y <= 0 or self.z <= 0:
                validation_result['errors'].append("Diamond dimensions (x, y, z) must be positive")
                validation_result['is_valid'] = False
            
            if not (0.1 <= self.x <= 15.0):
                validation_result['warnings'].append(f"X dimension {self.x} outside typical range (0.1-15.0)")
            
            if not (0.1 <= self.y <= 15.0):
                validation_result['warnings'].append(f"Y dimension {self.y} outside typical range (0.1-15.0)")
            
            if not (0.1 <= self.z <= 10.0):
                validation_result['warnings'].append(f"Z dimension {self.z} outside typical range (0.1-10.0)")
            
            # Validate categorical values
            valid_cuts = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            if self.cut not in valid_cuts:
                validation_result['errors'].append(f"Cut '{self.cut}' not in valid options: {valid_cuts}")
                validation_result['is_valid'] = False
            
            valid_colors = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
            if self.color not in valid_colors:
                validation_result['errors'].append(f"Color '{self.color}' not in valid options: {valid_colors}")
                validation_result['is_valid'] = False
            
            valid_clarities = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']
            if self.clarity not in valid_clarities:
                validation_result['errors'].append(f"Clarity '{self.clarity}' not in valid options: {valid_clarities}")
                validation_result['is_valid'] = False
            
            # Validate logical relationships
            if self.x > 0 and self.y > 0 and abs(self.x - self.y) > 2:
                validation_result['warnings'].append("Large difference between x and y dimensions")
            
            if self.z > 0 and self.x > 0 and self.z / self.x > 0.8:
                validation_result['warnings'].append("Unusually high z/x ratio - very thick diamond")
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            validation_result['is_valid'] = False
        
        return validation_result
    
    def to_dict(self) -> Dict[str, Union[float, str]]:
        """Convert to dictionary format"""
        return {
            'carat': self.carat,
            'cut': self.cut,
            'color': self.color,
            'clarity': self.clarity,
            'depth': self.depth,
            'table': self.table,
            'x': self.x,
            'y': self.y,
            'z': self.z
        }


class PredictPipeline:
    """
    Main prediction pipeline class
    """
    def __init__(self):
        self.config = PredictionPipelineConfig()
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessor artifacts"""
        try:
            # Load model
            if os.path.exists(self.config.model_path):
                self.model = load_object(self.config.model_path)
                logging.info(f"Model loaded from {self.config.model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {self.config.model_path}")
            
            # Load preprocessor
            if os.path.exists(self.config.preprocessor_path):
                self.preprocessor = load_object(self.config.preprocessor_path)
                logging.info(f"Preprocessor loaded from {self.config.preprocessor_path}")
            else:
                raise FileNotFoundError(f"Preprocessor not found at {self.config.preprocessor_path}")
                
        except Exception as e:
            raise Exception(f"Error loading artifacts: {str(e)}")
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction for diamond price
        
        Args:
            features: DataFrame with diamond features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Record start time for performance tracking
            import time
            start_time = time.time()
            
            # Preprocess features
            scaled_features = self.preprocessor.transform(features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Prepare result
            result = {
                'predicted_price': float(prediction[0]),
                'inference_time_ms': round(inference_time, 2),
                'model_info': {
                    'model_type': type(self.model).__name__,
                    'feature_count': scaled_features.shape[1]
                },
                'confidence_info': self._get_prediction_confidence(prediction[0])
            }
            
            logging.info(f"Prediction completed: ${result['predicted_price']:.2f} in {inference_time:.2f}ms")
            
            return result
            
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def predict_batch(self, features_list: List[pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Make batch predictions for multiple diamonds
        
        Args:
            features_list: List of DataFrames with diamond features
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            total_start_time = time.time()
            
            for i, features in enumerate(features_list):
                try:
                    result = self.predict(features)
                    result['batch_index'] = i
                    results.append(result)
                except Exception as e:
                    # Handle individual prediction failures
                    results.append({
                        'batch_index': i,
                        'error': str(e),
                        'predicted_price': None,
                        'inference_time_ms': None
                    })
            
            total_time = (time.time() - total_start_time) * 1000
            
            # Add batch summary
            successful_predictions = len([r for r in results if 'error' not in r])
            batch_summary = {
                'total_predictions': len(features_list),
                'successful_predictions': successful_predictions,
                'failed_predictions': len(features_list) - successful_predictions,
                'total_batch_time_ms': round(total_time, 2),
                'average_time_per_prediction_ms': round(total_time / len(features_list), 2)
            }
            
            return {
                'predictions': results,
                'batch_summary': batch_summary
            }
            
        except Exception as e:
            raise Exception(f"Error during batch prediction: {str(e)}")
    
    def _get_prediction_confidence(self, predicted_price: float) -> Dict[str, Any]:
        """
        Calculate prediction confidence metrics
        
        Args:
            predicted_price: Predicted diamond price
            
        Returns:
            Dictionary with confidence information
        """
        try:
            # Ensure predicted_price is a Python float for JSON serialization
            predicted_price = float(predicted_price)
            
            # Simple confidence scoring based on price ranges
            # This would be enhanced with actual model uncertainty quantification
            
            if predicted_price < 1000:
                confidence_level = "High"
                confidence_score = 0.9
                reasoning = "Low-price diamonds have consistent patterns"
            elif predicted_price < 5000:
                confidence_level = "Medium"
                confidence_score = 0.8
                reasoning = "Mid-range diamonds with moderate variability"
            elif predicted_price < 15000:
                confidence_level = "Medium"
                confidence_score = 0.75
                reasoning = "High-value diamonds with increased variability"
            else:
                confidence_level = "Low"
                confidence_score = 0.6
                reasoning = "Luxury diamonds with high market variability"
            
            # Price range estimation (±15% typical model uncertainty)
            uncertainty_percentage = 15
            price_range = {
                'lower_bound': float(predicted_price * (1 - uncertainty_percentage / 100)),
                'upper_bound': float(predicted_price * (1 + uncertainty_percentage / 100)),
                'uncertainty_percentage': uncertainty_percentage
            }
            
            return {
                'confidence_level': confidence_level,
                'confidence_score': confidence_score,
                'reasoning': reasoning,
                'price_range': price_range
            }
            
        except Exception as e:
            return {
                'confidence_level': 'Unknown',
                'confidence_score': 0.5,
                'reasoning': f'Error calculating confidence: {str(e)}',
                'price_range': None
            }
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from the model (if available)
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                # For tree-based models
                importance_values = self.model.feature_importances_
                
                # Create feature names (this should be enhanced to use actual feature names)
                feature_names = [f'feature_{i}' for i in range(len(importance_values))]
                
                importance_dict = {
                    name: float(importance) 
                    for name, importance in zip(feature_names, importance_values)
                }
                
                # Sort by importance
                sorted_importance = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
                
                return {
                    'available': True,
                    'feature_importance': sorted_importance,
                    'top_features': list(sorted_importance.keys())[:5]
                }
            
            elif hasattr(self.model, 'coef_'):
                # For linear models
                coef_values = self.model.coef_
                feature_names = [f'feature_{i}' for i in range(len(coef_values))]
                
                importance_dict = {
                    name: float(abs(coef)) 
                    for name, coef in zip(feature_names, coef_values)
                }
                
                sorted_importance = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
                
                return {
                    'available': True,
                    'feature_importance': sorted_importance,
                    'top_features': list(sorted_importance.keys())[:5]
                }
            
            else:
                return {
                    'available': False,
                    'message': f'Feature importance not available for {type(self.model).__name__}'
                }
                
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check pipeline health status
        """
        try:
            status = {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'preprocessor_loaded': self.preprocessor is not None,
                'model_type': type(self.model).__name__ if self.model else None,
                'artifacts': {
                    'model_path': self.config.model_path,
                    'model_exists': os.path.exists(self.config.model_path),
                    'preprocessor_path': self.config.preprocessor_path,
                    'preprocessor_exists': os.path.exists(self.config.preprocessor_path)
                }
            }
            
            # Test prediction with dummy data
            try:
                dummy_data = CustomData(
                    carat=1.0, cut='Ideal', color='E', clarity='VS1',
                    depth=61.5, table=55.0, x=6.0, y=6.0, z=3.7
                )
                dummy_df = dummy_data.get_data_as_data_frame()
                test_prediction = self.predict(dummy_df)
                status['test_prediction'] = {
                    'successful': True,
                    'predicted_price': test_prediction['predicted_price'],
                    'inference_time_ms': test_prediction['inference_time_ms']
                }
            except Exception as e:
                status['test_prediction'] = {
                    'successful': False,
                    'error': str(e)
                }
                status['status'] = 'degraded'
            
            return status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# Utility functions for API integration
def create_prediction_from_dict(data: Dict[str, Any]) -> CustomData:
    """
    Create CustomData object from dictionary
    
    Args:
        data: Dictionary with diamond features
        
    Returns:
        CustomData object
    """
    try:
        return CustomData(
            carat=float(data['carat']),
            cut=str(data['cut']),
            color=str(data['color']),
            clarity=str(data['clarity']),
            depth=float(data['depth']),
            table=float(data['table']),
            x=float(data['x']),
            y=float(data['y']),
            z=float(data['z'])
        )
    except Exception as e:
        raise Exception(f"Error creating prediction data: {str(e)}")


def validate_api_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate API input data
    
    Args:
        data: Dictionary with diamond features
        
    Returns:
        Validation result dictionary
    """
    try:
        # Check required fields
        required_fields = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {
                'is_valid': False,
                'errors': [f'Missing required fields: {missing_fields}']
            }
        
        # Create CustomData and validate
        custom_data = create_prediction_from_dict(data)
        return custom_data.validate_input()
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f'Validation error: {str(e)}']
        }


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize prediction pipeline
        print("Initializing prediction pipeline...")
        pipeline = PredictPipeline()
        
        # Health check
        print("\nPerforming health check...")
        health_status = pipeline.health_check()
        print(f"Pipeline status: {health_status['status']}")
        
        # Test single prediction
        print("\nTesting single prediction...")
        test_diamond = CustomData(
            carat=1.0,
            cut='Ideal',
            color='E',
            clarity='VS1',
            depth=61.5,
            table=55.0,
            x=6.0,
            y=6.0,
            z=3.7
        )
        
        # Validate input
        validation = test_diamond.validate_input()
        if validation['is_valid']:
            print("✅ Input validation passed")
        else:
            print(f"❌ Input validation failed: {validation['errors']}")
        
        # Make prediction
        test_df = test_diamond.get_data_as_data_frame()
        result = pipeline.predict(test_df)
        
        print(f"Predicted Price: ${result['predicted_price']:,.2f}")
        print(f"Inference Time: {result['inference_time_ms']} ms")
        print(f"Confidence: {result['confidence_info']['confidence_level']} ({result['confidence_info']['confidence_score']:.2f})")
        
        # Test feature importance
        print("\nTesting feature importance...")
        importance = pipeline.get_feature_importance()
        if importance['available']:
            print("Top 3 features:")
            for i, feature in enumerate(importance['top_features'][:3]):
                print(f"  {i+1}. {feature}: {importance['feature_importance'][feature]:.4f}")
        
        print("\n✅ Prediction pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()