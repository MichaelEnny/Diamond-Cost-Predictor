"""
Flask API for Diamond Price Predictor
Production-ready REST API with comprehensive endpoints
"""
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import logging

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.predict_pipeline import PredictPipeline, CustomData, create_prediction_from_dict, validate_api_input
from utils.common import setup_logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global prediction pipeline (loaded once at startup)
prediction_pipeline = None

# API configuration
API_VERSION = "v1"
API_TITLE = "Diamond Price Predictor API"
API_DESCRIPTION = "Production ML API for diamond price predictions using XGBoost"

# Performance tracking
request_count = 0
total_response_time = 0
error_count = 0


def initialize_pipeline():
    """Initialize the prediction pipeline at startup"""
    global prediction_pipeline
    try:
        logger.info("Initializing prediction pipeline...")
        prediction_pipeline = PredictPipeline()
        
        # Test pipeline
        health_status = prediction_pipeline.health_check()
        if health_status['status'] == 'healthy':
            logger.info("✅ Prediction pipeline initialized successfully")
            return True
        else:
            logger.error("❌ Pipeline health check failed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize prediction pipeline: {str(e)}")
        return False


def track_performance(response_time_ms: float, is_error: bool = False):
    """Track API performance metrics"""
    global request_count, total_response_time, error_count
    
    request_count += 1
    total_response_time += response_time_ms
    
    if is_error:
        error_count += 1


def create_error_response(message: str, status_code: int = 400, details: str = None) -> tuple:
    """Create standardized error response"""
    response = {
        'success': False,
        'error': {
            'message': message,
            'status_code': status_code,
            'timestamp': datetime.utcnow().isoformat(),
        }
    }
    
    if details:
        response['error']['details'] = details
    
    return jsonify(response), status_code


def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.utcnow().isoformat(),
        'api_version': API_VERSION
    }


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def home():
    """API home page with documentation"""
    return render_template('index.html')


@app.route('/api/v1/info', methods=['GET'])
def api_info():
    """Get API information and available endpoints"""
    try:
        start_time = time.time()
        
        api_info = {
            'title': API_TITLE,
            'description': API_DESCRIPTION,
            'version': API_VERSION,
            'endpoints': {
                'predict': {
                    'url': '/api/v1/predict',
                    'method': 'POST',
                    'description': 'Predict price for a single diamond',
                    'example_payload': {
                        'carat': 1.0,
                        'cut': 'Ideal',
                        'color': 'E',
                        'clarity': 'VS1',
                        'depth': 61.5,
                        'table': 55.0,
                        'x': 6.0,
                        'y': 6.0,
                        'z': 3.7
                    }
                },
                'predict_batch': {
                    'url': '/api/v1/predict/batch',
                    'method': 'POST',
                    'description': 'Predict prices for multiple diamonds',
                    'example_payload': {
                        'diamonds': [
                            # Array of diamond objects like in single predict
                        ]
                    }
                },
                'health': {
                    'url': '/api/v1/health',
                    'method': 'GET',
                    'description': 'Check API and model health status'
                }
            },
            'valid_values': {
                'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']
            },
            'feature_ranges': {
                'carat': [0.1, 10.0],
                'depth': [40.0, 80.0],
                'table': [40.0, 80.0],
                'x': [0.1, 15.0],
                'y': [0.1, 15.0],
                'z': [0.1, 10.0]
            }
        }
        
        response_time = (time.time() - start_time) * 1000
        track_performance(response_time)
        
        return jsonify(create_success_response(api_info, "API information retrieved"))
        
    except Exception as e:
        track_performance(0, is_error=True)
        return create_error_response(f"Error retrieving API info: {str(e)}", 500)


@app.route('/health', methods=['GET'])
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        start_time = time.time()
        
        if prediction_pipeline is None:
            return create_error_response("Prediction pipeline not initialized", 503)
        
        # Get pipeline health status
        pipeline_health = prediction_pipeline.health_check()
        
        # Calculate API performance metrics
        avg_response_time = (total_response_time / request_count) if request_count > 0 else 0
        error_rate = (error_count / request_count * 100) if request_count > 0 else 0
        
        health_status = {
            'api_status': 'healthy' if pipeline_health['status'] == 'healthy' else 'degraded',
            'pipeline_status': pipeline_health['status'],
            'model_loaded': pipeline_health['model_loaded'],
            'preprocessor_loaded': pipeline_health['preprocessor_loaded'],
            'model_type': pipeline_health['model_type'],
            'performance_metrics': {
                'total_requests': request_count,
                'error_count': error_count,
                'error_rate_percentage': round(error_rate, 2),
                'average_response_time_ms': round(avg_response_time, 2)
            },
            'test_prediction': pipeline_health.get('test_prediction', {}),
            'artifacts_status': pipeline_health.get('artifacts', {}),
            'uptime_info': {
                'api_version': API_VERSION,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        response_time = (time.time() - start_time) * 1000
        track_performance(response_time)
        
        # Return appropriate status code
        status_code = 200 if health_status['api_status'] == 'healthy' else 503
        
        return jsonify(create_success_response(health_status, "Health check completed")), status_code
        
    except Exception as e:
        track_performance(0, is_error=True)
        return create_error_response(f"Health check failed: {str(e)}", 500)


@app.route('/api/v1/predict', methods=['POST'])
def predict_diamond_price():
    """Predict price for a single diamond"""
    try:
        start_time = time.time()
        
        if prediction_pipeline is None:
            return create_error_response("Prediction pipeline not available", 503)
        
        # Get and validate request data
        if not request.is_json:
            return create_error_response("Request must be JSON", 400)
        
        data = request.get_json()
        if not data:
            return create_error_response("No data provided", 400)
        
        # Validate input data
        validation_result = validate_api_input(data)
        if not validation_result['is_valid']:
            return create_error_response(
                "Input validation failed", 
                400, 
                validation_result['errors']
            )
        
        # Create prediction data
        diamond_data = create_prediction_from_dict(data)
        diamond_df = diamond_data.get_data_as_data_frame()
        
        # Make prediction
        prediction_result = prediction_pipeline.predict(diamond_df)
        
        # Add validation warnings if any
        if validation_result.get('warnings'):
            prediction_result['input_warnings'] = validation_result['warnings']
        
        # Add request info
        prediction_result['request_info'] = {
            'input_data': diamond_data.to_dict(),
            'api_version': API_VERSION
        }
        
        response_time = (time.time() - start_time) * 1000
        track_performance(response_time)
        
        return jsonify(create_success_response(
            prediction_result, 
            f"Price predicted successfully: ${prediction_result['predicted_price']:,.2f}"
        ))
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        track_performance(response_time, is_error=True)
        return create_error_response(f"Prediction failed: {str(e)}", 500)


@app.route('/api/v1/predict/batch', methods=['POST'])
def predict_batch_diamond_prices():
    """Predict prices for multiple diamonds"""
    try:
        start_time = time.time()
        
        if prediction_pipeline is None:
            return create_error_response("Prediction pipeline not available", 503)
        
        # Get and validate request data
        if not request.is_json:
            return create_error_response("Request must be JSON", 400)
        
        data = request.get_json()
        if not data or 'diamonds' not in data:
            return create_error_response("No diamond data provided. Expected 'diamonds' array", 400)
        
        diamonds_data = data['diamonds']
        if not isinstance(diamonds_data, list):
            return create_error_response("'diamonds' must be an array", 400)
        
        if len(diamonds_data) == 0:
            return create_error_response("No diamonds in request", 400)
        
        if len(diamonds_data) > 100:  # Limit batch size
            return create_error_response("Batch size too large. Maximum 100 diamonds per request", 400)
        
        # Process each diamond
        batch_results = []
        validation_errors = []
        
        for i, diamond_data in enumerate(diamonds_data):
            try:
                # Validate individual diamond
                validation_result = validate_api_input(diamond_data)
                
                if validation_result['is_valid']:
                    # Create prediction data
                    diamond = create_prediction_from_dict(diamond_data)
                    diamond_df = diamond.get_data_as_data_frame()
                    
                    # Make prediction
                    prediction_result = prediction_pipeline.predict(diamond_df)
                    prediction_result['batch_index'] = i
                    prediction_result['input_data'] = diamond.to_dict()
                    
                    if validation_result.get('warnings'):
                        prediction_result['input_warnings'] = validation_result['warnings']
                    
                    batch_results.append(prediction_result)
                    
                else:
                    # Add validation error for this diamond
                    validation_errors.append({
                        'batch_index': i,
                        'errors': validation_result['errors']
                    })
                    
            except Exception as e:
                validation_errors.append({
                    'batch_index': i,
                    'errors': [f"Processing error: {str(e)}"]
                })
        
        # Calculate batch statistics
        successful_predictions = len(batch_results)
        total_predictions = len(diamonds_data)
        failed_predictions = total_predictions - successful_predictions
        
        if successful_predictions > 0:
            total_value = sum(result['predicted_price'] for result in batch_results)
            avg_price = total_value / successful_predictions
            min_price = min(result['predicted_price'] for result in batch_results)
            max_price = max(result['predicted_price'] for result in batch_results)
        else:
            total_value = avg_price = min_price = max_price = 0
        
        response_time = (time.time() - start_time) * 1000
        track_performance(response_time, is_error=(failed_predictions > 0))
        
        # Prepare response
        response_data = {
            'predictions': batch_results,
            'batch_summary': {
                'total_diamonds': total_predictions,
                'successful_predictions': successful_predictions,
                'failed_predictions': failed_predictions,
                'success_rate': round((successful_predictions / total_predictions) * 100, 2),
                'total_processing_time_ms': round(response_time, 2),
                'average_time_per_diamond_ms': round(response_time / total_predictions, 2)
            },
            'price_statistics': {
                'total_portfolio_value': round(total_value, 2),
                'average_price': round(avg_price, 2),
                'min_price': round(min_price, 2),
                'max_price': round(max_price, 2)
            } if successful_predictions > 0 else None,
            'validation_errors': validation_errors if validation_errors else None
        }
        
        message = f"Batch prediction completed: {successful_predictions}/{total_predictions} successful"
        return jsonify(create_success_response(response_data, message))
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        track_performance(response_time, is_error=True)
        return create_error_response(f"Batch prediction failed: {str(e)}", 500)


@app.route('/api/v1/model/info', methods=['GET'])
def get_model_info():
    """Get detailed model information and feature importance"""
    try:
        start_time = time.time()
        
        if prediction_pipeline is None:
            return create_error_response("Prediction pipeline not available", 503)
        
        # Get feature importance
        feature_importance = prediction_pipeline.get_feature_importance()
        
        # Get model configuration
        model_info = {
            'model_type': type(prediction_pipeline.model).__name__,
            'model_parameters': prediction_pipeline.model.get_params() if hasattr(prediction_pipeline.model, 'get_params') else {},
            'feature_importance': feature_importance,
            'training_info': {
                'target_accuracy': 0.95,
                'inference_target_ms': 200,
                'expected_features': 9,
                'model_version': API_VERSION
            },
            'preprocessing_info': {
                'scaler_type': 'StandardScaler',
                'outlier_handling': 'IQR method (3.0 threshold)',
                'feature_engineering': ['volume', 'dimension_ratios'],
                'categorical_encoding': 'Ordinal (domain-specific)'
            }
        }
        
        response_time = (time.time() - start_time) * 1000
        track_performance(response_time)
        
        return jsonify(create_success_response(model_info, "Model information retrieved"))
        
    except Exception as e:
        track_performance(0, is_error=True)
        return create_error_response(f"Error retrieving model info: {str(e)}", 500)


@app.route('/api/v1/validate', methods=['POST'])
def validate_diamond_data():
    """Validate diamond data without making a prediction"""
    try:
        start_time = time.time()
        
        # Get and validate request data
        if not request.is_json:
            return create_error_response("Request must be JSON", 400)
        
        data = request.get_json()
        if not data:
            return create_error_response("No data provided", 400)
        
        # Perform validation
        validation_result = validate_api_input(data)
        
        # Add additional context
        validation_result['validation_timestamp'] = datetime.utcnow().isoformat()
        validation_result['api_version'] = API_VERSION
        
        response_time = (time.time() - start_time) * 1000
        track_performance(response_time)
        
        message = "Validation passed" if validation_result['is_valid'] else "Validation failed"
        status_code = 200 if validation_result['is_valid'] else 400
        
        return jsonify(create_success_response(validation_result, message)), status_code
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        track_performance(response_time, is_error=True)
        return create_error_response(f"Validation failed: {str(e)}", 500)


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return create_error_response("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return create_error_response("Method not allowed", 405)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return create_error_response("Internal server error", 500)


# =============================================================================
# APPLICATION STARTUP
# =============================================================================

def create_app():
    """Application factory pattern"""
    
    # Initialize prediction pipeline
    if not initialize_pipeline():
        logger.error("Failed to initialize prediction pipeline. API may not function correctly.")
    
    return app


if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    # Initialize the application
    app = create_app()
    
    # Get configuration from environment variables
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"""
    {'='*60}
    Diamond Price Predictor API
    {'='*60}
    API Version: {API_VERSION}
    Host: {host}:{port}
    Debug Mode: {debug}
    
    Available Endpoints:
    * GET  /api/v1/info        - API information
    * GET  /api/v1/health      - Health check
    * POST /api/v1/predict     - Single prediction
    * POST /api/v1/predict/batch - Batch prediction
    * POST /api/v1/validate    - Data validation
    * GET  /api/v1/model/info  - Model information
    {'='*60}
    """)
    
    # Run the Flask application
    app.run(host=host, port=port, debug=debug)