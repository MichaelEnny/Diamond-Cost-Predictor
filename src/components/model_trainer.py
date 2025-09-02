"""
Model Training Engine for Diamond Price Predictor.

This module implements the ModelTrainer class with XGBoost optimization,
hyperparameter tuning, cross-validation, and comprehensive model evaluation.
Targets 95%+ prediction accuracy with MLflow experiment tracking.
"""

import os
import json
import pickle
import time
import logging
import warnings
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    train_test_split, StratifiedKFold, KFold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import ConfigurationManager, ModelTrainerConfig
from src.utils import save_object, load_object
from src.exception import CustomException
from src.logger import logging as custom_logging

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Model Training Engine with hyperparameter optimization and MLflow tracking.
    
    Features:
    - XGBoost with comprehensive hyperparameter tuning
    - Cross-validation for robust model selection
    - Multiple model comparison (XGBoost, Random Forest, Linear Regression)
    - 95%+ accuracy target validation
    - MLflow experiment tracking
    - Performance visualization and reporting
    """
    
    def __init__(self, config: Optional[ModelTrainerConfig] = None):
        """Initialize ModelTrainer with configuration.
        
        Args:
            config: Model training configuration. If None, loads from params.yaml
        """
        self.config = config or ConfigurationManager().get_model_trainer_config()
        self.logger = logging.getLogger(__name__)
        self.best_model = None
        self.best_score = 0.0
        self.best_params = {}
        self.training_results = {}
        
        # Create artifacts directory
        os.makedirs(self.config.root_dir, exist_ok=True)
        
        # Initialize MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("diamond_price_prediction")
            self.logger.info("MLflow experiment tracking initialized")
        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {e}. Continuing without MLflow.")
    
    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray) -> Dict[str, Any]:
        """
        Main method to initiate model training with hyperparameter optimization.
        
        Args:
            train_array: Training data array (features + target)
            test_array: Test data array (features + target)
            
        Returns:
            dict: Training results with model performance metrics
        """
        try:
            start_time = time.time()
            self.logger.info("Starting model training pipeline...")
            
            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            self.logger.info(f"Training data shape: {X_train.shape}")
            self.logger.info(f"Test data shape: {X_test.shape}")
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"model_training_{int(time.time())}"):
                
                # Log dataset information
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("test_samples", X_test.shape[0])
                mlflow.log_param("n_features", X_train.shape[1])
                
                # Train and evaluate models
                model_results = self.evaluate_models(X_train, y_train, X_test, y_test)
                
                # Select best model based on primary metric
                best_model_name = max(model_results.keys(), 
                                    key=lambda k: model_results[k][self.config.primary_metric])
                
                self.best_model = model_results[best_model_name]['model']
                self.best_score = model_results[best_model_name][self.config.primary_metric]
                
                # Perform hyperparameter tuning on best model
                if best_model_name == "XGBoost":
                    self.logger.info("Performing hyperparameter optimization on XGBoost...")
                    best_params = self.hyperparameter_tuning(X_train, y_train)
                    
                    # Retrain with best parameters
                    final_model = xgb.XGBRegressor(**best_params)
                    final_model.fit(X_train, y_train)
                    self.best_model = final_model
                    self.best_params = best_params
                
                # Final model evaluation
                final_metrics = self._evaluate_model(self.best_model, X_test, y_test)
                
                # Check accuracy target
                if final_metrics['r2_score'] >= self.config.target_accuracy:
                    self.logger.info(f"✅ Target accuracy achieved: {final_metrics['r2_score']:.4f} >= {self.config.target_accuracy}")
                else:
                    self.logger.warning(f"⚠️ Target accuracy not met: {final_metrics['r2_score']:.4f} < {self.config.target_accuracy}")
                
                # Log final metrics to MLflow
                for metric_name, value in final_metrics.items():
                    mlflow.log_metric(f"final_{metric_name}", value)
                
                # Save model
                self._save_model()
                
                # Calculate training time
                training_time = (time.time() - start_time) / 60
                mlflow.log_metric("training_time_minutes", training_time)
                
                self.logger.info(f"Model training completed in {training_time:.2f} minutes")
                
                # Prepare results
                self.training_results = {
                    'best_model_name': best_model_name,
                    'best_score': self.best_score,
                    'best_params': self.best_params,
                    'final_metrics': final_metrics,
                    'training_time_minutes': training_time,
                    'target_achieved': final_metrics['r2_score'] >= self.config.target_accuracy,
                    'model_path': self.config.trained_model_file_path
                }
                
                # Save training report
                self._save_training_report()
                
                return self.training_results
                
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise CustomException(e)
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            dict: Best hyperparameters found
        """
        try:
            self.logger.info("Starting hyperparameter optimization...")
            
            # Create base model
            base_model = xgb.XGBRegressor(random_state=self.config.random_state)
            
            # Get parameter grid
            param_grid = self._prepare_param_grid()
            
            # Choose optimization method
            if self.config.optimization_method == "grid_search":
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    scoring=self._get_scoring_metric(),
                    cv=self.config.cv_folds,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose
                )
            else:  # random_search
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    scoring=self._get_scoring_metric(),
                    cv=self.config.cv_folds,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    n_iter=50  # Number of parameter settings sampled
                )
            
            # Fit the search
            self.logger.info(f"Running {self.config.optimization_method} with {self.config.cv_folds}-fold CV...")
            search.fit(X_train, y_train)
            
            # Log results to MLflow
            mlflow.log_param("optimization_method", self.config.optimization_method)
            mlflow.log_param("cv_folds", self.config.cv_folds)
            mlflow.log_metric("best_cv_score", search.best_score_)
            mlflow.log_params(search.best_params_)
            
            self.logger.info(f"Best CV score: {search.best_score_:.4f}")
            self.logger.info(f"Best parameters: {search.best_params_}")
            
            return search.best_params_
            
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {str(e)}")
            raise CustomException(e)
    
    def evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models and return performance comparison.
        
        Args:
            X_train: Training features
            y_train: Training target  
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Model performance results
        """
        try:
            self.logger.info("Evaluating multiple models...")
            
            # Define models to evaluate
            models = {
                "XGBoost": xgb.XGBRegressor(
                    random_state=self.config.random_state,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1
                ),
                "Random Forest": RandomForestRegressor(
                    random_state=self.config.random_state,
                    n_estimators=100
                ),
                "Linear Regression": LinearRegression()
            }
            
            model_results = {}
            
            for model_name, model in models.items():
                self.logger.info(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                metrics = self._evaluate_model(model, X_test, y_test)
                
                # Add cross-validation score
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=self.config.cv_folds,
                    scoring=self._get_scoring_metric()
                )
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                metrics['model'] = model
                
                model_results[model_name] = metrics
                
                # Log to MLflow
                with mlflow.start_run(nested=True, run_name=f"{model_name}_evaluation"):
                    for metric_name, value in metrics.items():
                        if metric_name != 'model':
                            mlflow.log_metric(metric_name, value)
                    
                    if model_name == "XGBoost":
                        mlflow.xgboost.log_model(model, "model")
                    else:
                        mlflow.sklearn.log_model(model, "model")
            
            # Log model comparison
            self._log_model_comparison(model_results)
            
            return model_results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise CustomException(e)
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a single model and return comprehensive metrics."""
        y_pred = model.predict(X_test)
        
        # Calculate all metrics
        metrics = {}
        
        if "mean_absolute_error" in self.config.metrics:
            metrics["mean_absolute_error"] = mean_absolute_error(y_test, y_pred)
        
        if "mean_squared_error" in self.config.metrics:
            metrics["mean_squared_error"] = mean_squared_error(y_test, y_pred)
            
        if "root_mean_squared_error" in self.config.metrics:
            metrics["root_mean_squared_error"] = np.sqrt(mean_squared_error(y_test, y_pred))
        
        if "r2_score" in self.config.metrics:
            metrics["r2_score"] = r2_score(y_test, y_pred)
            
        if "mean_absolute_percentage_error" in self.config.metrics:
            metrics["mean_absolute_percentage_error"] = mean_absolute_percentage_error(y_test, y_pred)
        
        return metrics
    
    def _prepare_param_grid(self) -> Dict[str, List]:
        """Prepare parameter grid for hyperparameter optimization."""
        return {
            'n_estimators': self.config.xgboost_params['n_estimators'],
            'learning_rate': self.config.xgboost_params['learning_rate'],
            'max_depth': self.config.xgboost_params['max_depth'],
            'subsample': self.config.xgboost_params['subsample'],
            'colsample_bytree': self.config.xgboost_params['colsample_bytree'],
            'reg_alpha': self.config.xgboost_params['reg_alpha'],
            'reg_lambda': self.config.xgboost_params['reg_lambda']
        }
    
    def _get_scoring_metric(self) -> str:
        """Convert config primary metric to sklearn scoring format."""
        metric_mapping = {
            "r2_score": "r2",
            "mean_absolute_error": "neg_mean_absolute_error",
            "mean_squared_error": "neg_mean_squared_error",
            "root_mean_squared_error": "neg_root_mean_squared_error"
        }
        return metric_mapping.get(self.config.primary_metric, "r2")
    
    def _save_model(self):
        """Save the trained model to disk."""
        try:
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=self.best_model
            )
            
            # Log model to MLflow
            mlflow.xgboost.log_model(self.best_model, "best_model")
            
            self.logger.info(f"Model saved to {self.config.trained_model_file_path}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise CustomException(e)
    
    def _save_training_report(self):
        """Save comprehensive training report."""
        try:
            report_path = os.path.join(self.config.root_dir, "training_report.json")
            
            # Prepare serializable report
            report = {
                'best_model_name': self.training_results['best_model_name'],
                'best_score': float(self.training_results['best_score']),
                'best_params': self.training_results['best_params'],
                'final_metrics': {k: float(v) for k, v in self.training_results['final_metrics'].items()},
                'training_time_minutes': float(self.training_results['training_time_minutes']),
                'target_achieved': self.training_results['target_achieved'],
                'target_accuracy': self.config.target_accuracy,
                'model_path': self.training_results['model_path'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Log report as artifact to MLflow
            mlflow.log_artifact(report_path, "reports")
            
            self.logger.info(f"Training report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Report saving failed: {str(e)}")
    
    def _log_model_comparison(self, model_results: Dict[str, Dict[str, Any]]):
        """Log model comparison results."""
        self.logger.info("\n" + "="*50)
        self.logger.info("MODEL COMPARISON RESULTS")
        self.logger.info("="*50)
        
        for model_name, results in model_results.items():
            self.logger.info(f"\n{model_name}:")
            self.logger.info(f"  R² Score: {results.get('r2_score', 0):.4f}")
            self.logger.info(f"  MAE: {results.get('mean_absolute_error', 0):.2f}")
            self.logger.info(f"  RMSE: {results.get('root_mean_squared_error', 0):.2f}")
            self.logger.info(f"  CV Score: {results.get('cv_mean', 0):.4f} ± {results.get('cv_std', 0):.4f}")
        
        # Find best model
        best_model = max(model_results.keys(), 
                        key=lambda k: model_results[k].get('r2_score', 0))
        self.logger.info(f"\n🏆 Best Model: {best_model}")
        self.logger.info("="*50)