#!/usr/bin/env python3
"""
Demo script for Diamond Price Predictor ModelTrainer.

This script demonstrates the ModelTrainer functionality with sample data,
showcasing hyperparameter optimization, cross-validation, and model evaluation
to achieve the 95%+ accuracy target.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# Add src to path for imports
sys.path.append('src')

from src.components.model_trainer import ModelTrainer
from src.config import ConfigurationManager
from src.logger import logger

warnings.filterwarnings('ignore')


def create_sample_diamond_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Create synthetic diamond-like dataset for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with diamond-like features and price target
    """
    logger.info(f"Creating synthetic diamond dataset with {n_samples} samples...")
    
    # Generate base features using make_regression for realistic correlations
    X, y_base = make_regression(
        n_samples=n_samples,
        n_features=9,
        n_informative=9,
        noise=0.1,
        random_state=42
    )
    
    # Create realistic diamond features
    np.random.seed(42)
    
    # Carat (most important feature for price)
    carat = np.abs(X[:, 0]) * 0.5 + 0.3  # 0.3 to 2.5 carat range
    carat = np.clip(carat, 0.2, 5.0)
    
    # Cut (categorical: Fair, Good, Very Good, Premium, Ideal)
    cut_values = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut = np.random.choice(cut_values, size=n_samples, p=[0.05, 0.15, 0.25, 0.35, 0.2])
    
    # Color (categorical: D, E, F, G, H, I, J)
    color_values = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    color = np.random.choice(color_values, size=n_samples, p=[0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05])
    
    # Clarity (categorical)
    clarity_values = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    clarity = np.random.choice(clarity_values, size=n_samples, p=[0.05, 0.15, 0.20, 0.25, 0.15, 0.10, 0.08, 0.02])
    
    # Depth and Table percentages
    depth = np.clip(np.abs(X[:, 3]) * 5 + 60, 50, 75)
    table = np.clip(np.abs(X[:, 4]) * 8 + 55, 50, 70)
    
    # Physical dimensions (x, y, z) correlated with carat
    x = carat ** (1/3) * 6.2 + np.random.normal(0, 0.1, n_samples)
    y = x + np.random.normal(0, 0.05, n_samples)  # y slightly different from x
    z = x * 0.6 + np.random.normal(0, 0.05, n_samples)  # z is depth
    
    # Ensure positive dimensions
    x = np.abs(x)
    y = np.abs(y) 
    z = np.abs(z)
    
    # Create realistic price based on diamond characteristics
    # Price formula based on real diamond pricing factors
    cut_multiplier = {'Fair': 0.8, 'Good': 0.9, 'Very Good': 1.0, 'Premium': 1.1, 'Ideal': 1.2}
    color_multiplier = {'D': 1.3, 'E': 1.2, 'F': 1.1, 'G': 1.0, 'H': 0.9, 'I': 0.8, 'J': 0.7}
    clarity_multiplier = {'I1': 0.6, 'SI2': 0.7, 'SI1': 0.8, 'VS2': 0.9, 'VS1': 1.0, 'VVS2': 1.1, 'VVS1': 1.2, 'IF': 1.3}
    
    price = []
    for i in range(n_samples):
        base_price = (carat[i] ** 2) * 3000  # Carat squared relationship
        price_adj = (
            base_price * 
            cut_multiplier[cut[i]] * 
            color_multiplier[color[i]] * 
            clarity_multiplier[clarity[i]] *
            (1 + np.random.normal(0, 0.1))  # Add some noise
        )
        price.append(max(300, price_adj))  # Minimum price floor
    
    # Create DataFrame
    df = pd.DataFrame({
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z,
        'price': price
    })
    
    logger.info(f"Dataset created with shape: {df.shape}")
    logger.info(f"Price range: ${df['price'].min():.0f} - ${df['price'].max():.0f}")
    
    return df


def preprocess_diamond_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess diamond data for model training.
    
    Args:
        df: Raw diamond dataset
        
    Returns:
        tuple: (X_processed, y) arrays
    """
    logger.info("Preprocessing diamond dataset...")
    
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price'].values
    
    # Handle categorical variables
    categorical_features = ['cut', 'color', 'clarity']
    numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    
    # One-hot encode categorical features
    X_processed = pd.get_dummies(X, columns=categorical_features, drop_first=False)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_processed[numerical_features] = scaler.fit_transform(X_processed[numerical_features])
    
    logger.info(f"Processed features shape: {X_processed.shape}")
    logger.info(f"Feature columns: {list(X_processed.columns)}")
    
    return X_processed.values, y


def demonstrate_model_trainer():
    """Main demonstration of ModelTrainer functionality."""
    
    logger.info("="*60)
    logger.info("DIAMOND PRICE PREDICTOR - MODEL TRAINER DEMO")
    logger.info("="*60)
    
    try:
        # 1. Generate sample data
        logger.info("\n🎯 Step 1: Generating synthetic diamond dataset...")
        df = create_sample_diamond_data(n_samples=5000)
        
        # Show data sample
        logger.info("\nDataset Sample:")
        logger.info(f"\n{df.head()}")
        logger.info(f"\nDataset Info:")
        logger.info(f"{df.info()}")
        
        # 2. Preprocess data
        logger.info("\n🔧 Step 2: Preprocessing data...")
        X, y = preprocess_diamond_data(df)
        
        # 3. Split data
        logger.info("\n📊 Step 3: Splitting data for training...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Combine for ModelTrainer format (expects features + target in last column)
        train_array = np.c_[X_train, y_train]
        test_array = np.c_[X_test, y_test]
        
        logger.info(f"Training set: {train_array.shape}")
        logger.info(f"Test set: {test_array.shape}")
        
        # 4. Initialize ModelTrainer
        logger.info("\n🚀 Step 4: Initializing ModelTrainer...")
        config_manager = ConfigurationManager()
        model_trainer = ModelTrainer()
        
        # 5. Train model
        logger.info("\n🔥 Step 5: Starting model training with hyperparameter optimization...")
        logger.info("Target: 95%+ R² score accuracy")
        
        results = model_trainer.initiate_model_training(train_array, test_array)
        
        # 6. Display results
        logger.info("\n"+"="*60)
        logger.info("🏆 TRAINING RESULTS")
        logger.info("="*60)
        
        logger.info(f"✅ Best Model: {results['best_model_name']}")
        logger.info(f"✅ Best Score (R²): {results['best_score']:.4f}")
        logger.info(f"✅ Target Achieved: {results['target_achieved']}")
        logger.info(f"✅ Training Time: {results['training_time_minutes']:.2f} minutes")
        
        logger.info(f"\n📊 Final Model Metrics:")
        for metric, value in results['final_metrics'].items():
            logger.info(f"  • {metric}: {value:.4f}")
        
        if results['best_params']:
            logger.info(f"\n⚙️ Best Hyperparameters:")
            for param, value in results['best_params'].items():
                logger.info(f"  • {param}: {value}")
        
        # 7. Model validation
        if results['final_metrics']['r2_score'] >= 0.95:
            logger.info("\n🎉 SUCCESS: Model achieved 95%+ accuracy target!")
        else:
            logger.info(f"\n⚠️ Model accuracy ({results['final_metrics']['r2_score']:.4f}) below target (0.95)")
            logger.info("Consider: more data, feature engineering, or different algorithms")
        
        logger.info(f"\n📁 Model saved to: {results['model_path']}")
        
        # 8. Verification
        logger.info("\n🔍 Step 6: Verifying saved model...")
        if os.path.exists(results['model_path']):
            model_size = os.path.getsize(results['model_path']) / 1024  # KB
            logger.info(f"✅ Model file exists ({model_size:.1f} KB)")
        else:
            logger.error("❌ Model file not found!")
        
        logger.info("\n" + "="*60)
        logger.info("✅ DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    # Set up logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create necessary directories
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run demonstration
    results = demonstrate_model_trainer()
    
    print(f"\n🎯 Demo Summary:")
    print(f"  Model: {results['best_model_name']}")
    print(f"  Accuracy: {results['final_metrics']['r2_score']:.4f}")
    print(f"  Target Met: {'✅ Yes' if results['target_achieved'] else '❌ No'}")
    print(f"  Training Time: {results['training_time_minutes']:.2f} minutes")