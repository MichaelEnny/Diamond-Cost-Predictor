# Diamond Price Prediction - MLOps Pipeline

A comprehensive **Diamond Price Prediction** system leveraging advanced machine learning and MLOps practices with **MLflow**, **DVC**, **Airflow**, **Docker**, **Flask** and cloud deployment.

## Business Context
The diamond industry requires accurate price estimation models to support trading decisions, inventory valuation, and market analysis. This project delivers an automated ML pipeline that predicts diamond prices based on physical and quality characteristics, enabling data-driven pricing strategies for diamond retailers, wholesalers, and appraisers.

## Project Goals
Develop a production-ready machine learning system that:
- Accurately predicts diamond prices using 9 key characteristics
- Provides automated data processing and model retraining capabilities
- Offers both API and web interfaces for real-time predictions
- Implements MLOps best practices for model lifecycle management

## Data Overview
The diamond dataset contains comprehensive information about diamond characteristics and their corresponding market prices. Key features include: Carat weight, Cut grade, Color rating, Clarity assessment, Depth percentage, Table percentage, and dimensional measurements (x, y, z). 

## Technology Stack
- **Core Language**: [Python](https://www.python.org) for ML development and web services
- **Version Control**: [DVC](https://dvc.org/) for data and pipeline versioning
- **Experiment Management**: [MLflow](https://mlflow.org) for model tracking and registry
- **Workflow Orchestration**: [Apache Airflow](https://airflow.apache.org/) for automated pipeline execution
- **User Interfaces**: [Flask](https://flask.palletsprojects.com/en/3.0.x/) REST API and [Streamlit](https://streamlit.io/) interactive dashboard
- **Containerization**: [Docker](https://www.docker.com) for consistent deployment
- **CI/CD Pipeline**: [GitHub Actions](https://github.com/features/actions) for automated testing and deployment
- **Cloud Infrastructure**: [Azure Container Registry](https://azure.microsoft.com/en-in/products/container-registry) for container management

## System Architecture
The diamond price prediction system follows a modular MLOps architecture:

### Pipeline Components
- **Data Ingestion**: Automated data loading and validation
- **Feature Engineering**: Diamond characteristic preprocessing and transformation  
- **Model Training**: XGBoost regression with hyperparameter optimization
- **Model Evaluation**: Performance assessment and validation metrics
- **Model Registry**: MLflow-based model versioning and lifecycle management

### Deployment Infrastructure  
- **API Service**: Flask REST API for programmatic access
- **Web Interface**: Interactive Streamlit dashboard for business users
- **Orchestration**: Airflow DAGs for automated pipeline execution
- **Monitoring**: MLflow experiment tracking and model performance monitoring

## Key Features
- **Automated ML Pipeline**: End-to-end automation from data ingestion to model deployment
- **Real-time Predictions**: Instant diamond price estimates via web interface and API
- **Model Versioning**: Complete experiment tracking and model registry
- **Scalable Architecture**: Docker containerization for consistent deployment
- **Interactive Dashboard**: User-friendly interface for diamond price estimation

## Quick Start

### Environment Setup
```bash
# Create and activate conda environment
bash init_setup.sh
source activate ./env

# Install dependencies
python setup.py install
```

### Run Diamond Price Prediction Pipeline
```bash
# Execute complete ML pipeline
dvc repro

# Launch experiment tracking dashboard
mlflow ui
```

### Deploy Applications
```bash
# Start Flask API server (port 8000)
python app.py

# Launch Streamlit dashboard
streamlit run streamlit_app.py
```

### Orchestration with Airflow
```bash
# Initialize Airflow database
export AIRFLOW_HOME=./airflow
airflow db init

# Create admin user
airflow users create -e admin@example.com -f Admin -l User -p admin123 -u admin -r Admin

# Start Airflow services
bash start.sh
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual services
docker build -t diamond-price-api .
docker build -f Dockerfile.airflow -t diamond-airflow .
```

### Cloud Deployment (Azure)
```bash
# Deploy to Azure Container Registry
docker build -t diamondprediction.azurecr.io/diamond-api:latest .
docker login diamondprediction.azurecr.io
docker push diamondprediction.azurecr.io/diamond-api:latest
```

## Usage Examples

### API Predictions
```python
import requests

# Predict diamond price via API
diamond_data = {
    "carat": 1.2,
    "cut": "Premium", 
    "color": "G",
    "clarity": "VS1",
    "depth": 62.1,
    "table": 58.0,
    "x": 6.8,
    "y": 6.7,
    "z": 4.2
}

response = requests.post("http://localhost:8000/predict", json=diamond_data)
predicted_price = response.json()["predicted_price"]
```

### Model Performance
The XGBoost model achieves competitive performance on diamond price prediction:
- **R² Score**: ~0.95+ on test data  
- **RMSE**: Optimized through hyperparameter tuning
- **Features**: All 9 diamond characteristics contribute to predictions

---

**Diamond Price Prediction MLOps Pipeline** - Production-ready machine learning system for accurate diamond valuation
