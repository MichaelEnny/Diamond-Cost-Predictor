# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Production-ready Diamond Price Prediction system implementing MLOps best practices with MLflow, DVC, Airflow, Docker, Flask, and Streamlit. The system provides automated diamond price estimation based on 9 key characteristics (carat weight, cut grade, color rating, clarity assessment, depth percentage, table percentage, and dimensional measurements x, y, z) using optimized XGBoost regression models.

## Development Environment Setup

### Initial Environment Setup
```bash
bash init_setup.sh        # Creates conda env with Python 3.8
source activate ./venv     # Activates the environment 
python setup.py install   # Installs project as local package
```

### Development Requirements
Install from `requirements_dev.txt` which includes:
- Core ML libraries: pandas, scikit-learn, numpy, xgboost
- MLOps tools: mlflow==2.2.2, dvc, apache-airflow
- Dev tools: pytest, black, flake8, mypy, tox

## Core Commands

### ML Pipeline
```bash
dvc repro                           # Run complete ML pipeline with DVC versioning
python src/pipeline/training_pipeline.py  # Run training pipeline directly
```

### Model Experiment Tracking
```bash
mlflow ui                          # Launch MLflow UI for experiment tracking
```

### Web Applications
```bash
python app.py                      # Flask app on port 8000
streamlit run streamlit_app.py     # Streamlit app for interactive predictions
```

### Airflow Orchestration
```bash
# Set AIRFLOW_HOME environment variable to ./airflow
airflow db init                                    # Initialize Airflow database
airflow users create -e <email> -f <name> -l <lastname> -p <password> -u <username> -r admin
bash start.sh                                      # Start scheduler and webserver
```

### Docker Deployment
```bash
docker build -t <image_name> .               # Build main Flask app image
docker-compose up                            # Run multi-service deployment
docker build -f Dockerfile.airflow .         # Build Airflow image
docker build -f Dockerfile.flask .           # Build dedicated Flask image
```

## Architecture Overview

### Pipeline Components (src/components/)
- **DataIngestion**: Automated diamond dataset loading and train/test splitting (data_ingestion.py)
- **DataTransformation**: Diamond feature engineering and characteristic preprocessing (data_transformation.py)  
- **ModelTrainer**: XGBoost regression training with hyperparameter optimization and MLflow experiment tracking (model_trainer.py)
- **ModelEvaluation**: Diamond price prediction model performance assessment and validation (model_evaluation.py)

### Pipeline Orchestration (src/pipeline/)
- **TrainingPipeline**: Coordinated diamond price model training workflow execution
- **PredictionPipeline**: Real-time diamond price estimation with CustomData class for characteristic input handling

### Configuration Management
- **params.yaml**: XGBoost hyperparameters (n_estimators: 150, learning_rate: 0.8, etc.)
- **config.yaml**: Directory paths and file locations
- **dvc.yaml**: DVC pipeline definition with dependencies and outputs

### Data Flow
1. Diamond dataset → Data Ingestion → train/test CSV files with diamond characteristics
2. CSV files → Data Transformation → preprocessor.pkl + transformed diamond feature arrays  
3. Transformed diamond data → Model Training → trained XGBoost model.pkl + MLflow price prediction experiments
4. Model artifacts → Model Evaluation → diamond price prediction performance metrics
5. Trained artifacts used by prediction pipelines for real-time diamond price inference

### Airflow DAG Structure
- Sequential diamond pipeline tasks: diamond_data_ingestion → diamond_data_transformation → diamond_model_trainer → push_artifacts_to_s3
- Uses xcom for inter-task communication and artifact passing
- Scheduled daily execution with automated retry logic for robust diamond price model updates

### Artifact Management
All diamond price prediction artifacts stored in `artifacts/` directory:
- model.pkl (trained XGBoost diamond price regression model)
- preprocessor.pkl (diamond characteristic preprocessing pipeline)  
- model_name.yaml (diamond price model metadata and configuration)
- train.csv, test.csv, raw.csv (processed diamond dataset splits)

### Testing Structure
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Development tools configured: pytest, black (formatting), flake8 (linting), mypy (typing)

## Key Files for Diamond Price Prediction Modifications
- Diamond model hyperparameters: `params.yaml`
- Diamond pipeline configuration: `dvc.yaml` 
- Directory paths and artifact locations: `config.yaml`
- Diamond training workflow logic: `src/pipeline/training_pipeline.py`
- Web interfaces: `app.py` (Flask REST API), `streamlit_app.py` (Interactive Diamond Price Dashboard)
- Container deployment: `Dockerfile`, `docker-compose.yaml`