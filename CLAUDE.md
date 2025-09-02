# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Diamond Price Predictor System** - a production-ready ML platform that predicts diamond prices using XGBoost with comprehensive MLOps infrastructure. The system provides both REST API and web interface for diamond price predictions with 95%+ accuracy.

### Key Components
- **ML Pipeline**: XGBoost-based price prediction with 9 diamond characteristics (carat, cut, color, clarity, depth, table, x, y, z)
- **Flask REST API**: Production API service with `/predict` and `/predict/batch` endpoints
- **Streamlit Dashboard**: Interactive web interface for manual predictions and batch processing
- **MLOps Infrastructure**: MLflow experiment tracking, DVC versioning, Airflow orchestration

## Development Commands

Since this is an early-stage project, the core implementation is not yet complete. Based on the PRD and task breakdown:

### Expected Project Structure (to be implemented)
```
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   └── pipeline/
│       └── prediction_pipeline.py
├── app.py                  # Flask API server
├── streamlit_app.py       # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── Dockerfile.flask       # API containerization
├── docker-compose.yaml    # Multi-service orchestration
├── dvc.yaml              # DVC pipeline configuration
├── params.yaml           # Model hyperparameters
└── airflow/dags/         # Airflow DAGs
```

### Development Setup (planned)
```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# DVC initialization
dvc init
dvc remote add -d storage s3://diamond-predictor-artifacts

# MLflow server
mlflow ui --host 0.0.0.0 --port 5000

# API development server
python app.py

# Streamlit dashboard
streamlit run streamlit_app.py
```

### Testing Commands (to be implemented)
```bash
# Unit tests
pytest tests/ -v --cov=src --cov-report=html

# API integration tests
pytest tests/test_api.py -v

# Model performance tests
pytest tests/test_model.py -v

# Load testing
locust -f tests/load_test.py --host=http://localhost:5000
```

### Code Quality (planned)
```bash
# Formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
pylint src/

# Type checking
mypy src/
```

### Docker Operations (planned)
```bash
# Build containers
docker build -f Dockerfile.flask -t diamond-predictor-api .
docker-compose build

# Run services
docker-compose up -d

# Scale API service
docker-compose up --scale api=3
```

## Architecture Overview

### System Design
The application follows a microservices architecture with the following layers:

1. **Presentation Layer**: Streamlit dashboard + Flask REST API
2. **Application Layer**: Flask with prediction logic and model serving
3. **ML Layer**: XGBoost model with preprocessing pipeline
4. **Data Layer**: MLflow model registry and DVC artifact storage
5. **Infrastructure Layer**: Docker containers with orchestration

### Key Technical Decisions

**Model Choice**: XGBoost selected for high accuracy (95%+ target) and fast inference (<200ms)

**MLOps Stack**:
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data and model versioning
- **Airflow**: Pipeline orchestration and automated retraining
- **Docker**: Containerization for consistent deployments

**API Design**: RESTful endpoints with comprehensive validation:
- `POST /predict` - Single diamond prediction
- `POST /predict/batch` - Bulk diamond processing
- `GET /health` - System health monitoring
- `GET /model/info` - Model metadata

### Performance Requirements
- **API Response Time**: <200ms for single predictions
- **Throughput**: 1000+ requests/minute
- **Concurrent Users**: 100+ simultaneous users
- **Model Accuracy**: 95%+ on test dataset
- **System Uptime**: 99.9% availability SLA

## Development Phases

### Phase 1: MVP Development (Months 1-3)
**Current Status**: Planning/Early Development
- Month 1: ML pipeline development and optimization
- Month 2: Flask API implementation and containerization  
- Month 3: Streamlit dashboard and UX testing

### Phase 2: Production Readiness (Months 4-6)
- MLOps infrastructure (Airflow, DVC, monitoring)
- Security and authentication systems
- Performance optimization and load testing

### Phase 3: Market Launch (Months 7-9)
- Beta testing program
- Public launch and customer acquisition
- Growth optimization and revenue scaling

## Critical Dependencies

1. **Model Training Data**: Diamond dataset with 9 characteristics
2. **MLflow Server**: For experiment tracking and model registry
3. **DVC Remote Storage**: S3/GCS for artifact versioning
4. **Production Infrastructure**: Docker/Kubernetes deployment platform

## Security Considerations

- **API Authentication**: JWT token or API key-based authentication
- **Rate Limiting**: Configurable request throttling and abuse prevention
- **Data Privacy**: Secure handling of diamond specifications
- **Model Protection**: Encrypted model artifacts in registry

## Monitoring and Observability

**Model Monitoring**:
- Prediction accuracy drift detection
- Feature distribution monitoring
- Performance degradation alerts
- Automated retraining triggers

**System Monitoring**:
- API response times and error rates
- Resource utilization (CPU, memory)
- Request volume and patterns
- System health and availability

## Business Context

**Target Users**:
- Diamond retailers and appraisers
- Wholesalers requiring bulk pricing
- Online marketplaces needing pricing APIs
- Independent gemologists for validation

**Key Metrics**:
- Model accuracy: 95%+ prediction accuracy
- User adoption: 500+ monthly active users
- API usage: 100,000+ monthly requests
- Business impact: 90% time savings vs manual pricing

This is a high-value ML application targeting the $87.8B global diamond market with significant automation and efficiency benefits for industry professionals.