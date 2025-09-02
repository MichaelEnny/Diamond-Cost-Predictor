# Docker Usage Guide - Diamond Price Predictor

This guide explains how to use the Docker containerization setup for the Diamond Price Predictor system.

## Container Services

The system includes the following containerized services:

1. **API Service** (`api`) - Flask REST API for diamond price predictions
2. **Dashboard Service** (`dashboard`) - Streamlit web interface
3. **MLflow Service** (`mlflow`) - ML experiment tracking server

## Quick Start Commands

### Build and Start All Services
```bash
# Build containers and start services
docker-compose up --build

# Start in detached mode (background)
docker-compose up -d --build
```

### Start Individual Services
```bash
# Start only the API service
docker-compose up api

# Start API + MLflow (without dashboard)
docker-compose up api mlflow

# Start with dashboard
docker-compose up api dashboard
```

### Service Management
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View running services
docker-compose ps

# View service logs
docker-compose logs api
docker-compose logs dashboard
docker-compose logs mlflow
```

## Service Endpoints

When running, services will be available at:

- **API Service**: http://localhost:5000
  - Health check: `GET /health`
  - Single prediction: `POST /predict` 
  - Batch prediction: `POST /predict/batch`

- **Dashboard**: http://localhost:8501
  - Interactive web interface for predictions

- **MLflow UI**: http://localhost:5001
  - Experiment tracking and model registry

## Volume Mounts

The containers use the following persistent volumes:

- `./artifacts:/app/artifacts` - ML models and preprocessors
- `./logs:/app/logs` - Application logs
- `./models:/app/models` - Model files
- `./mlruns:/mlflow/mlruns` - MLflow tracking data

## Health Checks

All services include health checks that monitor:
- API service health endpoint
- Streamlit application status
- MLflow server availability

Check health status:
```bash
docker-compose ps
```

## Environment Variables

Key environment variables for configuration:

### API Service
- `FLASK_ENV=production`
- `FLASK_HOST=0.0.0.0`
- `FLASK_PORT=5000`
- `PYTHONPATH=/app/src`

### Dashboard Service
- `API_URL=http://api:5000`

## Development vs Production

### Development Mode
```bash
# Use development override
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Production Mode
```bash
# Default production configuration
docker-compose up -d
```

## Monitoring and Logs

### View Real-time Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f dashboard
```

### Container Resource Usage
```bash
# View resource usage
docker stats

# Service-specific stats
docker stats diamond-predictor-api
docker stats diamond-dashboard
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml if needed
2. **Build failures**: Check requirements.txt and Dockerfile syntax
3. **Health check failures**: Verify service endpoints are responding
4. **Volume permissions**: Ensure directories are writable

### Debug Commands
```bash
# Enter running container
docker-compose exec api bash
docker-compose exec dashboard bash

# Check container status
docker-compose ps
docker-compose top

# Rebuild specific service
docker-compose build --no-cache api
docker-compose up api
```

## Security Considerations

- Containers run with non-root user where possible
- Health checks use internal endpoints
- Environment variables for sensitive configuration
- Volume mounts are read-only where appropriate

## Next Steps

1. Integrate with CI/CD pipeline
2. Add monitoring and alerting
3. Configure load balancing for production
4. Implement backup strategies for persistent data