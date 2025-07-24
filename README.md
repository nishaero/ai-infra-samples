# Climate Temperature Prediction using NASA Earth Data

## Problem Description

This project addresses the critical need for accurate climate temperature prediction using satellite observations and meteorological data from NASA's Earth Observing System. The system predicts regional temperature anomalies by analyzing MODIS Land Surface Temperature data, precipitation patterns, and other climate indicators.

### Objectives
- Predict temperature anomalies for specific geographic regions
- Provide early warning capabilities for extreme temperature events
- Support climate monitoring and research initiatives
- Enable data-driven decision making for climate adaptation strategies

### Business Impact
- **Climate Monitoring**: Real-time tracking of temperature trends across regions
- **Early Warning Systems**: Proactive alerts for temperature anomalies
- **Research Support**: Data insights for climate scientists and researchers
- **Policy Support**: Evidence-based data for climate policy decisions

## Architecture Overview

The project implements a complete MLOps pipeline with the following components:

- **Data Pipeline**: Automated ingestion from NASA Earth Data APIs
- **ML Pipeline**: Time series forecasting using ensemble methods
- **Experiment Tracking**: MLflow for model versioning and metrics
- **Workflow Orchestration**: Airflow for pipeline automation
- **Model Deployment**: Containerized FastAPI service on AWS
- **Monitoring**: Real-time performance tracking with alerts
- **Infrastructure**: Terraform-managed AWS resources

## Features

- 🌍 **NASA Earth Data Integration**: Direct API access to MODIS and other satellite data
- 🤖 **Advanced ML Models**: Ensemble methods for robust temperature prediction
- 📊 **Experiment Tracking**: Full MLflow integration with model registry
- 🔄 **Automated Workflows**: Airflow orchestration for training and inference
- 🚀 **Cloud Deployment**: Scalable AWS infrastructure with containerization
- 📈 **Real-time Monitoring**: Comprehensive metrics and alerting system
- 🧪 **Quality Assurance**: Complete test suite with CI/CD pipeline

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (for cloud deployment)
- AWS CLI configured (for cloud deployment)
- Terraform installed (for infrastructure)

### Local Model Execution

To run the climate prediction model locally and generate visualizations:

```bash
# Clone the repository
git clone https://github.com/nishaero/ai-infra-samples.git
cd ai-infra-samples

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt

# Run the model with default parameters
python run_model.py

# Or with custom parameters
python run_model.py --start-date 2023-01-01 --end-date 2023-06-30 --bbox -125 30 -110 45 --output-dir ./results
```

#### What the Local Run Does:
1. 🌍 **Fetches Real Climate Data**: Downloads temperature and precipitation data from NOAA
2. 🤖 **Trains ML Models**: Trains Linear Regression + Random Forest ensemble
3. 📊 **Generates Predictions**: Creates temperature predictions and evaluates performance  
4. 📈 **Creates Visualizations**: Generates 5+ plots showing model performance
5. 📄 **Produces HTML Report**: Creates comprehensive report with plots and explanations

#### Output Files:
```
output/
├── climate_prediction_report.html    # Main report (open in browser)
└── plots/
    ├── actual_vs_predicted.png       # Prediction accuracy
    ├── residuals.png                 # Error analysis
    ├── feature_importance.png        # Important features
    ├── model_comparison.png          # Model performance
    └── temperature_distribution.png  # Data distribution
```

#### View Results:
```bash
# Open the HTML report in your browser
open output/climate_prediction_report.html  # macOS
# or
start output/climate_prediction_report.html  # Windows
# or  
xdg-open output/climate_prediction_report.html  # Linux
```

### Cloud Development Setup

For full MLOps pipeline development:

```bash
# Set up the environment
make setup

# Install dependencies
make install

# Run tests
make test

# Start local development
make dev
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data ingestion and preprocessing
│   ├── models/            # ML models and training
│   ├── api/               # FastAPI service
│   └── monitoring/        # Monitoring and metrics
├── tests/                 # Test suite
├── infrastructure/        # Terraform and Kubernetes configs
├── workflows/             # Airflow DAGs and orchestration
├── docker/                # Docker configurations
├── docs/                  # Documentation
└── requirements/          # Dependency specifications
```

## Documentation

- **[Local Model Execution Guide](LOCAL_README.md)** - Run the model locally with visualizations
- [Cloud Setup Guide](docs/cloud-setup.md) - AWS and infrastructure setup
- [Architecture Documentation](docs/architecture.md) - System design details
- [Deployment Guide](docs/deployment.md) - Production deployment instructions

## Development

### Code Quality
- **Linting**: Black, flake8, isort
- **Testing**: pytest with coverage reporting
- **Pre-commit hooks**: Automated code quality checks
- **CI/CD**: GitHub Actions for automated testing and deployment

### Local Development
```bash
# Set up pre-commit hooks
make hooks

# Run code formatting
make format

# Run linting
make lint

# Run tests with coverage
make test-coverage
```

## Deployment

### Local Development
```bash
make dev
```

### Production Deployment
```bash
# Deploy infrastructure
make deploy-infra

# Deploy application
make deploy-app

# Monitor deployment
make monitor
```

## Monitoring

The system includes comprehensive monitoring with:
- Model performance metrics
- Data drift detection  
- Infrastructure health checks
- Automated alerting for anomalies

Access monitoring dashboards at: `http://localhost:3000` (Grafana)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the full test suite
5. Submit a pull request

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA Earth Data for providing satellite observation data
- The open-source ML community for the tools and frameworks used