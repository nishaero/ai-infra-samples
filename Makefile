.PHONY: help setup install dev test test-coverage lint format hooks clean deploy-infra deploy-app monitor

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Set up the development environment
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"
	@echo "Run 'make install' after activating the virtual environment"

install: ## Install dependencies
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements/dev.txt
	pip install -e .

dev: ## Start local development environment
	@echo "Starting local development environment..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "Services started. Access:"
	@echo "  - API: http://localhost:8000"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Airflow: http://localhost:8080"
	@echo "  - Grafana: http://localhost:3000"

test: ## Run tests
	@echo "Running tests..."
	pytest tests/ -v

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	@echo "Running linting..."
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format: ## Format code
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

hooks: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	pre-commit install

clean: ## Clean up temporary files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

deploy-infra: ## Deploy infrastructure
	@echo "Deploying infrastructure..."
	cd infrastructure/terraform && terraform init && terraform plan && terraform apply

deploy-app: ## Deploy application
	@echo "Deploying application..."
	docker build -f docker/Dockerfile.api -t climate-prediction-api .
	kubectl apply -f infrastructure/kubernetes/

monitor: ## Show monitoring dashboards
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "MLflow: http://localhost:5000"
	@echo "Airflow: http://localhost:8080 (admin/admin)"

build: ## Build Docker images
	@echo "Building Docker images..."
	docker build -f docker/Dockerfile.api -t climate-prediction-api .
	docker build -f docker/Dockerfile.training -t climate-prediction-training .

start-services: ## Start all services
	@echo "Starting all services..."
	docker-compose -f docker/docker-compose.yml up -d

stop-services: ## Stop all services
	@echo "Stopping all services..."
	docker-compose -f docker/docker-compose.yml down

logs: ## Show service logs
	docker-compose -f docker/docker-compose.yml logs -f

data-fetch: ## Fetch NASA Earth data
	@echo "Fetching NASA Earth data..."
	python src/data/ingestion.py

train: ## Train the model
	@echo "Training model..."
	python workflows/scripts/train.py

validate: ## Validate the project setup
	@echo "Validating project setup..."
	python -c "import src.data, src.models, src.api; print('✓ All modules importable')"
	docker --version
	terraform --version
	@echo "✓ Project validation complete"