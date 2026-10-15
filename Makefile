# ==============================================================================
# Makefile: Closed-Loop Financial Risk MLOps Platform
# ==============================================================================

SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
TRIVY := trivy
TERRAFORM := terraform

.PHONY: help setup local-up local-down data train govern promote serve simulate-drift test security-scan tf-init tf-plan tf-apply tf-destroy clean

help:
	@echo "Available commands:"
	@echo "  make setup          - Initialize virtual environment and install pinned dependencies"
	@echo "  make local-up       - Spin up local MinIO, PostgreSQL, MLflow, Prometheus, Grafana stack"
	@echo "  make local-down     - Stop local Docker Compose services"
	@echo "  make data           - Generate synthetic transactions and preprocessed splits"
	@echo "  make train          - Train XGBoost model with MLflow Tracking & SHAP values"
	@echo "  make govern         - Run SageMaker Clarify-style fairness and explainability checks"
	@echo "  make promote        - Promote challenger to @production in MLflow Model Registry"
	@echo "  make serve          - Start FastAPI real-time inference microservice locally"
	@echo "  make simulate-drift - Inject covariate drift and run Evidently AI drift detection"
	@echo "  make test           - Run full pytest test suite with coverage report"
	@echo "  make security-scan  - Execute Aqua Security Trivy scan on codebase, Dockerfile, and Terraform"
	@echo "  make tf-init        - Initialize Terraform working directory"
	@echo "  make tf-plan        - Preview AWS infrastructure changes via Terraform"
	@echo "  make tf-apply       - Deploy production AWS infrastructure (ECS, Step Functions, S3, IAM)"
	@echo "  make tf-destroy     - Teardown all AWS cloud resources to maintain zero cloud bill"
	@echo "  make clean          - Remove temporary cache files and artifacts"

setup:
	@echo "==> Setting up Python virtual environment..."
	python3 -m venv .venv || py -m venv .venv || python -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "==> Environment ready."

local-up:
	@echo "==> Starting local MLOps services via Docker Compose..."
	docker compose up -d
	@echo "==> Services available:"
	@echo "    - MLflow Tracking:    http://localhost:5000"
	@echo "    - MinIO S3 Console:   http://localhost:9001"
	@echo "    - Grafana Dashboards: http://localhost:3000"
	@echo "    - Prometheus Metrics: http://localhost:9090"
	@echo "    - FastAPI Inference:  http://localhost:8000"
	@echo "    - Evidently Reports:  http://localhost:8001"

local-down:
	@echo "==> Stopping local Docker services..."
	docker compose down

data:
	@echo "==> Generating baseline synthetic transactions..."
	$(PYTHON) src/data/generate_synthetic_transactions.py --samples 15000
	@echo "==> Preprocessing and generating chronological feature splits..."
	$(PYTHON) src/data/preprocess.py

train:
	@echo "==> Training XGBoost fraud detection model with MLflow tracking..."
	$(PYTHON) src/training/train_xgboost.py

govern:
	@echo "==> Running pre-deployment fairness & SHAP explainability gate..."
	$(PYTHON) src/governance/evaluate_fairness_and_explainability.py

promote:
	@echo "==> Evaluating Champion vs Challenger in MLflow Model Registry..."
	$(PYTHON) src/governance/promote_model.py

serve:
	@echo "==> Starting FastAPI inference service on port 8000..."
	$(PYTHON) -m uvicorn src.inference.app:app --host 0.0.0.0 --port 8000 --reload

simulate-drift:
	@echo "==> Generating drifted production transaction stream..."
	$(PYTHON) src/data/generate_synthetic_transactions.py --samples 5000 --inject-drift
	@echo "==> Running Evidently AI data drift detector..."
	$(PYTHON) src/monitoring/drift_detector.py

test:
	@echo "==> Running pytest suite with code coverage..."
	$(PYTHON) -m pytest -v --cov=src --cov-report=term-missing tests/

security-scan:
	@echo "==> Running Aqua Security Trivy filesystem scan on dependencies..."
	$(TRIVY) fs --severity HIGH,CRITICAL --scanners vuln .
	@echo "==> Scanning Terraform configuration for misconfigurations..."
	$(TRIVY) config terraform/

tf-init:
	@echo "==> Initializing Terraform..."
	cd terraform && $(TERRAFORM) init

tf-plan:
	@echo "==> Generating Terraform execution plan..."
	cd terraform && $(TERRAFORM) plan -out=tfplan

tf-apply:
	@echo "==> Applying Terraform cloud deployment on AWS..."
	cd terraform && $(TERRAFORM) apply -auto-approve tfplan

tf-destroy:
	@echo "==> Tearing down AWS cloud resources to preserve cost..."
	cd terraform && $(TERRAFORM) destroy -auto-approve

clean:
	@echo "==> Cleaning cache directories..."
	rm -rf __pycache__ src/**/__pycache__ tests/**/__pycache__ .pytest_cache .coverage htmlcov
