<#
.SYNOPSIS
    Windows PowerShell CLI Helper for Financial Risk MLOps Platform.
.DESCRIPTION
    Mirrors all Makefile targets for native Windows execution without requiring GNU Make.
.EXAMPLE
    .\scripts\manage.ps1 data
    .\scripts\manage.ps1 train
    .\scripts\manage.ps1 test
#>

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$python = ".\.venv\Scripts\python.exe"
$pytest = ".\.venv\Scripts\pytest.exe"
$trivy = ".\bin\trivy.cmd"
$terraform = ".\bin\terraform.cmd"

switch ($Command.ToLower()) {
    "help" {
        Write-Host "Financial Risk MLOps CLI (Windows Helper):" -ForegroundColor Cyan
        Write-Host "  .\scripts\manage.ps1 data           - Generate synthetic data & feature splits"
        Write-Host "  .\scripts\manage.ps1 train          - Train XGBoost model with MLflow Tracking"
        Write-Host "  .\scripts\manage.ps1 govern         - Run pre-deployment fairness & SHAP gate"
        Write-Host "  .\scripts\manage.ps1 promote        - Promote Champion vs Challenger in MLflow"
        Write-Host "  .\scripts\manage.ps1 serve          - Start FastAPI real-time inference server"
        Write-Host "  .\scripts\manage.ps1 simulate-drift - Inject covariate drift and run Evidently AI"
        Write-Host "  .\scripts\manage.ps1 test           - Run full pytest test suite"
        Write-Host "  .\scripts\manage.ps1 security-scan  - Run Trivy vulnerability & IaC scans"
        Write-Host "  .\scripts\manage.ps1 local-up       - Start Docker Compose local stack"
        Write-Host "  .\scripts\manage.ps1 local-down     - Stop Docker Compose local stack"
        Write-Host "  .\scripts\manage.ps1 tf-plan        - Terraform plan preview"
        Write-Host "  .\scripts\manage.ps1 tf-apply       - Terraform deploy AWS infrastructure"
        Write-Host "  .\scripts\manage.ps1 tf-destroy     - Terraform destroy AWS infrastructure"
    }
    "data" {
        Write-Host "==> Generating baseline synthetic transactions..." -ForegroundColor Green
        & $python src/data/generate_synthetic_transactions.py --samples 15000
        Write-Host "==> Preprocessing and generating chronological feature splits..." -ForegroundColor Green
        & $python src/data/preprocess.py
    }
    "train" {
        Write-Host "==> Training XGBoost model..." -ForegroundColor Green
        & $python src/training/train_xgboost.py
    }
    "govern" {
        Write-Host "==> Running pre-deployment fairness and explainability gate..." -ForegroundColor Green
        & $python src/governance/evaluate_fairness_and_explainability.py
    }
    "promote" {
        Write-Host "==> Evaluating Champion vs Challenger in MLflow Registry..." -ForegroundColor Green
        & $python src/governance/promote_model.py
    }
    "serve" {
        Write-Host "==> Starting FastAPI inference service on port 8000..." -ForegroundColor Green
        & $python -m uvicorn src.inference.app:app --host 0.0.0.0 --port 8000 --reload
    }
    "simulate-drift" {
        Write-Host "==> Generating drifted production transaction stream..." -ForegroundColor Yellow
        & $python src/data/generate_synthetic_transactions.py --samples 5000 --inject-drift
        Write-Host "==> Running Evidently AI data drift detector..." -ForegroundColor Yellow
        & $python src/monitoring/drift_detector.py
    }
    "test" {
        Write-Host "==> Running pytest test suite..." -ForegroundColor Green
        & $pytest -v tests/
    }
    "security-scan" {
        Write-Host "==> Running Aqua Security Trivy filesystem scan on dependencies..." -ForegroundColor Green
        & $trivy fs --severity HIGH,CRITICAL --scanners vuln .
        Write-Host "==> Scanning Terraform configuration for misconfigurations..." -ForegroundColor Green
        & $trivy config terraform/
    }
    "local-up" {
        Write-Host "==> Starting local Docker Compose services..." -ForegroundColor Green
        docker compose up -d
        Write-Host "==> Local Stack Services Available:" -ForegroundColor Cyan
        Write-Host "    - MLflow Tracking:    http://localhost:5000"
        Write-Host "    - MinIO S3 Console:   http://localhost:9001 (minioadmin / minioadminpassword)"
        Write-Host "    - Grafana Dashboards: http://localhost:3000 (admin / admin)"
        Write-Host "    - Prometheus Metrics: http://localhost:9090"
        Write-Host "    - FastAPI Inference:  http://localhost:8000"
        Write-Host "    - Evidently Reports:  http://localhost:8001"
    }
    "local-down" {
        Write-Host "==> Stopping local Docker Compose services..." -ForegroundColor Yellow
        docker compose down
    }
    "tf-plan" {
        Set-Location terraform
        & $terraform init
        & $terraform plan -out=tfplan
        Set-Location ..
    }
    "tf-apply" {
        Set-Location terraform
        & $terraform apply -auto-approve tfplan
        Set-Location ..
    }
    "tf-destroy" {
        Set-Location terraform
        & $terraform destroy -auto-approve
        Set-Location ..
    }
    default {
        Write-Warning "Unknown command '$Command'. Run '.\scripts\manage.ps1 help' for available commands."
    }
}
