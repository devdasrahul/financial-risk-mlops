# Master AI Build Prompt: Closed-Loop Financial Risk MLOps & Agentic Compliance Platform on AWS

Copy and paste the master prompt below into an AI coding assistant (such as Cursor, Claude Code, Windsurf, GitHub Copilot, or Gemini) to generate the complete, production-grade portfolio repository tailored for the **UST MLOps Engineer (ML Engineer I — Job ID: 634702)** role.

## Part 1: Master System & Architecture Prompt (Copy-Paste Block)

```
You are a Principal MLOps Engineer and AWS Cloud Solutions Architect. Build a complete, production-ready, end-to-end MLOps portfolio repository titled:
"Closed-Loop Financial Risk MLOps Platform with Automated Drift Retraining & Agentic Audit Triage on AWS"

### 1. Project Objective & Real-World Problem Solved
Financial institutions face four critical production failures with fraud detection models:
1. Silent model degradation caused by covariate shift and evolving fraud patterns (data and concept drift).
2. Black-box predictions that fail regulatory compliance (AML/KYC) because they lack feature-level explainability and policy citations.
3. Fragile, manual handoffs between Data Science notebooks and DevOps infrastructure.
4. Security vulnerabilities in container images, over-privileged IAM roles, and unverified model fairness.

Build an automated, self-healing MLOps and LLMOps platform that:
- Ingests and validates payment transaction data using Apache Airflow and Kubeflow-compatible modular components.
- Orchestrates cloud training, hyperparameter tuning, SHAP explainability, and pre-deployment bias auditing via AWS Step Functions and Amazon SageMaker (with a local execution mode using Docker Compose).
- Tracks experiments, model lineage, and Champion/Challenger stage aliases (`@staging`, `@production`) in MLflow Model Registry backed by Amazon S3.
- Serves low-latency real-time fraud predictions and top SHAP risk factors via a containerized FastAPI microservice deployed on Amazon ECS (AWS Fargate) behind an Application Load Balancer.
- Routes high-risk flagged transactions (`fraud_probability >= 0.85`) to an AI compliance agent built with Amazon Bedrock AgentCore, AgentCore Gateway, and AgentCore Policy guardrails that retrieves AML/KYC regulatory rules via RAG and generates an auditable compliance decision memo.
- Continuously monitors live inference logs for data drift (Population Stability Index and Wasserstein distance) with Evidently AI and operational health with Prometheus and Grafana, emitting an Amazon EventBridge event when drift exceeds thresholds to automatically trigger the AWS Step Functions retraining pipeline.
- Provisions all AWS resources declaratively with Terraform and enforces shift-left DevSecOps in GitHub Actions using Aqua Security Trivy to scan Python dependencies, Docker images, and Terraform configurations before deployment.

### 2. Target Repository Structure
Generate the complete project following this exact directory layout:

- `README.md` (Architecture diagrams in Mermaid, quickstart guide, AWS teardown instructions, and governance documentation)
- `Makefile` (Commands: `make setup`, `make local-up`, `make train`, `make serve`, `make simulate-drift`, `make test`, `make security-scan`, `make tf-plan`, `make tf-apply`, `make tf-destroy`)
- `docker-compose.yml` (Local stack: MLflow Tracking Server, PostgreSQL backend store, MinIO S3-compatible artifact store, FastAPI Inference Service, Evidently Monitoring Service, Prometheus, and Grafana)
- `pyproject.toml` & `requirements.txt` (Pinned dependencies for Python 3.11)
- `airflow/dags/fraud_feature_pipeline_dag.py` (Batch/micro-batch ingestion, schema validation, and rolling window feature engineering)
- `src/data/generate_synthetic_transactions.py` (Generates realistic baseline training data and drifted production transaction streams with controlled covariate shift)
- `src/data/preprocess.py` (Feature transformations, velocity aggregations, and train/validation/reference split generation)
- `src/training/train_xgboost.py` (XGBoost training script integrated with MLflow Tracking, logging PR-AUC, F1, ROC-AUC, confusion matrix, SHAP summary artifacts, and dataset hashes)
- `src/governance/evaluate_fairness_and_explainability.py` (SageMaker Clarify & SHAP pre-deployment gate calculating Disparate Impact, Demographic Parity Difference, and global/local SHAP attributions, exporting an audit-ready Model Card JSON/Markdown)
- `src/governance/promote_model.py` (Champion vs. Challenger comparison script that promotes the candidate model to `@production` in MLflow Model Registry only if F1 >= Champion F1 and fairness thresholds pass)
- `src/inference/app.py` (Async FastAPI service with `/predict`, `/health`, `/model-info`, and `/metrics` Prometheus endpoints, dynamic `@production` model loading from MLflow, local SHAP calculation per request, and async background logging of inference payloads to S3)
- `src/inference/schemas.py` (Strict Pydantic v2 request/response contracts with field validation)
- `src/inference/Dockerfile` (Multi-stage, non-root, hardened Python 3.11-slim container image)
- `src/agentcore/compliance_triage_agent.py` (Amazon Bedrock AgentCore workflow that accepts high-risk transaction payloads + top 3 SHAP features, queries a local/Bedrock RAG knowledge base of AML/KYC policies via AgentCore Gateway, and produces a structured JSON/Markdown SAR compliance report)
- `src/agentcore/agentcore_policy.json` (Declarative AgentCore Policy and IAM least-privilege guardrails restricting agent tool calls and preventing PII leakage or prompt injection)
- `src/agentcore/knowledge_base/aml_kyc_policies.md` (Sample financial regulatory compliance guidelines for RAG retrieval)
- `src/monitoring/drift_detector.py` (Evidently AI job comparing live S3 inference logs against reference training distributions; outputs HTML/JSON drift reports and publishes a `FraudModelDriftDetected` event to Amazon EventBridge when PSI > 0.20 or >25% of features drift)
- `src/monitoring/prometheus.yml` & `src/monitoring/grafana_dashboard.json` (Scrape configs and dashboard tracking p95 latency, request throughput, fraud flag rate, and feature drift score)
- `statemachine/mlops_retraining_pipeline.asl.json` (AWS Step Functions Amazon States Language definition chaining Data Validation -> SageMaker Training -> Clarify Bias & Explainability Check -> Choice Gate -> Model Registration -> ECS Fargate Service Update -> SNS/Slack Notification, with exponential backoff `Retry` and `Catch` rollback blocks)
- `terraform/main.tf`, `terraform/variables.tf`, `terraform/outputs.tf`, `terraform/iam.tf`, `terraform/ecs.tf`, `terraform/step_functions.tf`, `terraform/eventbridge.tf` (Modular Terraform configuration for S3, ECR, ECS Fargate, ALB, Step Functions, EventBridge rule, CloudWatch log groups, and least-privilege IAM roles)
- `.github/workflows/ci_cd_devsecops.yml` (GitHub Actions workflow running `ruff`, `pytest`, `trivy fs`, `trivy config` on `./terraform`, Docker build, `trivy image` failing on CRITICAL/HIGH CVEs, ECR push, and Step Functions / ECS deployment)
- `tests/test_inference.py`, `tests/test_preprocessing.py`, `tests/test_governance_gate.py` (Comprehensive `pytest` suite)

### 3. Detailed Implementation Requirements & Coding Standards

#### A. Code Quality & Production Realism
- Write clean, modular, strongly typed Python 3.11 code with docstrings, `logging`, and explicit error handling.
- Never use `TODO` stubs or empty pass-through functions. Every script must be runnable out of the box.
- Provide a dual-mode configuration (`ENV=local` vs `ENV=aws`) via environment variables so the entire pipeline runs locally for free using `docker-compose` and synthetic data, while seamlessly targeting real AWS services (`boto3` for S3, Step Functions, SageMaker, ECS, EventBridge, and Bedrock AgentCore) when AWS credentials are provided.

#### B. Feature Engineering & Model Training (`src/data/` & `src/training/`)
- Use realistic payment fraud features: `transaction_amount`, `card_velocity_1h`, `card_velocity_24h`, `distance_from_home_km`, `merchant_risk_score`, `is_international`, `device_trust_score`, `account_age_days`, and a protected attribute `customer_age_group` (used strictly by the governance gate to test for demographic bias).
- In `train_xgboost.py`, handle class imbalance via `scale_pos_weight`, log all metrics (`f1_score`, `pr_auc`, `roc_auc`, `precision`, `recall`, `false_positive_rate`) and SHAP summary values to MLflow, and register the model under `fraud-detection-xgb`.

#### C. Governance & Responsible AI Gate (`src/governance/`)
- Calculate both predictive performance and fairness metrics: Disparate Impact Ratio (`DI >= 0.80`) and Statistical Parity Difference (`|SPD| <= 0.10`) across `customer_age_group`, mirroring [Amazon SageMaker Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-configure-processing-jobs.html).
- Generate a structured Model Card artifact (`model_card.json` and `model_card.md`) modeled after [Amazon SageMaker Model Cards](https://docs.aws.amazon.com/sagemaker/latest/dg/model-cards.html) containing model lineage, intended use, training dataset hash, threshold configuration, fairness evaluation results, and top global SHAP drivers.
- Fail the pipeline with a non-zero exit code if either the F1 threshold (`< 0.82`) or fairness gate (`DI < 0.80`) is violated.

#### D. Agentic RAG Compliance Layer (`src/agentcore/`)
- Implement `ComplianceTriageAgent` following the [Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html) pattern:
  1. Receive the flagged transaction ID, fraud probability, and top 3 local SHAP feature contributions from the FastAPI inference service.
  2. Query the AML/KYC policy knowledge base (`aml_kyc_policies.md`) to retrieve the matching regulatory rule (e.g., FinCEN High-Velocity Wire/Card Structuring Rule, Cross-Border Anomaly Policy).
  3. Enforce [Amazon Bedrock AgentCore Policy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy.html) checks (masking PII such as raw card numbers before invoking the model and verifying allowed tool actions).
  4. Return a deterministic, auditable JSON compliance report with `transaction_id`, `risk_tier`, `shap_explanation`, `cited_policy_section`, `recommended_action`, and `trace_id`. Include a local fallback mode so tests run without active Bedrock charges.

#### E. Drift Detection & Event-Driven Retraining (`src/monitoring/` & `statemachine/`)
- In `drift_detector.py`, use [Evidently AI](https://docs.evidentlyai.com/metrics/preset_data_drift) (`DataDriftPreset` and `TargetDriftPreset`) to compare `reference_data.parquet` against `production_inference_logs.parquet`.
- Provide a CLI command (`python src/data/generate_synthetic_transactions.py --inject-drift`) that simulates a sudden surge in cross-border high-velocity fraud so the drift detector trips during a live demo and emits the [Amazon EventBridge](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-what-is.html) payload to trigger [AWS Step Functions](https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html).

### 4. Output Delivery Plan
Because this is a complete multi-file engineering system, generate the files systematically in 5 sequential phases. Start immediately with **Phase 1** in full detail without skipping any code:
- **Phase 1:** Repository scaffolding (`README.md`, `Makefile`, `pyproject.toml`, `requirements.txt`, `docker-compose.yml`) and synthetic data + Airflow DAG (`src/data/`, `airflow/dags/`).
- **Phase 2:** Model training, MLflow Registry integration, SHAP explainability, and SageMaker Clarify governance gate (`src/training/`, `src/governance/`).
- **Phase 3:** FastAPI inference microservice, Prometheus metrics, Dockerfile, and Amazon Bedrock AgentCore RAG compliance triage module (`src/inference/`, `src/agentcore/`).
- **Phase 4:** Evidently AI drift detector, EventBridge trigger, and AWS Step Functions ASL state machine (`src/monitoring/`, `statemachine/`).
- **Phase 5:** Modular Terraform IaC (`terraform/`), GitHub Actions DevSecOps Trivy pipeline (`.github/workflows/`), and `pytest` unit test suite (`tests/`).

```

## Part 2: Follow-Up Prompts for Each Phase

If your AI coding assistant completes Phase 1 and waits for your next message, copy and paste these exact continuation prompts to generate every remaining file completely.

### Phase 2 Continuation Prompt (Training, MLflow & Governance)

```
Proceed to Phase 2. Generate the complete, production-ready code for:
1. `src/training/train_xgboost.py`: Full XGBoost training pipeline with MLflow Tracking (`mlflow.xgboost.log_model`), hyperparameter logging, PR-AUC/F1 evaluation, and SHAP explainer serialization.
2. `src/governance/evaluate_fairness_and_explainability.py`: Pre-deployment governance gate implementing SageMaker Clarify-style Disparate Impact (`DI >= 0.80`) and Statistical Parity Difference checks across `customer_age_group`, plus automated generation of `model_card.json` and `model_card.md`.
3. `src/governance/promote_model.py`: Champion vs. Challenger comparison script using `MlflowClient` that promotes the new model version to the `@production` alias only if it outperforms the current production model and passes the governance gate.
Write complete, runnable Python files with no placeholders.

```

### Phase 3 Continuation Prompt (FastAPI Serving & AgentCore Triage)

```
Proceed to Phase 3. Generate the complete, production-ready code for:
1. `src/inference/schemas.py`: Pydantic v2 schemas for `TransactionRequest`, `ShapContribution`, `ComplianceAuditSummary`, and `PredictionResponse`.
2. `src/inference/app.py`: FastAPI inference server that loads the `@production` model and SHAP explainer from MLflow, exposes `/predict`, `/health`, `/model-info`, and `/metrics` (Prometheus counter and latency histogram), logs predictions asynchronously to S3/MinIO, and invokes the AgentCore compliance agent when `fraud_probability >= 0.85`.
3. `src/inference/Dockerfile`: Hardened, multi-stage Python 3.11-slim Dockerfile running as a non-root user with a built-in `HEALTHCHECK`.
4. `src/agentcore/knowledge_base/aml_kyc_policies.md`, `src/agentcore/agentcore_policy.json`, and `src/agentcore/compliance_triage_agent.py`: Amazon Bedrock AgentCore compliance agent with PII masking guardrails, RAG policy retrieval over `aml_kyc_policies.md`, structured JSON audit output, and execution tracing.

```

### Phase 4 Continuation Prompt (Drift Monitoring & Step Functions Orchestration)

```
Proceed to Phase 4. Generate the complete, production-ready code for:
1. `src/monitoring/drift_detector.py`: Evidently AI drift monitoring script that compares `reference_data.parquet` against live inference logs, computes Population Stability Index (PSI) and feature drift share, saves an HTML/JSON report to S3, and emits a `FraudModelDriftDetected` event to Amazon EventBridge via `boto3` when drift exceeds thresholds.
2. `src/monitoring/prometheus.yml` and `src/monitoring/grafana_dashboard.json`: Observability configurations for tracking inference latency, HTTP error rates, fraud flag rates, and model drift status.
3. `statemachine/mlops_retraining_pipeline.asl.json`: Complete AWS Step Functions Amazon States Language (ASL) JSON state machine orchestrating Data Validation -> SageMaker Training Job -> SageMaker Clarify Bias & Explainability Evaluation -> Choice State (Fairness & F1 Gate) -> Model Registry Promotion -> Amazon ECS Fargate Rolling Deployment -> Failure/Success Notification, complete with `Retry` and `Catch` blocks.

```

### Phase 5 Continuation Prompt (Terraform IaC, DevSecOps CI/CD & Tests)

```
Proceed to Phase 5. Generate the complete, production-ready code for:
1. `terraform/main.tf`, `terraform/variables.tf`, `terraform/iam.tf`, `terraform/ecs.tf`, `terraform/step_functions.tf`, `terraform/eventbridge.tf`, and `terraform/outputs.tf`: Modular Terraform AWS configuration provisioning the S3 buckets, ECR repository with immutable tags, ECS Fargate cluster and service with Application Load Balancer, Step Functions state machine, EventBridge drift rule, and least-privilege IAM roles.
2. `.github/workflows/ci_cd_devsecops.yml`: GitHub Actions CI/CD workflow running `ruff` linting, `pytest` unit tests, `trivy fs` dependency scanning, `trivy config` Terraform security scanning, Docker build, `trivy image` vulnerability gate (failing on HIGH/CRITICAL CVEs), Amazon ECR push, and ECS/Step Functions deployment.
3. `tests/test_preprocessing.py`, `tests/test_governance_gate.py`, and `tests/test_inference.py`: Complete `pytest` unit and integration tests validating feature engineering, fairness threshold enforcement, PII masking in the AgentCore module, and `/predict` endpoint responses.

```

## Part 3: Quick Reference Matrix of Prompt Coverage vs. UST Job ID 634702

| **UST Job Requirement** | **Where the Prompt Implements It** | **Interview Artifact Produced** | 
| **Automate model training, testing, deployment & monitoring** | `statemachine/mlops_retraining_pipeline.asl.json` & `.github/workflows/ci_cd_devsecops.yml` | Visual Step Functions workflow & automated GitHub Actions pipeline | 
| **Model versioning, experiment tracking & model registry** | `src/training/train_xgboost.py` & `src/governance/promote_model.py` | [MLflow Tracking](https://www.google.com/search?q=https%3A%2F%2Fmlflow.org%2Fdocs%2Flatest%2Fml%2Ftracking%2F) UI & `@production` Champion/Challenger registry | 
| **Monitor model performance, drift & operational health** | `src/monitoring/drift_detector.py` & `src/monitoring/prometheus.yml` | [Evidently AI](https://www.google.com/search?q=https%3A%2F%2Fdocs.evidentlyai.com%2Fmetrics%2Fpreset_data_drift) HTML drift report, [Prometheus](https://www.google.com/search?q=https%3A%2F%2Fprometheus.io%2Fdocs%2Fintroduction%2Foverview%2F) metrics & [Amazon EventBridge](https://www.google.com/search?q=https%3A%2F%2Fdocs.aws.amazon.com%2Feventbridge%2Flatest%2Fuserguide%2Feb-what-is.html) rule | 
| **Establish governance, security, access control & auditability** | `src/governance/evaluate_fairness_and_explainability.py` & `terraform/iam.tf` | [Amazon SageMaker Clarify](https://www.google.com/search?q=https%3A%2F%2Fdocs.aws.amazon.com%2Fsagemaker%2Flatest%2Fdg%2Fclarify-configure-processing-jobs.html) bias gate, [Amazon SageMaker Model Cards](https://www.google.com/search?q=https%3A%2F%2Fdocs.aws.amazon.com%2Fsagemaker%2Flatest%2Fdg%2Fmodel-cards.html) & least-privilege IAM | 
| **Hands-on AWS: SageMaker, Step Functions, ECS & Docker** | `src/inference/Dockerfile`, `terraform/ecs.tf` & `terraform/step_functions.tf` | Hardened Docker container running on [Amazon ECS](https://www.google.com/search?q=https%3A%2F%2Fdocs.aws.amazon.com%2FAmazonECS%2Flatest%2Fdeveloperguide%2FWelcome.html) ([AWS Fargate](https://www.google.com/search?q=https%3A%2F%2Fdocs.aws.amazon.com%2FAmazonECS%2Flatest%2Fdeveloperguide%2FAWS_Fargate.html)) | 
| **Terraform IaC & DevSecOps CI/CD** | `terraform/*.tf` & `.github/workflows/ci_cd_devsecops.yml` | Declarative [Terraform AWS Provider](https://www.google.com/search?q=https%3A%2F%2Fregistry.terraform.io%2Fproviders%2Fhashicorp%2Faws%2Flatest%2Fdocs) modules & [Aqua Security Trivy](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Faquasecurity%2Ftrivy) security gates | 
| **Preferred: AgentCore, Generative AI, LLMOps & RAG** | `src/agentcore/compliance_triage_agent.py` & `src/agentcore/agentcore_policy.json` | [Amazon Bedrock AgentCore](https://www.google.com/search?q=https%3A%2F%2Fdocs.aws.amazon.com%2Fbedrock-agentcore%2Flatest%2Fdevguide%2Fwhat-is-bedrock-agentcore.html) RAG audit agent with [AgentCore Policy](https://www.google.com/search?q=https%3A%2F%2Fdocs.aws.amazon.com%2Fbedrock-agentcore%2Flatest%2Fdevguide%2Fpolicy.html) guardrails | 
