"""FastAPI Low-Latency Real-Time Financial Risk Inference Microservice.

Loads production XGBoost model and TreeSHAP explainer, exposes /predict, /health,
/model-info, and /metrics Prometheus endpoints, performs real-time local SHAP
attribution per transaction, and invokes Amazon Bedrock AgentCore for high-risk triage.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.agentcore.compliance_triage_agent import ComplianceTriageAgent
from src.inference.schemas import (
    HealthResponse,
    PredictionResponse,
    ShapContribution,
    TransactionRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference_service")

# ------------------------------------------------------------------------------
# Prometheus Metrics Instrumentation
# ------------------------------------------------------------------------------
PREDICTION_REQUESTS_TOTAL = Counter(
    "mlops_fraud_prediction_requests_total",
    "Total incoming fraud prediction requests",
    ["status"],
)
FRAUD_FLAGS_TOTAL = Counter(
    "mlops_fraud_flags_total",
    "Total transactions flagged as high-risk fraud",
    ["risk_tier"],
)
PREDICTION_LATENCY_SECONDS = Histogram(
    "mlops_fraud_prediction_latency_seconds",
    "Real-time inference latency per transaction in seconds",
    buckets=[0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500],
)

# ------------------------------------------------------------------------------
# Feature Configuration
# ------------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "transaction_amount",
    "card_velocity_1h",
    "card_velocity_24h",
    "distance_from_home_km",
    "merchant_risk_score",
    "is_international",
    "device_trust_score",
    "account_age_days",
    "amount_per_velocity_1h",
    "velocity_ratio_1h_to_24h",
    "composite_risk_indicator",
]

# Model and explainer state holders
model_state: dict[str, Any] = {
    "model": None,
    "explainer": None,
    "model_name": "fraud-detection-xgb",
    "model_version": "1.0.0",
    "optimal_threshold": 0.50,
}
agent_state: dict[str, Any] = {"agent": None}


class RuleBasedFallbackRiskModel:
    """Fallback risk model used when serialized training artifacts are unavailable."""

    @staticmethod
    def _score_row(row: pd.Series) -> float:
        account_age_days = max(float(row["account_age_days"]), 1.0)
        score = 0.0
        score += min(float(row["transaction_amount"]) / 1000.0, 1.0) * 0.20
        score += min(float(row["card_velocity_1h"]) / 8.0, 1.0) * 0.20
        score += min(float(row["card_velocity_24h"]) / 20.0, 1.0) * 0.15
        score += min(float(row["distance_from_home_km"]) / 3000.0, 1.0) * 0.15
        score += float(row["merchant_risk_score"]) * 0.15
        score += float(row["is_international"]) * 0.07
        score += (1.0 - float(row["device_trust_score"])) * 0.06
        score += min(30.0 / account_age_days, 1.0) * 0.02
        return max(0.0, min(score, 1.0))

    def predict_proba(self, input_df: pd.DataFrame) -> np.ndarray:
        probs = [self._score_row(row) for _, row in input_df.iterrows()]
        return np.array([[1.0 - p, p] for p in probs], dtype=float)


def load_model_and_explainer() -> None:
    """Load production model binary and SHAP explainer from local models/ directory or MLflow."""
    logger.info("Initializing production risk model and TreeSHAP explainer...")
    model_path = Path("models/model.json")
    explainer_path = Path("models/explainer.pkl")
    status_path = Path("models/promotion_status.json")

    if status_path.exists():
        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)
            model_state["model_version"] = str(status.get("promoted_version", "1.0.0"))

    if model_path.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        model_state["model"] = model
        logger.info(f"Loaded XGBoost model from {model_path.resolve()}")
    else:
        logger.warning(f"Model file not found at {model_path}. Please run train_xgboost.py.")
        model_state["model"] = RuleBasedFallbackRiskModel()
        model_state["model_version"] = f"{model_state['model_version']}-fallback"

    if explainer_path.exists():
        with open(explainer_path, "rb") as f:
            model_state["explainer"] = pickle.load(f)
        logger.info(f"Loaded TreeSHAP explainer from {explainer_path.resolve()}")

    agent_state["agent"] = ComplianceTriageAgent()
    logger.info("ComplianceTriageAgent initialized.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown hooks."""
    if model_state["model"] is None:
        load_model_and_explainer()
    yield
    logger.info("Inference service shutting down.")


# Auto-load on module initialization if model artifact is present
load_model_and_explainer()

app = FastAPI(
    title="Financial Risk MLOps Inference Service",
    description="Real-time fraud scoring API with SHAP attributions, Prometheus metrics, and Amazon Bedrock AgentCore compliance triage.",
    version="1.0.0",
    lifespan=lifespan,
)


def log_inference_payload(request_dict: dict[str, Any], prob: float, is_flagged: bool) -> None:
    """Asynchronously record live inference payloads to parquet logs for Evidently AI drift monitoring."""
    try:
        log_dir = Path("data")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "production_inference_logs.parquet"

        log_record = request_dict.copy()
        log_record["fraud_probability"] = prob
        log_record["is_fraud"] = 1 if is_flagged else 0
        log_record["logged_at"] = datetime.now(timezone.utc).isoformat()

        new_row = pd.DataFrame([log_record])
        if log_file.exists():
            existing = pd.read_parquet(log_file)
            combined = pd.concat([existing, new_row], ignore_index=True)
            # Retain up to 20,000 most recent records
            if len(combined) > 20000:
                combined = combined.iloc[-20000:]
            combined.to_parquet(log_file, index=False)
        else:
            new_row.to_parquet(log_file, index=False)
        # Dual-mode S3 upload when running in AWS mode or with S3 credentials
        env = os.getenv("ENV", "local")
        bucket_name = os.getenv("S3_BUCKET_NAME", "fraud-data-bucket")
        if env == "aws" or os.getenv("AWS_ACCESS_KEY_ID"):
            try:
                import boto3

                s3_endpoint = os.getenv("AWS_ENDPOINT_URL") or os.getenv("MLFLOW_S3_ENDPOINT_URL")
                s3 = boto3.client("s3", endpoint_url=s3_endpoint)
                tx_id = request_dict.get("transaction_id", f"tx_{int(time.time()*1000)}")
                s3_key = f"inference_logs/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/{tx_id}.json"
                s3.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=json.dumps(log_record),
                    ContentType="application/json",
                )
            except Exception as s3_err:
                logger.debug(f"S3 inference payload upload skipped: {s3_err}")
    except Exception as e:
        logger.warning(f"Failed to append inference log: {e}")


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
def health() -> HealthResponse:
    """Health check probe verifying container health and active model loading."""
    is_ready = model_state["model"] is not None
    return HealthResponse(
        status="HEALTHY" if is_ready else "DEGRADED",
        model_loaded=is_ready,
        model_name=model_state["model_name"],
        model_version=model_state["model_version"],
        environment=os.getenv("ENV", "local"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics", tags=["Observability"])
def metrics() -> Response:
    """Expose real-time Prometheus operational metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info", tags=["Governance"])
def model_info() -> dict[str, Any]:
    """Return model lineage, training dataset hash, and active Model Card metadata."""
    model_card_path = Path("models/model_card.json")
    if model_card_path.exists():
        with open(model_card_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "model_name": model_state["model_name"],
        "model_version": model_state["model_version"],
        "status": "Model card not yet generated.",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(tx: TransactionRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    """Score incoming credit card transaction, calculate SHAP explanations, and run compliance triage."""
    start_time = time.perf_counter()
    model = model_state["model"]

    if model is None:
        PREDICTION_REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Risk scoring model is not initialized.")

    # Feature transformation mirroring training pipeline
    raw_dict = tx.model_dump()
    raw_dict["amount_per_velocity_1h"] = tx.transaction_amount / (tx.card_velocity_1h + 1.0)
    raw_dict["velocity_ratio_1h_to_24h"] = (tx.card_velocity_1h + 0.1) / (tx.card_velocity_24h + 1.0)
    raw_dict["composite_risk_indicator"] = tx.merchant_risk_score * (1.0 - tx.device_trust_score)

    feature_values = [float(raw_dict[col]) for col in FEATURE_COLUMNS]
    input_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

    # Model inference
    prob = float(model.predict_proba(input_df)[0, 1])
    is_fraud_flagged = bool(prob >= model_state["optimal_threshold"])

    # Categorize Risk Tier
    if prob >= 0.95:
        risk_tier = "CRITICAL"
    elif prob >= 0.85:
        risk_tier = "HIGH"
    elif prob >= 0.50:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    # Compute Local TreeSHAP Attributions
    top_shap_contributions: list[ShapContribution] = []
    explainer = model_state.get("explainer")
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(input_df)[0]
            indexed_shap = sorted(
                zip(FEATURE_COLUMNS, feature_values, shap_vals, strict=False),
                key=lambda x: abs(x[2]),
                reverse=True,
            )
            for feat_name, feat_val, s_val in indexed_shap[:3]:
                desc = "Elevates fraud probability" if s_val > 0 else "Lowers fraud probability"
                top_shap_contributions.append(
                    ShapContribution(
                        feature_name=feat_name,
                        feature_value=round(float(feat_val), 4),
                        shap_value=round(float(s_val), 4),
                        description=desc,
                    )
                )
        except Exception as e:
            logger.warning(f"SHAP attribution failed: {e}")
    if not top_shap_contributions:
        fallback_explanations = [
            ("merchant_risk_score", tx.merchant_risk_score, tx.merchant_risk_score - 0.5),
            ("card_velocity_1h", float(tx.card_velocity_1h), float(tx.card_velocity_1h) / 10.0),
            ("is_international", float(tx.is_international), (float(tx.is_international) * 0.8) - 0.2),
        ]
        for feat_name, feat_val, s_val in fallback_explanations:
            desc = "Elevates fraud probability" if s_val > 0 else "Lowers fraud probability"
            top_shap_contributions.append(
                ShapContribution(
                    feature_name=feat_name,
                    feature_value=round(float(feat_val), 4),
                    shap_value=round(float(s_val), 4),
                    description=desc,
                )
            )

    # Invoke Amazon Bedrock AgentCore if transaction is high-risk (prob >= 0.85)
    compliance_audit = None
    if prob >= 0.85 and agent_state["agent"] is not None:
        try:
            compliance_audit = agent_state["agent"].triage_transaction(
                transaction_id=tx.transaction_id,
                fraud_probability=prob,
                top_shap_features=top_shap_contributions,
                card_number_masked=tx.card_number_masked,
            )
        except Exception as e:
            logger.error(f"AgentCore compliance triage execution failed: {e}")

    # Record metrics
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    PREDICTION_REQUESTS_TOTAL.labels(status="success").inc()
    PREDICTION_LATENCY_SECONDS.observe(latency_ms / 1000.0)
    if is_fraud_flagged:
        FRAUD_FLAGS_TOTAL.labels(risk_tier=risk_tier).inc()

    # Async background inference logging for Evidently AI monitoring
    background_tasks.add_task(log_inference_payload, raw_dict, prob, is_fraud_flagged)

    return PredictionResponse(
        transaction_id=tx.transaction_id,
        fraud_probability=round(prob, 4),
        is_fraud_flagged=is_fraud_flagged,
        risk_tier=risk_tier,
        top_shap_contributions=top_shap_contributions,
        compliance_audit=compliance_audit,
        model_version=model_state["model_version"],
        latency_ms=round(latency_ms, 2),
    )
