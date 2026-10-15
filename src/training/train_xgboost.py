"""Production XGBoost Model Training Pipeline with MLflow Tracking & SHAP Explainability.

Trains a fraud risk classifier handling extreme class imbalance via scale_pos_weight,
logs hyperparameters, PR-AUC, F1, ROC-AUC metrics, and registers the model artifact
to MLflow Model Registry under 'fraud-detection-xgb'.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_xgboost")

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
TARGET_COLUMN = "is_fraud"


def is_port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    """Fast socket probe to check if tracking server port is accepting connections."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def configure_mlflow() -> str:
    """Configure MLflow tracking URI based on environment (local fallback vs cloud/Docker)."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        try:
            parts = tracking_uri.replace("http://", "").replace("https://", "").split(":")
            host = parts[0]
            port = int(parts[1].split("/")[0]) if len(parts) > 1 else 80
            if not is_port_listening(host, port, timeout=0.4):
                local_dir = Path("mlruns").resolve().as_uri()
                mlflow.set_tracking_uri(local_dir)
                logger.info(f"Port {port} not listening. Using local file-based tracking store: {local_dir}")
                return local_dir
        except Exception:
            pass

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.search_experiments(max_results=1)
        logger.info(f"Connected to MLflow Tracking Server at: {tracking_uri}")
    except Exception as e:
        local_dir = Path("mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(local_dir)
        logger.warning(
            f"Unable to reach MLflow server at {tracking_uri} ({e}). "
            f"Fell back to local file-based tracking store: {local_dir}"
        )
    return mlflow.get_tracking_uri()


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.50) -> dict[str, float]:
    """Compute comprehensive performance metrics tailored for imbalanced fraud detection."""
    # Find optimal threshold maximizing F1 on validation distribution
    best_thresh = threshold
    best_f1 = -1.0
    for t in np.linspace(0.25, 0.75, 21):
        f = float(f1_score(y_true, (y_pred_prob >= t).astype(int), zero_division=0))
        if f > best_f1:
            best_f1 = f
            best_thresh = float(t)

    eval_thresh = best_thresh if best_f1 > 0 else threshold
    y_pred = (y_pred_prob >= eval_thresh).astype(int)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_pred_prob))
    roc_auc = float(roc_auc_score(y_true, y_pred_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "optimal_threshold": round(eval_thresh, 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def train_model(
    train_path: str = "data/train.parquet",
    val_path: str = "data/validation.parquet",
    model_dir: str = "models",
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.08,
) -> dict[str, Any]:
    """Train XGBoost fraud detection model with MLflow experiment tracking."""
    configure_mlflow()
    mlflow.set_experiment("fraud-detection-risk-engine")

    train_file = Path(train_path)
    val_file = Path(val_path)

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(f"Training datasets missing. Please run data preprocessing first.")

    logger.info(f"Loading datasets: {train_file} and {val_file}")
    df_train = pd.read_parquet(train_file)
    df_val = pd.read_parquet(val_file)

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN].values
    X_val = df_val[FEATURE_COLUMNS]
    y_val = df_val[TARGET_COLUMN].values

    # Handle extreme class imbalance dynamically
    num_neg = int(np.sum(y_train == 0))
    num_pos = int(np.sum(y_train == 1))
    scale_pos_weight = float(num_neg / max(1, num_pos))
    logger.info(f"Class imbalance calculation: Negatives={num_neg:,}, Positives={num_pos:,}, scale_pos_weight={scale_pos_weight:.2f}")

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "eval_metric": "aucpr",
        "random_state": 42,
        "objective": "binary:logistic",
        "n_jobs": -1,
    }

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="xgboost_fraud_champion_run") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow Run ID: {run_id}")

        mlflow.log_params(params)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # Evaluate probabilities
        val_probs = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, val_probs)
        mlflow.log_metrics(metrics)

        logger.info(
            f"Validation Results: F1={metrics['f1_score']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | "
            f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f}"
        )

        # Compute TreeSHAP Explainer for Model Explainability
        logger.info("Computing TreeSHAP values for local and global explainability...")
        explainer = shap.TreeExplainer(model)
        sample_val = X_val.sample(min(500, len(X_val)), random_state=42)
        shap_values = explainer.shap_values(sample_val)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        global_shap_importance = {
            col: float(val)
            for col, val in sorted(zip(FEATURE_COLUMNS, mean_abs_shap), key=lambda x: x[1], reverse=True)
        }
        logger.info(f"Top 3 Global SHAP Features: {list(global_shap_importance.keys())[:3]}")

        # Save artifacts locally
        model_artifact_path = out_dir / "model.json"
        explainer_artifact_path = out_dir / "explainer.pkl"
        metrics_artifact_path = out_dir / "evaluation_metrics.json"
        shap_summary_path = out_dir / "global_shap_importance.json"

        model.save_model(str(model_artifact_path))
        with open(explainer_artifact_path, "wb") as f:
            pickle.dump(explainer, f)
        with open(metrics_artifact_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(shap_summary_path, "w", encoding="utf-8") as f:
            json.dump(global_shap_importance, f, indent=2)

        # Log artifacts and model to MLflow
        mlflow.log_artifact(str(metrics_artifact_path))
        mlflow.log_artifact(str(shap_summary_path))

        # Register model under 'fraud-detection-xgb'
        try:
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                registered_model_name="fraud-detection-xgb",
            )
            logger.info("Registered model artifact to MLflow Model Registry as 'fraud-detection-xgb'.")
        except Exception as e:
            logger.warning(f"Could not register to remote MLflow Registry (falling back to local artifact log): {e}")
            mlflow.xgboost.log_model(xgb_model=model, artifact_path="model")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "global_shap_importance": global_shap_importance,
        "model_path": str(model_artifact_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model with MLflow Tracking.")
    parser.add_argument("--train-path", type=str, default="data/train.parquet", help="Path to training set")
    parser.add_argument("--val-path", type=str, default="data/validation.parquet", help="Path to validation set")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for model artifacts")
    parser.add_argument("--estimators", type=int, default=150, help="Number of boosting rounds")
    parser.add_argument("--depth", type=int, default=5, help="Max tree depth")
    parser.add_argument("--lr", type=float, default=0.08, help="Learning rate")
    args = parser.parse_args()

    train_model(
        train_path=args.train_path,
        val_path=args.val_path,
        model_dir=args.model_dir,
        n_estimators=args.estimators,
        max_depth=args.depth,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
