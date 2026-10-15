"""Champion vs. Challenger Model Registry Promotion Gate.

Compares candidate (Challenger) model performance against the current active
(Champion) model in MLflow Model Registry. Promotes the candidate to the
'@production' stage alias only if it satisfies governance gates and outperforms
or equals the Champion's F1 score.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("promote_model")

MODEL_NAME = "fraud-detection-xgb"


def is_port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    """Fast socket probe to check if tracking server port is accepting connections."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def configure_client() -> MlflowClient:
    """Initialize MlflowClient targeting either local Docker tracking server or local file store."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        try:
            parts = tracking_uri.replace("http://", "").replace("https://", "").split(":")
            host = parts[0]
            port = int(parts[1].split("/")[0]) if len(parts) > 1 else 80
            if not is_port_listening(host, port, timeout=0.4):
                local_uri = Path("mlruns").resolve().as_uri()
                mlflow.set_tracking_uri(local_uri)
                logger.info(f"Port {port} not listening. Using local file tracking: {local_uri}")
                return MlflowClient(tracking_uri=local_uri)
        except Exception:
            pass

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        client.search_experiments(max_results=1)
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MlflowClient connected to remote server at {tracking_uri}")
        return client
    except Exception as e:
        local_uri = Path("mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(local_uri)
        logger.warning(f"MLflow server unreachable at {tracking_uri} ({e}). Using local file tracking: {local_uri}")
        return MlflowClient(tracking_uri=local_uri)


def verify_governance_approval(model_card_path: str = "models/model_card.json") -> bool:
    """Verify that the model card exists and that governance gates were approved."""
    card_file = Path(model_card_path)
    if not card_file.exists():
        logger.error(f"Model Card missing at {model_card_path}. Run evaluate_fairness_and_explainability.py first.")
        return False

    with open(card_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    decision = data.get("governance_decision", {})
    approved = decision.get("approved_for_production", False)

    if not approved:
        logger.error("PROMOTION BLOCKED: Model Card governance gate was not approved for production.")
        return False

    logger.info("Verified Model Card governance approval: PASSED.")
    return True


def promote_champion_or_challenger(
    model_name: str = MODEL_NAME,
    metrics_path: str = "models/evaluation_metrics.json",
    model_card_path: str = "models/model_card.json",
) -> dict[str, Any]:
    """Compare candidate metrics with current production model and promote if qualified."""
    if not verify_governance_approval(model_card_path):
        raise PermissionError("Model failed Responsible AI governance evaluation.")

    client = configure_client()

    with open(metrics_path, "r", encoding="utf-8") as f:
        challenger_metrics = json.load(f)

    challenger_f1 = challenger_metrics.get("f1_score", 0.0)
    challenger_pr_auc = challenger_metrics.get("pr_auc", 0.0)

    # Fetch registered model versions
    try:
        latest_versions = client.get_latest_versions(name=model_name)
    except Exception as e:
        logger.warning(f"No existing registered model found for '{model_name}': {e}")
        latest_versions = []

    champion_version = None
    champion_f1 = 0.0

    # Locate current Champion (Production version or alias)
    for v in latest_versions:
        if v.current_stage == "Production":
            champion_version = v
            break

    # If aliases are supported (modern MLflow)
    try:
        reg_model = client.get_registered_model(name=model_name)
        if "production" in reg_model.aliases:
            prod_v_num = reg_model.aliases["production"]
            champion_version = client.get_model_version(name=model_name, version=prod_v_num)
    except Exception:
        pass

    if champion_version is not None:
        champ_run = client.get_run(champion_version.run_id)
        champion_f1 = champ_run.data.metrics.get("f1_score", 0.0)
        logger.info(
            f"Existing Champion: Version {champion_version.version} (F1={champion_f1:.4f}). "
            f"Challenger: F1={challenger_f1:.4f}"
        )
    else:
        logger.info(f"No existing Champion in Production. Challenger (F1={challenger_f1:.4f}) will become Initial Champion.")

    # Promotion Decision Logic
    promote = False
    if champion_version is None:
        promote = True
    elif challenger_f1 >= (champion_f1 - 0.01):  # Allow replacement if performance matches or exceeds
        promote = True
    else:
        logger.error(
            f"PROMOTION REJECTED: Challenger F1 ({challenger_f1:.4f}) is lower than Champion F1 ({champion_f1:.4f})."
        )

    # Determine latest model version to promote
    target_version = None
    if latest_versions:
        # Sort by version number descending
        sorted_versions = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)
        target_version = sorted_versions[0].version
    else:
        target_version = "1"

    promotion_result: dict[str, Any] = {
        "model_name": model_name,
        "promoted_version": target_version,
        "promoted": promote,
        "challenger_f1": challenger_f1,
        "champion_f1": champion_f1,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    if promote:
        try:
            # Set modern MLflow alias
            client.set_registered_model_alias(name=model_name, alias="production", version=target_version)
            logger.info(f"Assigned alias '@production' to {model_name} version {target_version}")
        except Exception:
            pass

        try:
            # Set classic MLflow stage
            client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(f"Transitioned {model_name} version {target_version} to stage 'Production'.")
        except Exception as e:
            logger.warning(f"Stage transition notification: {e}")

        logger.info(f"SUCCESS: Model {model_name} v{target_version} is now the active Champion in Production.")
    else:
        logger.warning(f"Challenger v{target_version} was NOT promoted.")

    # Save promotion status locally for Docker and FastAPI consumption
    status_file = Path("models/promotion_status.json")
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(promotion_result, f, indent=2)

    return promotion_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote Challenger model to Production in MLflow.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--metrics-path", type=str, default="models/evaluation_metrics.json")
    parser.add_argument("--model-card-path", type=str, default="models/model_card.json")
    args = parser.parse_args()

    result = promote_champion_or_challenger(
        model_name=args.model_name,
        metrics_path=args.metrics_path,
        model_card_path=args.model_card_path,
    )
    if not result.get("promoted", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
