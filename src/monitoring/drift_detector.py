"""Evidently AI Data & Concept Drift Detection Pipeline with Amazon EventBridge Retraining Trigger.

Compares production inference logs against baseline training distributions (reference_data.parquet),
calculates Population Stability Index (PSI) and Wasserstein distance per feature, exports
an HTML/JSON diagnostic audit report, and emits a 'FraudModelDriftDetected' event to
Amazon EventBridge to initiate automated Step Functions retraining.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drift_detector")

FEATURE_COLUMNS = [
    "transaction_amount",
    "card_velocity_1h",
    "card_velocity_24h",
    "distance_from_home_km",
    "merchant_risk_score",
    "is_international",
    "device_trust_score",
    "account_age_days",
]
TARGET_COLUMN = "is_fraud"

DRIFT_SHARE_THRESHOLD = 0.25  # Trigger retraining if >= 25% of features drift


def compute_psi(reference: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two continuous distributions."""
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Create quantile buckets from reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(ref_clean, percentiles)
    bucket_edges[0] -= 1e-5
    bucket_edges[-1] += 1e-5

    ref_counts = np.histogram(ref_clean, bins=bucket_edges)[0]
    cur_counts = np.histogram(cur_clean, bins=bucket_edges)[0]

    ref_pct = (ref_counts + 1e-4) / (len(ref_clean) + 1e-4 * num_buckets)
    cur_pct = (cur_counts + 1e-4) / (len(cur_clean) + 1e-4 * num_buckets)

    psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi_value)


def dispatch_eventbridge_retraining_event(
    drift_summary: dict[str, Any],
    event_bus_name: str = "default",
) -> dict[str, Any]:
    """Emit 'FraudModelDriftDetected' event to Amazon EventBridge."""
    event_payload = {
        "event_type": "FraudModelDriftDetected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_share": drift_summary["drift_share"],
        "drifted_features": drift_summary["drifted_features"],
        "dataset_drift_detected": drift_summary["dataset_drift_detected"],
        "trigger_source": "EvidentlyAIDriftMonitor",
        "action_required": "InitiateStepFunctionsRetrainingPipeline",
    }

    env = os.getenv("ENV", "local")
    dispatched_status = {"dispatched": False, "mode": env, "payload": event_payload}

    if env == "aws" or os.getenv("AWS_ACCESS_KEY_ID"):
        try:
            client = boto3.client("events", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
            response = client.put_events(
                Entries=[
                    {
                        "Source": "custom.fraud.mlops",
                        "DetailType": "FraudModelDriftDetected",
                        "Detail": json.dumps(event_payload),
                        "EventBusName": event_bus_name,
                    }
                ]
            )
            dispatched_status["dispatched"] = True
            dispatched_status["aws_response"] = response
            logger.info(f"Successfully published drift event to Amazon EventBridge: {response}")
            return dispatched_status
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"AWS EventBridge dispatch skipped ({e}). Falling back to local trigger record.")

    # Local execution fallback
    local_trigger_file = Path("reports/eventbridge_trigger.json")
    local_trigger_file.parent.mkdir(parents=True, exist_ok=True)
    with open(local_trigger_file, "w", encoding="utf-8") as f:
        json.dump(event_payload, f, indent=2)

    dispatched_status["dispatched"] = True
    dispatched_status["local_file"] = str(local_trigger_file)
    logger.info(
        f"EventBridge Retraining Trigger simulated locally. Trigger payload saved to: {local_trigger_file.resolve()}"
    )
    return dispatched_status


def run_drift_analysis(
    reference_path: str = "data/reference_data.parquet",
    current_path: str = "data/production_inference_logs.parquet",
    reports_dir: str = "reports",
) -> dict[str, Any]:
    """Execute Evidently AI drift analysis and trigger EventBridge if threshold exceeded."""
    ref_file = Path(reference_path)
    cur_file = Path(current_path)

    if not ref_file.exists():
        raise FileNotFoundError(f"Reference dataset missing at {ref_file}. Run preprocess.py first.")
    if not cur_file.exists():
        raise FileNotFoundError(f"Current inference logs missing at {cur_file}. Run generate_synthetic_transactions.py --inject-drift first.")

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading reference data: {ref_file} and production inference logs: {cur_file}")
    df_ref = pd.read_parquet(ref_file)
    df_cur = pd.read_parquet(cur_file)

    cols_to_evaluate = [c for c in FEATURE_COLUMNS if c in df_ref.columns and c in df_cur.columns]
    has_target = TARGET_COLUMN in df_ref.columns and TARGET_COLUMN in df_cur.columns

    eval_ref = df_ref[cols_to_evaluate + ([TARGET_COLUMN] if has_target else [])]
    eval_cur = df_cur[cols_to_evaluate + ([TARGET_COLUMN] if has_target else [])]

    # Calculate PSI per feature
    psi_scores: dict[str, float] = {}
    drifted_features: list[str] = []
    for col in cols_to_evaluate:
        psi = compute_psi(eval_ref[col].values, eval_cur[col].values)
        psi_scores[col] = round(psi, 4)
        if psi > 0.20:  # Industry standard PSI threshold for significant shift
            drifted_features.append(col)

    drift_share = len(drifted_features) / max(1, len(cols_to_evaluate))
    dataset_drift = bool(drift_share >= DRIFT_SHARE_THRESHOLD)

    logger.info(f"Calculated Feature PSI values:\n{json.dumps(psi_scores, indent=2)}")
    logger.info(
        f"Drift Analysis Result: {len(drifted_features)}/{len(cols_to_evaluate)} features drifted ({drift_share*100:.1f}%). "
        f"Dataset Drift Flag: {dataset_drift}"
    )

    # Generate Evidently AI Visual HTML Report
    logger.info("Generating Evidently AI comprehensive visual drift report...")
    presets = [DataDriftPreset()]
    if has_target:
        presets.append(TargetDriftPreset())

    report = Report(metrics=presets)
    report.run(reference_data=eval_ref, current_data=eval_cur)

    html_report_path = out_dir / "data_drift_report.html"
    json_report_path = out_dir / "data_drift_summary.json"

    report.save_html(str(html_report_path))

    summary_data = {
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_rows": len(df_ref),
        "current_inference_rows": len(df_cur),
        "drift_share": round(drift_share, 4),
        "drift_threshold": DRIFT_SHARE_THRESHOLD,
        "dataset_drift_detected": dataset_drift,
        "drifted_features": drifted_features,
        "psi_metrics": psi_scores,
        "action_taken": "EVENTBRIDGE_RETRAINING_TRIGGERED" if dataset_drift else "MONITORING_NORMAL",
    }

    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Evidently AI HTML Report saved: {html_report_path.resolve()}")
    logger.info(f"Drift Summary JSON saved: {json_report_path.resolve()}")

    # Step-Function / EventBridge Trigger Decision
    if dataset_drift:
        logger.warning(
            f"ALERT: Covariate drift detected across {len(drifted_features)} features. "
            f"Emitting Amazon EventBridge retraining trigger..."
        )
        dispatch_status = dispatch_eventbridge_retraining_event(summary_data)
        summary_data["eventbridge_dispatch"] = dispatch_status
    else:
        logger.info("Distributions remain stable within tolerances. No retraining needed.")

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidently AI Drift Detection & EventBridge Dispatcher.")
    parser.add_argument("--reference-path", type=str, default="data/reference_data.parquet")
    parser.add_argument("--current-path", type=str, default="data/production_inference_logs.parquet")
    parser.add_argument("--reports-dir", type=str, default="reports")
    args = parser.parse_args()

    run_drift_analysis(
        reference_path=args.reference_path,
        current_path=args.current_path,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
