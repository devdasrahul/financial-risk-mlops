"""Data Preprocessing & Feature Engineering Pipeline.

Transforms raw transactions into production-ready feature sets. Implements
strict chronological train/validation/test splits to avoid temporal data leakage,
computes scaling and velocity indicators, and generates reference datasets
for Evidently AI drift monitoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess")

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

PROTECTED_ATTRIBUTES = ["customer_age_group"]
TARGET_COLUMN = "is_fraud"
IDENTIFIER_COLUMNS = ["transaction_id", "timestamp", "customer_id", "card_number_masked"]


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA256 hash of dataset for lineage and Model Card tracking."""
    data_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate presence and basic types of expected columns."""
    required = set(FEATURE_COLUMNS + PROTECTED_ATTRIBUTES + [TARGET_COLUMN] + IDENTIFIER_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed. Missing required columns: {missing}")

    # Null value verification
    null_counts = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum()
    if null_counts.any():
        logger.warning(f"Detected null values in features:\n{null_counts[null_counts > 0]}")
        # Median imputation for numerical features if any nulls present
        for col in FEATURE_COLUMNS:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    logger.info("Schema and integrity validation passed successfully.")
    return True


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling feature ratios and normalized indicators."""
    processed = df.copy()

    # Amount-to-velocity ratio (structuring anomaly flag)
    processed["amount_per_velocity_1h"] = processed["transaction_amount"] / (
        processed["card_velocity_1h"] + 1.0
    )

    # High velocity ratio indicator
    processed["velocity_ratio_1h_to_24h"] = (processed["card_velocity_1h"] + 0.1) / (
        processed["card_velocity_24h"] + 1.0
    )

    # Risk compound score
    processed["composite_risk_indicator"] = (
        processed["merchant_risk_score"] * (1.0 - processed["device_trust_score"])
    )

    return processed


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform chronological train, validation, and test split.

    Uses chronological sorting to prevent lookahead bias in financial risk modeling.
    """
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
    n = len(df_sorted)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_idx].copy()
    val_df = df_sorted.iloc[train_idx:val_idx].copy()
    test_df = df_sorted.iloc[val_idx:].copy()

    logger.info(
        f"Data split: Train={len(train_df):,} ({train_df[TARGET_COLUMN].mean()*100:.2f}% fraud) | "
        f"Validation={len(val_df):,} ({val_df[TARGET_COLUMN].mean()*100:.2f}% fraud) | "
        f"Test={len(test_df):,} ({test_df[TARGET_COLUMN].mean()*100:.2f}% fraud)"
    )
    return train_df, val_df, test_df


def run_pipeline(
    raw_path: str = "data/raw_transactions.parquet",
    output_dir: str = "data",
) -> dict[str, Any]:
    """Execute end-to-end preprocessing pipeline."""
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {raw_path}. Run generate_synthetic_transactions.py first."
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw dataset from {raw_file}")
    df_raw = pd.read_parquet(raw_file)

    validate_schema(df_raw)
    df_features = engineer_features(df_raw)

    train_df, val_df, test_df = split_data(df_features)

    # Reference data for Evidently AI (baseline feature distribution)
    # The reference dataset matches the train feature distribution used for production model training
    reference_df = train_df.copy()

    train_path = out_path / "train.parquet"
    val_path = out_path / "validation.parquet"
    test_path = out_path / "test.parquet"
    ref_path = out_path / "reference_data.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    reference_df.to_parquet(ref_path, index=False)

    metadata = {
        "dataset_hash": compute_dataset_hash(train_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "reference_rows": len(reference_df),
        "features": FEATURE_COLUMNS + [
            "amount_per_velocity_1h",
            "velocity_ratio_1h_to_24h",
            "composite_risk_indicator",
        ],
        "protected_attributes": PROTECTED_ATTRIBUTES,
        "target": TARGET_COLUMN,
        "created_at": pd.Timestamp.now().isoformat(),
    }

    meta_file = out_path / "dataset_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Preprocessing completed. Metadata saved to {meta_file.resolve()}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess transaction data for MLOps pipeline.")
    parser.add_argument("--input", type=str, default="data/raw_transactions.parquet", help="Path to raw parquet")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    run_pipeline(raw_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
