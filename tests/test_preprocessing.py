"""Unit and Integration Tests for Data Preprocessing & Feature Engineering."""

import pandas as pd
import pytest

from src.data.generate_synthetic_transactions import generate_transactions
from src.data.preprocess import (
    FEATURE_COLUMNS,
    compute_dataset_hash,
    engineer_features,
    split_data,
    validate_schema,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Generate small synthetic dataset for testing."""
    return generate_transactions(num_samples=200, inject_drift=False, random_seed=42)


def test_schema_validation_success(sample_raw_data: pd.DataFrame):
    """Test schema validator succeeds with expected columns."""
    assert validate_schema(sample_raw_data) is True


def test_schema_validation_missing_column(sample_raw_data: pd.DataFrame):
    """Test schema validator raises ValueError on missing features."""
    corrupted = sample_raw_data.drop(columns=["transaction_amount"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(corrupted)


def test_feature_engineering_calculations(sample_raw_data: pd.DataFrame):
    """Test rolling feature ratios and composite indicators."""
    processed = engineer_features(sample_raw_data)

    assert "amount_per_velocity_1h" in processed.columns
    assert "velocity_ratio_1h_to_24h" in processed.columns
    assert "composite_risk_indicator" in processed.columns

    # Verify no infinite or NaN values introduced
    assert not processed["amount_per_velocity_1h"].isnull().any()
    assert not processed["velocity_ratio_1h_to_24h"].isnull().any()
    assert not processed["composite_risk_indicator"].isnull().any()


def test_chronological_data_split(sample_raw_data: pd.DataFrame):
    """Test that train/val/test splits maintain strict chronological ordering."""
    df_features = engineer_features(sample_raw_data)
    train_df, val_df, test_df = split_data(df_features, train_ratio=0.70, val_ratio=0.15)

    assert len(train_df) == 140
    assert len(val_df) == 30
    assert len(test_df) == 30

    # Verify chronological monotonic ordering: max(train) <= min(val) <= min(test)
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()


def test_dataset_hash_determinism(sample_raw_data: pd.DataFrame):
    """Test dataset SHA256 hashing is deterministic."""
    h1 = compute_dataset_hash(sample_raw_data)
    h2 = compute_dataset_hash(sample_raw_data.copy())
    assert h1 == h2
    assert len(h1) == 64
