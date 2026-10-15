"""Unit Tests for Responsible AI, Demographic Fairness & Model Card Generation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.governance.evaluate_fairness_and_explainability import (
    calculate_demographic_fairness,
    generate_model_card,
)


@pytest.fixture
def mock_evaluation_df() -> pd.DataFrame:
    """Create test dataframe with balanced demographic slices."""
    np.random.seed(42)
    n = 1000
    age_groups = ["18-25", "26-35", "36-50", "51-65", "65+"]
    return pd.DataFrame({
        "customer_age_group": np.random.choice(age_groups, size=n),
        "transaction_amount": np.random.uniform(10, 500, size=n),
    })


def test_fairness_evaluation_passes_on_uniform_predictions(mock_evaluation_df: pd.DataFrame):
    """Test fairness metrics pass when predictions have uniform rates across groups."""
    # Balanced predictions (~3% fraud flag rate across all groups)
    np.random.seed(42)
    predictions = (np.random.rand(len(mock_evaluation_df)) < 0.03).astype(int)

    fairness = calculate_demographic_fairness(mock_evaluation_df, predictions)

    assert fairness["gate_passed"] is True
    assert len(fairness["failure_reasons"]) == 0
    for _grp, metrics in fairness["groups"].items():
        assert metrics["disparate_impact_ratio"] >= 0.80
        assert abs(metrics["statistical_parity_difference"]) <= 0.10


def test_fairness_evaluation_fails_on_biased_predictions(mock_evaluation_df: pd.DataFrame):
    """Test fairness metrics fail when a demographic group is subjected to severe disparity."""
    # Heavily bias group '18-25' to be flagged 70% of the time, while other groups are flagged 1%
    df = mock_evaluation_df.copy()
    predictions = np.zeros(len(df), dtype=int)
    biased_mask = df["customer_age_group"] == "18-25"
    predictions[biased_mask] = (np.random.rand(biased_mask.sum()) < 0.70).astype(int)
    predictions[~biased_mask] = (np.random.rand((~biased_mask).sum()) < 0.01).astype(int)

    fairness = calculate_demographic_fairness(df, predictions)

    assert fairness["gate_passed"] is False
    assert len(fairness["failure_reasons"]) > 0


def test_model_card_generation(tmp_path: Path):
    """Test automated creation of Amazon SageMaker Model Card JSON and Markdown."""
    metrics = {
        "f1_score": 0.88,
        "pr_auc": 0.92,
        "roc_auc": 0.96,
        "precision": 0.86,
        "recall": 0.90,
        "false_positive_rate": 0.02,
    }
    fairness = {
        "gate_passed": True,
        "reference_group": "26-35",
        "groups": {
            "26-35": {
                "sample_count": 500,
                "approval_rate": 0.97,
                "flag_rate": 0.03,
                "disparate_impact_ratio": 1.0,
                "statistical_parity_difference": 0.0,
                "passed": True,
            }
        },
    }
    shap_importance = {
        "card_velocity_1h": 2.45,
        "merchant_risk_score": 1.82,
        "transaction_amount": 0.95,
    }
    metadata = {
        "dataset_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "train_rows": 10500,
        "val_rows": 2250,
    }

    json_path, md_path = generate_model_card(
        metrics=metrics,
        fairness=fairness,
        shap_importance=shap_importance,
        dataset_metadata=metadata,
        output_dir=str(tmp_path),
    )

    assert json_path.exists()
    assert md_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        card_data = json.load(f)

    assert card_data["model_overview"]["model_name"] == "fraud-detection-xgb"
    assert card_data["governance_decision"]["approved_for_production"] is True

    md_text = md_path.read_text(encoding="utf-8")
    assert "Amazon SageMaker Model Card" in md_text
    assert "APPROVED" in md_text
