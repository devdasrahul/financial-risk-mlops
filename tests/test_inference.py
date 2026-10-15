"""Integration Tests for FastAPI Inference Service & Bedrock AgentCore Triage."""

import pytest
from fastapi.testclient import TestClient

from src.agentcore.compliance_triage_agent import (
    AgentCorePolicyEnforcer,
    PolicyKnowledgeBaseRAG,
)
from src.inference.app import app
from src.inference.schemas import ShapContribution


@pytest.fixture
def client() -> TestClient:
    """Initialize FastAPI test client."""
    return TestClient(app)


def test_health_endpoint(client: TestClient):
    """Test /health endpoint returns HTTP 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["HEALTHY", "DEGRADED"]
    assert "timestamp" in data


def test_model_info_endpoint(client: TestClient):
    """Test /model-info returns governance and lineage metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_overview" in data or "model_name" in data


def test_predict_low_risk_transaction(client: TestClient):
    """Test /predict returns low probability and does not trigger compliance audit."""
    payload = {
        "transaction_id": "TX_TEST_LOW_001",
        "customer_id": "CUST_001",
        "card_number_masked": "411111******1234",
        "transaction_amount": 18.50,
        "card_velocity_1h": 0,
        "card_velocity_24h": 1,
        "distance_from_home_km": 3.5,
        "merchant_risk_score": 0.08,
        "is_international": 0,
        "device_trust_score": 0.98,
        "account_age_days": 420,
        "customer_age_group": "36-50",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "TX_TEST_LOW_001"
    assert data["fraud_probability"] < 0.50
    assert data["is_fraud_flagged"] is False
    assert data["risk_tier"] == "LOW"
    assert data["compliance_audit"] is None
    assert len(data["top_shap_contributions"]) > 0


def test_predict_high_risk_triggers_agentcore(client: TestClient):
    """Test /predict with high-risk attributes triggers Bedrock AgentCore compliance audit."""
    payload = {
        "transaction_id": "TX_TEST_HIGH_002",
        "customer_id": "CUST_002",
        "card_number_masked": "542418******9999",
        "transaction_amount": 950.00,
        "card_velocity_1h": 6,
        "card_velocity_24h": 14,
        "distance_from_home_km": 2400.0,
        "merchant_risk_score": 0.95,
        "is_international": 1,
        "device_trust_score": 0.05,
        "account_age_days": 12,
        "customer_age_group": "18-25",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "TX_TEST_HIGH_002"
    assert data["fraud_probability"] >= 0.85
    assert data["is_fraud_flagged"] is True
    assert data["risk_tier"] in ["HIGH", "CRITICAL"]

    # Verify AgentCore compliance audit block is populated
    audit = data["compliance_audit"]
    assert audit is not None
    assert audit["transaction_id"] == "TX_TEST_HIGH_002"
    assert audit["risk_tier"] in ["HIGH", "CRITICAL"]
    assert audit["audit_triggered"] is True
    assert "FinCEN" in audit["cited_policy_section"] or "Cross-Border" in audit["cited_policy_section"]
    assert audit["recommended_action"] in ["SUSPEND_CARD", "STEP_UP_AUTH", "FILE_SAR"]
    assert "trace_id" in audit
    assert audit["shap_explanation"] is not None and len(audit["shap_explanation"]) > 0
    assert len(audit["compliance_narrative"]) > 20


def test_agentcore_pii_masking():
    """Test AgentCore Policy enforcer masks unmasked PANs and SSNs."""
    enforcer = AgentCorePolicyEnforcer()
    raw_text = "Card 4111111111111111 associated with SSN 123-45-6789 had suspicious charges."
    masked = enforcer.mask_pii(raw_text)

    assert "4111111111111111" not in masked
    assert "123-45-6789" not in masked
    assert "[PROTECTED_CARD_NUMBER]" in masked
    assert "[REDACTED_SSN]" in masked


def test_agentcore_rag_policy_retrieval():
    """Test AgentCore RAG accurately retrieves policy section based on SHAP feature."""
    rag = PolicyKnowledgeBaseRAG()
    shap_features = [
        ShapContribution(
            feature_name="card_velocity_1h",
            feature_value=5.0,
            shap_value=2.8,
            description="High velocity burst",
        )
    ]
    matched = rag.retrieve_matching_policy(shap_features)
    assert "FinCEN High-Velocity" in matched["section"]
    assert matched["recommended_action"] == "SUSPEND_CARD"
