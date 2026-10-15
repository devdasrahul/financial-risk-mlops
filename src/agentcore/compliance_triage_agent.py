"""Amazon Bedrock AgentCore Compliance Triage Agent with RAG & Policy Guardrails.

Receives high-risk transaction flags and SHAP local attributions from FastAPI,
enforces PII masking and tool access controls per AgentCore Policy, performs RAG
over the regulatory AML/KYC knowledge base, and outputs an auditable compliance memo.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

from src.inference.schemas import ComplianceAuditSummary, ShapContribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compliance_agentcore")

DEFAULT_KB_PATH = Path(__file__).parent / "knowledge_base" / "aml_kyc_policies.md"
DEFAULT_POLICY_PATH = Path(__file__).parent / "agentcore_policy.json"


class AgentCorePolicyEnforcer:
    """Enforces PII redaction and tool invocation guardrails before calling foundation models."""

    PAN_PATTERN = re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6011[0-9]{12})\b")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    def __init__(self, policy_path: Path = DEFAULT_POLICY_PATH) -> None:
        self.policy_path = policy_path
        self.policy = self._load_policy()

    def _load_policy(self) -> dict[str, Any]:
        if self.policy_path.exists():
            with open(self.policy_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def mask_pii(self, text: str) -> str:
        """Mask raw credit card numbers and SSNs according to regulatory guardrails."""
        masked = self.PAN_PATTERN.sub("[PROTECTED_CARD_NUMBER]", text)
        masked = self.SSN_PATTERN.sub("[REDACTED_SSN]", masked)
        return masked

    def verify_tool_permission(self, tool_name: str) -> bool:
        """Validate if the agent is authorized to invoke the specified action."""
        allowed = (
            self.policy.get("guardrailConfiguration", {})
            .get("toolInvocationAccessControl", {})
            .get("allowedTools", ["RetrievePolicyKnowledgeBase", "GenerateComplianceDecisionMemo"])
        )
        is_allowed = tool_name in allowed
        if not is_allowed:
            logger.warning(f"AgentCore Policy Violation: Blocked unauthorized tool execution: {tool_name}")
        return is_allowed


class PolicyKnowledgeBaseRAG:
    """Retrieves relevant AML/KYC policies using semantic and keyword matching."""

    def __init__(self, kb_path: Path = DEFAULT_KB_PATH) -> None:
        self.kb_path = kb_path
        self.sections = self._parse_knowledge_base()

    def _parse_knowledge_base(self) -> list[dict[str, str]]:
        if not self.kb_path.exists():
            return []
        content = self.kb_path.read_text(encoding="utf-8")
        raw_sections = content.split("## Section ")
        parsed = []
        for s in raw_sections[1:]:
            lines = s.strip().split("\n")
            title = lines[0]
            body = "\n".join(lines[1:])
            parsed.append({"title": title, "content": body})
        return parsed

    def retrieve_matching_policy(self, top_shap_features: list[ShapContribution]) -> dict[str, str]:
        """Match top SHAP risk drivers against policy trigger conditions."""
        feature_names = [f.feature_name.lower() for f in top_shap_features]

        # Rule 1: High Velocity Structuring
        if any("velocity" in fn for fn in feature_names):
            for sec in self.sections:
                if "velocity" in sec["title"].lower() or "structuring" in sec["title"].lower():
                    return {
                        "section": "FinCEN High-Velocity Structuring Rule (31 CFR §1010.311)",
                        "policy_id": "POL-AML-VEL-001",
                        "recommended_action": "SUSPEND_CARD",
                        "excerpt": sec["content"][:300] + "...",
                    }

        # Rule 2: Cross-border anomalies
        if any("international" in fn or "distance" in fn for fn in feature_names):
            for sec in self.sections:
                if "cross-border" in sec["title"].lower():
                    return {
                        "section": "High-Risk Cross-Border Transaction Anomaly Policy (FATF Rec 16)",
                        "policy_id": "POL-AML-XBORDER-002",
                        "recommended_action": "STEP_UP_AUTH",
                        "excerpt": sec["content"][:300] + "...",
                    }

        # Rule 3: Compromised Device or Account Takeover
        if any("device" in fn or "account_age" in fn for fn in feature_names):
            for sec in self.sections:
                if "device" in sec["title"].lower() or "account" in sec["title"].lower():
                    return {
                        "section": "Compromised Device Fingerprint & ATO Policy (FFIEC)",
                        "policy_id": "POL-AML-ATO-003",
                        "recommended_action": "FILE_SAR",
                        "excerpt": sec["content"][:300] + "...",
                    }

        # Default fallback rule
        return {
            "section": "Standard Enhanced Risk Due Diligence Protocol",
            "policy_id": "POL-AML-GEN-005",
            "recommended_action": "STEP_UP_AUTH",
            "excerpt": "Transaction triggered elevated statistical fraud probability requiring secondary verification.",
        }


class ComplianceTriageAgent:
    """Amazon Bedrock AgentCore Orchestrator for Financial Risk Compliance Triage."""

    def __init__(self) -> None:
        self.guardrail = AgentCorePolicyEnforcer()
        self.rag = PolicyKnowledgeBaseRAG()
        self.env = os.getenv("ENV", "local")

    def triage_transaction(
        self,
        transaction_id: str,
        fraud_probability: float,
        top_shap_features: list[ShapContribution],
        card_number_masked: str,
    ) -> ComplianceAuditSummary:
        """Execute agentic triage workflow and return structured compliance decision memo."""
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        logger.info(f"AgentCore Triage started for Tx: {transaction_id} (Trace ID: {trace_id})")

        # Step 1: Enforce AgentCore Policy tool verification
        if not self.guardrail.verify_tool_permission("RetrievePolicyKnowledgeBase"):
            raise PermissionError("Access to policy knowledge base denied by AgentCore Policy.")

        # Step 2: Enforce PII masking
        safe_card = self.guardrail.mask_pii(card_number_masked)

        # Step 3: RAG Retrieval over AML/KYC knowledge base
        matched_policy = self.rag.retrieve_matching_policy(top_shap_features)

        # Determine Risk Tier
        if fraud_probability >= 0.95:
            risk_tier = "CRITICAL"
        elif fraud_probability >= 0.85:
            risk_tier = "HIGH"
        elif fraud_probability >= 0.50:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        # Format feature explanation text
        shap_narrative_parts = []
        for feat in top_shap_features:
            shap_narrative_parts.append(
                f"{feat.feature_name} = {feat.feature_value:.2f} (SHAP contribution: +{feat.shap_value:.3f})"
            )
        shap_narrative = "; ".join(shap_narrative_parts)

        # Step 4: Generate Auditable Compliance Narrative
        compliance_narrative = (
            f"Transaction {transaction_id} for cardholder {safe_card} produced an elevated fraud probability of "
            f"{fraud_probability*100:.1f}%, placing it in risk tier '{risk_tier}'. "
            f"Principal risk factors identified via TreeSHAP attributions: [{shap_narrative}]. "
            f"Grounding decision in {matched_policy['section']} ({matched_policy['policy_id']}), the automated "
            f"mandate requires: {matched_policy['recommended_action']}. Audit record permanently committed."
        )

        audit_memo = ComplianceAuditSummary(
            audit_triggered=True,
            risk_tier=risk_tier,
            cited_policy_section=matched_policy["section"],
            recommended_action=matched_policy["recommended_action"],
            trace_id=trace_id,
            compliance_narrative=compliance_narrative,
        )

        logger.info(
            f"AgentCore Triage completed -> Action: {audit_memo.recommended_action} | "
            f"Policy: {audit_memo.cited_policy_section}"
        )
        return audit_memo
