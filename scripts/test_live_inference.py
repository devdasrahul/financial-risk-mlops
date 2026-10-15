"""Live AWS Cloud Verification & Video Demo Script for Financial Risk MLOps.

Queries the deployed Application Load Balancer (ALB) on AWS ECS Fargate
to demonstrate live real-time fraud scoring, local TreeSHAP attributions,
and automated AML/KYC compliance audit triage memos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_ALB_URL = os.getenv(
    "ALB_URL",
    "http://alb-20261015160028850100000007-627408025.us-east-1.elb.amazonaws.com",
)


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_health(base_url: str) -> bool:
    print_header("1. PROBING SERVICE HEALTH (/health)")
    url = f"{base_url.rstrip('/')}/health"
    print(f"Target: {url}")
    try:
        t0 = time.perf_counter()
        req = urllib.request.Request(url, headers={"User-Agent": "MLOps-LiveDemo/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed = (time.perf_counter() - t0) * 1000.0
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Status Code: {resp.status} OK ({elapsed:.2f} ms)")
            print(json.dumps(data, indent=2))
            return data.get("status") == "HEALTHY"
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_normal_transaction(base_url: str) -> None:
    print_header("2. TEST: NORMAL LEGITIMATE TRANSACTION (LOW RISK)")
    url = f"{base_url.rstrip('/')}/predict"
    payload = {
        "transaction_id": "TX_20261015_NORMAL_01",
        "timestamp": "2026-10-15T16:00:00Z",
        "customer_id": "CUST_99014",
        "card_number_masked": "411111******1234",
        "transaction_amount": 45.00,
        "card_velocity_1h": 1,
        "card_velocity_24h": 2,
        "distance_from_home_km": 5.0,
        "merchant_risk_score": 0.10,
        "is_international": 0,
        "device_trust_score": 0.90,
        "account_age_days": 365,
        "customer_age_group": "36-50",
    }
    print("Sending Normal Payload: $45.00 purchase from trusted device near home...")
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "MLOps-LiveDemo/1.0"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed = (time.perf_counter() - t0) * 1000.0
            result = json.loads(resp.read().decode("utf-8"))
            print(f"Roundtrip Latency: {elapsed:.2f} ms | Model Latency: {result.get('latency_ms', 'N/A')} ms")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"FAILED: {e}")


def test_fraud_transaction(base_url: str) -> None:
    print_header("3. TEST: HIGH-RISK SUSPICIOUS FRAUD (SHAP + COMPLIANCE AUDIT)")
    url = f"{base_url.rstrip('/')}/predict"
    payload = {
        "transaction_id": "TX_20261015_FRAUD_99",
        "timestamp": "2026-10-15T16:00:00Z",
        "customer_id": "CUST_004128",
        "card_number_masked": "411111******8821",
        "transaction_amount": 4800.00,
        "card_velocity_1h": 7,
        "card_velocity_24h": 15,
        "distance_from_home_km": 3200.0,
        "merchant_risk_score": 0.92,
        "is_international": 1,
        "device_trust_score": 0.05,
        "account_age_days": 10,
        "customer_age_group": "26-35",
    }
    print("Sending Fraud Attack: $4,800.00, 7 velocity/hr, untrusted device, 3,200 km away...")
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "MLOps-LiveDemo/1.0"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed = (time.perf_counter() - t0) * 1000.0
            result = json.loads(resp.read().decode("utf-8"))
            print(f"Roundtrip Latency: {elapsed:.2f} ms | Model Latency: {result.get('latency_ms', 'N/A')} ms")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"FAILED: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Live AWS MLOps Inference Service")
    parser.add_argument("--url", default=DEFAULT_ALB_URL, help="Base URL of ALB")
    parser.add_argument("--mode", choices=["all", "health", "normal", "fraud"], default="all")
    args = parser.parse_args()

    print(f"\n[+] Connecting to live AWS ALB: {args.url}")

    if args.mode in ("all", "health"):
        check_health(args.url)
    if args.mode in ("all", "normal"):
        test_normal_transaction(args.url)
    if args.mode in ("all", "fraud"):
        test_fraud_transaction(args.url)

    print("\n[+] Live Verification Complete.\n")


if __name__ == "__main__":
    main()
