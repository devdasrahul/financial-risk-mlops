"""Synthetic Transaction Generator for Financial Fraud Detection.

Generates realistic payment transaction datasets with authentic fraud dynamics,
imbalanced class distribution (~2-3% fraud), protected demographic attributes for
fairness auditing, and an optional covariate/concept drift injection mode.
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_synthetic_transactions")


def generate_transactions(
    num_samples: int = 15000,
    inject_drift: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic credit card payment transactions.

    Args:
        num_samples: Number of transaction records to generate.
        inject_drift: If True, injects significant covariate and concept drift
            (surges in international transactions, elevated velocity, abnormal amounts).
        random_seed: Seed for reproducibility.

    Returns:
        pd.DataFrame containing the generated transaction dataset.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    customer_ids = [f"CUST_{i:06d}" for i in range(1, max(500, num_samples // 10))]
    age_groups = ["18-25", "26-35", "36-50", "51-65", "65+"]
    age_weights = [0.15, 0.35, 0.30, 0.15, 0.05]

    customer_age_map = {
        cid: np.random.choice(age_groups, p=age_weights) for cid in customer_ids
    }

    records: list[dict[str, Any]] = []

    for i in range(num_samples):
        tx_id = f"TX_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{i:07d}"
        tx_timestamp = start_date + timedelta(seconds=int(i * (30 * 86400 / num_samples)))
        customer_id = random.choice(customer_ids)
        age_group = customer_age_map[customer_id]

        if not inject_drift:
            # Baseline realistic payment behavior
            is_international = 1 if np.random.rand() < 0.05 else 0
            merchant_risk = float(np.clip(np.random.beta(2, 8), 0.01, 0.99))
            device_trust = float(np.clip(np.random.beta(8, 2), 0.01, 1.0))
            account_age_days = int(np.random.exponential(scale=600) + 30)

            # Amounts: Log-normal distribution typical of retail cards
            if is_international:
                amount = float(np.exp(np.random.normal(4.8, 0.8)))
                velocity_1h = int(np.random.poisson(lam=0.8))
                velocity_24h = int(np.random.poisson(lam=2.5))
                distance_km = float(np.random.exponential(scale=850.0))
            else:
                amount = float(np.exp(np.random.normal(3.8, 0.9)))
                velocity_1h = int(np.random.poisson(lam=0.3))
                velocity_24h = int(np.random.poisson(lam=1.5))
                distance_km = float(np.random.exponential(scale=15.0))

            # Realistic ground truth fraud assignment based on syndicated patterns
            # ~2.5% authentic fraud rate with high signal across velocity, merchant risk, device trust, and amount
            is_fraud_candidate = (
                (np.random.rand() < 0.024)
                or (is_international and np.random.rand() < 0.12)
            )

            if is_fraud_candidate:
                is_fraud = 1
                amount = float(np.exp(np.random.normal(5.8, 0.6)))  # Higher fraud amounts ($200 - $800+)
                velocity_1h = int(np.random.randint(2, 6))          # Rapid velocity structuring
                velocity_24h = int(velocity_1h + np.random.randint(2, 7))
                merchant_risk = float(np.clip(np.random.beta(7, 3), 0.55, 0.99))
                device_trust = float(np.clip(np.random.beta(2, 7), 0.05, 0.40))
                distance_km = float(np.random.exponential(scale=450.0))
            else:
                is_fraud = 0
                amount = float(np.exp(np.random.normal(3.7, 0.7)))  # Normal retail amounts ($20 - $90)
                velocity_1h = int(np.random.choice([0, 1], p=[0.85, 0.15]))
                velocity_24h = int(velocity_1h + np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
                merchant_risk = float(np.clip(np.random.beta(2, 8), 0.01, 0.45))
                device_trust = float(np.clip(np.random.beta(8, 2), 0.60, 0.99))
                distance_km = float(np.random.exponential(scale=12.0))

        else:
            # DRIFT INJECTION: Simulating coordinated cross-border syndicate attack
            # Covariate shift: surge in international volume, elevated velocities, abnormal amounts
            is_international = 1 if np.random.rand() < 0.42 else 0  # 42% vs baseline 5%
            merchant_risk = float(np.clip(np.random.beta(5, 4), 0.10, 0.99))
            device_trust = float(np.clip(np.random.beta(4, 5), 0.01, 0.95))
            account_age_days = int(np.random.exponential(scale=180) + 10)  # Newer accounts

            is_fraud_candidate = (
                (np.random.rand() < 0.075)
                or (is_international and np.random.rand() < 0.25)
            )
            if is_fraud_candidate:
                is_fraud = 1
                amount = float(np.exp(np.random.normal(6.4, 0.7)))
                velocity_1h = int(np.random.randint(4, 10))
                velocity_24h = int(velocity_1h + np.random.randint(5, 15))
                distance_km = float(np.random.exponential(scale=2500.0))
            else:
                is_fraud = 0
                amount = float(np.exp(np.random.normal(4.8, 0.9)))
                velocity_1h = int(np.random.poisson(lam=1.5))
                velocity_24h = int(np.random.poisson(lam=3.5))
                distance_km = float(np.random.exponential(scale=45.0))

        # Masked card number for PII compliance testing (simulates real ingest)
        bin_num = random.choice(["411111", "542418", "378282", "601100"])
        last_four = f"{random.randint(1000, 9999)}"
        card_masked = f"{bin_num}******{last_four}"

        records.append({
            "transaction_id": tx_id,
            "timestamp": tx_timestamp.isoformat(),
            "customer_id": customer_id,
            "card_number_masked": card_masked,
            "transaction_amount": round(max(1.0, amount), 2),
            "card_velocity_1h": max(0, velocity_1h),
            "card_velocity_24h": max(velocity_1h, velocity_24h),
            "distance_from_home_km": round(max(0.1, distance_km), 2),
            "merchant_risk_score": round(merchant_risk, 4),
            "is_international": is_international,
            "device_trust_score": round(device_trust, 4),
            "account_age_days": max(1, account_age_days),
            "customer_age_group": age_group,
            "is_fraud": is_fraud,
        })

    df = pd.DataFrame(records)
    fraud_rate = df["is_fraud"].mean() * 100.0
    logger.info(
        f"Generated {len(df):,} transactions (Drift={inject_drift}). "
        f"Fraud instances: {df['is_fraud'].sum():,} ({fraud_rate:.2f}%)"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic financial transaction data.")
    parser.add_argument("--samples", type=int, default=15000, help="Number of samples to generate")
    parser.add_argument("--inject-drift", action="store_true", help="Inject covariate & concept drift")
    parser.add_argument("--output-dir", type=str, default="data", help="Target output directory")
    parser.add_argument("--filename", type=str, default="", help="Optional output filename override")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_transactions(num_samples=args.samples, inject_drift=args.inject_drift)

    if args.filename:
        filename = args.filename
    else:
        filename = "production_inference_logs.parquet" if args.inject_drift else "raw_transactions.parquet"

    target_path = out_dir / filename
    df.to_parquet(target_path, index=False)
    logger.info(f"Dataset successfully written to: {target_path.resolve()}")


if __name__ == "__main__":
    main()
