# Financial Crime Compliance & AML/KYC Policy Knowledge Base

This policy repository is maintained by the Global Financial Crimes Compliance (FCC) and Anti-Money Laundering (AML) division. It provides binding decision rules for automated transaction triage and Suspicious Activity Report (SAR) filing.

---

## Section 1: FinCEN High-Velocity Structuring Rule (31 CFR §1010.311)

### Policy ID: `POL-AML-VEL-001`
**Regulatory Authority:** Financial Crimes Enforcement Network (FinCEN), USA  
**Applicability:** Retail cards, wire transfers, and electronic payment instruments.

### Trigger Criteria:
- **Velocity Threshold:** $\ge 3$ consecutive transactions within a 1-hour window.
- **Aggregate Volume:** Exceeding $1,000 cumulatively or exhibiting incremental amounts structured below reporting thresholds ($10,000 CTR rule).
- **Risk Indicator:** Rapid bursts of transactions through online or unverified POS terminals.

### Mandated Actions:
1. **Immediate Action:** Temporarily suspend card authorization for online checkout (`SUSPEND_CARD`).
2. **Investigation Protocol:** Route transaction payload and 24-hour ledger history to Level 2 Compliance Investigator.
3. **Regulatory Filing:** If structuring intent is verified within 24 hours, generate and submit Form 111 (Suspicious Activity Report - SAR) under Category: *Structuring / Rapid Movement of Funds*.

---

## Section 2: High-Risk Cross-Border Transaction Anomaly Policy

### Policy ID: `POL-AML-XBORDER-002`
**Regulatory Authority:** Financial Action Task Force (FATF) Recommendation 16 & Office of Foreign Assets Control (OFAC)  
**Applicability:** Transactions where merchant country does not match cardholder domicile or home GPS geolocation.

### Trigger Criteria:
- **International Indicator:** `is_international = 1` AND `distance_from_home_km > 500`.
- **Merchant Risk Factor:** Merchant category risk score $\ge 0.70$ (e.g., cryptocurrency on-ramps, high-value electronics, foreign wire services).
- **Temporal Anomaly:** Impossible travel speed (transaction occurring > 1,000 km from last domestic authorization within 2 hours).

### Mandated Actions:
1. **Immediate Action:** Trigger real-time Out-of-Band Step-Up Authentication (`STEP_UP_AUTH`) via biometric push notification or SMS OTP.
2. **Declined Response:** If authentication challenge fails or is ignored within 180 seconds, decline authorization and place temporary fraud hold.
3. **Escalation:** Flag customer record for Enhanced Due Diligence (EDD) review.

---

## Section 3: Compromised Device Fingerprint & Account Takeover (ATO) Policy

### Policy ID: `POL-AML-ATO-003`
**Regulatory Authority:** Federal Financial Institutions Examination Council (FFIEC) Authentication Guidance  
**Applicability:** E-commerce, mobile application, and virtual terminal payment requests.

### Trigger Criteria:
- **Device Trust Score:** `device_trust_score < 0.30` (indicating newly seen device ID, proxy/VPN routing, emulator detection, or headless browser session).
- **Account Age Anomaly:** `account_age_days < 60` paired with sudden transaction volume surge.
- **Behavioral Shift:** High transaction amount relative to customer's historical 30-day baseline.

### Mandated Actions:
1. **Immediate Action:** Block transaction authorization (`FILE_SAR` / `SUSPEND_CARD`).
2. **Account Security:** Invalidate active session tokens, mandate immediate password reset upon next login.
3. **Audit Trail:** Log device telemetry, client IP, and User-Agent headers to immutable compliance audit ledger.

---

## Section 4: Standard Clean Transaction Protocol (De Minimis Rule)

### Policy ID: `POL-AML-CLEAN-004`
**Applicability:** Routine low-risk consumer retail payments.

### Trigger Criteria:
- Domestic merchant (`is_international = 0`).
- Normal velocity ($\le 1$ transaction in past hour).
- High device trust score ($\ge 0.70$).
- Low merchant risk ($\le 0.35$).

### Mandated Actions:
1. **Action:** Instantaneous straight-through processing (`APPROVE`).
2. **Audit Requirement:** Standard transaction logging to analytical warehouse for periodic batch monitoring.
