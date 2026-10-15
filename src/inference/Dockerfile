# ==============================================================================
# Multi-Stage Hardened Production Dockerfile for Financial Risk Inference Service
# Complies with Aqua Security Trivy DevSecOps & Least-Privilege Execution Standards
# ==============================================================================

# Stage 1: Build stage for dependency compilation
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Hardened Runtime Container
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user and group
RUN groupadd -g 10001 mlops && \
    useradd -u 10001 -g mlops -m -s /bin/bash mlopsuser

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/mlopsuser/.local
ENV PATH=/home/mlopsuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source code and models
COPY --chown=mlopsuser:mlops src /app/src
COPY --chown=mlopsuser:mlops models /app/models
COPY --chown=mlopsuser:mlops data /app/data

# Switch to non-root user
USER 10001:10001

EXPOSE 8000

# Docker Healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
