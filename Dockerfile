# Multi-stage production image for Diabetes Risk Prediction API
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Diabetes Risk Prediction API"
LABEL org.opencontainers.image.description="Clinical decision-support API for Type 2 Diabetes risk stratification (research use only)"
LABEL org.opencontainers.image.version="1.0.0"

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Application code
COPY --chown=appuser:appuser . .

# Ensure expected directories exist
RUN mkdir -p models data/raw data/processed reports/figures reports/metrics logs \
    && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]