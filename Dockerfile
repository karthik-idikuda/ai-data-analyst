# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build. Wheels are compiled in the builder and only the installed
# packages are copied forward, so the runtime image carries no compilers.
# The container runs as a non-root user.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt


# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 analyst

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=analyst:analyst core/ ./core/
COPY --chown=analyst:analyst api/ ./api/
COPY --chown=analyst:analyst ui/ ./ui/
COPY --chown=analyst:analyst evals/ ./evals/
COPY --chown=analyst:analyst scripts/ ./scripts/
COPY --chown=analyst:analyst tests/ ./tests/
COPY --chown=analyst:analyst data/ ./data/
COPY --chown=analyst:analyst requirements.txt .env.example ./

USER analyst

# Streamlit by default; docker-compose overrides this for the API service.
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=200"]
