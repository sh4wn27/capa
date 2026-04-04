# ── CAPA Python backend ───────────────────────────────────────────────────────
# Multi-stage build:
#   builder  — installs dependencies with uv into a virtual-env
#   runtime  — copies the venv + source, runs uvicorn
#
# Build:
#   docker build -t capa-backend .
#
# Run (dev, mock mode):
#   docker run -p 8000:8000 capa-backend
#
# Run (with a trained checkpoint):
#   docker run -p 8000:8000 \
#     -v /path/to/runs:/app/runs:ro \
#     -v /path/to/data:/app/data:ro \
#     -e CAPA_CHECKPOINT=/app/runs/best/model.pt \
#     capa-backend
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy only the files needed to resolve dependencies first (layer-cache friendly)
COPY pyproject.toml .
COPY README.md .

# Create a virtual environment and install all dependencies (no editable install yet)
RUN uv venv /opt/venv && \
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --no-dev --no-install-project

# Copy source and install the project itself
COPY capa/ capa/
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --no-dev


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash capa

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source (needed for the FastAPI app import path)
COPY --from=builder /build/capa /app/capa
COPY --from=builder /build/pyproject.toml /app/pyproject.toml

# Activate venv by prepending to PATH
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Directories that may be mounted at runtime (model checkpoint, HDF5 cache)
RUN mkdir -p /app/runs /app/data/processed /app/data/raw && \
    chown -R capa:capa /app

USER capa

EXPOSE 8000

# Healthcheck — the /health endpoint is always available (even in mock mode)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# uvicorn with a single worker; for multi-worker deployments override CMD or
# use gunicorn with the uvicorn worker class.
CMD ["uvicorn", "capa.api.predict:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
