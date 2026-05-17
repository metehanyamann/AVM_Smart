# ============================================================
# AVM Smart Track — Production Dockerfile
# FastAPI + ONNX Runtime + OpenCV (headless)
# ============================================================

FROM python:3.11-slim AS base

# System deps for OpenCV-headless, PostgreSQL client, and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- dependency layer (cached unless requirements.txt changes) ----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------- application layer ----------
COPY . .

# Create models directory (ONNX models auto-download on first startup)
RUN mkdir -p /app/backend/infrastructure/models

# Expose the FastAPI port
EXPOSE 8000

# Health check — hit the /api/v1/health/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/health || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "backend.api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info"]
