# Multi-stage Dockerfile for Enhanced Multi-Agent PostgreSQL Analysis System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --user-group --uid 1000 multiagent
WORKDIR /app

# Development stage
FROM base as development
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY --chown=multiagent:multiagent src/ src/
COPY --chown=multiagent:multiagent tests/ tests/
COPY --chown=multiagent:multiagent streamlit_app.py .
COPY --chown=multiagent:multiagent .env.template .env

USER multiagent

# Expose ports
EXPOSE 8501 8000 9090

# Default command (can be overridden)
CMD ["python", "src/main.py"]

# Production stage
FROM base as production

# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn uvicorn[standard] \
    && apt-get purge -y --auto-remove gcc \
    && apt-get clean

# Copy application code
COPY --chown=multiagent:multiagent src/ src/
COPY --chown=multiagent:multiagent streamlit_app.py .
COPY --chown=multiagent:multiagent .env.template .env

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; from src.utils.config import Config; print('Health check passed')" || exit 1

USER multiagent

# Expose ports
EXPOSE 8501 8000 9090

# Production command
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]