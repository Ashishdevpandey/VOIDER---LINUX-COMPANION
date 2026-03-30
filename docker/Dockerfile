# AI OS - Local AI Assistant
# Multi-stage Dockerfile for production deployment

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

LABEL maintainer="AI OS Team"
LABEL description="Local AI Assistant with RAG for Linux"
LABEL version="1.0.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 aios && \
    mkdir -p /app/data /app/logs && \
    chown -R aios:aios /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=aios:aios backend/ ./backend/
COPY --chown=aios:aios cli/ ./cli/
COPY --chown=aios:aios ui/ ./ui/
COPY --chown=aios:aios config/ ./config/
COPY --chown=aios:aios requirements.txt .

# Create necessary directories
RUN mkdir -p data/vector_store logs

# Switch to non-root user
USER aios

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AIOS_HOST=0.0.0.0
ENV AIOS_PORT=8000

# Entry point
ENTRYPOINT ["uvicorn", "backend.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
