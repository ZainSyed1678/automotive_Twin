# ==============================================================================
# AI-Powered Automotive Digital Twin: Container Definition
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install curl for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and data assets
COPY model/ ./model/
COPY rag/ ./rag/
COPY casegen/ ./casegen/
COPY agents/ ./agents/
COPY mcp_server/ ./mcp_server/
COPY eval/ ./eval/
COPY app/ ./app/
COPY data/ ./data/
COPY pytest.ini ./

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Healthcheck targeting FastAPI endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command launches FastAPI backend
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
