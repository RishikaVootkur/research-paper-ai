# ============================================================
# Dockerfile for Research Paper Intelligence Platform
# ============================================================
# This builds a single image that can run either the FastAPI
# backend or the Streamlit frontend (controlled by CMD).
#
# Build: docker build -t research-paper-ai .
# Run API: docker run -p 8000:8000 research-paper-ai api
# Run UI:  docker run -p 8501:8501 research-paper-ai ui

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
# gcc and python3-dev are needed for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create data directories
RUN mkdir -p data/raw_papers data/processed data/ml data/logs models

# Expose ports for both services
EXPOSE 8000 8501

# Default command (can be overridden by docker-compose)
CMD ["python", "demo.py"]