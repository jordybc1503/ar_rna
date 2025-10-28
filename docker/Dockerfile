FROM python:3.10-slim

LABEL maintainer="jordybc1503@example.com"
LABEL description="Mantaro gap-filling: AR, RNA y Hybrid AR+RNA"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY .env.example .env

# Create data directories
RUN mkdir -p data/raw data/interim data/processed experiments/mlruns experiments/results

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:./experiments/mlruns

# Default command
CMD ["python", "-m", "src.main", "--help"]
