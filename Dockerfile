FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Create necessary directories with appropriate permissions
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/metrics \
    && chmod -R 777 /app/data /app/models /app/metrics

# Add a non-root user
RUN useradd -m appuser
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports for API and frontend
EXPOSE 8000 8080

# Default command
CMD ["uvicorn", "src.serving.enhanced_api:app", "--host", "0.0.0.0", "--port", "8000"]