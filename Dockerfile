FROM python:3.9-slim-buster

LABEL maintainer="MediNex AI Team <info@medinex.life>"
LABEL description="MediNex AI - Medical Knowledge Assistant"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/data/knowledge \
    /app/data/contributors \
    /app/data/revenue \
    /app/data/models/versions \
    /app/data/models/packages \
    /app/data/models/deployments \
    /app/data/tmp \
    /app/cache \
    /app/logs \
    /app/models/imaging

# Copy requirements first for better caching
COPY ai/requirements.txt /app/ai/requirements.txt
RUN pip install --no-cache-dir -r ai/requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8000

# Set up entrypoint
ENTRYPOINT ["python", "app.py"]

# Default command
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"] 