FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl unzip build-essential \
    python3-pip python3-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY app/ /app/app
WORKDIR /app

# Expose FastAPI port
EXPOSE 8080

# Supervisor config for FastAPI (optional, can run directly if desired)
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Default command: run FastAPI via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
