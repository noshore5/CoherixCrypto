# Use a lightweight base image
FROM python:3.11-slim

# Set environment variables to reduce prompts and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app


# Set working directory
WORKDIR /app

# Install system-level dependencies needed for scientific Python libs
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libfftw3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose port
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
