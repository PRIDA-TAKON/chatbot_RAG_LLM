# Use an NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=TRUE
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Install Python (if not already present, or ensure correct version)
# and system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is updated and available for python3.9
RUN python3 -m pip install --upgrade pip

# Install Python dependencies (base)
COPY requirements_base.txt .
RUN python3 -m pip install --no-cache-dir --default-timeout=1000 -r requirements_base.txt

# Install Python dependencies (ML specific)
COPY requirements_ml.txt .
RUN python3 -m pip install --no-cache-dir --default-timeout=1000 -r requirements_ml.txt

# Copy application code and serve script
COPY . $APP_HOME

# Copy the FAISS index
COPY faiss_index $APP_HOME/faiss_index

# Expose the port for the FastAPI application
EXPOSE 8080

# Define the entrypoint script
ENTRYPOINT ["/bin/bash", "-c", "python3 serve.py"]
