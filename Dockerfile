# ========================
# STAGE 1: Build environment
# ========================
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for pandas/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libatlas-base-dev \
 && rm -rf /var/lib/apt/lists/*

# ========================
# STAGE 2: Install Python deps
# ========================
FROM base AS builder

# Copy only dependency list first (for layer caching)
COPY requirements ./requirements

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements

# ========================
# STAGE 3: Final runtime image
# ========================
FROM base AS runtime

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12

# Copy source code (this layer changes most often)
COPY src ./src
COPY data ./data
COPY docker-compose.yml .
COPY README.md .

# Create output directory
RUN mkdir -p /app/data/out

# Default command: run your pipeline
CMD ["python", "-m", "src.app.cli.pipeline"]
