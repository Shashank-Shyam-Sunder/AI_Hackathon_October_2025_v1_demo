# ====== Minimal Dockerfile for AI Cost Estimator ======
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy everything (scripts, data, requirements, etc.)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements

# Run the pipeline directly when the container starts
CMD ["python", "-m", "src.app.cli.pipeline"]
