# Base: slim Python for small image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1

# System deps (optional but useful for scientific stacks)
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency manifests first
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default entrypoint: run training; can be overridden at runtime
ENTRYPOINT ["python", "main.py"]

CMD ["--checkpoint_dir", "models", "--lr", "1e-3", "--epochs", "3"]
