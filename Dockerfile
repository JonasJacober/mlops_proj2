# ============================================================
# Dockerfile — DistilBERT GLUE trainer (MRPC by default)
# - Default: CUDA runtime image with PyTorch preinstalled
# - Alternative: CPU-only build (see build args below)
# ============================================================
# ---------- build-time defaults (can be overridden with --build-arg) ----------
ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

FROM ${BASE_IMAGE} AS runtime

ARG APP_USER=app
ARG APP_UID=1000
ARG APP_GID=1000
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Runtime env
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE} \
    PYTHONUNBUFFERED=${PYTHONUNBUFFERED} \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    HF_DATASETS_CACHE=/cache/hf/datasets \
    WANDB_DIR=/cache/wandb

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user & dirs
RUN groupadd -g ${APP_GID} ${APP_USER} \
 && useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER} \
 && mkdir -p /app ${HF_HOME} ${WANDB_DIR} /app/checkpoints \
 && chown -R ${APP_UID}:${APP_GID} /app ${HF_HOME} ${WANDB_DIR}

WORKDIR /app
USER ${APP_USER}

# Python deps
COPY --chown=${APP_UID}:${APP_GID} requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Project files
COPY --chown=${APP_UID}:${APP_GID} . /app

VOLUME ["/cache/hf", "/cache/wandb", "/app/checkpoints"]

ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["python", "main.py", "--task_name", "mrpc", "--epochs", "3", "--devices", "1", "--accelerator", "auto"]
