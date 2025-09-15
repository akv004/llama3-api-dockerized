FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/envs/appenv/bin:/opt/conda/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash","-lc"]

# Minimal OS deps for Python/SSL/runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl bzip2 git \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      tzdata && \
    rm -rf /var/lib/apt/lists/*

# Create target env dir
RUN mkdir -p /opt/conda/envs/appenv

# Copy your packed host env and unpack it
COPY appenv.tar.gz /tmp/appenv.tar.gz
RUN cd /opt/conda/envs/appenv && \
    tar -xzf /tmp/appenv.tar.gz && \
    # fix absolute paths in scripts/libs to container paths
    /opt/conda/envs/appenv/bin/conda-unpack || true

# Non-root user
ARG APP_UID=1001
RUN useradd -m -u ${APP_UID} -s /bin/bash appuser

USER appuser
WORKDIR /home/appuser/app

# Now it's safe to chown on copy
COPY --chown=appuser:appuser app/ .

# Expose your API port
EXPOSE 8002

# Launch with your existing module/port
CMD ["python","-m","uvicorn","llama3_api_langchain_rag_redis_sse:app","--host","0.0.0.0","--port","8002","--workers","1"]
