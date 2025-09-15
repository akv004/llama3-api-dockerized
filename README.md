1. Pack your environment on the host

As we have running host conda environment that work with llam3 API, we will pack it and use it in the docker image.

Activate the environment you use for llama3-api:

```bash
conda activate synapse-serve-llama3   # replace with your env name
```


Pack it :
```
conda-pack -n synapse-serve-llama3 -o appenv.tar.gz --ignore-missing-files

```


This creates a self-contained tarball of your environment (appenv.tar.gz).

⚠️ Make sure you run this inside the repo root (where your Dockerfile and app/ are), or move the tarball there afterwards:

```bash
mv appenv.tar.gz /home/amit/projects/PycharmProjects/llama3-api-dockerized

```

2. Dockerfile (Ubuntu 24.04 + your packed env)

Save this as Dockerfile at your repo root:
```Dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/envs/appenv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash","-lc"]

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl bzip2 git \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      tzdata && \
    rm -rf /var/lib/apt/lists/*

# Target env directory
RUN mkdir -p /opt/conda/envs/appenv

# Copy the packed env and unpack
COPY appenv.tar.gz /tmp/appenv.tar.gz
RUN cd /opt/conda/envs/appenv && \
    tar -xzf /tmp/appenv.tar.gz && \
    /opt/conda/envs/appenv/bin/conda-unpack || true

# Use existing user (avoid UID conflicts)
WORKDIR /home/appuser/app
COPY app/ .

EXPOSE 8002
CMD ["python","-m","uvicorn","llama3_api_langchain_rag_redis_sse:app","--host","0.0.0.0","--port","8002","--workers","1"]
```

3. docker-compose.yaml

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: llama3-redis
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis_data:/data
    restart: unless-stopped

  llama3-api:
    build: .
    container_name: llama3-api
    depends_on:
      - redis
    ports:
      - "8002:8002"
    volumes:
      - /home/amit/projects/.cache/huggingface:/home/appuser/.cache/huggingface
      - /home/amit/projects/logs:/home/appuser/app/logs
      - /home/amit/projects/my_rag_data:/home/appuser/app/faiss_index
    environment:
      - REDIS_URL=redis://redis:6379/0
      - BASE_DIR=/home/appuser/app
      - HF_HOME=/home/appuser/.cache/huggingface
      - HUGGINGFACE_HUB_CACHE=/home/appuser/.cache/huggingface/hub
      - FAISS_DIR=/home/appuser/app/faiss_index
      - EMBED_MODEL=BAAI/bge-small-en-v1.5
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    restart: unless-stopped
    gpus: all

```

4. Build and run

Clean up old builds:

```bash
docker compose down --remove-orphans
docker image prune -af
```

build: 

```bash
docker compose build --no-cache
```

run:

```bash
docker compose up -d
```


check logs:

```bash
docker compose logs llama3-api -f
```

health check:

```bash
curl http://127.0.0.1:8002/health
```








