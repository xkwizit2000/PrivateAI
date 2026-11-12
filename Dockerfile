# --- Stage 1: record the exact commit/branch for the /status report ----------------------
# Reads only HEAD/refs/packed-refs (see .dockerignore exceptions); requirements.txt in the
# COPY keeps it valid when there is no .git at all (then the version is "unknown").
# Optional build args GIT_COMMIT / GIT_BRANCH override what is read from .git.
FROM python:3.12-slim AS version
ARG GIT_COMMIT=""
ARG GIT_BRANCH=""
COPY requirements.txt .git* /gitmeta/
COPY scripts/gen_version.py /gen_version.py
RUN python /gen_version.py /gitmeta /VERSION

# --- Stage 2: the bot image --------------------------------------------------------------
FROM python:3.12-slim

# Node.js provides `npx` for the example filesystem MCP server.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MCP_CONFIG_PATH=/app/mcp.json \
    MEMORY_DIR=/data/sessions \
    RAG_DIR=/data/rag \
    SESSION_SETTINGS_PATH=/data/session_settings.json

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY --from=version /VERSION /app/VERSION

# Run as an unprivileged user and persist memory to a mounted volume.
RUN useradd --create-home --uid 10001 privateai \
    && mkdir -p /data/sessions /data/rag \
    && chown -R privateai:privateai /app /data
USER privateai

VOLUME ["/data"]

CMD ["python", "-m", "src"]
