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
    RAG_DIR=/data/rag

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Run as an unprivileged user and persist memory to a mounted volume.
RUN useradd --create-home --uid 10001 privateai \
    && mkdir -p /data/sessions /data/rag \
    && chown -R privateai:privateai /app /data
USER privateai

VOLUME ["/data"]

CMD ["python", "-m", "src"]
