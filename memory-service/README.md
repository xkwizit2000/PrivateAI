# memory-service (prototype)

External long-term memory for the PrivateAI bot: immutable SQLite archive of every turn, hybrid
(FTS5 keyword + in-process CPU embeddings) retrieval, curated facts, hierarchical dated summaries,
an MCP server with tiny tool schemas, a thin HTTP API and a CLI. **Prototype: not wired into the bot.**
Design, measurements and the bot-side integration plan: [`docs/memory-service-design.md`](../docs/memory-service-design.md).

* Python 3.12 (Docker), `mcp==1.3.0` (same as the bot's client), `fastembed` (ONNX Runtime, no torch),
  default model `snowflake/snowflake-arctic-embed-xs` (384-d, ~87 MB on disk, baked into the image).
* One process = HTTP API **and** MCP over SSE on port 8765. No host port is published in the example compose.

## Run it

```bash
cd memory-service
docker build --target test -t memsvc-test .          # builds + runs the unit tests (synthetic data only)
docker build --target runtime -t privateai-memory .  # service image (model baked in, runtime is offline)

# backfill from bot transcripts (JSONL: {timestamp, role, content}) into ./data/memory.db
mkdir -p data && chmod 700 data
docker run --rm -v "$PWD/data:/data" -v /path/to/sessions:/in:ro privateai-memory \
    python -m memory_service ingest /in

docker run -d --name privateai-memory -v "$PWD/data:/data" -e AUTH_TOKEN=change-me privateai-memory
docker exec privateai-memory python -m memory_service search "what did we decide about the database" -k 3
docker exec privateai-memory python -m memory_service stats
```

(Run as your own uid with `--user $(id -u):$(id -g) -e HOME=/tmp` if the data dir is owned by you; the image's
default user is uid 10002.) See `docker-compose.example.yml` for a hardened service definition (internal network,
resource limits, healthcheck) — it is **not** part of the repo's live `docker-compose.yml`.

### CLI

`python -m memory_service <cmd>`: `serve` (HTTP+MCP/SSE) · `mcp-stdio` · `ingest PATH...` · `search QUERY [-k N --since --until --session --mode hybrid|vector|keyword]` ·
`stats` · `facts [--add STATEMENT --category C]` · `get --turn ID | --date YYYY-MM-DD` · `consolidate [--llm ollama|extractive] [--force]` · `reembed` · `purge TURN_ID...`

### HTTP API (Bearer token if `AUTH_TOKEN` is set; `/healthz` is open)

`POST /v1/search` · `POST /v1/recall` (gated auto-injection) · `POST /v1/turns` (append turns) · `GET /v1/core?session=` ·
`POST /v1/conversation` · `GET|POST /v1/facts` · `GET /v1/stats` · MCP at `/sse` (+ `/messages/`).

### MCP tools (5; ~1.5 KB of schema JSON)

`search_memory(query, k=5, since="", until="", session="")` · `get_conversation(around_turn_id=0, date="", window=3)` ·
`remember_fact(statement, category="note")` · `list_facts(category="")` · `memory_stats()`.
Outputs are hard-capped (search 1,800 chars, conversation 3,000, facts 1,500).

### Configuration (env vars; see `memory_service/config.py`)

`DB_PATH` `EMBED_MODEL` `EMBED_CACHE_DIR` `EMBED_THREADS` `AUTH_TOKEN` `SEARCH_MAX_CHARS` `SEARCH_SNIPPET_CHARS`
`GATE_CUE_MIN_COS` `GATE_STRONG_COS` `IDLE_SECONDS` `OLLAMA_HOST` `OLLAMA_MODEL` `REDACT` ...
Changing `EMBED_MODEL` on an existing DB is refused until you run `reembed` (vectors are model-specific).

## Tests

* `tests/` — 41 unit/integration tests, synthetic data only (`pytest`; they run during `docker build --target test`).
  Includes a real MCP-over-SSE round trip using the **mcp 1.3.0 client** and the tool-schema size budget (< 2,000 chars).
* `integration/bot_hub_check.py` — run inside the *bot's own image*: drives the service through the bot's real
  `src.mcp_client.McpHub` (+ a proposed SSE-capable subclass), checks schema size as the bot sends it, idle keep-alive and
  restart/reconnect behaviour.
* `eval/` — evaluation scripts used for the measurements in the design doc (they read transcripts you point them at;
  nothing from real transcripts is stored in this repo).

## Pinned versions

`requirements.txt` holds the direct pins; `requirements.lock.txt` is the full `pip freeze` that was verified on
`python:3.12-slim`. `mcp==1.3.0` only imports with the pydantic 2.11-era stack (newer pydantic breaks
`mcp.server.fastmcp`), hence `pydantic==2.11.7`, `starlette==0.46.2`, `sse-starlette==2.3.5`.
