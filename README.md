# PrivateAI

PrivateAI is a headless personal assistant for Ubuntu Server. Telegram is the
remote interface, Ollama provides local or remote model inference, and Model
Context Protocol (MCP) servers expose scoped tools.

The bot currently supports:

- Telegram user-ID authorization
- Local or network-accessible Ollama inference
- Ollama tool calling through local stdio MCP servers
- Read-only MCP mode by default
- Durable per-chat transcripts and rolling summaries
- Telegram-safe response chunking and sanitized error messages

## Architecture

```text
Your Telegram client
        │
        ▼
Telegram Bot API
        │
        ▼
PrivateAI bot / agent host
  ├── src/bot.py          authorization and Telegram messaging
  ├── src/agent.py        Ollama reasoning and tool-call loop
  ├── src/mcp_client.py   MCP server lifecycle and tool routing
  └── src/memory.py       transcript and rolling-summary storage
        │                       │
        │ HTTP                  │ stdio
        ▼                       ▼
Ollama model host          Scoped MCP servers
(local or remote GPU)      (filesystem, future tools)
```

The bot and MCP servers can run on a smaller tool host while Ollama runs on a
dedicated GPU server. Your normal interaction with the whole system remains
through Telegram.

## Requirements

- Python 3.10+
- Ubuntu Server 24.04+ recommended
- Ollama with a tool-capable model, such as `gemma4:12b`
- Node.js, npm, and `npx` for the example filesystem MCP server
- A Telegram bot token and your numeric Telegram user ID

## Installation

```bash
git clone https://github.com/xkwizit2000/PrivateAI.git
cd PrivateAI

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Telegram setup

1. Open the official Telegram bot `@BotFather`.
2. Send `/newbot` and follow its prompts.
3. Copy the generated HTTP API token.
4. Open `@userinfobot`, send `/start`, and copy your numeric user ID.

Do not use your Telegram username for `ALLOWED_USER_ID`; the application
compares numeric IDs.

## Environment configuration

Copy the complete example and replace its placeholder values:

```bash
cp .env.example .env
```

Required variables:

```env
TELEGRAM_TOKEN=your_botfather_token
ALLOWED_USER_ID=your_numeric_telegram_id
```

Available variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEGRAM_TOKEN` | none | Telegram Bot API token; required |
| `ALLOWED_USER_ID` | `0` | Only Telegram user permitted to use the bot; required |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Ollama model tag |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Local or remote Ollama API |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | Hard timeout per Ollama chat call |
| `OLLAMA_STUCK_WARN_SECONDS` | `30` | Log a stuck warning while still waiting |
| `OLLAMA_THINK` | `0` | Default model thinking/reasoning (`1`/`0`; overridable with `/think`) |
| `MCP_CONFIG_PATH` | `<project>/mcp.json` | Local MCP server configuration |
| `MAX_TOOL_ITERATIONS` | `8` | Maximum model/tool loops per request |
| `MCP_READONLY` | `1` | Hide tools whose names appear to modify data |
| `MEMORY_DIR` | `<project>/data/sessions` | Durable per-chat memory directory |
| `MEMORY_RECENT_TURNS` | `4` | Recent turns inserted into each model prompt |
| `MEMORY_SUMMARIZE_EVERY` | `6` | Completed turns between summary updates |
| `MEMORY_MAX_CONTEXT_CHARS` | `12000` | Soft memory-text budget per prompt |
| `RAG_ENABLED` | `1` | Retrieve older relevant turns via embeddings |
| `RAG_DIR` | `<project>/data/rag` | Per-session SQLite vector store directory |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Embedding model on `OLLAMA_HOST` |
| `RAG_TOP_K` | `4` | Max retrieved chunks per prompt |
| `RAG_MAX_CONTEXT_CHARS` | `4000` | Soft character budget for retrieved snippets |
| `RAG_MIN_SCORE` | `0.25` | Drop weak cosine similarity matches |

The `.env` file is gitignored and must never be committed.

## Ollama setup

### Ollama on the same machine

```bash
ollama pull gemma4:12b
ollama pull nomic-embed-text
ollama serve
```

Keep the default:

```env
OLLAMA_HOST=http://127.0.0.1:11434
```

Each successful chat call logs performance stats (`prompt_tok/s`, `gen_tok/s`,
durations). While a call is in flight, the bot logs a stuck warning every
`OLLAMA_STUCK_WARN_SECONDS` (default 30s). If the call exceeds
`OLLAMA_TIMEOUT_SECONDS` (default 180s), it fails with a clear timeout error
instead of hanging indefinitely.

Gemma 4 and similar models may "think" before answering, which is slower.
PrivateAI defaults to `OLLAMA_THINK=0` so reasoning mode is off. Set
`OLLAMA_THINK=1` only when you want that behavior as the default.

Per Telegram chat, override it without restarting:

```text
/status
/think status
/think on
/think off
/verbose status
/verbose on
/verbose off
/timeout status
/timeout 600
/timeout reset
```

`/status` lists every per-chat setting (think, verbose, timeout, and any
future ones), plus Ollama host/chat model and currently loaded models from
`GET /api/ps`. Each of `/think`, `/verbose`, and `/timeout` also accepts
`status` (or no args) for that setting alone.

With `/think on` and `/verbose on`, model reasoning streams into the temporary
status message. With `/verbose off`, status stays as the simple progress
indicator.

When enabling `/think on`, raise `/timeout` for that chat so deep reasoning
has enough time. Per-chat choices are stored in `data/session_settings.json`.

### Ollama on a dedicated GPU server

On the GPU server, expose Ollama only on a trusted private network:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
ollama pull gemma4:12b
ollama pull nomic-embed-text
```

On the PrivateAI bot/tool host:

```env
OLLAMA_HOST=http://gpu-server.internal:11434
```

Use a firewall, WireGuard, Tailscale, or an SSH tunnel. Ollama does not provide
an authentication boundary suitable for direct public Internet exposure.

## MCP tool configuration

Copy the example configuration:

```bash
cp mcp.json.example mcp.json
```

Edit its allowed filesystem root:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/absolute/path/the/agent/may/access"
      ]
    }
  }
}
```

At startup, PrivateAI launches each configured MCP server over stdio, discovers
its tools, converts their schemas for Ollama, and routes model tool calls back
to the correct server.

Security notes:

- `mcp.json` is gitignored because it defines the agent's capabilities.
- Use a narrowly scoped directory, never `/`, as the filesystem root.
- `MCP_READONLY=1` is the default and hides tools with write-like names.
- The read-only filter is a safety layer, not a security sandbox.
- Do not add unrestricted shell tools without command allowlists and explicit
  user approval.
- An empty or missing `mcp.json` leaves the bot in chat-only mode.
- The current MCP client supports local stdio servers. Remote HTTP MCP
  transport is not implemented yet.

## Durable memory

Ollama models do not remember previous API requests. PrivateAI therefore keeps
memory in the application layer.

For every completed turn, the agent immediately appends:

```text
data/sessions/<telegram-chat-id>.jsonl
```

After `MEMORY_SUMMARIZE_EVERY` completed turns, a background Ollama request
updates:

```text
data/sessions/<telegram-chat-id>.summary.json
```

Each new prompt receives:

1. The system instructions
2. The compact rolling summary
3. Relevant older turns retrieved by conversation RAG (when enabled)
4. A limited number of recent user/assistant turns
5. The current Telegram message

This gives a small-context model durable continuity without loading the entire
conversation into VRAM. The character budget is an approximation rather than
an exact tokenizer-based context limit.

## Conversation RAG

PrivateAI embeds past conversation turns with an Ollama embedding model (default
`nomic-embed-text` on the same `OLLAMA_HOST` as the chat model) and stores
vectors in per-session SQLite files:

```text
data/rag/<telegram-chat-id>.sqlite3
```

On each request the agent embeds the user message, ranks earlier chunks by
cosine similarity, and injects the top matches (excluding turns already present
in the recent window). After each reply, new turns are indexed in the
background. On startup, existing `data/sessions/*.jsonl` transcripts are
backfilled into the RAG store.

Pull the embedding model on the GPU/Ollama host:

```bash
ollama pull nomic-embed-text
```

Optional `.env` settings:

```env
RAG_ENABLED=1
RAG_DIR=/path/to/privateai/data/rag
RAG_EMBED_MODEL=nomic-embed-text
RAG_TOP_K=4
RAG_MAX_CONTEXT_CHARS=4000
RAG_MIN_SCORE=0.25
```

Set `RAG_ENABLED=0` to disable retrieval and indexing. Document/file corpus
indexing is not implemented yet — RAG currently covers conversation history
only.

Memory and RAG files are gitignored, permissioned for the local user, and may
contain private conversation content. Store them on protected storage and back
up `MEMORY_DIR` and `RAG_DIR` if they must survive host failure.

## Running PrivateAI

Activate the environment and start the bot:

```bash
source .venv/bin/activate
python -m src
```

If Ollama is local, ensure `ollama serve` is already running. For persistent
operation, run PrivateAI under `systemd`, a container supervisor, or `tmux`.

On startup, logs report:

- Whether durable memory is enabled
- Whether conversation RAG is enabled
- Which MCP tools were loaded
- Whether the bot is running in chat-only mode
- Telegram polling status

## Docker

The image bundles Python and Node.js (for `npx`-based MCP servers). Secrets and
tool configuration are provided at runtime rather than baked into the image.

### Docker Compose (recommended)

```bash
cp .env.example .env      # fill in TELEGRAM_TOKEN and ALLOWED_USER_ID
cp mcp.json.example mcp.json
docker compose up -d --build
docker compose logs -f
```

Compose mounts `mcp.json` read-only, stores memory in the `privateai-data`
volume, and defaults `OLLAMA_HOST` to `http://host.docker.internal:11434` so the
container can reach an Ollama server on the Docker host. Override it for a
remote GPU host:

```bash
OLLAMA_HOST=http://gpu-server.internal:11434 docker compose up -d
```

### Plain Docker

```bash
docker build -t privateai:latest .
docker run -d --name privateai \
  --env-file .env \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway \
  -v "$PWD/mcp.json:/app/mcp.json:ro" \
  -v privateai-data:/data \
  privateai:latest
```

### Container notes

- `MCP_CONFIG_PATH` defaults to `/app/mcp.json`, `MEMORY_DIR` to
  `/data/sessions`, and `RAG_DIR` to `/data/rag` inside the image.
- A filesystem MCP root must be a path that exists **inside** the container.
  Mount the host directory (e.g. `-v /srv/work:/srv/work`) and use that path in
  `mcp.json`.
- The container runs as a non-root user; the `/data` volume keeps memory and
  RAG indexes across restarts.

## Testing

Run the current unit tests:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The tests currently cover transcript persistence, recent-turn loading,
summary thresholds, summary loading, session-path validation, and conversation
RAG index/retrieve behavior.

## Project layout

```text
PrivateAI/
├── .env.example          complete environment template
├── mcp.json.example      scoped MCP server template
├── Dockerfile            container image (Python + Node for MCP)
├── docker-compose.yml    compose service with memory volume
├── requirements.txt
├── src/
│   ├── __main__.py       `python -m src` entry point
│   ├── agent.py          Ollama agent/tool loop
│   ├── bot.py            Telegram application
│   ├── config.py         environment configuration
│   ├── mcp_client.py     MCP client hub
│   ├── memory.py         durable session memory
│   └── rag.py            conversation RAG store
└── tests/
    ├── test_memory.py
    └── test_rag.py
```

## Current limitations

- Only one Telegram user ID is authorized.
- MCP connections currently use local stdio transport only.
- There is no confirmation workflow for write-capable tools.
- RAG indexes conversation turns only, not arbitrary workspace documents.
- There is no packaged `systemd` unit yet.
